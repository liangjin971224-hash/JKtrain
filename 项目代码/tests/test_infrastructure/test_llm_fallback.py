"""LLM 主备双通道测试（P4）。

★ 这个文件的核心不是「能调通」，而是「**什么情况算异常、什么时候该切**」。

双重保险最容易做错的地方：
    · 该切的时候没切（千问挂了还在重试，用户等 30 秒）
    · 不该切的时候切了（用户问题本身有问题，切过去也一样失败，反而慢一倍）
    · 切了但没记录（出故障不知道切过，排障时白找）

所以这里重点测**分类**：哪些异常该切、哪些不该切。
"""
from __future__ import annotations

import pytest

from domain.ports import GenerationResult, LanguageModelPort
from infrastructure.exceptions import LLMCallError, LLMResponseFormatError


# ===================================================================
# 测试替身：可控行为的假端点
# ===================================================================
class _StubLLM(LanguageModelPort):
    """可指定「调用结果」的假模型。

    用于精确模拟：成功 / 超时 / 限流 / 5xx / 返回格式错。
    """

    def __init__(self, name: str, behaviour: str = "ok",
                 reply: str = "") -> None:
        """
        Args:
            name: 端点名（qwen / deepseek）。
            behaviour: ok | timeout | rate_limit | server_error |
                       bad_response | auth_error
            reply: ok 时的返回文本。
        """
        self.name = name
        self.behaviour = behaviour
        self.reply = reply or f"来自 {name} 的回答"
        self.calls: list[str] = []          # 记录被调用的方法

    def generate(self, prompt: str, *, system_prompt: str = "",
                 temperature: float = 0.0,
                 max_tokens: int = 1024) -> GenerationResult:
        self.calls.append("generate")
        b = self.behaviour
        if b == "ok":
            return GenerationResult(text=self.reply, model=self.name,
                                    finish_reason="stop")
        if b == "timeout":
            raise LLMCallError("请求超时", model=self.name)
        if b == "rate_limit":
            raise LLMCallError("429 Too Many Requests", model=self.name)
        if b == "server_error":
            raise LLMCallError("503 Service Unavailable", model=self.name)
        if b == "auth_error":
            raise LLMCallError("401 Unauthorized（Key 无效）", model=self.name)
        if b == "bad_response":
            raise LLMResponseFormatError("返回体不符合 OpenAI 协议")
        raise AssertionError(f"未知 behaviour: {b}")

    def classify(self, text: str, labels: list[str], *,
                 allow_unknown: bool = True) -> str:
        self.calls.append("classify")
        if self.behaviour != "ok":
            # 复用 generate 的异常逻辑
            self.generate("x")
        return labels[0] if labels else "unknown"

    def is_available(self) -> bool:
        return self.behaviour == "ok"

    def generate_stream(self, prompt: str, *, system_prompt: str = "",
                        temperature: float = 0.0,
                        max_tokens: int = 1024):
        """流式：复用 generate 的异常逻辑，成功后按 3 字切片产出。

        切片是为了模拟真实 SSE 的「一次只来几个字」。
        """
        self.calls.append("generate_stream")
        result = self.generate(
            prompt, system_prompt=system_prompt,
            temperature=temperature, max_tokens=max_tokens)
        text = result.text
        for i in range(0, len(text), 3):
            yield text[i:i + 3]


# ===================================================================
# 1. 正常路径
# ===================================================================
class TestPrimaryWorks:
    def test_uses_primary_only(self) -> None:
        """主模型正常时，不应调用备用。"""
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        primary = _StubLLM("qwen", "ok", "千问的回答")
        backup = _StubLLM("deepseek", "ok", "DS 的回答")

        llm = FallbackLanguageModel(primary=primary, backups=[backup])
        result = llm.generate("问题")

        assert result.text == "千问的回答"
        assert result.model == "qwen"
        assert backup.calls == [], "主模型正常时不该调备用"
        assert llm.last_used == "qwen"

    def test_classify_uses_primary(self) -> None:
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        primary = _StubLLM("qwen")
        backup = _StubLLM("deepseek")
        llm = FallbackLanguageModel(primary=primary, backups=[backup])

        label = llm.classify("文本", ["A", "B"])
        assert label == "A"
        assert backup.calls == []


# ===================================================================
# 2. ★ 什么情况该切换
# ===================================================================
class TestWhenToFailover:
    """主模型异常时必须切到备用。"""

    @pytest.mark.parametrize("behaviour", [
        "timeout", "rate_limit", "server_error", "auth_error", "bad_response",
    ])
    def test_failover_on_primary_failure(self, behaviour: str) -> None:
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        primary = _StubLLM("qwen", behaviour)
        backup = _StubLLM("deepseek", "ok", "DS 兜底的回答")

        llm = FallbackLanguageModel(primary=primary, backups=[backup],
                                    primary_retries=0)
        result = llm.generate("问题")

        assert result.text == "DS 兜底的回答", (
            f"主模型 {behaviour} 时应切到备用，实际返回 {result.text!r}")
        assert result.model == "deepseek"
        assert llm.last_used == "deepseek"
        assert llm.last_error, "切换时必须记录原始错误，便于排障"

    def test_all_endpoints_fail_raises(self) -> None:
        """全部失败时必须抛异常，不能返回空字符串。

        返回空串会被上层当成「模型说没内容」，掩盖了真实故障。
        """
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        llm = FallbackLanguageModel(
            primary=_StubLLM("qwen", "timeout"),
            backups=[_StubLLM("deepseek", "server_error")],
            primary_retries=0,
        )
        with pytest.raises(LLMCallError) as exc:
            llm.generate("问题")

        msg = str(exc.value)
        assert "qwen" in msg and "deepseek" in msg, (
            "异常信息必须列出尝试过哪些端点，否则排障不知道挂了几个")

    def test_retries_before_failover(self) -> None:
        """配置了重试时，应先重试主模型再切。"""
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        primary = _StubLLM("qwen", "timeout")
        backup = _StubLLM("deepseek", "ok")

        llm = FallbackLanguageModel(primary=primary, backups=[backup],
                                    primary_retries=2)
        llm.generate("问题")

        # 1 次首次 + 2 次重试 = 3 次
        assert len(primary.calls) == 3, (
            f"重试 2 次应共调 3 次，实际 {len(primary.calls)} 次")

    def test_zero_retries_switches_immediately(self) -> None:
        """重试 0 次时，主模型失败一次就切（最快响应）。"""
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        primary = _StubLLM("qwen", "timeout")
        backup = _StubLLM("deepseek", "ok")

        llm = FallbackLanguageModel(primary=primary, backups=[backup],
                                    primary_retries=0)
        llm.generate("问题")
        assert len(primary.calls) == 1

    def test_partial_retry_success_no_failover(self) -> None:
        """重试成功后不应切换——避免不必要的模型切换导致回答风格突变。"""
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        class _FlakyThenOK(_StubLLM):
            def __init__(self) -> None:
                super().__init__("qwen", "timeout")
                self._n = 0

            def generate(self, prompt: str, **kw) -> GenerationResult:
                self.calls.append("generate")
                self._n += 1
                if self._n == 1:
                    raise LLMCallError("首次超时", model="qwen")
                return GenerationResult(text="重试成功的回答", model="qwen")

        primary = _FlakyThenOK()
        backup = _StubLLM("deepseek", "ok")

        llm = FallbackLanguageModel(primary=primary, backups=[backup],
                                    primary_retries=1)
        result = llm.generate("问题")

        assert result.text == "重试成功的回答"
        assert backup.calls == [], "重试成功就不该切备用"


# ===================================================================
# 3. 备用链
# ===================================================================
class TestBackupChain:
    def test_multiple_backups_in_order(self) -> None:
        """多个备用应按顺序尝试，前一个成功就停。"""
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        llm = FallbackLanguageModel(
            primary=_StubLLM("qwen", "timeout"),
            backups=[_StubLLM("deepseek", "server_error"),
                     _StubLLM("local", "ok", "本地的回答")],
            primary_retries=0,
        )
        result = llm.generate("问题")
        assert result.text == "本地的回答"

    def test_no_backup_raises_immediately(self) -> None:
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        llm = FallbackLanguageModel(
            primary=_StubLLM("qwen", "timeout"), backups=[],
            primary_retries=0)
        with pytest.raises(LLMCallError):
            llm.generate("问题")

    def test_is_available_reflects_any_endpoint(self) -> None:
        """任一端点可用即视为可用。"""
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        llm = FallbackLanguageModel(
            primary=_StubLLM("qwen", "auth_error"),
            backups=[_StubLLM("deepseek", "ok")],
        )
        assert llm.is_available() is True

        dead = FallbackLanguageModel(
            primary=_StubLLM("qwen", "auth_error"),
            backups=[_StubLLM("deepseek", "auth_error")],
        )
        assert dead.is_available() is False


# ===================================================================
# 4. 可观测性
# ===================================================================
class TestObservability:
    """切换必须留痕，否则故障排查时不知道发生过什么。"""

    def test_records_used_endpoint(self) -> None:
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        llm = FallbackLanguageModel(
            primary=_StubLLM("qwen", "ok"), backups=[])
        llm.generate("x")
        assert llm.last_used == "qwen"

    def test_records_last_error_on_failover(self) -> None:
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        llm = FallbackLanguageModel(
            primary=_StubLLM("qwen", "rate_limit"),
            backups=[_StubLLM("deepseek", "ok")], primary_retries=0)
        llm.generate("x")
        assert "429" in llm.last_error

    def test_stats_counts_calls(self) -> None:
        """统计每个端点被调次数，用于监控主备比例。"""
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        llm = FallbackLanguageModel(
            primary=_StubLLM("qwen", "ok"), backups=[_StubLLM("deepseek", "ok")])
        llm.generate("a")
        llm.generate("b")

        stats = llm.stats()
        assert stats["qwen"]["success"] == 2
        assert stats["deepseek"]["success"] == 0

    def test_stats_counts_failures(self) -> None:
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        llm = FallbackLanguageModel(
            primary=_StubLLM("qwen", "timeout"),
            backups=[_StubLLM("deepseek", "ok")], primary_retries=0)
        llm.generate("x")

        stats = llm.stats()
        assert stats["qwen"]["failed"] == 1
        assert stats["deepseek"]["success"] == 1


# ===================================================================
# 5. 不该被切换「吞掉」的情况
# ===================================================================
class TestDoNotSwallow:
    def test_original_error_preserved(self) -> None:
        """全部失败时，异常里必须能看到每个端点的原始错误。

        否则只报「LLM 调用失败」，排障时不知道是 Key 问题还是网络问题。
        """
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel

        llm = FallbackLanguageModel(
            primary=_StubLLM("qwen", "auth_error"),
            backups=[_StubLLM("deepseek", "timeout")],
            primary_retries=0,
        )
        with pytest.raises(LLMCallError) as exc:
            llm.generate("x")

        msg = str(exc.value)
        assert "401" in msg, "应包含千问的鉴权错误"
        assert "超时" in msg, "应包含 DeepSeek 的超时错误"
