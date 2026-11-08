"""LLM 真实调用集成测试（P4）。

★ 这些测试会**真实消耗 API 额度**，所以：
    · 打 `@pytest.mark.integration`，日常 pytest 默认跳过
    · 每次只发极短请求（max_tokens ≤ 32）
    · 全部用 temperature=0，保证可复现

为什么必须有真实调用测试：
    假替身只能验证「切换逻辑」对不对，
    验证不了「协议实现对不对」——请求体字段名、响应解析路径、
    错误码分类，这些只有真调用才能暴露。
    本文件曾抓出「DeepSeek 返回的实际 model 名与请求不同」这类细节。
"""
from __future__ import annotations

import pytest

from config.settings import Settings
from infrastructure.exceptions import LLMCallError
from infrastructure.llm.fallback_adapter import FallbackLanguageModel
from infrastructure.llm.openai_compatible_adapter import OpenAICompatibleAdapter

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def settings() -> Settings:
    s = Settings()
    if not s.llm_enabled:
        pytest.skip("未配置 LLM Key，跳过真实调用测试")
    return s


def _make(name: str, settings: Settings) -> OpenAICompatibleAdapter:
    """按名字构造真实适配器。"""
    if name == "qwen":
        if not settings.qwen_api_key:
            pytest.skip("未配置 QWEN_API_KEY")
        return OpenAICompatibleAdapter(
            name="qwen", api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url, model=settings.qwen_model,
            timeout_sec=settings.llm_timeout_sec)
    if not settings.deepseek_api_key:
        pytest.skip("未配置 DEEPSEEK_API_KEY")
    return OpenAICompatibleAdapter(
        name="deepseek", api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url, model=settings.deepseek_model,
        timeout_sec=settings.llm_timeout_sec)


# ===================================================================
# 单端点：协议实现正确性
# ===================================================================
class TestProtocol:
    @pytest.mark.parametrize("name", ["qwen", "deepseek"])
    def test_generate_returns_text(self, settings: Settings, name: str) -> None:
        """真实调用能返回非空文本，且 finish_reason 正常。"""
        llm = _make(name, settings)
        result = llm.generate("回复两个字：可用", max_tokens=16)

        assert result.text.strip(), "返回文本不能为空"
        assert result.finish_reason in ("stop", "length")
        assert result.model, "必须回传实际模型名"

    @pytest.mark.parametrize("name", ["qwen", "deepseek"])
    def test_token_usage_reported(self, settings: Settings, name: str) -> None:
        """token 用量要能读到——成本统计依赖它。"""
        llm = _make(name, settings)
        result = llm.generate("你好", max_tokens=8)
        assert result.prompt_tokens > 0
        assert result.completion_tokens >= 0

    @pytest.mark.parametrize("name", ["qwen", "deepseek"])
    def test_system_prompt_respected(self, settings: Settings, name: str) -> None:
        """system_prompt 要真的生效（不是被丢掉）。"""
        llm = _make(name, settings)
        result = llm.generate(
            "回答",
            system_prompt="你只能说「收到」两个字，不要有任何其他内容。",
            max_tokens=16,
        )
        assert "收到" in result.text, f"system_prompt 未生效: {result.text!r}"

    @pytest.mark.parametrize("name", ["qwen", "deepseek"])
    def test_temperature_zero_is_deterministic(self, settings: Settings,
                                               name: str) -> None:
        """temperature=0 时同输入应得到基本相同的结果。

        合规场景要求回答稳定可复现，不能这次拦住了下次又没拦住。
        """
        llm = _make(name, settings)
        a = llm.generate("1+1等于几？只回答数字。", max_tokens=8).text.strip()
        b = llm.generate("1+1等于几？只回答数字。", max_tokens=8).text.strip()
        assert a == b, f"temperature=0 却不稳定: {a!r} vs {b!r}"

    @pytest.mark.parametrize("name", ["qwen", "deepseek"])
    def test_classify_returns_valid_label(self, settings: Settings,
                                          name: str) -> None:
        """分类必须返回候选标签之一或 unknown。"""
        llm = _make(name, settings)
        labels = ["禁忌咨询", "产品功效", "业务流程", "开放问答"]
        label = llm.classify("客户在吃阿司匹林，能吃鱼油吗", labels)
        assert label in labels or label == "unknown", f"返回了非法标签: {label}"

    def test_classify_security_intent_correct(self, settings: Settings) -> None:
        """安全类查询应被分到「禁忌咨询」——P3 的权重表依赖这个。"""
        llm = _make("qwen", settings)
        labels = ["禁忌咨询", "产品功效", "业务流程", "开放问答"]
        label = llm.classify("我爸有高血压在吃降压药，能吃这个鱼油吗", labels)
        assert label == "禁忌咨询", (
            f"安全查询被分到「{label}」，会导致三路召回权重错配"
            f"（禁忌类应提高图谱权重到 0.6）"
        )

    def test_is_available_true_with_key(self, settings: Settings) -> None:
        llm = _make("qwen", settings)
        assert llm.is_available() is True


# ===================================================================
# 真实故障切换
# ===================================================================
class TestLiveFailover:
    def test_invalid_key_fails_over_to_backup(self, settings: Settings) -> None:
        """★ 主端点 Key 无效时，必须自动切到备用并成功返回。

        这是「双重保险」的端到端验证：
        不是测切换逻辑（那个用假替身测过了），
        而是测「真实的 401 会不会被正确分类并触发切换」。
        """
        if not settings.deepseek_api_key:
            pytest.skip("未配置备用 Key，无法验证切换")

        broken = OpenAICompatibleAdapter(
            name="qwen-broken", api_key="sk-invalid-key-for-test",
            base_url=settings.qwen_base_url, model=settings.qwen_model,
            timeout_sec=15)
        backup = _make("deepseek", settings)

        llm = FallbackLanguageModel(
            primary=broken, backups=[backup], primary_retries=0)
        result = llm.generate("回复两个字：可用", max_tokens=16)

        assert result.text.strip(), "切换后应拿到有效回答"
        assert llm.last_used == "deepseek"
        assert llm.last_error and "401" in llm.last_error, (
            "必须记录原始 401，否则排障时不知道是 Key 问题")

        stats = llm.stats()
        assert stats["qwen-broken"]["failed"] == 1
        assert stats["deepseek"]["success"] == 1

    def test_all_invalid_raises(self, settings: Settings) -> None:
        """全部 Key 无效时必须抛异常，不能静默返回空。"""
        bad1 = OpenAICompatibleAdapter(
            name="bad1", api_key="sk-bad-1",
            base_url=settings.qwen_base_url, model=settings.qwen_model,
            timeout_sec=10)
        bad2 = OpenAICompatibleAdapter(
            name="bad2", api_key="sk-bad-2",
            base_url=settings.deepseek_base_url, model=settings.deepseek_model,
            timeout_sec=10)

        llm = FallbackLanguageModel(primary=bad1, backups=[bad2],
                                    primary_retries=0)
        with pytest.raises(LLMCallError) as exc:
            llm.generate("test", max_tokens=8)

        msg = str(exc.value)
        assert "bad1" in msg and "bad2" in msg, "异常应列出所有尝试过的端点"


# ===================================================================
# 装配根：真实端点顺序
# ===================================================================
class TestAssemblyLive:
    def test_container_llm_is_fallback_adapter(self, settings: Settings) -> None:
        from config.assembly import build_container

        c = build_container(settings)
        try:
            assert c.llm is not None
            assert isinstance(c.llm, FallbackLanguageModel)
            assert c.llm.primary.name == "qwen", "主端点应为千问"
            assert [b.name for b in c.llm.backups] == ["deepseek"]
        finally:
            c.close()

    def test_container_llm_actually_works(self, settings: Settings) -> None:
        """从容器拿到的 LLM 必须真的能调通（不是个空壳）。"""
        from config.assembly import build_container

        c = build_container(settings)
        try:
            result = c.llm.generate("回复两个字：可用", max_tokens=16)
            assert result.text.strip()
        finally:
            c.close()


# ===================================================================
# 流式真实调用
# ===================================================================
class TestLiveStream:
    """真实 SSE 流式调用。

    ★ 重点不是「能不能拿到内容」，而是：
      ① SSE 解析正确（帧格式、[DONE] 标记、delta 路径）
      ② 确实是**逐块**到达，不是一次性返回
    """

    @pytest.mark.parametrize("name", ["qwen", "deepseek"])
    def test_stream_yields_multiple_pieces(self, settings: Settings,
                                           name: str) -> None:
        llm = _make(name, settings)
        pieces = list(llm.generate_stream(
            "用两句话介绍保健食品", max_tokens=80))

        assert pieces, "流式应产出内容"
        assert len(pieces) > 1, (
            f"应分多块到达，实际 {len(pieces)} 块。"
            f"若只有 1 块，说明 SSE 没正确解析或服务端没开流式。"
        )
        assert "".join(pieces).strip()

    def test_stream_first_piece_faster_than_full(self,
                                                 settings: Settings) -> None:
        """★ 首块应明显早于全文完成（这是流式的全部意义）。"""
        import time

        llm = _make("qwen", settings)
        t0 = time.perf_counter()
        first_at = None
        n = 0
        for _ in llm.generate_stream(
                "用三句话说明保健食品的批准功能", max_tokens=200):
            if first_at is None:
                first_at = time.perf_counter() - t0
            n += 1
        total = time.perf_counter() - t0

        assert first_at is not None
        assert n > 1
        assert first_at < total * 0.8, (
            f"首块 {first_at:.2f}s，总 {total:.2f}s —— 差距太小，"
            f"流式没起作用"
        )

    def test_stream_fallback_on_bad_key(self, settings: Settings) -> None:
        """主端点 Key 无效时，流式也要能切到备用。"""
        if not settings.deepseek_api_key:
            pytest.skip("未配置备用 Key")

        broken = OpenAICompatibleAdapter(
            name="qwen-broken", api_key="sk-invalid",
            base_url=settings.qwen_base_url, model=settings.qwen_model,
            timeout_sec=15)
        backup = _make("deepseek", settings)

        llm = FallbackLanguageModel(primary=broken, backups=[backup])
        pieces = list(llm.generate_stream("回复两个字：可用", max_tokens=16))

        assert "".join(pieces).strip()
        assert llm.last_used == "deepseek"

    def test_stream_all_bad_raises(self, settings: Settings) -> None:
        """全部端点流式失败必须抛异常，不能静默产出空流。"""
        bad1 = OpenAICompatibleAdapter(
            name="bad1", api_key="sk-bad-1",
            base_url=settings.qwen_base_url, model=settings.qwen_model,
            timeout_sec=10)
        llm = FallbackLanguageModel(primary=bad1, backups=[])
        with pytest.raises(LLMCallError):
            list(llm.generate_stream("test", max_tokens=8))
