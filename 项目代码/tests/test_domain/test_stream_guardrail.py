"""流式输出下的护栏（P4 优化阶段）。

★ 这个文件解决的是一个**架构冲突**，不是性能优化。

冲突：
    流式 = 边生成边展示（首字 0.5s，解决 P95 达标问题）
    护栏 = 输出必经节点（约束 #1，违规内容不得输出）

    天真做法（边收边 emit）会让违规内容在检测前就展示给用户，
    护栏形同虚设 —— 直接违反约束 #1。

解法：**分句缓冲 + 逐句过护栏 + 违规即截断**。

    LLM token 流
        ↓  缓冲到句末标点（。！？换行）
    一整句文本
        ↓  过 OutputGuardrail
    通过 → emit       不通过 → 截断 + 转拒答，后续全部丢弃

    已 emit 的每一句都通过了护栏，所以「输出必经护栏」仍然成立。
    代价：首字延迟从 0.1s 变成「第一句生成完」，实测仍在 0.5–1s。

★ 为什么按句而不是按 token 检测：
    单看一个 token 永远看不全禁用词（「降」「血」「脂」分开都不违规）。
    按句是能保证「检测单元语义完整」的最小粒度。
"""
from __future__ import annotations

import pytest

from domain.services.output_guardrail import OutputGuardrail
from domain.services.stream_guard import SentenceBuffer, StreamGuard


# ===================================================================
# 1. 分句缓冲
# ===================================================================
class TestSentenceBuffer:
    """把 token 流拼成完整句子。"""

    def test_splits_on_period(self) -> None:
        buf = SentenceBuffer()
        out = buf.feed("第一句话。第二句")
        assert out == ["第一句话。"]          # 第二句未完结，留在缓冲
        assert buf.flush() == ["第二句"]

    def test_splits_on_multiple_punctuation(self) -> None:
        buf = SentenceBuffer()
        out = buf.feed("A。B！C？D")
        assert out == ["A。", "B！", "C？"]
        assert buf.flush() == ["D"]

    def test_splits_on_newline(self) -> None:
        """markdown 常见：列表项之间用换行，每行是一句。"""
        buf = SentenceBuffer()
        out = buf.feed("- 第一项\n- 第二项\n- 第三")
        assert out == ["- 第一项\n", "- 第二项\n"]
        assert buf.flush() == ["- 第三"]

    def test_keeps_punctuation(self) -> None:
        """标点必须保留——影响可读性，也可能影响规则匹配。"""
        buf = SentenceBuffer()
        assert buf.feed("你好。") == ["你好。"]

    def test_token_by_token_feeding(self) -> None:
        """模拟真实 SSE：一次只来几个字。"""
        buf = SentenceBuffer()
        tokens = ["保", "健", "食", "品", "不", "是", "药", "物", "。",
                  "不能", "代替", "药物", "。"]
        out: list[str] = []
        for t in tokens:
            out.extend(buf.feed(t))
        out.extend(buf.flush())
        assert out == ["保健食品不是药物。", "不能代替药物。"]

    def test_empty_input(self) -> None:
        buf = SentenceBuffer()
        assert buf.feed("") == []
        assert buf.flush() == []

    def test_flush_idempotent(self) -> None:
        """flush 后缓冲应清空，再 flush 返回空。"""
        buf = SentenceBuffer()
        buf.feed("残句")
        assert buf.flush() == ["残句"]
        assert buf.flush() == []

    def test_long_sentence_force_flush(self) -> None:
        """★ 超长句子必须强制切分。

        实测风险：LLM 可能生成一个 800 字不带句号的段落，
        若一直缓冲不 emit，用户会长时间看不到任何内容，
        流式就退化成「等更久」。
        """
        buf = SentenceBuffer(max_buffer_chars=50)
        out = buf.feed("啊" * 50)
        assert out, "超过 max_buffer_chars 必须强制吐出，不能一直憋着"
        assert "".join(out) == "啊" * 50, "强制吐出的内容不能丢"
        assert buf.pending == "", "吐出后缓冲应清空"

    def test_partial_force_flush(self) -> None:
        """喂 120 字（上限 50）应分多次吐出，不丢内容。"""
        buf = SentenceBuffer(max_buffer_chars=50)
        out = buf.feed("啊" * 120)
        assert len(out) >= 2, f"应至少切两段，实际 {len(out)} 段"
        assert "".join(out) + buf.pending == "啊" * 120


# ===================================================================
# 2. 流式护栏 ★ 核心
# ===================================================================
class TestStreamGuard:
    """分句过护栏才 emit。"""

    @pytest.fixture()
    def guard(self) -> StreamGuard:
        return StreamGuard(guardrail=OutputGuardrail())

    def test_clean_stream_all_emitted(self, guard: StreamGuard) -> None:
        """合规内容全部放行，顺序不变。"""
        text = (
            "这款鱼油有助于维持血脂健康水平。"
            "保健食品不是药物，不能代替药物治疗疾病。"
        )
        out = list(guard.filter(text))
        joined = "".join(out)
        assert "有助于维持血脂" in joined
        assert "不能代替药物治疗疾病" in joined

    def test_violation_truncates_stream(self, guard: StreamGuard) -> None:
        """★ 出现违规即截断，后续内容一律不输出。

        这是最关键的行为：一旦有违规，后面再合规也不能补回来——
        因为用户已经看到前文，拼起来可能构成违规。
        """
        text = (
            "这款鱼油是保健食品。"
            "它能降血脂，很多人吃了都说好。"      # ← 违规句
            "保健食品不是药物，不能代替药物治疗疾病。"  # ← 合规，但不能再输出
        )
        out = list(guard.filter(text))
        joined = "".join(out)

        assert "降血脂" not in joined, "违规内容被输出了"
        assert "很多人吃了都说好" not in joined
        assert "不能代替药物治疗疾病" not in joined, (
            "截断后不应继续输出后续内容"
        )
        assert guard.blocked is True
        assert guard.block_reason

    def test_first_sentence_clean_second_violating(self, guard: StreamGuard) -> None:
        """第一句合规可以 emit，第二句违规时截断。

        这是分句缓冲的价值：合规部分能尽早给用户，不用全等。
        """
        text = "这款鱼油是保健食品。它能治疗糖尿病。保健食品不是药物。"
        out = list(guard.filter(text))
        joined = "".join(out)

        assert "这款鱼油是保健食品。" in joined, "合规的第一句应已输出"
        assert "治疗糖尿病" not in joined
        assert guard.blocked is True

    def test_ordinary_food_claim_truncates(self) -> None:
        """普通食品功能声称也要拦（约束 #2）。"""
        from domain.value_objects import ProductCategory

        guard = StreamGuard(guardrail=OutputGuardrail(),
                            category=ProductCategory.ORDINARY_FOOD)
        out = list(guard.filter("每份含蛋白质 12g。这款有助于增强免疫力。"))
        joined = "".join(out)

        assert "每份含蛋白质" in joined
        assert "增强免疫力" not in joined
        assert guard.blocked is True

    def test_missing_warning_not_blocking_stream(self) -> None:
        """★ 缺警示语**不能**在流式过程中拦截。

        原因：回答是逐句生成的，警示语通常在最后一句。
        若每句都要求含警示语，第一句就会被拦 —— 全错。

        处理：缺警示语只能在**全流结束后**判定（见 test_final_check）。
        """
        from domain.value_objects import ProductCategory

        guard = StreamGuard(guardrail=OutputGuardrail(),
                            category=ProductCategory.HEALTH_FOOD)
        out = list(guard.filter("这款鱼油有助于维持血脂健康水平。"))
        joined = "".join(out)

        assert "有助于维持血脂" in joined, (
            "首句不含警示语是正常的，不能因此拦截"
        )
        assert guard.blocked is False
        # 但最终检查要能发现缺警示语
        assert guard.final_verdict is not None
        assert guard.final_verdict.blocked is True, (
            "全流结束后必须发现缺警示语"
        )

    def test_accumulates_full_text(self, guard: StreamGuard) -> None:
        """要保留完整文本，供最终检查与日志使用。"""
        text = "第一句。第二句。"
        list(guard.filter(text))
        assert guard.full_text == text

    def test_reset_between_calls(self, guard: StreamGuard) -> None:
        """同一个 StreamGuard 可复用（避免每次请求都建对象）。"""
        list(guard.filter("这款鱼油是保健食品。"))
        assert not guard.blocked
        guard.reset()

        list(guard.filter("这个能降血糖。"))
        assert guard.blocked

    def test_block_reason_is_readable(self, guard: StreamGuard) -> None:
        """拦截原因要能读懂，便于前端展示与排障。"""
        list(guard.filter("这个能降血糖。"))
        assert guard.block_reason
        assert "禁用" in guard.block_reason or "违规" in guard.block_reason


# ===================================================================
# 2b. ★ 逗号切分不能破坏跨段规则（真机踩坑回归）
# ===================================================================
class TestContextTailAcrossChunks:
    """在逗号处提前切分后，否定性陈述不能被误拦。

    ★ 这是真机跑出来的回归：
        为降低首块延迟，允许在逗号处切分。
        「它不用于降血脂治疗，」被切在逗号后，
        若切成「它不用于」+「降血脂治疗，」，后半段单独检测就是违规命中
        —— 而原句是完全合规的边界说明。

    修法：检测时带上前文 20 字作为上下文，
    让「不用于 + 降血脂」的紧邻关系恢复。
    """

    @pytest.mark.parametrize("text", [
        "它不用于降血脂治疗，也不具有护心作用。",
        "该产品不具备治疗作用，不能代替药物治疗疾病。",
        "其功能不包括降血糖，仅限于维持血脂健康水平。",
        "不能声称治疗疾病，只能说明批准功能。",
    ])
    def test_negated_claim_not_blocked_across_chunks(self, text: str) -> None:
        guard = StreamGuard(guardrail=OutputGuardrail())
        # 逐字喂，强制在每个可能的逗号处切分
        out = "".join(guard.filter(iter(text)))
        assert not guard.blocked, (
            f"合规的否定性陈述被误拦: {text}\n"
            f"原因: {guard.block_reason}"
        )
        assert "不用于" in out or "不具备" in out or "不包括" in out \
            or "不能声称" in out

    def test_real_violation_still_blocked_across_chunks(self) -> None:
        """带上下文不能变成「什么都放行」。"""
        guard = StreamGuard(guardrail=OutputGuardrail())
        out = "".join(guard.filter(iter("这款鱼油是保健食品，它能降血脂，效果很好。")))
        assert "降血脂" not in out
        assert guard.blocked is True

    def test_context_does_not_leak_into_output(self) -> None:
        """上下文只用于检测，不能重复拼进输出。"""
        text = "第一句话。它不用于治疗高血脂症。第三句话。"
        guard = StreamGuard(guardrail=OutputGuardrail())
        out = "".join(guard.filter(iter(text)))
        # 输出里每句只出现一次（没被上下文重复）
        assert out.count("第一句话") == 1
        assert out.count("不用于治疗") == 1

    def test_first_chunk_arrives_before_full_sentence(self) -> None:
        """★ 首块应在第一整句完成之前就到达（这是降延迟的关键）。"""
        # 一句很长的话，中间有逗号
        long_sentence = (
            "康源牌鱼油软胶囊是保健食品，它的批准功能是有助于维持血脂健康水平，"
            "适宜人群是血脂偏高者，不适宜人群包括出血倾向者。"
        )
        guard = StreamGuard(guardrail=OutputGuardrail())

        first_chunk_at = None
        for i, ch in enumerate(long_sentence):
            for _ in guard.filter(iter(ch)):
                if first_chunk_at is None:
                    first_chunk_at = i

        assert first_chunk_at is not None
        assert first_chunk_at < len(long_sentence) // 2, (
            f"首块在第 {first_chunk_at} 字才到（共 {len(long_sentence)} 字），"
            f"说明逗号切分没生效，首块延迟仍然很高"
        )


# ===================================================================
# 3. 端到端：模拟 stream → guard → emit
# ===================================================================
class TestStreamingPipeline:
    def test_first_token_latency_improved(self) -> None:
        """★ 分句缓冲后，首句应在第一句完成时立即吐出。

        这验证「流式确实比等全文快」——不是看总耗时，
        而是看「第一段内容到达时间」远早于「全部完成时间」。
        """
        sentence1 = "这款鱼油是保健食品，批准功能是有助于维持血脂健康水平。"
        sentence2 = "保健食品不是药物，不能代替药物治疗疾病。"

        guard = StreamGuard(guardrail=OutputGuardrail())
        arrivals: list[tuple[int, str]] = []

        # 模拟：先来第一句的 token，再停顿，再来第二句
        for i, ch in enumerate(sentence1):
            for chunk in guard.filter(ch):
                arrivals.append((i, chunk))
        first_batch_end = len(arrivals)

        for i, ch in enumerate(sentence2):
            for chunk in guard.filter(ch):
                arrivals.append((i, chunk))

        assert first_batch_end > 0, (
            "第一句的 token 流走完时应已有内容 emit，"
            "否则流式退化成等全文"
        )
        assert len(arrivals) >= 2, "两句话应分别 emit"

    def test_multiline_markdown_stream(self) -> None:
        """真实回答常是多行 markdown，每行都应独立过护栏。"""
        text = (
            "康源牌鱼油软胶囊是保健食品。\n"
            "批准功能：有助于维持血脂健康水平。\n"
            "- 适宜人群：血脂偏高者\n"
            "- 不适宜人群：出血倾向者\n"
        )
        guard = StreamGuard(guardrail=OutputGuardrail())
        out = "".join(guard.filter(text))
        assert "批准功能" in out
        assert "不适宜人群" in out
        assert not guard.blocked
