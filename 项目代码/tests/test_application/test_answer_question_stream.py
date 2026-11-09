"""流式问答用例测试。

重点：**流式仍必须走护栏**（约束 #1 不因性能优化而放松）。

三条必须守住：
    1. 召回为空 → 不调 LLM，直接产出拒答（一条流都不断）
    2. 违规内容 → 在 emit 之前被截断
    3. 缺警示语 → 全流结束后才发现，需要补发说明
"""
from __future__ import annotations

import pytest

from application.use_cases.answer_question import AnswerQuestionUseCase
from domain.ports import LanguageModelPort
from domain.value_objects import (
    ApprovalClaim,
    ChunkMeta,
    Contraindication,
    KeywordHit,
    PermissionLevel,
    ProductCategory,
    ProductFacts,
    Severity,
    VectorHit,
)
from tests.fakes import (
    FailOnCallLLM,
    FakeGraphQueryPort,
    FakeKeywordSearchPort,
    FakeLanguageModelPort,
    FakeVectorSearchPort,
)


# ===================================================================
# 夹具
# ===================================================================
def _fish_oil() -> ProductFacts:
    return ProductFacts(
        product_id="P001", name="康源牌鱼油软胶囊",
        category="保健食品", approval_no="国食健注G20250012",
        claims=(ApprovalClaim(efficacy_id="EFF_19",
                             text="有助于维持血脂（胆固醇/甘油三酯）健康水平",
                             catalog_no=19),),
        contraindications=(
            Contraindication("CI_BLEED", "出血倾向者和出血性疾病患者",
                             Severity.CRITICAL, ""),
        ),
        aliases=("鱼油",),
    )


def _chunks() -> list[ChunkMeta]:
    return [
        ChunkMeta(chunk_id="D005_c001", doc_id="D005",
                  text="保健食品只能声称 24 项功能",
                  title="功能声称规范",
                  permission_level=PermissionLevel.INTERNAL),
    ]


class _FixedVector(FakeVectorSearchPort):
    def __init__(self, hits: list[VectorHit]) -> None:
        super().__init__([h.chunk for h in hits])
        self._fixed = hits

    def search(self, query, *, collection="", top_k=20,
               filter_expr=None, allowed_permissions=None):
        self.calls.append("search")
        pool = self._fixed
        if allowed_permissions is not None:
            allowed = {str(p) for p in allowed_permissions}
            pool = [h for h in pool
                    if str(h.chunk.permission_level) in allowed]
        return pool[:top_k]


class _FixedKeyword(FakeKeywordSearchPort):
    def __init__(self, hits: list[KeywordHit]) -> None:
        super().__init__([h.chunk for h in hits])
        self._fixed = hits

    def search(self, query, *, top_k=20, allowed_permissions=None):
        self.calls.append("search")
        pool = self._fixed
        if allowed_permissions is not None:
            allowed = {str(p) for p in allowed_permissions}
            pool = [h for h in pool
                    if str(h.chunk.permission_level) in allowed]
        return pool[:top_k]


class _ChunkedLLM(LanguageModelPort):
    """按指定粒度切片产出的假模型（模拟 SSE）。"""

    def __init__(self, text: str, *, chunk: int = 4) -> None:
        self._text = text
        self._chunk = chunk
        self.calls: list[str] = []

    def generate(self, prompt, *, system_prompt="", temperature=0.0,
                 max_tokens=1024):
        from domain.ports import GenerationResult
        self.calls.append("generate")
        return GenerationResult(text=self._text, model="fake")

    def generate_stream(self, prompt, *, system_prompt="", temperature=0.0,
                        max_tokens=1024):
        self.calls.append("generate_stream")
        for i in range(0, len(self._text), self._chunk):
            yield self._text[i:i + self._chunk]

    def classify(self, text, labels, *, allow_unknown=True):
        return labels[0] if labels else "unknown"

    def is_available(self) -> bool:
        return True


def _build(llm=None, **overrides) -> AnswerQuestionUseCase:
    kwargs = dict(
        graph=FakeGraphQueryPort({"P001": _fish_oil()}),
        vector=_FixedVector([VectorHit(chunk=c, score=0.72)
                             for c in _chunks()]),
        keyword=_FixedKeyword([KeywordHit(chunk=c, score=20.0)
                               for c in _chunks()]),
        llm=llm or FakeLanguageModelPort(
            reply=("这款鱼油有助于维持血脂健康水平。"
                   "保健食品不是药物，不能代替药物治疗疾病。")),
    )
    kwargs.update(overrides)
    return AnswerQuestionUseCase(**kwargs)


W = "保健食品不是药物，不能代替药物治疗疾病。"


# ===================================================================
# 1. 正常流式
# ===================================================================
class TestNormalStream:
    def test_yields_multiple_chunks(self) -> None:
        """应分多次产出，不是一次性给全文。"""
        uc = _build(llm=_ChunkedLLM(
            f"这款鱼油是保健食品。批准功能是有助于维持血脂健康水平。{W}",
            chunk=4))
        chunks = list(uc.execute_stream(
            "鱼油有什么功能", vector_collection="training",
            product_ids=("P001",)))

        text_parts = [c for c in chunks if c.kind == "text"]
        assert len(text_parts) > 1, f"应分多次产出，实际 {len(text_parts)} 次"
        assert "有助于维持血脂" in "".join(c.text for c in text_parts)

    def test_done_event_has_meta(self) -> None:
        """最后必须有一个 done 事件带溯源信息。"""
        uc = _build(llm=_ChunkedLLM(
            f"这款鱼油是保健食品。{W}", chunk=5))
        chunks = list(uc.execute_stream(
            "鱼油有什么功能", vector_collection="training",
            product_ids=("P001",)))

        done = [c for c in chunks if c.kind == "done"]
        assert len(done) == 1
        assert done[0].meta
        assert "cited_chunk_ids" in done[0].meta

    def test_chunks_assembled_equals_full(self) -> None:
        """所有 text 片段拼起来应等于完整回答。"""
        full = f"这款鱼油是保健食品。批准功能是有助于维持血脂健康水平。{W}"
        uc = _build(llm=_ChunkedLLM(full, chunk=3))
        chunks = list(uc.execute_stream(
            "鱼油有什么功能", vector_collection="training",
            product_ids=("P001",)))

        assembled = "".join(c.text for c in chunks if c.kind == "text")
        assert assembled == full


# ===================================================================
# 2. ★ 召回为空 → 不调 LLM
# ===================================================================
class TestNoHitsStream:
    def test_refuses_without_llm(self) -> None:
        """流式路径同样不能调 LLM。"""
        uc = AnswerQuestionUseCase(
            graph=FakeGraphQueryPort({}),
            vector=_FixedVector([]),
            keyword=_FixedKeyword([]),
            llm=FailOnCallLLM(),
        )
        chunks = list(uc.execute_stream("知识库里没有的问题",
                                        vector_collection="training"))

        assert chunks, "至少要产出拒答内容"
        done = [c for c in chunks if c.kind == "done"]
        assert done and done[0].refused is True
        emitted = "".join(c.text for c in chunks if c.kind == "text")
        assert "没有找到" in emitted or "暂无" in emitted
        # 拒答必须给下一步建议（incremental-delivery 的要求）
        assert any(h in emitted for h in ("咨询", "主管", "换个说法"))

    def test_low_vector_score_also_refuses(self) -> None:
        """向量分数低于阈值同样视为无召回。"""
        uc = AnswerQuestionUseCase(
            graph=FakeGraphQueryPort({}),
            vector=_FixedVector([VectorHit(chunk=_chunks()[0], score=0.31)]),
            keyword=_FixedKeyword([]),
            llm=FailOnCallLLM(),
        )
        chunks = list(uc.execute_stream("无关问题",
                                        vector_collection="training"))
        done = [c for c in chunks if c.kind == "done"]
        assert done[0].refused is True


# ===================================================================
# 3. ★ 违规内容必须在 emit 前截断
# ===================================================================
class TestStreamGuardrail:
    def test_violation_truncated_before_emit(self) -> None:
        """违规句不得出现在任何 text 事件里。"""
        text = (
            "这款鱼油是保健食品。"          # 合规
            "它能降血脂，很多人吃了都说好。"   # 违规
            f"{W}"                          # 合规，但不能再输出
        )
        uc = _build(llm=_ChunkedLLM(text, chunk=2))
        chunks = list(uc.execute_stream(
            "鱼油有什么功能", vector_collection="training",
            product_ids=("P001",)))

        emitted = "".join(c.text for c in chunks if c.kind == "text")
        assert "降血脂" not in emitted, "违规内容泄漏"
        assert "很多人" not in emitted

        done = [c for c in chunks if c.kind == "done"]
        assert done[0].refused is True
        assert done[0].meta.get("blocked") is True

    def test_clean_prefix_emitted_before_block(self) -> None:
        """违规前的合规内容应当已经产出（这是流式的价值）。"""
        text = "这款鱼油是保健食品。它能治疗糖尿病。"
        uc = _build(llm=_ChunkedLLM(text, chunk=3))
        chunks = list(uc.execute_stream(
            "鱼油有什么功能", vector_collection="training",
            product_ids=("P001",)))

        emitted = "".join(c.text for c in chunks if c.kind == "text")
        assert "这款鱼油是保健食品" in emitted, (
            "合规前缀应已产出，否则流式没有意义"
        )
        assert "治疗糖尿病" not in emitted

    def test_ordinary_food_claim_blocked_in_stream(self) -> None:
        """普通食品功能声称在流式中也要拦（约束 #2）。"""
        from domain.value_objects import ProductFacts as PF

        ordinary = PF(product_id="P016", name="复合蛋白固体饮料",
                      category="普通食品")
        uc = AnswerQuestionUseCase(
            graph=FakeGraphQueryPort({"P016": ordinary}),
            vector=_FixedVector([VectorHit(chunk=c, score=0.72)
                                 for c in _chunks()]),
            keyword=_FixedKeyword([KeywordHit(chunk=c, score=20.0)
                                   for c in _chunks()]),
            llm=_ChunkedLLM("每份含蛋白质 12g。这款有助于增强免疫力。",
                            chunk=3),
        )
        chunks = list(uc.execute_stream(
            "这个固体饮料能增强免疫力吗", vector_collection="training",
            product_ids=("P016",)))

        emitted = "".join(c.text for c in chunks if c.kind == "text")
        assert "增强免疫力" not in emitted
        done = [c for c in chunks if c.kind == "done"]
        assert done[0].refused is True


# ===================================================================
# 4. 缺警示语：全流结束后才发现
# ===================================================================
class TestMissingWarningAfterStream:
    def test_missing_warning_repaired_at_end(self) -> None:
        """缺警示语无法在中途判定，全流结束后补发，不再标拒答。

        ★ 与非流式的「补回警示语」保持一致：
          用户已经看到一条有用的回答，最后补上法定警示语即可，
          不应因为「警示语在最后一句」把整条回答判成拒答。
        """
        uc = _build(llm=_ChunkedLLM("这款鱼油有助于维持血脂健康水平。",
                                    chunk=4))
        chunks = list(uc.execute_stream(
            "鱼油有什么功能", vector_collection="training",
            product_ids=("P001",)))

        done = [c for c in chunks if c.kind == "done"]
        assert done[0].refused is False
        assert done[0].meta.get("repaired_warning") is True
        emitted = "".join(c.text for c in chunks if c.kind == "text")
        assert "不能代替药物治疗疾病" in emitted

    def test_warning_present_passes(self) -> None:
        uc = _build(llm=_ChunkedLLM(
            f"这款鱼油有助于维持血脂健康水平。{W}", chunk=4))
        chunks = list(uc.execute_stream(
            "鱼油有什么功能", vector_collection="training",
            product_ids=("P001",)))

        done = [c for c in chunks if c.kind == "done"]
        assert done[0].refused is False
        emitted = "".join(c.text for c in chunks if c.kind == "text")
        assert "不能代替药物治疗疾病" in emitted


# ===================================================================
# 5. LLM 故障
# ===================================================================
class TestStreamFailure:
    def test_llm_error_yields_refusal(self) -> None:
        """LLM 挂掉时产出拒答 + done(refused)，不静默结束。"""
        class _Dead(LanguageModelPort):
            def generate(self, *a, **kw):
                from infrastructure.exceptions import LLMCallError
                raise LLMCallError("挂了")

            def generate_stream(self, *a, **kw):
                from infrastructure.exceptions import LLMCallError
                raise LLMCallError("挂了")
                yield  # pragma: no cover

            def classify(self, *a, **kw):
                return "unknown"

            def is_available(self):
                return False

        uc = _build(llm=_Dead())
        chunks = list(uc.execute_stream(
            "鱼油有什么功能", vector_collection="training",
            product_ids=("P001",)))

        done = [c for c in chunks if c.kind == "done"]
        assert done[0].refused is True
        assert done[0].refusal_reason


# ===================================================================
# 6. 事件结构
# ===================================================================
class TestStreamEvent:
    def test_text_event_format(self) -> None:
        uc = _build(llm=_ChunkedLLM(f"第一句。{W}", chunk=4))
        chunks = list(uc.execute_stream(
            "鱼油有什么功能", vector_collection="training",
            product_ids=("P001",)))

        for c in chunks:
            assert c.kind in ("text", "done")
            if c.kind == "text":
                assert c.text

    def test_done_is_last(self) -> None:
        uc = _build(llm=_ChunkedLLM(f"第一句。{W}", chunk=4))
        chunks = list(uc.execute_stream(
            "鱼油有什么功能", vector_collection="training",
            product_ids=("P001",)))

        assert chunks[-1].kind == "done"
