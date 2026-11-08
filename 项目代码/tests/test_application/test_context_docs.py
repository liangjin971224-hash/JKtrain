"""参考资料条数（P1-b `context_docs`）测试。

守住的语义：
    1. 默认仍是 5 条进 prompt（代码默认值 = 已验证行为）
    2. `context_docs` 可放大/缩小，且只影响 prompt，不影响召回窗口
    3. 召回不足时按实际条数组装，不报错
    4. 非法值（0/负数）钳到 1，绝不允许「参考资料区为空」

背景：用 app_path_head_v1 的 gold_rank 直接算上下文覆盖——
    5 条 = 177、8 条 = 181、10 条 = 185（案例库 26→28→30）。
    但多出来的段落可能是噪声，所以默认值必须等答案级 A/B 通过才改。
"""
from __future__ import annotations

from application.use_cases.answer_question import AnswerQuestionUseCase
from domain.value_objects import ChunkMeta, PermissionLevel, VectorHit
from tests.fakes import FakeLanguageModelPort, FakeVectorSearchPort

QUERY = "客户嫌贵怎么办"

_COMPLIANT_REPLY = "保健食品不是药物，不能代替药物治疗疾病。"


class _FixedVector(FakeVectorSearchPort):
    """按给定顺序返回固定分数的向量替身。"""

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


def _chunks(n: int = 6) -> list[ChunkMeta]:
    return [
        ChunkMeta(
            chunk_id=f"D00{i}_c001", doc_id=f"D00{i}",
            text=f"第{i}条参考资料正文：客户嫌贵时先讲价值再谈价格。",
            title=f"资料{i}",
            permission_level=PermissionLevel.INTERNAL,
        )
        for i in range(1, n + 1)
    ]


def _uc(*, context_docs: int | None = None, n: int = 6) -> tuple[
        AnswerQuestionUseCase, FakeLanguageModelPort]:
    llm = FakeLanguageModelPort(reply=_COMPLIANT_REPLY)
    hits = [VectorHit(chunk=c, score=0.9 - i * 0.05)
            for i, c in enumerate(_chunks(n))]
    kwargs = {}
    if context_docs is not None:
        kwargs["context_docs"] = context_docs
    uc = AnswerQuestionUseCase(
        vector=_FixedVector(hits),
        keyword=None,
        llm=llm,
        **kwargs,
    )
    return uc, llm


def _prompt(llm: FakeLanguageModelPort) -> str:
    assert llm.prompts, "应该调用过 LLM"
    return llm.prompts[0]


def test_default_is_five_docs() -> None:
    uc, llm = _uc()
    uc.execute(QUERY, vector_collection="health_training_docs")

    prompt = _prompt(llm)
    assert "[1] 《资料1》" in prompt
    assert "[5] 《资料5》" in prompt
    assert "[6] 《资料6》" not in prompt


def test_context_docs_can_shrink() -> None:
    uc, llm = _uc(context_docs=3)
    uc.execute(QUERY, vector_collection="health_training_docs")

    prompt = _prompt(llm)
    assert "[3] 《资料3》" in prompt
    assert "[4] 《资料4》" not in prompt


def test_context_docs_can_expand_within_window() -> None:
    uc, llm = _uc(context_docs=6)
    r = uc.execute(QUERY, vector_collection="health_training_docs", top_k=20)

    prompt = _prompt(llm)
    assert "[6] 《资料6》" in prompt
    # 召回窗口本身不因 prompt 条数变化而缩水
    assert len(r.cited_chunk_ids) == 6


def test_fewer_hits_than_context_docs() -> None:
    uc, llm = _uc(context_docs=10, n=2)
    uc.execute(QUERY, vector_collection="health_training_docs")

    prompt = _prompt(llm)
    assert "[2] 《资料2》" in prompt
    assert "[3] " not in prompt


def test_invalid_context_docs_clamped_to_one() -> None:
    uc, llm = _uc(context_docs=0)
    uc.execute(QUERY, vector_collection="health_training_docs")

    prompt = _prompt(llm)
    assert "[1] 《资料1》" in prompt
    assert "[2] " not in prompt
