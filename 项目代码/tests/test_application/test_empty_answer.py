"""A3：空答案不变量。

守住的语义：
    `refused=False` 时正文必须非空。推理模型可能把 max_tokens 全花在
    reasoning 上（finish_reason=length），此时宁可拒答，也不能返回空答案。
"""
from __future__ import annotations

from application.use_cases.answer_question import AnswerQuestionUseCase
from domain.value_objects import ChunkMeta, PermissionLevel, VectorHit
from tests.fakes import FakeLanguageModelPort, FakeVectorSearchPort


class _FixedVector(FakeVectorSearchPort):
    def __init__(self, hits: list[VectorHit]) -> None:
        super().__init__([h.chunk for h in hits])
        self._fixed = hits

    def search(self, query, *, collection="", top_k=20,
               filter_expr=None, allowed_permissions=None):
        self.calls.append("search")
        return self._fixed[:top_k]


def _uc(reply: str) -> tuple[AnswerQuestionUseCase, FakeLanguageModelPort]:
    chunk = ChunkMeta(
        chunk_id="D001_c001", doc_id="D001",
        text="客户嫌贵时先讲价值再谈价格。",
        title="接待话术", permission_level=PermissionLevel.INTERNAL,
    )
    llm = FakeLanguageModelPort(reply=reply)
    uc = AnswerQuestionUseCase(
        vector=_FixedVector([VectorHit(chunk=chunk, score=0.9)]),
        keyword=None,
        llm=llm,
    )
    return uc, llm


def test_empty_completion_becomes_refusal() -> None:
    uc, _ = _uc("")
    r = uc.execute("客户嫌贵怎么办", vector_collection="health_training_docs")

    assert r.refused is True
    assert r.answer.strip(), "拒答话术不能为空"
    assert "空" in r.refusal_reason or "正文" in r.refusal_reason


def test_empty_stream_marks_refused() -> None:
    uc, _ = _uc("")
    events = list(uc.execute_stream(
        "客户嫌贵怎么办", vector_collection="health_training_docs"))

    texts = [e for e in events if e.kind == "text" and (e.text or "").strip()]
    done = events[-1]
    assert texts == []
    assert done.kind == "done"
    assert done.refused is True


def test_non_empty_still_normal() -> None:
    uc, _ = _uc("先讲价值，再谈价格。保健食品不是药物，不能代替药物治疗疾病。")
    r = uc.execute("客户嫌贵怎么办", vector_collection="health_training_docs")

    assert r.refused is False
    assert r.answer.strip()
