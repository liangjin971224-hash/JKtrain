"""重排在**用例编排**中的接入测试。

只测编排，不测模型质量（那是 test_rerank_adapter 的事）：

| 场景 | 断言 |
|---|---|
| 未注入 reranker | 走 RRF 原顺序，取前 top_k（与改造前完全一致） |
| 注入 reranker | 被调用；候选数 = rerank_candidates；结果数 = rerank_top_k |
| reranker 抛异常 | **仍然出答案**，降级为原顺序 |
| 提问原话 | 传给 reranker 的是用户原话，不是改写后的检索文本 |
"""
from __future__ import annotations

from domain.value_objects import ChunkMeta, VectorHit
from application.use_cases.answer_question import AnswerQuestionUseCase
from tests.fakes import (ExplodingRerankPort, FakeGraphQueryPort,
                         FakeKeywordSearchPort, FakeLanguageModelPort,
                         FakeRerankPort, FakeVectorSearchPort)

_REPLY = ("康源牌鱼油软胶囊是保健食品，批准功能是"
          "「有助于维持血脂（胆固醇/甘油三酯）健康水平」。"
          "保健食品不是药物，不能代替药物治疗疾病。")


def _chunks(n: int = 10) -> list[ChunkMeta]:
    return [
        ChunkMeta(chunk_id=f"c{i}", doc_id="D001",
                  text=f"第{i}段：鱼油相关资料内容。", title="鱼油",
                  section=f"节{i}", permission_level="public")
        for i in range(n)
    ]


class _FixedVector(FakeVectorSearchPort):
    """返回固定分数的向量替身。

    ★ 为什么需要它：`FakeVectorSearchPort` 用字符重合度模拟相似度，
      分数只有个位数，会低于 `MIN_VECTOR_SCORE`(0.50) 而被判为「无有效召回」，
      导致根本走不到重排。这里固定给 0.9，确保召回非空。
    """

    def __init__(self, chunks: list[ChunkMeta], *, score: float = 0.9) -> None:
        super().__init__(chunks)
        self._fixed = [VectorHit(chunk=c, score=score) for c in chunks]

    def search(self, query, *, collection="", top_k=20,
               filter_expr=None, allowed_permissions=None):
        self.calls.append("search")
        pool = self._fixed
        if allowed_permissions is not None:
            allowed = {str(p) for p in allowed_permissions}
            pool = [h for h in pool
                    if str(h.chunk.permission_level) in allowed]
        return pool[:top_k]


def _build(**overrides):
    kwargs = dict(
        graph=FakeGraphQueryPort({}),
        vector=_FixedVector(_chunks(10)),
        keyword=FakeKeywordSearchPort([]),
        llm=FakeLanguageModelPort(reply=_REPLY),
    )
    kwargs.update(overrides)
    return AnswerQuestionUseCase(**kwargs)


def _fused_docs(n: int = 10) -> list:
    """构造 n 条 FusedHit（有正文）。"""
    from domain.value_objects import FusedHit

    return [
        FusedHit(chunk_id=f"c{i}", fused_score=1.0 - i * 0.01,
                 chunk=ChunkMeta(chunk_id=f"c{i}", doc_id="D001",
                                 text=f"第{i}段：鱼油相关资料内容。",
                                 title="鱼油", section=f"节{i}",
                                 permission_level="public"))
        for i in range(n)
    ]


class TestPickDocsWithoutReranker:
    """未启用重排：`_pick_docs` 必须等价于改造前的 `docs[:5]`。"""

    def test_uses_rrf_order(self) -> None:
        uc = _build(reranker=None, rerank_top_k=5)
        picked = uc._pick_docs("鱼油有什么功能", _fused_docs(10))
        assert [h.chunk_id for h in picked] == ["c0", "c1", "c2", "c3", "c4"]

    def test_top_k_respected(self) -> None:
        uc = _build(reranker=None, rerank_top_k=2)
        picked = uc._pick_docs("鱼油有什么功能", _fused_docs(10))
        assert len(picked) == 2

    def test_tolerates_fewer_docs_than_top_k(self) -> None:
        uc = _build(reranker=None, rerank_top_k=5)
        picked = uc._pick_docs("鱼油", _fused_docs(2))
        assert len(picked) == 2


class TestPickDocsWithReranker:
    def test_reranker_is_called(self) -> None:
        fake = FakeRerankPort()
        uc = _build(reranker=fake)
        uc._pick_docs("鱼油有什么功能", _fused_docs(10))
        assert len(fake.calls) == 1

    def test_candidate_count_is_bounded(self) -> None:
        """送重排的候选数必须被 rerank_candidates 限制。

        ★ 这是延迟控制的关键：cross-encoder 延迟随条数线性增长，
          不限制就等于把全部召回结果都过一遍模型。
        """
        fake = FakeRerankPort()
        uc = _build(reranker=fake, rerank_candidates=4, rerank_top_k=2)
        uc._pick_docs("鱼油有什么功能", _fused_docs(10))

        call = fake.calls[0]
        assert len(call["candidate_ids"]) == 4
        assert call["top_k"] == 2

    def test_gets_original_question(self) -> None:
        """传给重排的必须是**用户原话**。

        重排要判断的是「这段是否回答了用户问的问题」；
        传改写后的检索文本，判定的就不是用户意图了。
        """
        fake = FakeRerankPort()
        uc = _build(reranker=fake)
        uc._pick_docs("在吃阿司匹林能吃鱼油吗", _fused_docs(5))
        assert fake.calls[0]["query"] == "在吃阿司匹林能吃鱼油吗"

    def test_result_count_matches_top_k(self) -> None:
        fake = FakeRerankPort()
        uc = _build(reranker=fake, rerank_top_k=3)
        picked = uc._pick_docs("鱼油有什么功能", _fused_docs(10))
        assert len(picked) == 3


class TestDegrade:
    """★ 最重要的一组：重排出任何问题，都不能让用户拿不到答案。"""

    def test_reranker_exception_degrades(self) -> None:
        boom = ExplodingRerankPort()
        uc = _build(reranker=boom)
        picked = uc._pick_docs("鱼油有什么功能", _fused_docs(10))

        assert boom.called == 1, "应该尝试调用过重排"
        assert [h.chunk_id for h in picked] == ["c0", "c1", "c2", "c3", "c4"]

    def test_exception_still_answers_end_to_end(self) -> None:
        """端到端：重排炸了，问答依然出答案（不是拒答）。"""
        boom = ExplodingRerankPort()
        uc = _build(reranker=boom)
        res = uc.execute("鱼油有什么功能", vector_collection="training")

        assert boom.called >= 1
        assert res.refused is False
        assert res.answer

    def test_exception_does_not_skip_guardrail(self) -> None:
        """重排失败 → 护栏仍必须生效（不能因为降级就跳过合规判定）。"""
        uc = _build(
            reranker=ExplodingRerankPort(),
            llm=FakeLanguageModelPort(reply="这个产品能降血脂，效果很好。"),
        )
        res = uc.execute("鱼油有什么功能", vector_collection="training")
        assert "效果很好" not in res.answer or res.refused

    def test_empty_recall_does_not_call_reranker(self) -> None:
        """召回为空时不调用重排（没东西可排），且仍走拒答。"""
        fake = FakeRerankPort()
        uc = AnswerQuestionUseCase(
            graph=FakeGraphQueryPort({}),
            vector=_FixedVector([]),
            keyword=FakeKeywordSearchPort([]),
            llm=FakeLanguageModelPort(reply=_REPLY),
            reranker=fake,
        )
        res = uc.execute("一个生僻问题", vector_collection="training")

        assert res.refused is True
        assert fake.calls == [], "召回为空时不应调用重排"
