"""rerank 接入的三态测试（docs/TDD_检索质量优化 §4.3 降级矩阵）。

    1. rerank 未装配（None）     → 行为与历史完全一致
    2. rerank 正常               → 顺序改变，但**集合不变**（只重排，不增删）
    3. rerank 故障（抛异常）      → 降级为融合原序，问答不中断

全部用假端口，零外部依赖，秒级。
"""
from __future__ import annotations

from application.use_cases.answer_question import AnswerQuestionUseCase
from application.use_cases.retrieve import RetrieveUseCase
from domain.value_objects import ChunkMeta, KeywordHit, PermissionLevel, VectorHit
from tests.fakes import (
    FakeGraphQueryPort,
    FakeKeywordSearchPort,
    FakeRerankPort,
    FakeVectorSearchPort,
)


# ===================================================================
# 夹具（与 test_retrieve.py 同款：分数用真实量级，避免与阈值脱节）
# ===================================================================
def _products():
    from domain.value_objects import ProductFacts

    return {
        "P001": ProductFacts(
            product_id="P001", name="康源牌鱼油软胶囊",
            category="保健食品", approval_no="国食健注G20250012",
            aliases=("鱼油",),
        ),
    }


def _chunks() -> list[ChunkMeta]:
    """5 个块：制造足够长的融合列表，验证窗口截断与尾部保序。"""
    return [
        ChunkMeta(chunk_id=f"D00X_c{i:03d}", doc_id="D00X",
                  text=f"知识块 {i} 的正文内容", title=f"标题{i}",
                  permission_level=PermissionLevel.INTERNAL)
        for i in range(5)
    ]


def _vector_hits() -> list[VectorHit]:
    return [VectorHit(chunk=c, score=0.72) for c in _chunks()]


def _keyword_hits() -> list[KeywordHit]:
    return [KeywordHit(chunk=c, score=20.0) for c in _chunks()[:2]]


class _StubVector(FakeVectorSearchPort):
    def __init__(self, hits: list[VectorHit]) -> None:
        super().__init__([h.chunk for h in hits])
        self._fixed = hits

    def search(self, query, *, collection="", top_k=20,
               filter_expr=None, allowed_permissions=None):
        self.calls.append("search")
        return self._fixed[:top_k]


class _StubKeyword(FakeKeywordSearchPort):
    def __init__(self, hits: list[KeywordHit]) -> None:
        super().__init__([h.chunk for h in hits])
        self._fixed = hits

    def search(self, query, *, top_k=20, allowed_permissions=None):
        self.calls.append("search")
        return self._fixed[:top_k]


def _use_case(**overrides) -> RetrieveUseCase:
    kwargs = dict(
        graph=FakeGraphQueryPort(_products()),
        vector=_StubVector(_vector_hits()),
        keyword=_StubKeyword(_keyword_hits()),
    )
    kwargs.update(overrides)
    return RetrieveUseCase(**kwargs)


def _run(uc: RetrieveUseCase) -> list[str]:
    r = uc.execute("鱼油有什么功能", vector_collection="training")
    return [h.chunk_id for h in r.fused]


# ===================================================================
# 1. 未装配：行为与历史一致
# ===================================================================
class TestRerankOff:
    def test_no_rerank_works(self) -> None:
        ids = _run(_use_case())
        assert len(ids) == 5                       # 三路融合去重后 5 个块

    def test_rerank_port_not_called_when_absent(self) -> None:
        fake = FakeRerankPort()
        # 装配了 fake 但候选不足 2 条时不调用 —— 用单向量命中构造
        uc = _use_case(
            vector=_StubVector(_vector_hits()[:1]),
            keyword=_StubKeyword([]),
            rerank=fake,
        )
        _run(uc)
        assert fake.calls == []


# ===================================================================
# 2. 正常重排：顺序可变，集合不变
# ===================================================================
class TestRerankOn:
    def test_preferred_chunk_moves_to_front(self) -> None:
        baseline = _run(_use_case())
        fake = FakeRerankPort(preferred_chunk_id="D00X_c004")
        ids = _run(_use_case(rerank=fake))
        assert ids[0] == "D00X_c004"
        assert ids != baseline                     # 顺序确实变了
        assert sorted(ids) == sorted(baseline)     # ★ 集合必须完全一致

    def test_reverse_order(self) -> None:
        baseline = _run(_use_case())
        ids = _run(_use_case(rerank=FakeRerankPort(order="reverse")))
        assert ids == list(reversed(baseline))

    def test_window_and_tail(self) -> None:
        """窗口外的候选保留融合原序（rerank_candidate_k 截断语义）。"""
        # 窗口=3：前 3 名可被重排，第 4、5 名必须仍在最后两位
        baseline = _run(_use_case())
        fake = FakeRerankPort(preferred_chunk_id=baseline[2])
        ids = _run(_use_case(rerank=fake, rerank_candidate_k=3))
        assert ids[0] == baseline[2]               # 窗口内的被顶到最前
        assert ids[3:] == baseline[3:]             # 窗口外原序保留

    def test_empty_text_candidates_skip_model(self) -> None:
        """无正文的候选不送模型，按原序排在窗口尾部。

        ★ 只剩 1 条有正文候选时（无可重排对象）连模型都不调——省一次推理。
        """
        chunks = [
            ChunkMeta(chunk_id="EMPTY_c001", doc_id="E", text="",
                      permission_level=PermissionLevel.INTERNAL),
            ChunkMeta(chunk_id="FULL_c002", doc_id="E", text="有正文",
                      permission_level=PermissionLevel.INTERNAL),
        ]
        fake = FakeRerankPort(order="reverse")
        uc = _use_case(
            vector=_StubVector([VectorHit(chunk=c, score=0.72) for c in chunks]),
            keyword=_StubKeyword([]),
            rerank=fake,
        )
        ids = _run(uc)
        # 只有一条有正文 → 无可重排对象，模型一次都不调
        assert fake.calls == []
        # 集合不变，空正文块仍在结果里
        assert sorted(ids) == ["EMPTY_c001", "FULL_c002"]

    def test_model_sees_only_text_candidates(self) -> None:
        """有 ≥2 条正文候选时，模型只收到有正文的，空正文块按原序垫尾。"""
        chunks = [
            ChunkMeta(chunk_id="EMPTY_c001", doc_id="E", text="",
                      permission_level=PermissionLevel.INTERNAL),
            ChunkMeta(chunk_id="FULL_c002", doc_id="E", text="正文二",
                      permission_level=PermissionLevel.INTERNAL),
            ChunkMeta(chunk_id="FULL_c003", doc_id="E", text="正文三",
                      permission_level=PermissionLevel.INTERNAL),
        ]
        fake = FakeRerankPort(preferred_chunk_id="FULL_c003")
        uc = _use_case(
            vector=_StubVector([VectorHit(chunk=c, score=0.72) for c in chunks]),
            keyword=_StubKeyword([]),
            rerank=fake,
        )
        ids = _run(uc)
        assert len(fake.calls) == 1
        assert sorted(fake.calls[0][1]) == ["FULL_c002", "FULL_c003"]
        # 窗口内：被偏好的在前，空正文垫在窗口尾部；集合不变
        assert ids[0] == "FULL_c003"
        assert ids[-1] == "EMPTY_c001"
        assert sorted(ids) == ["EMPTY_c001", "FULL_c002", "FULL_c003"]

    def test_contract_no_add_no_drop(self) -> None:
        """端口契约：返回候选与输入不增不减不重复。"""
        fake = FakeRerankPort(order="reverse")
        out = fake.rerank("q", [("a", "文本a"), ("b", "文本b"), ("c", "文本c")])
        assert sorted(h.chunk_id for h in out) == ["a", "b", "c"]
        assert len({h.chunk_id for h in out}) == 3


# ===================================================================
# 3. 故障降级：问答不中断
# ===================================================================
class TestRerankFailure:
    def test_exception_falls_back_to_fusion_order(self) -> None:
        baseline = _run(_use_case())
        ids = _run(_use_case(rerank=FakeRerankPort(raise_on_call=True)))
        assert ids == baseline                     # 原序返回
        r = _use_case(rerank=FakeRerankPort(raise_on_call=True)).execute(
            "鱼油有什么功能", vector_collection="training")
        assert not r.is_degraded                   # 不算路级失败（不进 degraded）
        assert not r.failed

    def test_rerank_error_maps_to_503_name(self) -> None:
        """RerankError 在接口层 503 名单里（errors.py 按类名匹配）。"""
        from interfaces.http.errors import is_infrastructure_error
        from infrastructure.exceptions import RerankError

        assert is_infrastructure_error(RerankError("x"))


# ===================================================================
# 4. 接线：AnswerQuestionUseCase 透传
# ===================================================================
class TestWiring:
    def test_answer_question_passes_rerank_through(self) -> None:
        fake = FakeRerankPort()
        uc = AnswerQuestionUseCase(
            graph=FakeGraphQueryPort(_products()),
            vector=_StubVector(_vector_hits()),
            keyword=_StubKeyword(_keyword_hits()),
            rerank=fake,
        )
        # 白盒断言：透传到内部 retriever（防未来重构悄悄丢参数）
        assert uc._retriever._rerank is fake

    def test_answer_question_default_no_rerank(self) -> None:
        uc = AnswerQuestionUseCase(
            vector=_StubVector(_vector_hits()),
            keyword=_StubKeyword(_keyword_hits()),
        )
        assert uc._retriever._rerank is None
