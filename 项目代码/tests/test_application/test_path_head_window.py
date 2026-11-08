"""「每路头部进窗口」测试（P1-a 落地）。

守住的语义：
    1. 窗口构图 = 向量头 1 + 关键词头 1 + RRF 回填（覆盖优先）
    2. 头部与 RRF 重复时跨路去重，且不丢回填
    3. **只对配置的 collection** 生效（案例库/题库保持原构图）
    4. 配置置空 = 回滚纯 RRF
    5. 与稀疏并集重叠时稀疏并集优先（不叠加两种构图）

背景（205 题离线模拟，2026-11-07）：
    文档库纯 RRF 105/131 → 每路头 1 = 111/131；案例库 21 → 22（不如稀疏
    并集 26），题库 40/40 不变。所以策略必须按库选，这里测的就是「分库」。
"""
from __future__ import annotations

from application.use_cases.retrieve import RetrieveUseCase
from domain.value_objects import ChunkMeta, PermissionLevel
from tests.fakes import FakeKeywordSearchPort, FakeSparseSearchPort, FakeVectorSearchPort

CASE = "health_case_library"
DOCS = "health_training_docs"

QUERY = "客户嫌贵怎么办"


def _chunk(cid: str, text: str, perm: PermissionLevel = PermissionLevel.RESTRICTED):
    return ChunkMeta(
        chunk_id=cid, doc_id=cid.split("_")[0], text=text,
        title=cid, permission_level=perm,
    )


def _corpus() -> dict[str, ChunkMeta]:
    # 向量路按「查询去重字符重合度」打分（FakeVectorSearchPort）：
    #   A/B/K 都是 7/7=1.0（稳定排序保持列表顺序），C 是 6/7。
    # 关键词路按「查询 2-gram 命中数」打分（FakeKeywordSearchPort）：
    #   K 与 A 都是 5，靠列表顺序让 K 当上关键词头。
    return {
        "A": _chunk("A_m1", "客户嫌贵怎么办"),
        "B": _chunk("B_m1", "怎么办？客户嫌贵"),
        "C": _chunk("C_m1", "客户嫌贵怎么"),
        "K": _chunk("K_m1", "客户嫌贵怎么办？先算日均成本"),
    }


def _use_case(
    *,
    docs_configured: bool = True,
    keyword: bool = True,
    sparse: FakeSparseSearchPort | None = None,
    sparse_collections: frozenset[str] = frozenset({CASE}),
    keyword_first: bool = False,
) -> RetrieveUseCase:
    c = _corpus()
    vector_pool = [c["A"], c["B"], c["K"], c["C"]]
    # 关键词列表顺序 = 同分时的名次顺序
    keyword_pool = [c["K"], c["A"], c["B"], c["C"]]
    if keyword_first:
        keyword_pool = [c["A"], c["K"], c["B"], c["C"]]
    return RetrieveUseCase(
        vector=FakeVectorSearchPort(vector_pool),
        keyword=FakeKeywordSearchPort(keyword_pool) if keyword else None,
        sparse=sparse,
        sparse_collections=sparse_collections,
        path_head_collections=frozenset({DOCS}) if docs_configured else frozenset(),
        # FakeKeywordSearchPort 的分数是 1~6 的小整数，必须放过绝对阈值
        min_keyword_score=0.0,
    )


def test_docs_window_is_path_heads_plus_rrf_backfill():
    r = _use_case().execute(QUERY, vector_collection=DOCS, top_k=10, fused_limit=3)

    # 纯 RRF 序是 A, B, K；每路头 1 把关键词头 K 提到第 2 位。
    assert [h.chunk_id for h in r.fused] == ["A_m1", "K_m1", "B_m1"]


def test_unconfigured_collection_keeps_pure_rrf():
    r = _use_case().execute(QUERY, vector_collection=CASE, top_k=10, fused_limit=3)

    assert [h.chunk_id for h in r.fused] == ["A_m1", "B_m1", "K_m1"]


def test_empty_config_rolls_back_to_pure_rrf():
    r = _use_case(docs_configured=False).execute(
        QUERY, vector_collection=DOCS, top_k=10, fused_limit=3)

    assert [h.chunk_id for h in r.fused] == ["A_m1", "B_m1", "K_m1"]


def test_duplicate_head_not_repeated_and_backfill_continues():
    # 关键词头 = 向量头 A → 窗口不能出现两次 A，回填照常
    r = _use_case(keyword_first=True).execute(
        QUERY, vector_collection=DOCS, top_k=10, fused_limit=3)

    ids = [h.chunk_id for h in r.fused]
    assert ids == ["A_m1", "B_m1", "K_m1"]
    assert len(ids) == len(set(ids))


def test_head_entry_reuses_rrf_metadata():
    r = _use_case().execute(QUERY, vector_collection=DOCS, top_k=10, fused_limit=3)

    k = next(h for h in r.fused if h.chunk_id == "K_m1")
    # K 在向量/关键词两路都命中 → sources 保留两路，分数是 RRF 原分
    assert k.sources == ("vector", "keyword")
    assert k.fused_score > 0.0
    assert k.chunk is not None


def test_keyword_path_missing_still_builds_window():
    r = _use_case(keyword=False).execute(
        QUERY, vector_collection=DOCS, top_k=10, fused_limit=3)

    assert "keyword" in r.degraded
    assert [h.chunk_id for h in r.fused] == ["A_m1", "B_m1", "K_m1"]


def test_sparse_union_takes_precedence_when_collections_overlap():
    c = _corpus()
    s = _chunk("S_m1", "必答要素 权重 判定说明 正则")

    def build(path_head: frozenset[str]) -> RetrieveUseCase:
        return RetrieveUseCase(
            vector=FakeVectorSearchPort([c["A"], c["B"], c["K"], c["C"]]),
            keyword=FakeKeywordSearchPort([c["K"], c["A"]]),
            sparse=FakeSparseSearchPort([s]),
            sparse_collections=frozenset({DOCS}),
            path_head_collections=path_head,
            min_keyword_score=0.0,
        )

    overlap = build(frozenset({DOCS})).execute(
        QUERY, vector_collection=DOCS, top_k=10, fused_limit=4)
    sparse_only = build(frozenset()).execute(
        QUERY, vector_collection=DOCS, top_k=10, fused_limit=4)

    # 两库重叠时行为必须与「只配稀疏并集」完全一致 → 稀疏优先
    ids = [h.chunk_id for h in overlap.fused]
    assert ids == [h.chunk_id for h in sparse_only.fused]
    assert "S_m1" in ids
