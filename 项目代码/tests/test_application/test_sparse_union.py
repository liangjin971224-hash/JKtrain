"""稀疏并集窗口测试（P0-3 落地）。

守住的语义：
    1. 窗口构图 = RRF 头部 + 稀疏未重复命中 + RRF 尾部回填
    2. 稀疏**只对配置的 collection** 生效（文档库不开）
    3. 稀疏失败只降级该路，不影响向量/关键词
    4. 稀疏自己的 floor 与 dense 的 0.50 互不干扰（IP 尺度不同）
    5. 稀疏条目带 sources=("sparse",) 且 fused_score=0.0（不参与 RRF 排序）

背景（205 题实测）：案例库纯向量 R@5=21/34，稀疏 25/34，并集 28/34；
若把稀疏当第 4 路丢进 RRF 会稀释回 21/34。所以这里测的是"并集"而不是"融合"。
"""
from __future__ import annotations

import pytest

from application.use_cases.retrieve import PATH_SPARSE, RetrieveUseCase
from domain.value_objects import ChunkMeta, PermissionLevel
from tests.fakes import FakeSparseSearchPort, FakeVectorSearchPort

CASE = "health_case_library"
DOCS = "health_training_docs"


def _chunk(cid: str, text: str, perm: PermissionLevel = PermissionLevel.RESTRICTED):
    return ChunkMeta(
        chunk_id=cid, doc_id=cid.split("_")[0], text=text,
        title=cid, permission_level=perm,
    )


@pytest.fixture()
def dense_chunks() -> list[ChunkMeta]:
    # 假向量端口按「与查询的**去重字符**重合度」打分（重合数 / 查询字符数），
    # 所以用逐字递减的文本制造确定顺序；三条都要 ≥ MIN_VECTOR_SCORE(0.50)
    # 才不会被阈值滤掉（7/7、6/7、4/7）。
    return [
        _chunk("C01_m1", "客户嫌贵怎么办"),
        _chunk("C01_m2", "客户嫌贵怎么"),
        _chunk("C01_m3", "客户嫌贵"),
    ]


@pytest.fixture()
def sparse_chunks() -> list[ChunkMeta]:
    # 稀疏独有：向量路按字重合度抓不到的表格型块
    return [
        _chunk("C01_m2", "客户嫌贵怎么"),                  # 与向量路重复
        _chunk("C02_m2", "必答要素 权重 判定说明 正则"),
    ]


def _use_case(dense, sparse, **kw) -> RetrieveUseCase:
    return RetrieveUseCase(
        vector=FakeVectorSearchPort(dense),
        sparse=sparse,
        sparse_collections=frozenset({CASE}),
        **kw,
    )


def test_window_is_head_plus_sparse_plus_backfill(dense_chunks, sparse_chunks):
    uc = _use_case(dense_chunks, FakeSparseSearchPort(sparse_chunks))
    r = uc.execute("客户嫌贵怎么办", vector_collection=CASE, top_k=10, fused_limit=4)

    ids = [h.chunk_id for h in r.fused]
    # 头部 2 条来自 RRF，第 3 条是稀疏独有块，第 4 条是 RRF 回填
    assert ids[:2] == ["C01_m1", "C01_m2"]
    assert ids[2] == "C02_m2"
    assert len(ids) == len(set(ids)) == 4


def test_sparse_entries_are_marked_and_scoreless(dense_chunks, sparse_chunks):
    uc = _use_case(dense_chunks, FakeSparseSearchPort(sparse_chunks))
    r = uc.execute("客户嫌贵怎么办", vector_collection=CASE, top_k=10, fused_limit=4)

    sparse_hit = next(h for h in r.fused if h.chunk_id == "C02_m2")
    assert sparse_hit.sources == (PATH_SPARSE,)
    # 稀疏条目不带 RRF 分数——避免下游把 IP 分数当余弦分数排序
    assert sparse_hit.fused_score == 0.0
    assert sparse_hit.chunk is not None


def test_duplicates_across_paths_not_repeated(dense_chunks):
    # 稀疏只返回一个与 RRF 重复的块 → 窗口不应出现重复
    uc = _use_case(dense_chunks, FakeSparseSearchPort([dense_chunks[1]]))
    r = uc.execute("客户嫌贵怎么办", vector_collection=CASE, top_k=10, fused_limit=4)
    ids = [h.chunk_id for h in r.fused]
    assert ids == ["C01_m1", "C01_m2", "C01_m3"]


def test_sparse_skipped_for_unconfigured_collection(dense_chunks, sparse_chunks):
    sparse = FakeSparseSearchPort(sparse_chunks)
    uc = _use_case(dense_chunks, sparse)
    r = uc.execute("客户嫌贵怎么办", vector_collection=DOCS, top_k=10, fused_limit=4)

    assert sparse.calls == []          # 一次都没被调用
    assert r.sparse_hits == ()
    assert "sparse" not in r.elapsed_ms


def test_sparse_failure_degrades_only_that_path(dense_chunks):
    uc = _use_case(dense_chunks, FakeSparseSearchPort(fail_on_call=True))
    r = uc.execute("客户嫌贵怎么办", vector_collection=CASE, top_k=10, fused_limit=4)

    assert "sparse" in r.degraded
    assert r.fused                       # 向量路仍然给出结果
    assert r.sparse_hits == ()


def test_sparse_uses_its_own_floor_not_cosine_threshold(dense_chunks, sparse_chunks):
    # IP 分数 0.01 < floor 0.02 → 全部被滤掉；这不影响 dense 的 0.50
    sparse = FakeSparseSearchPort(sparse_chunks, score=0.01)
    uc = _use_case(dense_chunks, sparse)
    r = uc.execute("客户嫌贵怎么办", vector_collection=CASE, top_k=10, fused_limit=4)

    assert r.sparse_hits == ()
    assert [h.chunk_id for h in r.fused][:2] == ["C01_m1", "C01_m2"]


def test_sparse_rescues_when_dense_empty(dense_chunks):
    # 向量路返回空（例如 dense 降级/无命中）时，稀疏仍能撑起窗口
    uc = RetrieveUseCase(
        vector=FakeVectorSearchPort([]),
        sparse=FakeSparseSearchPort(dense_chunks[:1]),
        sparse_collections=frozenset({CASE}),
    )
    r = uc.execute("客户嫌贵怎么办", vector_collection=CASE, top_k=10, fused_limit=4)
    assert [h.chunk_id for h in r.fused] == ["C01_m1"]
    assert not r.failed
