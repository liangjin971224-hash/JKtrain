"""三路召回用例测试（P3）。

重点不是「能召回」，而是**降级行为与阈值语义**：
    1. 单路失败 → 其余路继续（容错）
    2. 三路全失败 → failed=True（P4 据此拒答，不调 LLM）
    3. 向量低分结果必须被阈值滤掉（否则 P4 的拒答永远触发不了）

全部用假端口，零外部依赖，秒级。
"""
from __future__ import annotations

import pytest

from application.use_cases.retrieve import (
    PATH_GRAPH,
    PATH_KEYWORD,
    PATH_VECTOR,
    RetrieveUseCase,
)
from domain.value_objects import (
    ChunkMeta,
    KeywordHit,
    PermissionLevel,
    ProductFacts,
    VectorHit,
)
from tests.fakes import (
    FakeGraphQueryPort,
    FakeKeywordSearchPort,
    FakeVectorSearchPort,
)


# ===================================================================
# 夹具数据
# ===================================================================
def _products() -> dict[str, ProductFacts]:
    return {
        "P001": ProductFacts(
            product_id="P001", name="康源牌鱼油软胶囊",
            category="保健食品", approval_no="国食健注G20250012",
            aliases=("鱼油",),
        ),
    }


def _chunks() -> list[ChunkMeta]:
    return [
        ChunkMeta(chunk_id="D005_c001", doc_id="D005",
                  text="保健食品只能声称 24 项功能", title="功能声称规范",
                  permission_level=PermissionLevel.INTERNAL),
        ChunkMeta(chunk_id="D005_c002", doc_id="D005",
                  text="绝对化表述严禁使用", title="禁用表述",
                  permission_level=PermissionLevel.PUBLIC),
        ChunkMeta(chunk_id="Q001", doc_id="Q001",
                  text="产品类别辨识题目", title="题目",
                  permission_level=PermissionLevel.RESTRICTED),
    ]


def _vector_hits(*, high: bool = True) -> list[VectorHit]:
    """构造向量命中。high=False 时分数低于阈值。"""
    score = 0.72 if high else 0.31
    return [VectorHit(chunk=c, score=score) for c in _chunks()]


class _StubVector(FakeVectorSearchPort):
    """可指定返回固定命中的向量替身。"""

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


class _StubKeyword(FakeKeywordSearchPort):
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


def _keyword_hits() -> list[KeywordHit]:
    """关键词命中。

    ★ 分数必须高于 `MIN_KEYWORD_SCORE`（实测标定的 14.0）。
      用真实量级的分数，避免测试数据与生产阈值脱节——
      P3 初期这里写的 5.0，后来加绝对阈值时全部误挂，
      根源就是测试数据不是真实量级。
    """
    return [KeywordHit(chunk=c, score=20.0) for c in _chunks()[:2]]


def _use_case(**overrides) -> RetrieveUseCase:
    """默认三路全可用。"""
    kwargs = dict(
        graph=FakeGraphQueryPort(_products()),
        vector=_StubVector(_vector_hits()),
        keyword=_StubKeyword(_keyword_hits()),
    )
    kwargs.update(overrides)
    return RetrieveUseCase(**kwargs)


# ===================================================================
# 1. 正常路径
# ===================================================================
class TestNormal:
    def test_three_paths_all_hit(self) -> None:
        r = _use_case().execute("鱼油", intent="产品功效",
                                vector_collection="training")
        assert len(r.graph_hits) == 1
        assert len(r.vector_hits) == 3
        assert len(r.keyword_hits) == 2
        assert not r.failed
        assert not r.is_degraded

    def test_fused_not_empty(self) -> None:
        r = _use_case().execute("鱼油", vector_collection="training")
        assert len(r.fused) > 0

    def test_weights_recorded(self) -> None:
        """权重必须记录在结果里，便于回放「为什么这条排第一」。"""
        r = _use_case().execute("鱼油", intent="禁忌咨询",
                                vector_collection="training")
        assert len(r.weights) == 3
        assert abs(sum(r.weights) - 1.0) < 1e-6
        # 安全类意图：图谱权重应最高
        assert r.weights[0] >= 0.5

    def test_sources_recorded(self) -> None:
        """每条融合结果要能说出「来自哪几路」。"""
        r = _use_case().execute("鱼油", vector_collection="training")
        for hit in r.fused:
            assert hit.sources, f"{hit.chunk_id} 未记录来源"

    def test_elapsed_recorded(self) -> None:
        r = _use_case().execute("鱼油", vector_collection="training")
        assert set(r.elapsed_ms) == {PATH_GRAPH, PATH_VECTOR, PATH_KEYWORD}
        assert all(v >= 0 for v in r.elapsed_ms.values())

    def test_empty_query(self) -> None:
        r = _use_case().execute("", vector_collection="training")
        assert r.failed
        assert r.fused == ()


# ===================================================================
# 2. 单路降级（P3 核心）
# ===================================================================
class TestDegradation:
    """三路中任一路挂掉，整体仍可用。"""

    def test_graph_failure_degrades(self) -> None:
        """图谱挂掉 → 向量/关键词仍出结果。"""
        r = _use_case(
            graph=FakeGraphQueryPort(_products(), fail_on_call=True),
        ).execute("鱼油", vector_collection="training")

        assert not r.failed, "图谱失败不应导致整体失败"
        assert PATH_GRAPH in r.degraded
        assert len(r.vector_hits) > 0 or len(r.keyword_hits) > 0
        assert len(r.fused) > 0

    def test_vector_failure_degrades(self) -> None:
        r = _use_case(
            vector=FakeVectorSearchPort(_chunks(), fail_on_call=True),
        ).execute("鱼油", vector_collection="training")

        assert not r.failed
        assert PATH_VECTOR in r.degraded
        assert len(r.fused) > 0

    def test_keyword_failure_degrades(self) -> None:
        r = _use_case(
            keyword=FakeKeywordSearchPort(_chunks(), fail_on_call=True),
        ).execute("鱼油", vector_collection="training")

        assert not r.failed
        assert PATH_KEYWORD in r.degraded

    def test_two_paths_failure_still_works(self) -> None:
        """两路挂掉，剩一路也要能出结果。"""
        r = _use_case(
            graph=FakeGraphQueryPort(_products(), fail_on_call=True),
            vector=FakeVectorSearchPort(_chunks(), fail_on_call=True),
        ).execute("鱼油", vector_collection="training")

        assert not r.failed
        assert set(r.degraded) == {PATH_GRAPH, PATH_VECTOR}
        assert len(r.keyword_hits) > 0

    def test_degraded_reason_recorded(self) -> None:
        """降级原因必须能看出是哪种异常，不能只说「失败」。"""
        r = _use_case(
            graph=FakeGraphQueryPort(_products(), fail_on_call=True),
        ).execute("鱼油", vector_collection="training")

        reason = r.degraded[PATH_GRAPH]
        assert "AssertionError" in reason
        assert reason.strip()

    def test_all_three_fail_marks_failed(self) -> None:
        """三路全失败 → failed=True（P4 必须据此拒答）。"""
        r = _use_case(
            graph=FakeGraphQueryPort(_products(), fail_on_call=True),
            vector=FakeVectorSearchPort(_chunks(), fail_on_call=True),
            keyword=FakeKeywordSearchPort(_chunks(), fail_on_call=True),
        ).execute("鱼油", vector_collection="training")

        assert r.failed is True
        assert len(r.degraded) == 3
        assert r.fused == ()

    def test_port_not_assembled_degrades(self) -> None:
        """端口未装配（P0 时 graph/vector 都是 None）应降级而不是崩。"""
        r = RetrieveUseCase(
            graph=None, vector=None, keyword=_StubKeyword(_keyword_hits()),
        ).execute("鱼油", vector_collection="training")

        assert not r.failed
        assert PATH_GRAPH in r.degraded
        assert PATH_VECTOR in r.degraded
        assert "未装配" in r.degraded[PATH_GRAPH]


# ===================================================================
# 3. 分数阈值（★ P4 拒答的前提）
# ===================================================================
class TestScoreThreshold:
    """向量永远返回 Top-N，低分结果必须被阈值滤掉。"""

    def test_low_score_vector_hits_filtered(self) -> None:
        """向量 Top1 只有 0.31（低于阈值 0.50）→ 该路视为无有效召回。"""
        r = _use_case(
            vector=_StubVector(_vector_hits(high=False)),
            graph=None, keyword=None,
        ).execute("知识库里完全没有的问题", vector_collection="training")

        assert r.vector_hits == (), (
            "低分向量结果必须被滤掉。否则 P4 会以为「召回到了」，"
            "把无关内容喂给 LLM 生成答案。"
        )
        assert r.failed is True

    def test_high_score_vector_hits_kept(self) -> None:
        r = _use_case(
            vector=_StubVector(_vector_hits(high=True)),
            graph=None, keyword=None,
        ).execute("保健食品功能", vector_collection="training")

        assert len(r.vector_hits) == 3
        assert not r.failed

    def test_custom_threshold(self) -> None:
        """阈值可配置，便于后续调优。"""
        r = _use_case(
            vector=_StubVector(_vector_hits(high=False)),
            graph=None, keyword=None,
            min_vector_score=0.3,
        ).execute("查询", vector_collection="training")

        assert len(r.vector_hits) == 3, "阈值降到 0.3 后 0.31 的结果应被保留"

    def test_threshold_boundary_inclusive(self) -> None:
        """恰好等于阈值时应保留（>= 而非 >）。"""
        chunk = _chunks()[0]
        r = _use_case(
            vector=_StubVector([VectorHit(chunk=chunk, score=0.50)]),
            graph=None, keyword=None, min_vector_score=0.50,
        ).execute("查询", vector_collection="training")

        assert len(r.vector_hits) == 1


# ===================================================================
# 4. 权限过滤透传
# ===================================================================
class TestPermissionPassthrough:
    """权限参数必须透传到每一路，不能在用例层丢掉。"""

    def test_trainee_cannot_see_restricted(self) -> None:
        r = _use_case().execute(
            "题", vector_collection="training",
            allowed_permissions=("public", "internal"),
        )
        ids = {h.chunk.chunk_id for h in r.vector_hits}
        assert "Q001" not in ids, "学员不应看到 restricted 的题目"

    def test_manager_sees_restricted(self) -> None:
        r = _use_case().execute(
            "题", vector_collection="training",
            allowed_permissions=("public", "internal", "restricted"),
        )
        ids = {h.chunk.chunk_id for h in r.vector_hits}
        assert "Q001" in ids

    def test_keyword_also_filtered(self) -> None:
        r = _use_case().execute(
            "题", vector_collection="training",
            allowed_permissions=("public",),
        )
        ids = {h.chunk.chunk_id for h in r.keyword_hits}
        assert "Q001" not in ids


# ===================================================================
# 5. 融合语义
# ===================================================================
class TestFusion:
    def test_fused_limit_respected(self) -> None:
        r = _use_case().execute("鱼油", vector_collection="training",
                                fused_limit=2)
        assert len(r.fused) <= 2

    def test_fused_sorted_desc(self) -> None:
        r = _use_case().execute("鱼油", vector_collection="training")
        scores = [h.fused_score for h in r.fused]
        assert scores == sorted(scores, reverse=True)

    def test_doc_in_multi_paths_ranks_higher(self) -> None:
        """同时出现在向量与关键词路的结果，应比只出现一路的排前。"""
        r = _use_case().execute("鱼油", vector_collection="training")
        multi = [h for h in r.fused if len(h.sources) >= 2]
        single = [h for h in r.fused if len(h.sources) == 1]
        if multi and single:
            assert multi[0].fused_score > single[-1].fused_score

    def test_graph_result_not_in_fused(self) -> None:
        """★ 图谱结果（产品事实）不应进入文档排名。

        原因：产品事实是权威结构化数据，走「事实卡」直接呈现；
        文档块是供 LLM 参考的文本。两者混在一个排名里会导致：
          · 出现 `chunk=None` 的条目，下游按 fused 取内容时踩空
          · 一个分数同时表达「产品对不对」和「文档相关不相关」，语义混乱

        图谱的正确性由 `graph_hits` 单独承载，权重在上层体现。
        """
        r = _use_case().execute("鱼油", vector_collection="training")

        fused_ids = {h.chunk_id for h in r.fused}
        assert "P001" not in fused_ids, "产品 ID 不应出现在文档排名里"
        assert all(h.chunk is not None for h in r.fused), (
            "fused 里不应有 chunk=None 的条目"
        )

        # 但图谱结果必须保留给上层用
        assert [x.product_id for x in r.graph_hits] == ["P001"]

    def test_fused_only_contains_chunks(self) -> None:
        """fused 里每一条都必须有完整 chunk 元数据。"""
        r = _use_case().execute("鱼油", vector_collection="training")
        for h in r.fused:
            assert h.chunk is not None
            assert h.chunk.chunk_id
            assert h.chunk.text

    def test_top_doc_ids_helper(self) -> None:
        r = _use_case().execute("鱼油", vector_collection="training")
        assert len(r.top_doc_ids(3)) <= 3


# ===================================================================
# 6. 参数校验
# ===================================================================
class TestValidation:
    def test_missing_collection_degrades_not_crashes(self) -> None:
        """未传 collection 时该路降级，不影响其他路。"""
        r = _use_case().execute("鱼油", vector_collection="")
        assert PATH_VECTOR in r.degraded
        assert "vector_collection" in r.degraded[PATH_VECTOR]

    def test_unknown_intent_falls_back(self) -> None:
        """意图分类出错时不能崩。"""
        r = _use_case().execute("鱼油", intent="完全不认识的意图",
                                vector_collection="training")
        assert not r.failed
        assert abs(sum(r.weights) - 1.0) < 1e-6
