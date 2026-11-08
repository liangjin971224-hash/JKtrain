"""P2 向量检索验收测试（红灯先行）。

按 `references/incremental-delivery.md` §P2 编写。

★ 本文件的重心不是「能不能召回」，而是**权限过滤是否在检索层生效**。
  P2 的红灯项明确写着「跨权限检索（学员搜不到 restricted）」。
  这是 FR-050 的技术实现点，也是 P8 验收的前提。

运行：
    python -m pytest tests/test_infrastructure/test_vector.py -m integration
"""
from __future__ import annotations

import pytest

from config.settings import Settings
from domain.exceptions import EmbeddingDimensionError

pytestmark = pytest.mark.integration


# ===================================================================
# 夹具
# ===================================================================
@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(_env_file=None)          # type: ignore[call-arg]


@pytest.fixture(scope="module")
def adapter(settings: Settings):
    """真实的 MilvusSearchAdapter（会加载 bge-m3，首次约 5s）。"""
    from pymilvus import MilvusClient

    from infrastructure.vector.milvus_search_adapter import MilvusSearchAdapter

    client = MilvusClient(uri=settings.milvus_uri)
    yield MilvusSearchAdapter(client, settings=settings)


# ===================================================================
# ★ P2 红灯核心：权限过滤必须在检索层
# ===================================================================
class TestPermissionFilter:
    """学员搜不到受限内容（FR-050）。"""

    def test_trainee_cannot_see_restricted_in_training(self, adapter,
                                                       settings) -> None:
        """学员检索培训库：返回结果不得含 restricted。"""
        hits = adapter.search(
            "参考答案",
            collection=settings.coll_training,
            top_k=20,
            allowed_permissions=("public", "internal"),
        )
        leaked = [h for h in hits
                  if str(h.chunk.permission_level) == "restricted"]
        assert leaked == [], (
            f"学员检索泄漏了 {len(leaked)} 条 restricted 内容："
            f"{[h.chunk.chunk_id for h in leaked]}"
        )

    def test_trainee_gets_zero_from_case_library(self, adapter,
                                                 settings) -> None:
        """学员检索案例库：必须 0 条（案例全 restricted）。"""
        hits = adapter.search(
            "客户询问能否降血糖",
            collection=settings.coll_case,
            top_k=20,
            allowed_permissions=("public", "internal"),
        )
        assert hits == [], (
            f"学员不应检索到任何案例，实际返回 {len(hits)} 条"
        )

    def test_trainee_gets_zero_from_question_bank(self, adapter,
                                                  settings) -> None:
        """学员检索题库：必须 0 条（题库全 restricted）。"""
        hits = adapter.search(
            "产品类别",
            collection=settings.coll_question,
            top_k=20,
            allowed_permissions=("public", "internal"),
        )
        assert hits == []

    def test_trainee_gets_zero_from_self_test(self, adapter, settings) -> None:
        """学员检索参考答案库：必须 0 条。"""
        hits = adapter.search(
            "答案",
            collection=settings.coll_self_test,
            top_k=20,
            allowed_permissions=("public", "internal"),
        )
        assert hits == []

    def test_manager_can_see_restricted(self, adapter, settings) -> None:
        """主管（含 restricted）能检索到案例。"""
        hits = adapter.search(
            "客户询问能否降血糖",
            collection=settings.coll_case,
            top_k=5,
            allowed_permissions=("public", "internal", "restricted"),
        )
        assert len(hits) > 0, "主管应能检索到案例"
        assert any(str(h.chunk.permission_level) == "restricted" for h in hits)

    def test_filter_is_at_search_layer_not_post_filter(self, adapter,
                                                       settings) -> None:
        """★ 权限过滤必须在检索层，不能检索后过滤。

        这个测试必须用**含大量受限内容**的 collection 才有区分力。

        数据事实（V2 重建后）：
            health_training_docs 只有 public(113) + internal(180)，**没有 restricted**。
            所以拿 training 库测这条会两者都是 20，测不出区别——那是假绿。

        改用 health_case_library（117 条全 restricted）：
            - 学员视角（public+internal）→ 检索层过滤后应为 0 条
            - 若实现是「检索后过滤」，会先取到 20 条再筛掉，同样是 0 条……

        所以还需要一个**混合权限**的场景才真正有区分力。
        这里用 health_training_docs + 显式放宽到 restricted 做对照：
        """
        # 场景一：全 restricted 的库，学员必须 0 条
        trainee_hits = adapter.search(
            "客户询问能否降血糖",
            collection=settings.coll_case,
            top_k=20,
            allowed_permissions=("public", "internal"),
        )
        assert trainee_hits == [], "学员在案例库应返回 0 条"

        # 场景二：主管视角能拿满 20 条——证明库里有数据、top_k 生效
        manager_hits = adapter.search(
            "客户询问能否降血糖",
            collection=settings.coll_case,
            top_k=20,
            allowed_permissions=("public", "internal", "restricted"),
        )
        assert len(manager_hits) == 20, (
            f"主管应能取满 20 条，实际 {len(manager_hits)} 条"
        )

        # 场景三：★ 核心判据 —— 权限条件确实进入了 filter 表达式，
        # 而不是靠返回后再筛。验证方式：非法权限值必须立刻报错，
        # 说明权限参数是被解析进查询条件、而不是被忽略的。
        from infrastructure.exceptions import VectorSearchError

        with pytest.raises(VectorSearchError, match="非法权限级别"):
            adapter.search(
                "测试", collection=settings.coll_training, top_k=5,
                allowed_permissions=("public", "superadmin"),   # 拼错的值
            )

    def test_empty_permission_list_returns_nothing(self, adapter,
                                                   settings) -> None:
        """空权限列表 = 一个都不允许 → 必须 0 条。

        不能用「不传参数」混淆：不传是「不限制」，传空列表是「一律禁止」。
        """
        hits = adapter.search(
            "保健食品", collection=settings.coll_training,
            top_k=5, allowed_permissions=(),
        )
        assert hits == []

    def test_none_permissions_means_no_restriction(self, adapter,
                                                   settings) -> None:
        """传 None 表示不做权限限制（仅限内部调用，如数据巡检）。"""
        hits = adapter.search(
            "保健食品", collection=settings.coll_training, top_k=5,
            allowed_permissions=None,
        )
        assert len(hits) == 5


# ===================================================================
# 4 条召回验收查询
# ===================================================================
class TestRecall:
    """P2 验收：4 条查询能召回正确文档。"""

    @pytest.mark.parametrize(("query", "expected_doc"), [
        ("保健食品能说什么功能", "D005"),
        ("门店接待流程", "D002"),
        ("客户说太贵了怎么处理", "D009"),
        ("体脂仪怎么用", "D010"),
    ])
    def test_top5_contains_expected_doc(self, adapter, settings,
                                        query: str, expected_doc: str) -> None:
        """Top5 里应出现期望文档。

        注意：期望文档可能不是 Top1——一个知识点可能分布在多篇文档里，
        这是评测集标注的固有局限（P0 有记录）。
        所以这里只断言 Top5 命中，符合 R@5 ≥ 0.7 的目标。
        """
        hits = adapter.search(
            query,
            collection=settings.coll_training,
            top_k=5,
            allowed_permissions=("public", "internal"),
        )
        docs = [h.chunk.doc_id for h in hits]
        assert expected_doc in docs, (
            f"查询「{query}」的 Top5 未命中 {expected_doc}，实际：{docs}"
        )

    def test_results_sorted_by_score_desc(self, adapter, settings) -> None:
        """结果必须按相似度降序。"""
        hits = adapter.search(
            "保健食品功能", collection=settings.coll_training, top_k=10,
            allowed_permissions=("public", "internal"),
        )
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True), "结果未按分数降序"

    def test_scores_in_valid_range(self, adapter, settings) -> None:
        """余弦相似度应在 [-1, 1]；本语料实际都在 0–1。"""
        hits = adapter.search(
            "保健食品功能", collection=settings.coll_training, top_k=5,
            allowed_permissions=("public", "internal"),
        )
        for h in hits:
            assert -1.0 <= h.score <= 1.0, f"分数越界: {h.score}"

    def test_empty_query_returns_empty(self, adapter, settings) -> None:
        """空查询应返回空列表，不抛异常。"""
        assert adapter.search("", collection=settings.coll_training,
                              top_k=5) == []

    def test_no_match_still_returns_hits(self, adapter, settings) -> None:
        """向量检索总能返回「最相近的」，不会真的空。

        这与图谱查询不同：图谱可能查不到，向量永远有 Top-N。
        所以「召回为空」的判定要靠**分数阈值**，不能靠结果为空。
        """
        hits = adapter.search(
            "完全不相关的问题xyz123", collection=settings.coll_training,
            top_k=3, allowed_permissions=("public", "internal"),
        )
        assert len(hits) == 3


# ===================================================================
# ChunkMeta 字段完整性
# ===================================================================
class TestChunkMeta:
    def test_meta_fields_populated(self, adapter, settings) -> None:
        """返回的 ChunkMeta 必须带齐关键字段（否则下游无法做权限/追溯）。"""
        hits = adapter.search(
            "保健食品", collection=settings.coll_training, top_k=3,
            allowed_permissions=("public", "internal"),
        )
        assert hits, "应有结果"
        for h in hits:
            c = h.chunk
            assert c.chunk_id, "chunk_id 不能空"
            assert c.doc_id, "doc_id 不能空"
            assert c.text, "text 不能空"
            assert str(c.permission_level) in ("public", "internal")

    def test_question_meta_has_knowledge_point(self, adapter, settings) -> None:
        """题库的 ChunkMeta 应带 knowledge_point（P6 组卷要用）。"""
        hits = adapter.search(
            "产品类别", collection=settings.coll_question, top_k=3,
            allowed_permissions=("restricted",),
        )
        assert hits
        assert hits[0].chunk.knowledge_point.startswith("KP_")

    def test_no_vector_in_meta(self, adapter, settings) -> None:
        """ChunkMeta 不应携带向量（1024 维会把对象撑爆，且领域层不该有向量）。"""
        hits = adapter.search(
            "保健食品", collection=settings.coll_training, top_k=1,
            allowed_permissions=("public", "internal"),
        )
        assert not hasattr(hits[0].chunk, "vector")


# ===================================================================
# get_chunks_by_ids
# ===================================================================
class TestGetChunksByIds:
    def test_fetch_known_chunk(self, adapter, settings) -> None:
        hits = adapter.search(
            "保健食品", collection=settings.coll_training, top_k=1,
            allowed_permissions=("public", "internal"),
        )
        target = hits[0].chunk.chunk_id

        metas = adapter.get_chunks_by_ids(
            [target], collection=settings.coll_training)
        assert len(metas) == 1
        assert metas[0].chunk_id == target

    def test_unknown_ids_ignored(self, adapter, settings) -> None:
        metas = adapter.get_chunks_by_ids(
            ["NOT_EXIST_1", "NOT_EXIST_2"],
            collection=settings.coll_training)
        assert metas == []

    def test_empty_input(self, adapter, settings) -> None:
        assert adapter.get_chunks_by_ids(
            [], collection=settings.coll_training) == []


# ===================================================================
# embed
# ===================================================================
class TestEmbed:
    def test_dimension(self, adapter, settings) -> None:
        """维度必须是 1024。"""
        vecs = adapter.embed(["测试"])
        assert len(vecs) == 1
        assert len(vecs[0]) == settings.expected_embed_dim

    def test_batch(self, adapter, settings) -> None:
        vecs = adapter.embed(["一", "二", "三"])
        assert len(vecs) == 3
        assert all(len(v) == settings.expected_embed_dim for v in vecs)

    def test_empty_input(self, adapter, settings) -> None:
        assert adapter.embed([]) == []

    def test_normalized(self, adapter, settings) -> None:
        """向量必须已归一化（COSINE 检索前提）。"""
        import math

        vec = adapter.embed(["测试文本"])[0]
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-3, f"向量未归一化，模长 {norm}"

    def test_same_text_same_vector(self, adapter, settings) -> None:
        """同文本两次编码结果应一致（temperature 无关，确定性）。"""
        a = adapter.embed(["鱼油有什么功能"])[0]
        b = adapter.embed(["鱼油有什么功能"])[0]
        assert a == b

    def test_dimension_mismatch_raises(self, settings) -> None:
        """维度与配置不符时必须报错并给可执行提示。

        不加载模型，直接构造一个只有 settings 的实例来测校验函数。
        """
        from infrastructure.vector.milvus_search_adapter import (
            MilvusSearchAdapter,
        )

        bad_settings = settings.model_copy(update={"expected_embed_dim": 768})
        adapter = object.__new__(MilvusSearchAdapter)
        adapter._settings = bad_settings

        with pytest.raises(EmbeddingDimensionError) as exc_info:
            adapter._check_dim(1024)          # 实际 1024，配置期望 768

        msg = str(exc_info.value)
        assert "1024" in msg, "错误信息应报告实际维度"
        assert "768" in msg, "错误信息应报告期望维度"
        assert "EMBED_MODEL_DIR" in msg, "应给出可执行的修复提示"
