"""Milvus 适配器的非集成测试（不连库、不加载模型）。

覆盖三处容易做错、且错了不会立刻暴露的地方：
    1. 权限过滤表达式的组装（拼错就是越权）
    2. 输出字段必须按 collection schema 动态取（写死会直接报错）
    3. 模型必须惰性加载（模块导入时加载会拖慢所有 import 方）
"""
from __future__ import annotations

import pytest

from config.settings import Settings
from domain.exceptions import EmbeddingDimensionError
from infrastructure.exceptions import VectorSearchError
from infrastructure.vector.milvus_search_adapter import (
    MilvusSearchAdapter,
    _to_chunk_meta,
)


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,                          # type: ignore[call-arg]
        embed_model_dir=str(tmp_path / "model"),
        expected_embed_dim=1024,
    )


@pytest.fixture()
def adapter(settings: Settings) -> MilvusSearchAdapter:
    """只构造不连库：client 传 None，只测不触网的方法。"""
    return MilvusSearchAdapter(None, settings=settings)   # type: ignore[arg-type]


# ===================================================================
# 权限过滤表达式组装（★ 越权的源头）
# ===================================================================
class TestBuildFilter:
    def test_trainee_permissions(self, adapter) -> None:
        expr = adapter._build_filter(("public", "internal"), None)
        assert 'permission_level in ["public", "internal"]' in expr
        assert 'review_status == "approved"' in expr

    def test_manager_includes_restricted(self, adapter) -> None:
        expr = adapter._build_filter(
            ("public", "internal", "restricted"), None)
        assert "restricted" in expr

    def test_none_means_no_permission_constraint(self, adapter) -> None:
        """None = 不限制权限（仅内部调用）。"""
        expr = adapter._build_filter(None, None)
        assert "permission_level" not in expr
        assert 'review_status == "approved"' in expr

    def test_empty_tuple_means_deny_all(self, adapter) -> None:
        """空元组 = 一个都不允许，必须生成不可能成立的条件。

        ★ 不能和 None 混淆：None 是「不限制」，() 是「一律禁止」。
        若实现把空元组当成 None，就会变成越权全开。
        """
        expr = adapter._build_filter((), None)
        assert "__none__" in expr

    def test_invalid_permission_raises(self, adapter) -> None:
        """非法权限值必须报错。

        拼错成 "publik" 时若静默忽略，会变成「不过滤」→ 越权。
        """
        with pytest.raises(VectorSearchError, match="非法权限级别"):
            adapter._build_filter(("public", "publik"), None)

    def test_extra_filter_combined_with_and(self, adapter) -> None:
        expr = adapter._build_filter(
            ("public",), 'category == "training"')
        assert "and" in expr
        assert 'category == "training"' in expr
        # 权限条件不能被额外 filter 覆盖掉
        assert 'permission_level in ["public"]' in expr

    def test_review_only_can_be_disabled(self, adapter) -> None:
        """巡检场景可能要包含未审核内容。"""
        expr = adapter._build_filter(("public",), None, review_only=False)
        assert "review_status" not in expr

    def test_double_quotes_not_breakable(self, adapter) -> None:
        """权限值受白名单约束，注入不进 filter。"""
        with pytest.raises(VectorSearchError):
            adapter._build_filter(('public" or "1"=="1',), None)


# ===================================================================
# 输出字段动态获取（★ 写死会直接报错）
# ===================================================================
class TestOutputFields:
    def test_question_collection_includes_kp_fields(self, settings) -> None:
        """题库 collection 必须带上 knowledge_point 等字段。"""

        class _FakeClient:
            def describe_collection(self, name: str) -> dict:
                return {"fields": [
                    {"name": "id"}, {"name": "vector"},
                    {"name": "chunk_id"}, {"name": "text"},
                    {"name": "doc_id"}, {"name": "title"},
                    {"name": "section"}, {"name": "category"},
                    {"name": "permission_level"}, {"name": "review_status"},
                    {"name": "data_origin"}, {"name": "source_file"},
                    {"name": "chunk_index"},
                    {"name": "knowledge_point"}, {"name": "question_type"},
                    {"name": "difficulty"},
                ]}

        adapter = MilvusSearchAdapter(_FakeClient(), settings=settings)
        fields = adapter._output_fields("health_question_bank")

        assert "knowledge_point" in fields
        assert "question_type" in fields
        assert "difficulty" in fields
        # ★ vector 绝不能出现在输出字段里
        assert "vector" not in fields
        assert "id" not in fields

    def test_training_collection_excludes_kp_fields(self, settings) -> None:
        """培训库没有 knowledge_point，指定它会报
        `MilvusException: field knowledge_point not exist`。"""

        class _FakeClient:
            def describe_collection(self, name: str) -> dict:
                return {"fields": [
                    {"name": "id"}, {"name": "vector"},
                    {"name": "chunk_id"}, {"name": "text"},
                    {"name": "doc_id"}, {"name": "title"},
                    {"name": "section"}, {"name": "category"},
                    {"name": "permission_level"}, {"name": "review_status"},
                    {"name": "data_origin"}, {"name": "source_file"},
                    {"name": "chunk_index"},
                ]}

        adapter = MilvusSearchAdapter(_FakeClient(), settings=settings)
        fields = adapter._output_fields("health_training_docs")

        assert "knowledge_point" not in fields
        assert "chunk_id" in fields
        assert "vector" not in fields

    def test_fields_are_cached(self, settings) -> None:
        """schema 探测应缓存，避免每次检索都 describe。"""
        calls = {"n": 0}

        class _FakeClient:
            def describe_collection(self, name: str) -> dict:
                calls["n"] += 1
                return {"fields": [{"name": "chunk_id"}, {"name": "text"}]}

        adapter = MilvusSearchAdapter(_FakeClient(), settings=settings)
        adapter._output_fields("x")
        adapter._output_fields("x")
        adapter._output_fields("x")
        assert calls["n"] == 1, "describe_collection 被重复调用，缓存未生效"


# ===================================================================
# 惰性加载（★ 导入代价）
# ===================================================================
class TestLazyLoading:
    def test_model_not_loaded_at_construction(self, adapter) -> None:
        """构造时不得加载模型（`import ST` 占启动 87%）。"""
        assert adapter._model is None

    def test_embed_empty_does_not_load_model(self, adapter) -> None:
        """空输入直接返回，不该触发模型加载。"""
        assert adapter.embed([]) == []
        assert adapter._model is None


# ===================================================================
# 空查询与参数校验（不触网）
# ===================================================================
class TestSearchGuards:
    def test_empty_query_returns_empty_without_search(self, adapter) -> None:
        """空查询直接返回，不调 client.search（client 是 None 会炸）。"""
        assert adapter.search("", collection="x") == []
        assert adapter.search("   ", collection="x") == []

    def test_zero_top_k_returns_empty(self, adapter) -> None:
        assert adapter.search("查询", collection="x", top_k=0) == []

    def test_invalid_permission_checked_before_embedding(self, adapter) -> None:
        """参数非法应在编码前就报错，不浪费一次模型推理。"""
        with pytest.raises(VectorSearchError, match="非法权限级别"):
            adapter.search("查询", collection="x",
                           allowed_permissions=("bad",))

    def test_get_chunks_empty(self, adapter) -> None:
        assert adapter.get_chunks_by_ids([], collection="x") == []


# ===================================================================
# ChunkMeta 转换
# ===================================================================
class TestToChunkMeta:
    def test_drops_vector(self) -> None:
        """即使上游误传 vector，也不能进领域对象。"""
        meta = _to_chunk_meta({
            "chunk_id": "D001_c000", "doc_id": "D001", "text": "正文",
            "vector": [0.1] * 1024, "permission_level": "public",
        })
        assert not hasattr(meta, "vector")

    def test_null_fields_become_empty(self) -> None:
        """Milvus 可能返回 None（字段存在但值为空），不能变成字符串 "None"。"""
        meta = _to_chunk_meta({
            "chunk_id": "x", "doc_id": "x", "text": "t",
            "title": None, "section": None, "permission_level": None,
        })
        assert meta.title == ""
        assert meta.section == ""
        # 权限缺失时降级为 internal（不升不降，保守）
        assert meta.permission_level == "internal"

    def test_question_fields(self) -> None:
        meta = _to_chunk_meta({
            "chunk_id": "Q001", "doc_id": "Q001", "text": "题",
            "knowledge_point": "KP_PROD_CATEGORY",
            "question_type": "单选", "difficulty": 2,
        })
        assert meta.knowledge_point == "KP_PROD_CATEGORY"
        assert meta.question_type == "单选"
        assert meta.difficulty == 2

    def test_missing_question_fields_default_zero(self) -> None:
        """培训库的行没有 question 字段，应默认 0 而不是报错。"""
        meta = _to_chunk_meta({
            "chunk_id": "D001", "doc_id": "D001", "text": "t",
        })
        assert meta.difficulty == 0
        assert meta.knowledge_point == ""


# ===================================================================
# 维度校验
# ===================================================================
class TestCheckDim:
    def test_match_passes(self, adapter) -> None:
        adapter._check_dim(1024)           # 不抛

    def test_mismatch_raises_with_hint(self, adapter) -> None:
        with pytest.raises(EmbeddingDimensionError) as exc_info:
            adapter._check_dim(768)
        msg = str(exc_info.value)
        assert "768" in msg and "1024" in msg
        assert "EMBED_MODEL_DIR" in msg
