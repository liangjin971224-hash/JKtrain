"""测试替身自身的测试。

为什么替身也要测：
    如果 Fake 的实现有 bug，所有依赖它的应用层测试都会得出错误结论，
    而且错误是**静默**的——测试全绿但实际行为不对。
    尤其权限过滤、唯一约束、文件白名单这三处，替身必须与真实现同语义。
"""
from __future__ import annotations

import pytest

from domain.exceptions import (
    DuplicateDailySummaryError,
    FileTooLargeError,
    PathTraversalError,
    UnsupportedFileTypeError,
)
from domain.ports import DailySummaryRow, MissLogRow
from domain.value_objects import (
    ChunkMeta,
    PermissionLevel,
    ProductFacts,
    Severity,
    Contraindication,
)
from tests.fakes import (
    FailOnCallLLM,
    FakeCachePort,
    FakeGraphQueryPort,
    FakeKeywordSearchPort,
    FakeRepositoryPort,
    FakeVectorSearchPort,
    InMemoryFileStorage,
)


# ===================================================================
# 图谱替身
# ===================================================================
class TestFakeGraph:
    """排除法与别名归一必须真实现。"""

    def _products(self) -> dict[str, ProductFacts]:
        return {
            "P001": ProductFacts(
                product_id="P001", name="康源牌鱼油软胶囊",
                category="保健食品", approval_no="国食健注G20250012",
                contraindications=(
                    Contraindication("CI_BLEED", "出血倾向", Severity.CRITICAL),
                ),
                aliases=("鱼油",),
            ),
            "P002": ProductFacts(
                product_id="P002", name="康源牌辅酶Q10软胶囊",
                category="保健食品", approval_no="国食健注G20250013",
                aliases=("Q10",),
            ),
            "P016": ProductFacts(
                product_id="P016", name="康源牌复合蛋白固体饮料",
                category="普通食品",
            ),
        }

    def test_excluding_removes_bleeding_risk(self) -> None:
        """排除法：含出血禁忌的鱼油必须被排除。"""
        graph = FakeGraphQueryPort(self._products())
        names = {p.name for p in graph.find_products_excluding(["CI_BLEED"])}
        assert "康源牌鱼油软胶囊" not in names
        assert "康源牌辅酶Q10软胶囊" in names

    def test_excluding_skips_ordinary_food(self) -> None:
        """普通食品没有批准功能，不该出现在功效反查/排除结果里。"""
        graph = FakeGraphQueryPort(self._products())
        ids = {p.product_id for p in graph.find_products_excluding(["CI_BLEED"])}
        assert "P016" not in ids

    def test_resolve_alias(self) -> None:
        graph = FakeGraphQueryPort(self._products())
        hit = graph.resolve_alias("鱼油")
        assert hit is not None and hit.product_id == "P001"

    def test_resolve_alias_miss_returns_none(self) -> None:
        graph = FakeGraphQueryPort(self._products())
        assert graph.resolve_alias("不存在的名字") is None

    def test_check_contraindication_returns_hits_not_raises(self) -> None:
        """禁忌命中是**返回值**不是异常（业务结果 vs 编程错误的区分）。"""
        graph = FakeGraphQueryPort(self._products())
        hits = graph.check_contraindication("P001", ["CI_BLEED"])
        assert len(hits) == 1
        assert hits[0].severity is Severity.CRITICAL

    def test_check_contraindication_no_hit_is_empty(self) -> None:
        graph = FakeGraphQueryPort(self._products())
        assert graph.check_contraindication("P001", ["CI_RENAL"]) == []

    def test_check_unknown_product_returns_empty(self) -> None:
        """产品不存在时返回空列表，不抛——与端口契约一致。"""
        graph = FakeGraphQueryPort(self._products())
        assert graph.check_contraindication("P999", ["CI_BLEED"]) == []


# ===================================================================
# 向量替身：权限过滤必须真生效
# ===================================================================
class TestFakeVector:
    """★ 权限过滤如果替身不生效，P8 的测试就是假绿。"""

    def _chunks(self) -> list[ChunkMeta]:
        return [
            ChunkMeta(chunk_id="D001_c001", doc_id="D001", text="保健食品能说什么功能",
                      permission_level=PermissionLevel.PUBLIC),
            ChunkMeta(chunk_id="D005_qa001", doc_id="D005", text="保健食品功能声称答案",
                      permission_level=PermissionLevel.RESTRICTED),
            ChunkMeta(chunk_id="C01_s00", doc_id="C01", text="客户询问鱼油的案例",
                      permission_level=PermissionLevel.RESTRICTED),
        ]

    def test_trainee_cannot_see_restricted(self) -> None:
        vec = FakeVectorSearchPort(self._chunks())
        hits = vec.search("保健食品功能",
                          allowed_permissions=("public", "internal"))
        assert all(h.chunk.permission_level != "restricted" for h in hits)
        assert [h.chunk.chunk_id for h in hits] == ["D001_c001"]

    def test_manager_sees_restricted(self) -> None:
        vec = FakeVectorSearchPort(self._chunks())
        hits = vec.search("保健食品功能",
                          allowed_permissions=("public", "internal", "restricted"))
        assert len(hits) == 2

    def test_no_filter_returns_all(self) -> None:
        vec = FakeVectorSearchPort(self._chunks())
        hits = vec.search("保健食品功能", allowed_permissions=None)
        assert len(hits) == 2

    def test_embed_dim(self) -> None:
        vec = FakeVectorSearchPort([], dim=1024)
        assert len(vec.embed(["x"])[0]) == 1024


# ===================================================================
# 关键词替身
# ===================================================================
class TestFakeKeyword:
    def test_permission_filter(self) -> None:
        chunks = [
            ChunkMeta(chunk_id="D001", doc_id="D001", text="国食健注G20250012",
                      permission_level=PermissionLevel.PUBLIC),
            ChunkMeta(chunk_id="Q001", doc_id="Q001", text="国食健注G20250012 是哪款",
                      permission_level=PermissionLevel.RESTRICTED),
        ]
        kw = FakeKeywordSearchPort(chunks)
        hits = kw.search("国食健注", allowed_permissions=("public",))
        assert [h.chunk.chunk_id for h in hits] == ["D001"]


# ===================================================================
# 仓储替身：唯一约束必须真生效
# ===================================================================
class TestFakeRepository:
    def test_daily_summary_draft_overwritable(self) -> None:
        """★ 同人同天「草稿」可以反复覆盖保存（FR-045）。

        行为变更（2026-10-23）：
          原测试断言「同人同天第二条必须被拒」，
          但那是错的理解 —— FR-045 要求的「每人每天只允许一条」
          指的是**最终只有一条记录**，而不是「第二次保存报错」。

          草稿本来就要能反复修改，覆盖保存正是其实现方式。
          真正要拦的是「已提交还想改」。
        """
        repo = FakeRepositoryPort()
        repo.save_daily_summary(
            DailySummaryRow(trainee_no="T001", summary_date="2026-10-22",
                            content="第一版"))
        repo.save_daily_summary(
            DailySummaryRow(trainee_no="T001", summary_date="2026-10-22",
                            content="第二版"))

        saved = repo.get_daily_summary("T001", "2026-10-22")
        assert saved is not None
        assert saved.content == "第二版"

    def test_daily_summary_submitted_frozen(self) -> None:
        """★ 已提交的不能再被覆盖（这才是核心不变量）。"""
        from domain.exceptions import DailySummaryFrozenError

        repo = FakeRepositoryPort()
        repo.save_daily_summary(
            DailySummaryRow(trainee_no="T001", summary_date="2026-10-22",
                            content="正文", status="submitted"))

        with pytest.raises(DailySummaryFrozenError):
            repo.save_daily_summary(
                DailySummaryRow(trainee_no="T001", summary_date="2026-10-22",
                                content="想偷偷改"))

    def test_different_day_ok(self) -> None:
        repo = FakeRepositoryPort()
        repo.save_daily_summary(
            DailySummaryRow(trainee_no="T001", summary_date="2026-10-22"))
        repo.save_daily_summary(
            DailySummaryRow(trainee_no="T001", summary_date="2026-10-23"))
        assert repo.get_daily_summary("T001", "2026-10-23") is not None

    def test_miss_log_auto_timestamp(self) -> None:
        repo = FakeRepositoryPort()
        repo.append_miss_log(MissLogRow(id="M1", query="知识库里没有的问题"))
        assert repo.miss_logs[0].created_at != ""

    def test_list_questions_filter(self) -> None:
        repo = FakeRepositoryPort()
        repo.seed_questions([
            {"chunk_id": "Q001", "knowledge_point": "KP_A", "difficulty": 1},
            {"chunk_id": "Q002", "knowledge_point": "KP_A", "difficulty": 3},
            {"chunk_id": "Q003", "knowledge_point": "KP_B", "difficulty": 1},
        ])
        assert len(repo.list_questions(knowledge_point="KP_A")) == 2
        assert len(repo.list_questions(difficulty=1)) == 2
        assert len(repo.list_questions(knowledge_point="KP_A", difficulty=1)) == 1


# ===================================================================
# 文件存储替身：三条安全要求
# ===================================================================
class TestInMemoryFileStorage:
    def test_rejects_exe(self) -> None:
        """上传 .exe 必须被拒（P6 红灯项）。"""
        fs = InMemoryFileStorage()
        with pytest.raises(UnsupportedFileTypeError):
            fs.save(content=b"MZ", file_name="virus.exe",
                    category="submissions", sub_path="2026/11")

    def test_rejects_oversize(self) -> None:
        fs = InMemoryFileStorage()
        big = b"0" * (fs.MAX_SIZE + 1)
        with pytest.raises(FileTooLargeError):
            fs.save(content=big, file_name="a.pdf",
                    category="submissions", sub_path="2026/11")

    def test_rejects_path_traversal(self) -> None:
        """`../` 必须被挡（sub_path 来自用户输入）。"""
        fs = InMemoryFileStorage()
        with pytest.raises(PathTraversalError):
            fs.save(content=b"%PDF", file_name="a.pdf",
                    category="submissions", sub_path="../../../etc")

    def test_accepts_pdf_and_roundtrip(self) -> None:
        fs = InMemoryFileStorage()
        stored = fs.save(content=b"%PDF-1.4 test", file_name="答卷.pdf",
                         category="submissions", sub_path="2026/11/E001")
        assert stored.mime_type == "application/pdf"
        assert fs.read(stored.relative_path) == b"%PDF-1.4 test"
        assert fs.exists(stored.relative_path) is True

    def test_delete_is_idempotent(self) -> None:
        fs = InMemoryFileStorage()
        fs.delete("not/exist.pdf")           # 不应抛
        stored = fs.save(content=b"x", file_name="a.png",
                         category="exports", sub_path="tmp")
        fs.delete(stored.relative_path)
        fs.delete(stored.relative_path)
        assert fs.exists(stored.relative_path) is False


# ===================================================================
# 缓存替身
# ===================================================================
class TestFakeCache:
    def test_set_get(self) -> None:
        c = FakeCachePort()
        c.set("k", {"a": 1})
        assert c.get("k") == {"a": 1}

    def test_miss_returns_none(self) -> None:
        assert FakeCachePort().get("nope") is None

    def test_clear_prefix(self) -> None:
        c = FakeCachePort()
        c.set("drill:1:a", 1)
        c.set("drill:1:b", 2)
        c.set("other", 3)
        assert c.clear_prefix("drill:1:") == 2
        assert c.get("other") == 3

    def test_ttl_expires(self) -> None:
        c = FakeCachePort()
        c.set("k", "v", ttl_sec=0)          # 立即过期
        assert c.get("k") is None


# ===================================================================
# FailOnCallLLM：证明"不调 LLM"这条架构约束
# ===================================================================
class TestFailOnCallLLM:
    """这是本项目最重要的测试手法。"""

    def test_generate_raises(self) -> None:
        with pytest.raises(AssertionError, match="不应调用 LLM"):
            FailOnCallLLM().generate("prompt")

    def test_classify_raises(self) -> None:
        with pytest.raises(AssertionError):
            FailOnCallLLM().classify("text", ["a", "b"])

    def test_is_available_true(self) -> None:
        """可用但一调就炸——正好证明「有能力但未被使用」。"""
        assert FailOnCallLLM().is_available() is True
