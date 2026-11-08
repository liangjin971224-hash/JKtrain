"""P1 图谱查询验收测试（红灯先行）。

按 `references/incremental-delivery.md` §P1 的 6 条验收查询编写。
本文件在 adapter 尚未实现时必须**全部失败**。

运行方式：
    python -m pytest tests/test_infrastructure/test_graph.py -m integration

★ 为什么打 integration 标记：
   它连真 Neo4j，属于 10% 的集成测试。日常 `pytest` 默认跳过，
   阶段验收时显式跑 `-m integration`。

★ 为什么用真库而不是 fake：
   adapter 的职责是「把 Cypher 结果转成领域对象」，
   这个转换只有在真实数据结构上验证才有意义。
   用 fake 测 adapter 等于测自己编的数据。
"""
from __future__ import annotations

import pytest

from config.settings import Settings
from domain.value_objects import Severity

pytestmark = pytest.mark.integration


# ===================================================================
# 夹具：真连 Neo4j
# ===================================================================
@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(_env_file=None)          # type: ignore[call-arg]


@pytest.fixture(scope="module")
def adapter(settings: Settings):
    """真实的 Neo4jGraphAdapter。"""
    from neo4j import GraphDatabase

    from infrastructure.graph.neo4j_graph_adapter import Neo4jGraphAdapter

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    try:
        yield Neo4jGraphAdapter(driver, settings.neo4j_database)
    finally:
        driver.close()


# ===================================================================
# 验收查询 1：完整产品画像
# ===================================================================
class TestGetProductFacts:
    """`get_product_facts("P001")` 返回鱼油完整画像。"""

    def test_p001_basic_fields(self, adapter) -> None:
        facts = adapter.get_product_facts("P001")
        assert facts is not None, "P001 应存在"
        assert facts.product_id == "P001"
        assert facts.name == "康源牌鱼油软胶囊"
        assert facts.category == "保健食品"
        assert facts.approval_no == "国食健注G20250012"

    def test_p001_has_six_contraindications(self, adapter) -> None:
        """验收要求：含 6 个不适宜人群。"""
        facts = adapter.get_product_facts("P001")
        assert facts is not None
        assert len(facts.contraindications) == 6, (
            f"P001 应有 6 个不适宜人群，实际 {len(facts.contraindications)}: "
            f"{[c.name for c in facts.contraindications]}"
        )
        ids = {c.contra_id for c in facts.contraindications}
        assert ids == {"CI_CHILD", "CI_PREG", "CI_LACT",
                       "CI_BLEED", "CI_LIVER", "CI_SEAFOOD"}

    def test_p001_claims_are_catalog_text(self, adapter) -> None:
        """批准功能必须是目录原文，不能是改写。"""
        facts = adapter.get_product_facts("P001")
        assert facts is not None
        assert len(facts.claims) == 1
        claim = facts.claims[0]
        assert claim.efficacy_id == "EFF_19"
        assert "有助于维持血脂" in claim.text
        assert claim.catalog_no == 19

    def test_p001_aliases(self, adapter) -> None:
        facts = adapter.get_product_facts("P001")
        assert facts is not None
        assert "鱼油" in facts.aliases
        assert "深海鱼油" in facts.aliases

    def test_p001_usage(self, adapter) -> None:
        facts = adapter.get_product_facts("P001")
        assert facts is not None
        assert facts.usage is not None
        assert "每日" in facts.usage.dosage
        assert "4.0g" in facts.usage.daily_limit

    def test_p001_suitable_for(self, adapter) -> None:
        facts = adapter.get_product_facts("P001")
        assert facts is not None
        assert "血脂偏高者" in facts.suitable_for

    def test_critical_contraindications_extracted(self, adapter) -> None:
        """critical 级禁忌必须能单独取出（P4 护栏要用）。"""
        facts = adapter.get_product_facts("P001")
        assert facts is not None
        crit = facts.critical_contraindications
        assert len(crit) >= 1
        assert all(c.severity is Severity.CRITICAL for c in crit)

    def test_ordinary_food_has_no_claims(self, adapter) -> None:
        """普通食品 P016：claims 必须为空（约束 #2 的数据层保证）。"""
        facts = adapter.get_product_facts("P016")
        assert facts is not None
        assert facts.category == "普通食品"
        assert facts.claims == ()
        assert facts.is_health_food is False

    def test_unknown_product_returns_none(self, adapter) -> None:
        """不存在的产品返回 None，不抛异常。"""
        assert adapter.get_product_facts("P999") is None


# ===================================================================
# 验收查询 2：按功能反查
# ===================================================================
class TestFindByEfficacy:
    def test_eff_19_returns_p001(self, adapter) -> None:
        """验收要求：`find_products_by_efficacy("EFF_19")` 返回 P001。"""
        refs = adapter.find_products_by_efficacy("EFF_19")
        ids = {r.product_id for r in refs}
        assert "P001" in ids, f"EFF_19（血脂）应含 P001，实际 {ids}"

    def test_only_health_food_returned(self, adapter) -> None:
        """只返回保健食品——普通食品没有批准功能。"""
        refs = adapter.find_products_by_efficacy("EFF_19")
        assert all(r.category == "保健食品" for r in refs)

    def test_unknown_efficacy_returns_empty(self, adapter) -> None:
        assert adapter.find_products_by_efficacy("EFF_NOT_EXIST") == []


# ===================================================================
# 验收查询 3：排除法（★ P1 核心）
# ===================================================================
class TestFindProductsExcluding:
    """这是图谱相对向量检索的核心优势，必须验证。"""

    def test_excludes_bleeding_risk(self, adapter) -> None:
        """验收要求：`find_products_excluding(["CI_BLEED"])` **不含 P001**。"""
        refs = adapter.find_products_excluding(["CI_BLEED"])
        ids = {r.product_id for r in refs}

        assert "P001" not in ids, (
            "鱼油有出血禁忌，必须被排除。若此断言失败，说明排除法失效——"
            "这会导致向在服抗凝药的客户推荐鱼油，是安全事故级别的问题。"
        )

    def test_returns_other_health_food(self, adapter) -> None:
        """验收要求：含其他保健食品。"""
        refs = adapter.find_products_excluding(["CI_BLEED"])
        ids = {r.product_id for r in refs}
        assert len(refs) == 14, f"应为 14 个保健食品，实际 {len(refs)}"
        assert "P002" in ids, "辅酶Q10 无出血禁忌，应保留"

    def test_ordinary_food_never_returned(self, adapter) -> None:
        """排除法只针对保健食品（普通食品无批准功能，不该出现在这里）。"""
        refs = adapter.find_products_excluding(["CI_BLEED"])
        assert all(r.category == "保健食品" for r in refs)
        ids = {r.product_id for r in refs}
        assert "P016" not in ids

    def test_multiple_contra_ids(self, adapter) -> None:
        """多禁忌组合：CI_BLEED + CI_RENAL。

        P001 忌出血、P004/P005 忌肾病，三个都应被排除。
        """
        refs = adapter.find_products_excluding(["CI_BLEED", "CI_RENAL"])
        ids = {r.product_id for r in refs}
        assert "P001" not in ids
        assert "P004" not in ids
        assert "P005" not in ids

    def test_empty_input_returns_all_health_food(self, adapter) -> None:
        """不排除任何禁忌时，应返回全部 15 个保健食品。"""
        refs = adapter.find_products_excluding([])
        assert len(refs) == 15


# ===================================================================
# 验收查询 4：别名归一
# ===================================================================
class TestResolveAlias:
    def test_resolve_yu_you(self, adapter) -> None:
        """验收要求：`resolve_alias("鱼油")` 返回 P001。"""
        ref = adapter.resolve_alias("鱼油")
        assert ref is not None
        assert ref.product_id == "P001"

    def test_resolve_by_full_name(self, adapter) -> None:
        """标准全名也应能解析（客户偶尔会说全称）。"""
        ref = adapter.resolve_alias("康源牌鱼油软胶囊")
        assert ref is not None
        assert ref.product_id == "P001"

    def test_resolve_multiple_aliases(self, adapter) -> None:
        """多个常见口语名都应能映射。"""
        cases = {"Q10": "P002", "蛋白粉": None, "钙片": None,
                 "褪黑素": None, "益生菌": None}
        for alias, expected in cases.items():
            ref = adapter.resolve_alias(alias)
            assert ref is not None, f"别名「{alias}」应能解析"
            if expected:
                assert ref.product_id == expected

    def test_unknown_alias_returns_none(self, adapter) -> None:
        assert adapter.resolve_alias("不存在的产品名") is None

    def test_whitespace_tolerated(self, adapter) -> None:
        """客户输入常带空格。"""
        ref = adapter.resolve_alias("  鱼油  ")
        assert ref is not None
        assert ref.product_id == "P001"


# ===================================================================
# 验收查询 5：禁忌检查
# ===================================================================
class TestCheckContraindication:
    def test_p001_no_renal_contraindication(self, adapter) -> None:
        """验收要求：`check_contraindication("P001", ["CI_RENAL"])` 返回空。

        P001 不忌肾病——这个"返回空"是**正常业务结果**，不是异常。
        """
        hits = adapter.check_contraindication("P001", ["CI_RENAL"])
        assert hits == []

    def test_p001_has_bleeding_contraindication(self, adapter) -> None:
        hits = adapter.check_contraindication("P001", ["CI_BLEED"])
        assert len(hits) == 1
        hit = hits[0]
        assert hit.product_id == "P001"
        assert hit.contra_id == "CI_BLEED"
        assert hit.severity is Severity.CRITICAL
        assert hit.reason, "禁忌必须带 reason（47/47 全带）"
        assert "出血" in hit.name

    def test_p004_has_renal_contraindication(self, adapter) -> None:
        """P004 乳清蛋白粉忌肾病。"""
        hits = adapter.check_contraindication("P004", ["CI_RENAL"])
        assert len(hits) == 1
        assert hits[0].contra_id == "CI_RENAL"

    def test_multiple_ids_partial_match(self, adapter) -> None:
        """传多个禁忌，只返回命中的那几个。"""
        hits = adapter.check_contraindication(
            "P001", ["CI_RENAL", "CI_BLEED", "CI_PREG"])
        ids = {h.contra_id for h in hits}
        assert ids == {"CI_BLEED", "CI_PREG"}

    def test_unknown_product_returns_empty(self, adapter) -> None:
        assert adapter.check_contraindication("P999", ["CI_BLEED"]) == []

    def test_empty_input(self, adapter) -> None:
        assert adapter.check_contraindication("P001", []) == []


# ===================================================================
# 验收查询 6：需求反查（多跳）
# ===================================================================
class TestFindProductsByNeed:
    def test_need_lipid_returns_p001(self, adapter) -> None:
        """NEED_LIPID（血脂健康关注）经 Efficacy 多跳应找到 P001。"""
        refs = adapter.find_products_by_need("NEED_LIPID")
        ids = {r.product_id for r in refs}
        assert "P001" in ids, f"NEED_LIPID 应含 P001，实际 {ids}"

    def test_unknown_need_returns_empty(self, adapter) -> None:
        assert adapter.find_products_by_need("NEED_NOT_EXIST") == []


# ===================================================================
# 相似产品对比
# ===================================================================
class TestFindSimilarProducts:
    def test_p001_similar_to_p002(self, adapter) -> None:
        """验收要求：`find_similar_products("P001")` 返回与 P002 的对比。"""
        pairs = adapter.find_similar_products("P001")
        assert len(pairs) >= 1
        others = {p.other_product_id for p in pairs}
        assert "P002" in others, f"P001 应与 P002 相似，实际 {others}"

    def test_difference_is_objective(self, adapter) -> None:
        """差异必须是客观描述，不含优劣评价。"""
        pairs = adapter.find_similar_products("P001")
        for p in pairs:
            assert p.difference, "差异说明不能为空"
            # 不出现主观优劣词
            for word in ("更好", "更优", "推荐买", "首选", "最强"):
                assert word not in p.difference, (
                    f"差异说明含主观评价「{word}」: {p.difference}"
                )

    def test_lesson_point_present(self, adapter) -> None:
        pairs = adapter.find_similar_products("P001")
        assert any(p.lesson_point for p in pairs), "教学要点不能全空"

    def test_unknown_product_returns_empty(self, adapter) -> None:
        assert adapter.find_similar_products("P999") == []


# ===================================================================
# 领域纯净性：adapter 不得让 SDK 类型穿越
# ===================================================================
class TestDomainPurity:
    def test_returns_only_domain_objects(self, adapter) -> None:
        """返回值的类型必须全部来自 domain，不含 neo4j 类型。"""
        from domain.value_objects import (
            ContraindicationHit,
            ProductFacts,
            ProductRef,
            SimilarPair,
        )

        facts = adapter.get_product_facts("P001")
        assert isinstance(facts, ProductFacts)

        refs = adapter.find_products_excluding(["CI_BLEED"])
        assert all(isinstance(r, ProductRef) for r in refs)

        hits = adapter.check_contraindication("P001", ["CI_BLEED"])
        assert all(isinstance(h, ContraindicationHit) for h in hits)

        pairs = adapter.find_similar_products("P001")
        assert all(isinstance(p, SimilarPair) for p in pairs)
