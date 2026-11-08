"""领域层单元测试。零 IO、零外部依赖，必须秒级跑完。

覆盖 P0 定的值对象语义——这些是后续所有阶段的判定基础，
先在 P0 把它们钉死，避免后面各处各自解释。
"""
from __future__ import annotations

import pytest

from domain.value_objects import (
    ApprovalClaim,
    Contraindication,
    PermissionLevel,
    ProductCategory,
    ProductFacts,
    Severity,
)


# ===================================================================
# 产品类别：约束 #2 的基础
# ===================================================================
class TestProductCategory:
    """普通食品不得有任何功能声称。"""

    def test_str_enum_matches_neo4j_value(self) -> None:
        """StrEnum 必须能直接与图谱里的中文字符串比较。

        Neo4j 里 category 属性存的就是「保健食品」/「普通食品」。
        """
        assert ProductCategory("保健食品") is ProductCategory.HEALTH_FOOD
        assert ProductCategory("普通食品") is ProductCategory.ORDINARY_FOOD

    def test_only_health_food_allows_claim(self) -> None:
        """只有保健食品允许功能声称。"""
        assert ProductCategory.HEALTH_FOOD.allows_efficacy_claim is True
        assert ProductCategory.ORDINARY_FOOD.allows_efficacy_claim is False

    def test_unknown_value_raises(self) -> None:
        """未知类别必须报错，不能默默当成合法值。"""
        with pytest.raises(ValueError):
            ProductCategory("膳食补充剂")


# ===================================================================
# 严重度：P4 护栏与 P5 评阅的判定依据
# ===================================================================
class TestSeverity:
    """只有 critical 无条件拦截。"""

    def test_only_critical_must_block(self) -> None:
        assert Severity.CRITICAL.must_block is True
        assert Severity.HIGH.must_block is False
        assert Severity.MEDIUM.must_block is False

    def test_order_is_monotonic(self) -> None:
        """排序权重必须严格递增，便于取「最严重的一项」。"""
        assert Severity.MEDIUM.order < Severity.HIGH.order < Severity.CRITICAL.order

    def test_values_align_with_graph(self) -> None:
        """取值必须与 Neo4j 的 severity 属性完全一致（小写）。"""
        assert Severity.CRITICAL == "critical"
        assert Severity.HIGH == "high"
        assert Severity.MEDIUM == "medium"


# ===================================================================
# ProductFacts：问答链路的核心输入
# ===================================================================
def _fish_oil() -> ProductFacts:
    """P001 鱼油的真实数据形状（来自 jiankangshujuku）。"""
    return ProductFacts(
        product_id="P001",
        name="康源牌鱼油软胶囊",
        category="保健食品",
        approval_no="国食健注G20250012",
        dosage_form="软胶囊",
        spec="1.0g/粒×60粒",
        claims=(
            ApprovalClaim(
                efficacy_id="EFF_19",
                text="有助于维持血脂（胆固醇/甘油三酯）健康水平",
                catalog_no=19,
            ),
        ),
        contraindications=(
            Contraindication("CI_CHILD", "少年儿童", Severity.HIGH, ""),
            Contraindication("CI_PREG", "孕妇", Severity.HIGH, ""),
            Contraindication("CI_BLEED", "出血倾向者和出血性疾病患者",
                             Severity.CRITICAL, "出血风险"),
            Contraindication("CI_SEAFOOD", "海产品过敏者",
                             Severity.CRITICAL, "过敏风险"),
        ),
        aliases=("鱼油", "深海鱼油"),
    )


class TestProductFacts:
    """产品画像的判定语义。"""

    def test_health_food_recognized(self) -> None:
        assert _fish_oil().is_health_food is True
        assert _fish_oil().allows_efficacy_claim is True

    def test_ordinary_food_rejects_claim(self) -> None:
        """普通食品：is_health_food=False，不允许声称。"""
        p = ProductFacts(
            product_id="P016",
            name="康源牌复合蛋白固体饮料",
            category="普通食品",
            approval_no="",
        )
        assert p.is_health_food is False
        assert p.allows_efficacy_claim is False

    def test_unknown_category_fails_safe(self) -> None:
        """未知类别必须按「不允许声称」处理（fail-safe）。

        这是关键：脏数据不能让普通食品蒙混过关变成保健食品。
        """
        p = ProductFacts(
            product_id="PX",
            name="脏数据产品",
            category="未知类别",
        )
        assert p.category_enum is None
        assert p.is_health_food is False
        assert p.allows_efficacy_claim is False

    def test_critical_contraindications_filtered(self) -> None:
        """critical 禁忌必须能单独取出——这是必须明示给用户的部分。"""
        crit = _fish_oil().critical_contraindications
        assert len(crit) == 2
        assert {c.contra_id for c in crit} == {"CI_BLEED", "CI_SEAFOOD"}

    def test_frozen_immutable(self) -> None:
        """值对象不可变：防止被下游意外改写。"""
        p = _fish_oil()
        with pytest.raises((AttributeError, TypeError)):
            p.name = "改个名字"          # type: ignore[misc]

    def test_ordinary_food_has_no_claims(self) -> None:
        """普通食品的 claims 必须为空——图谱里普通食品 CLAIMS 关系 0 条。"""
        p = ProductFacts(
            product_id="P016", name="复合蛋白固体饮料", category="普通食品",
        )
        assert p.claims == ()


# ===================================================================
# 权限级别：P8 的基础，但 P0 就要定死
# ===================================================================
class TestPermissionLevel:
    """学员 / 主管的可见范围。"""

    def test_trainee_cannot_see_restricted(self) -> None:
        allowed = PermissionLevel.trainee_allowed()
        assert PermissionLevel.RESTRICTED not in allowed
        assert PermissionLevel.CONFIDENTIAL not in allowed
        assert PermissionLevel.PUBLIC in allowed
        assert PermissionLevel.INTERNAL in allowed

    def test_manager_sees_restricted(self) -> None:
        allowed = PermissionLevel.manager_allowed()
        assert PermissionLevel.RESTRICTED in allowed
        # 但 confidential 也不给——那是更高层
        assert PermissionLevel.CONFIDENTIAL not in allowed

    def test_values_align_with_milvus(self) -> None:
        """取值必须与 Milvus 的 permission_level 字段一致。"""
        assert PermissionLevel.PUBLIC == "public"
        assert PermissionLevel.INTERNAL == "internal"
        assert PermissionLevel.RESTRICTED == "restricted"
