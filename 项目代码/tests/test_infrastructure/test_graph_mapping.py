"""适配器转换函数测试。

为什么不连真库也要测：
    `_to_severity` / `_get_int` / `_to_usage` 这些转换函数决定了
    「脏数据会怎样表现」。真库里数据是干净的，测不出降级路径；
    而线上出问题的恰恰是脏数据。

    ★ 重点是 severity 的 fail-safe 语义：
      未知值必须降级为 critical（从严），而不是 medium（放行）。
"""
from __future__ import annotations

import pytest

from domain.value_objects import Severity
from infrastructure.graph.neo4j_graph_adapter import (
    _get_int,
    _to_product_facts,
    _to_product_ref,
    _to_severity,
    _to_usage,
)


class _Node(dict):
    """模拟 neo4j.Node 的 .get() 行为。"""


# ===================================================================
# severity 转换：安全语义的核心
# ===================================================================
class TestToSeverity:
    """★ 未知值必须 fail-safe 到 critical。"""

    @pytest.mark.parametrize("raw", ["critical", "CRITICAL", " Critical ",
                                     "critical\n"])
    def test_critical_variants(self, raw: str) -> None:
        assert _to_severity(raw) is Severity.CRITICAL

    @pytest.mark.parametrize(("raw", "expected"), [
        ("high", Severity.HIGH),
        (" medium ", Severity.MEDIUM),
        ("HIGH", Severity.HIGH),
    ])
    def test_known_values(self, raw: str, expected: Severity) -> None:
        assert _to_severity(raw) is expected

    @pytest.mark.parametrize("raw", ["urgent", "严重", "", None, 123, "unknown"])
    def test_unknown_fails_safe_to_critical(self, raw) -> None:
        """未知值必须降到 critical（从严拦截），不能降到 medium（放行）。

        若此测试失败，意味着脏数据能让 critical 禁忌逃过 P4 护栏，
        这是安全事故级别的问题。
        """
        assert _to_severity(raw) is Severity.CRITICAL, (
            f"severity={raw!r} 未按 fail-safe 处理。"
            f"若降成 medium，该禁忌将绕过输出护栏。"
        )

    def test_fail_safe_means_must_block(self) -> None:
        """降级后的值必须真的触发拦截。"""
        assert _to_severity("脏数据").must_block is True


# ===================================================================
# 整数与用法转换
# ===================================================================
class TestGetInt:
    def test_valid(self) -> None:
        assert _get_int(_Node(retail_price=298), "retail_price") == 298

    def test_string_number(self) -> None:
        assert _get_int(_Node(x="120"), "x") == 120

    @pytest.mark.parametrize("node", [_Node(), _Node(x=None), _Node(x="abc"),
                                      _Node(x="")])
    def test_missing_or_invalid_returns_none(self, node: _Node) -> None:
        assert _get_int(node, "x") is None


class TestToUsage:
    def test_empty_returns_none(self) -> None:
        assert _to_usage([]) is None

    def test_maps_all_fields(self) -> None:
        usage = _to_usage([_Node(
            dosage="每日2次，每次2粒",
            daily_limit="不超过4.0g",
            note="随餐服用",
        )])
        assert usage is not None
        assert usage.dosage == "每日2次，每次2粒"
        assert usage.daily_limit == "不超过4.0g"
        assert usage.note == "随餐服用"

    def test_missing_fields_become_empty_string(self) -> None:
        """字段缺失不能抛，应是空串（下游按空串判断是否展示）。"""
        usage = _to_usage([_Node(dosage="每日1次")])
        assert usage is not None
        assert usage.daily_limit == ""
        assert usage.note == ""

    def test_takes_first_usage(self) -> None:
        """数据层每个产品恰好 1 个 Usage；多于一个时取第一个。"""
        usage = _to_usage([_Node(dosage="A"), _Node(dosage="B")])
        assert usage is not None
        assert usage.dosage == "A"


# ===================================================================
# 画像组装：OPTIONAL MATCH 的空值处理
# ===================================================================
class TestToProductFacts:
    def test_full_assembly(self) -> None:
        facts = _to_product_facts(
            node=_Node(product_id="P001", name="鱼油", category="保健食品",
                       approval_no="国食健注G20250012", spec="1.0g×60"),
            efficacies=[_Node(efficacy_id="EFF_19", text="", name="有助于维持血脂",
                              catalog_no=19, catalog_version="2023")],
            ingredients=[_Node(name="鱼油")],
            usages=[_Node(dosage="每日2次")],
            targets=[_Node(name="血脂偏高者")],
            contras=[_Node(contra_id="CI_BLEED", name="出血倾向",
                           severity="critical", reason="影响凝血")],
            aliases=[_Node(name="鱼油")],
        )
        assert facts.product_id == "P001"
        assert facts.is_health_food is True
        assert len(facts.claims) == 1
        assert facts.claims[0].catalog_no == 19
        assert facts.ingredients == ("鱼油",)
        assert facts.usage is not None
        assert facts.suitable_for == ("血脂偏高者",)
        assert facts.contraindications[0].severity is Severity.CRITICAL
        assert facts.aliases == ("鱼油",)

    def test_ordinary_food_no_claims(self) -> None:
        """普通食品：所有关系都空，必须组装成空元组而不是 [None]。"""
        facts = _to_product_facts(
            node=_Node(product_id="P016", name="固体饮料", category="普通食品"),
            efficacies=[], ingredients=[], usages=[],
            targets=[], contras=[], aliases=[],
        )
        assert facts.claims == ()
        assert facts.contraindications == ()
        assert facts.usage is None
        assert facts.is_health_food is False
        assert facts.allows_efficacy_claim is False

    def test_missing_category_fails_safe(self) -> None:
        """category 缺失时不允许声称（fail-safe）。"""
        facts = _to_product_facts(
            node=_Node(product_id="PX", name="脏数据"),
            efficacies=[], ingredients=[], usages=[],
            targets=[], contras=[], aliases=[],
        )
        assert facts.category == ""
        assert facts.is_health_food is False
        assert facts.allows_efficacy_claim is False


class TestToProductRef:
    def test_basic(self) -> None:
        ref = _to_product_ref({
            "product_id": "P001", "name": "鱼油",
            "category": "保健食品", "approval_no": "G20250012",
        })
        assert ref.product_id == "P001"
        assert ref.approval_no == "G20250012"

    def test_null_approval_no(self) -> None:
        """普通食品可能没有批准文号（null），不能变成字符串 "None"。"""
        ref = _to_product_ref({
            "product_id": "P016", "name": "固体饮料",
            "category": "普通食品", "approval_no": None,
        })
        assert ref.approval_no == ""
