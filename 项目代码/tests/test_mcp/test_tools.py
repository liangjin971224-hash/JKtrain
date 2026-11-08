"""MCP 工具行为测试（不接真库，用假端口）。

★ 测什么
  这里是「工具返回结构对不对」，不是「越权拦没拦住」（后者在 test_identity.py）。
  两边分开：越权是安全逻辑，这里是业务契约。

★ 为什么用假端口
  真库（Neo4j + Milvus + MySQL + LLM）启动要 10s+，且任一挂掉测试就失败。
  工具层的职责是「把用例结果转成结构化 JSON」，用假数据就能验证。
"""
from __future__ import annotations

import json

import pytest

from domain.value_objects.approval_claim import ApprovalClaim, Contraindication
from domain.value_objects.product_facts import ProductFacts, Usage
from domain.value_objects.role import Role
from domain.value_objects.severity import Severity
from interfaces.mcp.identity import McpIdentity
from interfaces.mcp.prompts import SCENES, expression_polish_template
from interfaces.mcp.resources import read_product
from interfaces.mcp.tools.drill import _clamp_days, my_drill_reviews
from interfaces.mcp.tools.graph import ALLOWED_RELATIONS, graph_entity


# ======================================================================
# 假实现
# ======================================================================
def _facts() -> ProductFacts:
    return ProductFacts(
        product_id="P001",
        name="康源牌鱼油软胶囊",
        category="保健食品",
        approval_no="国食健注G20250012",
        claims=(
            ApprovalClaim(
                efficacy_id="EFF_19",
                text="有助于维持血脂（胆固醇/甘油三酯）健康水平",
                catalog_no=19,
                catalog_version="2023",
            ),
        ),
        ingredients=("鱼油", "明胶"),
        suitable_for=("血脂偏高者",),
        contraindications=(
            Contraindication(
                contra_id="CI_BLEED",
                name="出血倾向者和出血性疾病患者",
                severity=Severity.CRITICAL,
                reason="鱼油可能影响凝血功能，与抗凝药同服增加出血风险",
            ),
            Contraindication(
                contra_id="CI_PREG",
                name="孕妇",
                severity=Severity.HIGH,
                reason="孕期营养需专业评估",
            ),
        ),
        usage=Usage(dosage="每日2次，每次2粒", daily_limit="4粒"),
    )


class _FakeGraph:
    def __init__(self) -> None:
        self._f = _facts()

    def resolve_alias(self, alias: str):
        if alias in ("鱼油", "P001"):
            class Ref:
                product_id = "P001"
            return Ref()
        return None

    def get_product_facts(self, product_id: str):
        return self._f if product_id in ("P001", "鱼油") else None


class _FakeRepo:
    def __init__(self, rows=None) -> None:
        self._rows = rows or []

    def list_drill_reviews(self, *, trainee_id, since="", until="",
                           with_items=False):
        return self._rows


class _Row:
    def __init__(self, **kw) -> None:
        self.drill_id = kw.get("drill_id", "d1")
        self.trainee_id = kw.get("trainee_id", "E001")
        self.case_id = kw.get("case_id", "C07")
        self.case_type = kw.get("case_type", "禁忌询问（contraindication）")
        self.duration_sec = kw.get("duration_sec", 0)
        self.element_hit = kw.get("element_hit", 2)
        self.element_total = kw.get("element_total", 4)
        self.risk_count = kw.get("risk_count", 1)
        self.critical_count = kw.get("critical_count", 1)
        self.created_at = kw.get("created_at", "2026-10-21 14:30:00")


class _C:
    def __init__(self, graph=None, repo=None) -> None:
        self.graph = graph
        self.repository = repo
        self.settings = None
        self.vector = None
        self.keyword = None
        self.llm = None
        self.rerank = None


def _t() -> McpIdentity:
    return McpIdentity(employee_no="E001", role=Role.TRAINEE)


# ======================================================================
# graph_entity
# ======================================================================
class TestGraphEntity:
    def test_returns_product_and_relations(self) -> None:
        d = json.loads(graph_entity(_C(graph=_FakeGraph()), _t(), name="鱼油"))
        assert d["found"] is True
        assert d["product"]["product_id"] == "P001"
        # 批准功能必须原文照抄
        assert d["product"]["claims_text"] == [
            "有助于维持血脂（胆固醇/甘油三酯）健康水平"
        ]
        assert "CONTRAINDICATED_FOR" in d["relations"]

    def test_contraindication_carries_severity(self) -> None:
        d = json.loads(graph_entity(_C(graph=_FakeGraph()), _t(), name="鱼油"))
        contras = d["relations"]["CONTRAINDICATED_FOR"]
        bleed = next(x for x in contras if "出血" in x["name"])
        assert bleed["severity"] == "critical"
        assert bleed["must_block"] is True
        # ★ 原因要带上，否则 AI 无法解释「为什么不能吃」
        assert "抗凝药" in bleed["reason"]

    def test_unknown_name_returns_not_found(self) -> None:
        d = json.loads(
            graph_entity(_C(graph=_FakeGraph()), _t(), name="不存在的产品"))
        assert d["found"] is False
        assert "note" in d

    def test_relation_whitelist_filters_input(self) -> None:
        """LLM 传了非法关系名，应被过滤掉而不是报错。"""
        d = json.loads(graph_entity(
            _C(graph=_FakeGraph()), _t(),
            name="鱼油", with_relations=["CLAIMS", "DROP_TABLE"],
        ))
        assert d["found"] is True
        assert "CLAIMS" in d["relations"]
        assert "DROP_TABLE" not in d["relations"]

    def test_all_allowed_relations_defined(self) -> None:
        assert ALLOWED_RELATIONS == {
            "ALIAS_OF", "CONTRAINDICATED_FOR", "CLAIMS", "SUITABLE_FOR"}


# ======================================================================
# product resource
# ======================================================================
class TestProductResource:
    def test_returns_full_profile(self) -> None:
        d = json.loads(read_product(_C(graph=_FakeGraph()), _t(), "P001"))
        assert d["name"] == "康源牌鱼油软胶囊"
        assert d["approval_no"] == "国食健注G20250012"
        assert d["claims"][0]["catalog_no"] == 19

    def test_marks_claims_as_regulatory_text(self) -> None:
        """★ 必须标注「批准功能是法规原文」，防止下游改写。"""
        d = json.loads(read_product(_C(graph=_FakeGraph()), _t(), "P001"))
        assert d["_meta"]["claims_are_regulatory_text"] is True

    def test_missing_product(self) -> None:
        d = json.loads(read_product(_C(graph=_FakeGraph()), _t(), "P999"))
        assert "error" in d


# ======================================================================
# my_drill_reviews
# ======================================================================
class TestMyDrillReviews:
    def test_returns_rows(self) -> None:
        repo = _FakeRepo([_Row()])
        d = json.loads(my_drill_reviews(_C(repo=repo), _t(), days=7))
        assert d["trainee_id"] == "E001"
        assert d["count"] == 1
        assert d["reviews"][0]["case_id"] == "C07"

    def test_no_score_field(self) -> None:
        """★ 红线：不给学员打分。

        摘要卡不能出现 score / verdict / grade，MCP 返回也遵守。
        """
        repo = _FakeRepo([_Row()])
        d = json.loads(my_drill_reviews(_C(repo=repo), _t(), days=7))
        blob = json.dumps(d, ensure_ascii=False).lower()
        for banned in ("score", "verdict", "grade", "得分", "评级"):
            assert banned not in blob, f"出现了禁止的评分字段: {banned}"

    def test_empty_returns_hint(self) -> None:
        d = json.loads(my_drill_reviews(_C(repo=_FakeRepo([])), _t(), days=7))
        assert d["count"] == 0
        assert "没有演练记录" in d["note"]

    @pytest.mark.parametrize("value,expect", [
        (0, 1), (-5, 1), (1, 1), (30, 30), (90, 90), (999, 90), (1e9, 90),
    ])
    def test_days_clamped(self, value, expect) -> None:
        assert _clamp_days(value) == expect

    def test_created_at_str_or_datetime(self) -> None:
        """created_at 可能是 str 也可能是 datetime，都要能序列化。"""
        d1 = json.loads(my_drill_reviews(
            _C(repo=_FakeRepo([_Row(created_at="2026-10-21 14:30:00")])),
            _t(), days=7))
        assert d1["reviews"][0]["created_at"] == "2026-10-21 14:30:00"

        import datetime
        d2 = json.loads(my_drill_reviews(
            _C(repo=_FakeRepo([_Row(created_at=datetime.datetime(2026, 10, 21, 14, 30))])),
            _t(), days=7))
        assert "2026-10-21" in d2["reviews"][0]["created_at"]


# ======================================================================
# expression_polish
# ======================================================================
class TestExpressionPolish:
    def test_contains_all_three_constraints(self) -> None:
        p = expression_polish_template("测试文本", "daily_summary")
        assert "不改任何数字" in p
        assert "不添加任何事实" in p
        assert "不加强语气" in p

    def test_contains_original_text(self) -> None:
        p = expression_polish_template("客户命中率 4/6", "daily_summary")
        assert "客户命中率 4/6" in p

    def test_scene_specific_rule(self) -> None:
        p1 = expression_polish_template("x", "monthly_report")
        p2 = expression_polish_template("x", "drill_comment")
        assert "对外" in p1
        assert "不给分数" in p2
        assert p1 != p2

    def test_unknown_scene_falls_back(self) -> None:
        p = expression_polish_template("x", "not_a_scene")
        assert SCENES["daily_summary"] in p
