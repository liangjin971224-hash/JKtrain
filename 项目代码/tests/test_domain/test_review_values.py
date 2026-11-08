"""摘要卡与案例规格值对象（P5 红灯先行）。

★ 约束 #3 的守护点：**摘要卡不含分数、不含结论**。

  系统只提供判断依据，判定由人来下。
  这是需求明确要求的——培训场景里「系统给分」会带来两个问题：
    ① 学员会去优化分数，而不是优化话术
    ② 一旦分数有争议，系统成了背锅方，而它其实只是规则匹配

  所以有 `test_no_score_field` 这类测试直接 grep 字段名。
"""
from __future__ import annotations

import dataclasses

import pytest

from domain.value_objects import (
    CaseSpec,
    ElementCheck,
    ElementSpec,
    ForbiddenRule,
    RiskItem,
    RiskLevel,
    SummaryCard,
    TraineeTurn,
)


# ===================================================================
# 1. 案例规格
# ===================================================================
class TestCaseSpec:
    def test_minimal_spec(self) -> None:
        spec = CaseSpec(
            case_id="C01",
            title="客户询问是否有鱼油",
            opening_line="你们有鱼油吗？",
        )
        assert spec.case_id == "C01"
        assert spec.elements == ()
        assert spec.forbidden_rules == ()

    def test_required_elements_filtered(self) -> None:
        """必需要素 vs 建议要素要能区分——判定规则不同。"""
        spec = CaseSpec(
            case_id="C01", title="t", opening_line="o",
            elements=(
                ElementSpec("E1", "明确回答", weight=20, required=True),
                ElementSpec("E6", "提及服药建议", weight=10, required=False),
            ),
        )
        required = spec.required_elements
        assert len(required) == 1
        assert required[0].element_id == "E1"

    def test_element_weight_positive(self) -> None:
        with pytest.raises(ValueError, match="权重"):
            ElementSpec("E1", "x", weight=-1, required=True)


# ===================================================================
# 2. 风险项
# ===================================================================
class TestRiskItem:
    def test_location_is_turn_and_sentence(self) -> None:
        """风险必须能定位到「第几轮第几句」（P5 验收要求）。"""
        item = RiskItem(
            level=RiskLevel.CRITICAL,
            rule_type="违规宣称",
            matched="降血脂",
            turn_index=2,
            sentence_index=1,
            reason="属医疗术语",
        )
        assert item.location == "第2轮第1句"

    def test_turn_index_is_one_based(self) -> None:
        """对外展示的轮次从 1 开始（第 0 轮会被当成「没定位到」）。"""
        item = RiskItem(level=RiskLevel.HIGH, rule_type="x", matched="y",
                        turn_index=0, sentence_index=0)
        with pytest.raises(ValueError, match="轮次"):
            _ = item.location

    def test_critical_flag(self) -> None:
        crit = RiskItem(level=RiskLevel.CRITICAL, rule_type="x",
                        matched="y", turn_index=1, sentence_index=1)
        high = RiskItem(level=RiskLevel.HIGH, rule_type="x",
                        matched="y", turn_index=1, sentence_index=1)
        assert crit.is_critical is True
        assert high.is_critical is False


# ===================================================================
# 3. 要素判定
# ===================================================================
class TestElementCheck:
    def test_covered(self) -> None:
        check = ElementCheck(
            element_id="E1", element_name="明确回答", weight=20,
            required=True, covered=True, evidence="有的，康源牌鱼油软胶囊",
        )
        assert check.covered is True
        assert check.missing_required is False

    def test_missing_required(self) -> None:
        check = ElementCheck(
            element_id="E5", element_name="加警示语", weight=15,
            required=True, covered=False,
        )
        assert check.missing_required is True

    def test_missing_optional_not_blocking(self) -> None:
        """建议要素缺失不算「缺必答」。"""
        check = ElementCheck(
            element_id="E7", element_name="不适宜人群", weight=10,
            required=False, covered=False,
        )
        assert check.missing_required is False


# ===================================================================
# 4. ★ 摘要卡：不含分数与结论
# ===================================================================
class TestSummaryCardNoScore:
    def _card(self) -> SummaryCard:
        return SummaryCard(
            trainee_id="T001", case_id="C01", duration_sec=180,
            risk_items=(RiskItem(level=RiskLevel.CRITICAL,
                                 rule_type="违规宣称", matched="降血脂",
                                 turn_index=1, sentence_index=1),),
            element_coverage=(
                ElementCheck("E1", "明确回答有", 20, True, True, "有的"),
                ElementCheck("E5", "加警示语", 15, True, False),
            ),
            pending_human=("是否属于「回避推诿」需人工判断",),
        )

    def test_no_score_field(self) -> None:
        """★ 约束 #3：摘要卡不得有 score 字段。"""
        fields = {f.name for f in dataclasses.fields(SummaryCard)}
        assert "score" not in fields
        assert not any("score" in f for f in fields)

    def test_no_verdict_field(self) -> None:
        """★ 约束 #3：摘要卡不得有 verdict / 结论字段。"""
        fields = {f.name for f in dataclasses.fields(SummaryCard)}
        for banned in ("verdict", "result", "grade", "pass_fail"):
            assert banned not in fields

    def test_no_total_score_property(self) -> None:
        """连「算出总分」的属性也不能有——那等价于给分。"""
        card = self._card()
        public = [a for a in dir(card) if not a.startswith("_")]
        for name in public:
            assert "score" not in name.lower(), (
                f"摘要卡有疑似分数属性: {name}"
            )

    def test_critical_items_accessible(self) -> None:
        """虽不给分，但要能筛出 critical 供人工重点关注。"""
        card = self._card()
        assert len(card.critical_items) == 1
        assert card.critical_items[0].matched == "降血脂"

    def test_missing_required_accessible(self) -> None:
        card = self._card()
        assert len(card.missing_required_elements) == 1
        assert card.missing_required_elements[0].element_id == "E5"

    def test_has_critical_violation(self) -> None:
        """用于「红线案例一票否决」的展示，但仍是布尔而非分数。"""
        assert self._card().has_critical_violation is True

    def test_to_display_dict_no_score_key(self) -> None:
        """展示用字典也不能偷偷夹带分数。"""
        d = self._card().to_display_dict()
        assert "score" not in d
        assert "verdict" not in d

    def test_display_dict_structure(self) -> None:
        """展示结构要有四个区块（P5 要求「四区块齐全」）。"""
        d = self._card().to_display_dict()
        assert "风险提示" in d
        assert "要素覆盖" in d
        assert "待人工确认" in d
        assert "基础信息" in d


# ===================================================================
# 5. 学员对话轮次
# ===================================================================
class TestTraineeTurn:
    def test_turn_holds_sentences(self) -> None:
        turn = TraineeTurn(turn_index=1, text="有的。这是保健食品。")
        assert turn.sentences == ["有的。", "这是保健食品。"]

    def test_turn_index_validation(self) -> None:
        with pytest.raises(ValueError, match="轮次"):
            TraineeTurn(turn_index=0, text="x")

    def test_empty_text(self) -> None:
        turn = TraineeTurn(turn_index=1, text="")
        assert turn.sentences == []
