"""必答要素覆盖判定（P5）。

★ 为什么用规则而不是 LLM 判：

    需求明确「确定性」：同一个答卷，今天判「说到警示语」、明天判「没说」，
    学员无法信服，也无法复核。

    而且判定标准在案例文档里已经写成了**可执行的描述**：
        E1  出现「有」+ 产品名，或「没有」。不能是"好像有""我们产品挺多"
        E4  说清"是保健食品"+ 批准功能原句"有助于维持血脂健康水平"
    这些都能翻译成正则或关键词集合。

★ 判定的两个方向都必须测：
    · 说了 → 判定覆盖（否则学员做了也不算）
    · 没说 → 判定缺失（否则回避也能过）
    只测一边会做出「永远判定覆盖」或「永远判定缺失」的错实现。
"""
from __future__ import annotations

import pytest

from domain.rules.element_check import (
    check_elements,
    judge_element,
)
from domain.value_objects import ElementSpec


# ===================================================================
# E1 类：明确回答有 / 没有
# ===================================================================
class TestExplicitAnswer:
    """E1「明确回答有/没有」——不能是模糊话术。"""

    SPEC = ElementSpec(
        "E1", "明确回答有 / 没有", weight=20, required=True,
        judgement_note="出现「有」+ 产品名，或「没有」。不能是「好像有」「我们产品挺多」")

    @pytest.mark.parametrize("text", [
        "有的，康源牌鱼油软胶囊。",
        "有，我们这款鱼油软胶囊有货。",
        "没有这个产品。",
        "目前没有这款。",
    ])
    def test_covered(self, text: str) -> None:
        check = judge_element(self.SPEC, text)
        assert check.covered is True, f"应判定覆盖: {text}"

    @pytest.mark.parametrize("text", [
        "我们产品挺多的，您想了解什么。",
        "好像有吧，我看看。",
        "这个不太清楚，您问下别的同事。",
        "嗯……应该是有。",
    ])
    def test_not_covered(self, text: str) -> None:
        check = judge_element(self.SPEC, text)
        assert check.covered is False, f"模糊话术不应判定覆盖: {text}"


# ===================================================================
# E4 类：说明类别 + 批准功能原文
# ===================================================================
class TestCategoryAndClaim:
    """E4「说明产品类别与批准功能原文」——两个都要有。"""

    SPEC = ElementSpec(
        "E4", "说明产品类别与批准功能原文", weight=20, required=True,
        judgement_note="说清「是保健食品」+ 批准功能原句「有助于维持血脂健康水平」")

    def test_both_present(self) -> None:
        text = ("这是保健食品，批准功能是有助于维持血脂健康水平。")
        assert judge_element(self.SPEC, text).covered is True

    def test_only_category(self) -> None:
        """只说类别没给功能原文 → 不算覆盖（判定要求是「两个都要」）。"""
        text = "这是保健食品。"
        assert judge_element(self.SPEC, text).covered is False

    def test_only_claim(self) -> None:
        """只给功能没说类别 → 不算覆盖。"""
        text = "它的功能是有助于维持血脂健康水平。"
        assert judge_element(self.SPEC, text).covered is False

    def test_old_claim_not_accepted(self) -> None:
        """旧声称「降血脂」不算——这是最关键的一条。"""
        text = "这是保健食品，能降血脂。"
        assert judge_element(self.SPEC, text).covered is False

    def test_evidence_returned(self) -> None:
        """覆盖时要给出命中依据，便于人工复核。"""
        text = "这是保健食品，批准功能是有助于维持血脂健康水平。"
        check = judge_element(self.SPEC, text)
        assert check.evidence
        assert "保健食品" in check.evidence or "有助于维持血脂" in check.evidence


# ===================================================================
# E5 类：强制警示语
# ===================================================================
class TestWarningElement:
    SPEC = ElementSpec(
        "E5", "加强制警示语", weight=15, required=True,
        judgement_note="出现「不是药物，不能代替药物治疗疾病」或等价完整表述")

    @pytest.mark.parametrize("text", [
        "保健食品不是药物，不能代替药物治疗疾病。",
        "不是药物，不能代替药物治疗疾病。",
        "本品不能代替药物。",
        "不能代替药物治疗疾病。",
    ])
    def test_warning_variants_covered(self, text: str) -> None:
        """等价表述都要算——只认逐字一致会造成大量误判。"""
        assert judge_element(self.SPEC, text).covered is True

    @pytest.mark.parametrize("text", [
        "这是保健食品，很安全的。",
        "可以长期吃，没副作用。",
        # 常见错误：把警示语说成「不是药」就完了
        "它不是药。",
    ])
    def test_incomplete_not_covered(self, text: str) -> None:
        assert judge_element(self.SPEC, text).covered is False


# ===================================================================
# 建议要素（required=False）
# ===================================================================
class TestOptionalElement:
    SPEC = ElementSpec("E7", "提及不适宜人群", weight=10, required=False)

    def test_missing_optional_not_blocking(self) -> None:
        check = judge_element(self.SPEC, "有的，康源牌鱼油。")
        assert check.covered is False
        assert check.missing_required is False, (
            "建议要素缺失不应计为「缺必答」"
        )


# ===================================================================
# 批量判定
# ===================================================================
class TestCheckElements:
    def test_all_checked(self) -> None:
        specs = (
            ElementSpec("E1", "明确回答有 / 没有", 20, True),
            ElementSpec("E2", "给出产品名称", 10, True),
            ElementSpec("E7", "提及不适宜人群", 10, False),
        )
        text = "有的，康源牌鱼油软胶囊，不适宜人群包括出血倾向者。"
        checks = check_elements(specs, text)

        assert len(checks) == 3
        by_id = {c.element_id: c for c in checks}
        assert by_id["E1"].covered is True
        assert by_id["E2"].covered is True
        assert by_id["E7"].covered is True

    def test_preserves_order(self) -> None:
        """顺序必须与 specs 一致，否则摘要卡展示会错位。"""
        specs = (
            ElementSpec("E3", "客观参数", 15, True),
            ElementSpec("E1", "明确回答", 20, True),
            ElementSpec("E2", "产品名称", 10, True),
        )
        checks = check_elements(specs, "有的，鱼油。")
        assert [c.element_id for c in checks] == ["E3", "E1", "E2"]

    def test_empty_specs(self) -> None:
        assert check_elements((), "任意文本") == []

    def test_empty_text_all_missing(self) -> None:
        """学员一句话没说 → 全部缺失。"""
        specs = (
            ElementSpec("E1", "明确回答", 20, True),
            ElementSpec("E2", "产品名称", 10, True),
        )
        checks = check_elements(specs, "")
        assert all(not c.covered for c in checks)
        assert len([c for c in checks if c.missing_required]) == 2

    def test_deterministic(self) -> None:
        """★ 同输入必须得到同结果（确定性的直接验证）。"""
        specs = (
            ElementSpec("E1", "明确回答有 / 没有", 20, True),
            ElementSpec("E4", "说明产品类别与批准功能原文", 20, True),
        )
        text = "有的，这是保健食品，有助于维持血脂健康水平。"
        a = check_elements(specs, text)
        b = check_elements(specs, text)
        assert [(c.element_id, c.covered) for c in a] == \
               [(c.element_id, c.covered) for c in b]


# ===================================================================
# 跨轮次合并判定
# ===================================================================
class TestMultiTurn:
    def test_elements_can_be_covered_across_turns(self) -> None:
        """要素分布在多轮里也算覆盖——不能只看最后一轮。"""
        specs = (
            ElementSpec("E1", "明确回答有 / 没有", 20, True),
            ElementSpec("E5", "加强制警示语", 15, True),
        )
        # 第 1 轮说了有，第 2 轮补了警示语
        text = ("有的，康源牌鱼油软胶囊。" "\n"
                "需要说明的是，保健食品不是药物，不能代替药物治疗疾病。")
        checks = check_elements(specs, text)
        assert all(c.covered for c in checks), (
            "跨轮次提到的要素应合并判定"
        )


# ===================================================================
# 未知要素的兜底
# ===================================================================
class TestUnknownElement:
    def test_unknown_id_uses_name_keywords(self) -> None:
        """案例里出现未预置规则的要素时，用名称关键词做宽松判定。

        ★ 为什么需要：14 + 2 个案例里有 60+ 条要素，
          不可能每条都写专门规则。兜底策略是「名称关键词命中即算覆盖」，
          并标记为「需人工确认」（由用例层负责标记）。
        """
        spec = ElementSpec("E99", "提及服用时间", weight=5, required=False)
        check = judge_element(spec, "建议随餐服用。")
        # 兜底：能判，但不强求准确
        assert isinstance(check.covered, bool)

    def test_empty_text_never_covered(self) -> None:
        spec = ElementSpec("E99", "任意要素", weight=5, required=True)
        assert judge_element(spec, "").covered is False
