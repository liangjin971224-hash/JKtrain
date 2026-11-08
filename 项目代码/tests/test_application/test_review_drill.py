"""演练评阅用例测试（P5）。

★ 三条必须守住：

  1. 摘要卡不含分数（约束 #3）——用 grep 字段名验证，不靠人工检查
  2. critical 检出率 100%
  3. 要素覆盖与「待人工确认」都被正确标记：
     兜底判定的要素必须进「待人工确认」，不能装成确定的结论
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from application.use_cases.review_drill import ReviewDrillUseCase
from domain.value_objects import (
    CaseSpec,
    ElementSpec,
    ForbiddenRule,
    RiskLevel,
    SummaryCard,
    TraineeTurn,
)


# ===================================================================
# 夹具
# ===================================================================
def _c01() -> CaseSpec:
    return CaseSpec(
        case_id="C01",
        title="客户询问是否有鱼油",
        opening_line="你们有鱼油吗？",
        elements=(
            ElementSpec("E1", "明确回答有 / 没有", 20, True),
            ElementSpec("E2", "给出产品名称", 10, True),
            ElementSpec("E3", "给出客观参数", 15, True),
            ElementSpec("E4", "说明产品类别与批准功能原文", 20, True),
            ElementSpec("E5", "加强制警示语", 15, True),
            ElementSpec("E6", "提及服药时建议咨询医生", 10, False),
            ElementSpec("E7", "提及不适宜人群", 10, False),
        ),
        forbidden_rules=(
            ForbiddenRule("违规宣称", r"降血脂|降胆固醇|治疗|改善血脂|调节血脂",
                          RiskLevel.CRITICAL),
            ForbiddenRule("效果承诺", r"肯定有效|保证|一定管用|很快见效",
                          RiskLevel.HIGH),
            ForbiddenRule("主动推销", r"要不要买|带两盒|现在有活动|买二送一",
                          RiskLevel.CRITICAL),
            ForbiddenRule("回避推诿", r"这个我不清楚|你问医生吧",
                          RiskLevel.HIGH),
        ),
    )


BEST_ANSWER = (
    "有的，康源牌鱼油软胶囊，每粒含鱼油1000mg，EPA+DHA共300mg。"
    "这是保健食品，批准功能是有助于维持血脂健康水平。"
    "保健食品不是药物，不能代替药物治疗疾病。"
    "不适宜人群包括出血倾向者、肝功能不全、海产品过敏、孕妇乳母和儿童。"
    "您在吃降压药的话建议先咨询医生。"
)

BAD_ANSWER = (
    "有的！我们这款鱼油特别好，很多人吃了血脂都降下来了，"
    "您可以先带两盒试试，现在还有活动。"
)


# ===================================================================
# 1. 优秀答卷
# ===================================================================
class TestBestAnswer:
    def test_all_required_covered(self) -> None:
        uc = ReviewDrillUseCase()
        card = uc.execute(
            _c01(), [TraineeTurn(1, BEST_ANSWER)],
            trainee_id="T001", duration_sec=120)

        missing = [c.element_id for c in card.missing_required_elements]
        assert missing == [], f"优秀答卷不应缺必答要素，实际缺 {missing}"

    def test_no_critical_risk(self) -> None:
        uc = ReviewDrillUseCase()
        card = uc.execute(
            _c01(), [TraineeTurn(1, BEST_ANSWER)], trainee_id="T001")

        assert card.has_critical_violation is False, (
            f"优秀答卷不应有 critical 违规，实际 "
            f"{[(r.matched, r.location) for r in card.critical_items]}"
        )


# ===================================================================
# 2. ★ 不合格答卷：critical 必须全部检出
# ===================================================================
class TestBadAnswer:
    def test_critical_detected(self) -> None:
        uc = ReviewDrillUseCase()
        card = uc.execute(
            _c01(), [TraineeTurn(1, BAD_ANSWER)],
            trainee_id="T002", duration_sec=60)

        assert card.has_critical_violation is True
        assert len(card.critical_items) >= 2, (
            f"该答卷至少 2 处 critical（降血脂 / 带两盒），"
            f"实际 {len(card.critical_items)} 处"
        )

    def test_missing_required_elements(self) -> None:
        uc = ReviewDrillUseCase()
        card = uc.execute(
            _c01(), [TraineeTurn(1, BAD_ANSWER)], trainee_id="T002")

        missing = {c.element_id for c in card.missing_required_elements}
        # 缺参数、缺功能原文、缺警示语
        assert "E3" in missing
        assert "E4" in missing
        assert "E5" in missing

    def test_risk_locations_present(self) -> None:
        """每条风险都必须有可读定位。"""
        uc = ReviewDrillUseCase()
        card = uc.execute(
            _c01(), [TraineeTurn(1, BAD_ANSWER)], trainee_id="T002")

        for r in card.risk_items:
            assert r.location.startswith("第")
            assert "轮" in r.location and "句" in r.location


# ===================================================================
# 3. ★ 约束 #3：无分数泄漏
# ===================================================================
class TestNoScoreLeak:
    def test_dataclass_fields(self) -> None:
        """摘要卡不能有 score / verdict 字段。"""
        fields = {f.name for f in dataclasses.fields(SummaryCard)}
        for banned in ("score", "verdict", "grade", "result", "pass_fail"):
            assert banned not in fields

    def test_no_score_in_display_dict(self) -> None:
        uc = ReviewDrillUseCase()
        card = uc.execute(_c01(), [TraineeTurn(1, BAD_ANSWER)],
                          trainee_id="T002")
        d = card.to_display_dict()

        serialized = json.dumps(d, ensure_ascii=False).lower()
        assert '"score"' not in serialized
        assert '"verdict"' not in serialized

    def test_no_numeric_total(self) -> None:
        """不能通过属性偷偷算总分。"""
        uc = ReviewDrillUseCase()
        card = uc.execute(_c01(), [TraineeTurn(1, BAD_ANSWER)],
                          trainee_id="T002")

        public = [a for a in dir(card) if not a.startswith("_")]
        for name in public:
            assert "score" not in name.lower()
            assert "total" not in name.lower() or "total_" not in name.lower()

    def test_to_json_excludes_scores(self) -> None:
        uc = ReviewDrillUseCase()
        card = uc.execute(_c01(), [TraineeTurn(1, BAD_ANSWER)],
                          trainee_id="T002")
        text = uc.to_json(card)
        low = text.lower()
        assert '"score"' not in low
        assert '"verdict"' not in low


# ===================================================================
# 4. 待人工确认
# ===================================================================
class TestPendingHuman:
    def test_fallback_element_flagged(self) -> None:
        """只能靠兜底判定的要素，必须进「待人工确认」。

        ★ 为什么重要：兜底判定是宽松的关键词匹配，会有误判。
          如果把它当成确定结论展示，学员会据错误信息改进话术。
          标为「待确认」后，人工会去看一眼。
        """
        spec = CaseSpec(
            "C99", "t", "o",
            elements=(ElementSpec("E99", "提及服用时间的建议", 5, False),),
        )
        uc = ReviewDrillUseCase()
        card = uc.execute(spec, [TraineeTurn(1, "建议随餐服用。")],
                          trainee_id="T003")

        assert any("E99" in p for p in card.pending_human), (
            f"兜底判定的要素应进待人工确认，实际: {card.pending_human}"
        )

    def test_builtin_element_not_flagged(self) -> None:
        """有内置规则的要素不需要人工复核。"""
        spec = CaseSpec(
            "C99", "t", "o",
            elements=(ElementSpec("E1", "明确回答有 / 没有", 20, True),),
        )
        uc = ReviewDrillUseCase()
        card = uc.execute(spec, [TraineeTurn(1, "有的，康源牌鱼油软胶囊。")],
                          trainee_id="T003")

        assert not any("E1" in p for p in card.pending_human)

    def test_partial_optional_flagged(self) -> None:
        """建议要素只部分满足时也要人工看一下。"""
        spec = CaseSpec(
            "C99", "t", "o",
            elements=(
                ElementSpec("E1", "明确回答有 / 没有", 20, True),
                ElementSpec("E7", "提及不适宜人群", 10, False),
            ),
        )
        uc = ReviewDrillUseCase()
        card = uc.execute(spec, [TraineeTurn(1, "有的，康源牌鱼油软胶囊。")],
                          trainee_id="T003")

        assert any("E7" in p for p in card.pending_human)


# ===================================================================
# 5. 边界
# ===================================================================
class TestEdgeCases:
    def test_empty_turns(self) -> None:
        """学员一句话没说 → 全部要素缺失，但不崩。"""
        uc = ReviewDrillUseCase()
        card = uc.execute(_c01(), [], trainee_id="T004")

        assert len(card.missing_required_elements) == 5
        assert card.risk_items == ()

    def test_empty_text_turn(self) -> None:
        uc = ReviewDrillUseCase()
        card = uc.execute(_c01(), [TraineeTurn(1, "")], trainee_id="T004")
        assert len(card.missing_required_elements) == 5

    def test_multi_turn_merge(self) -> None:
        """要素跨轮次提到也要合并判定。"""
        turns = [
            TraineeTurn(1, "有的，康源牌鱼油软胶囊。"),
            TraineeTurn(2, "这是保健食品，批准功能是有助于维持血脂健康水平。"),
            TraineeTurn(3, "保健食品不是药物，不能代替药物治疗疾病。"),
        ]
        uc = ReviewDrillUseCase()
        card = uc.execute(_c01(), turns, trainee_id="T005")

        missing = {c.element_id for c in card.missing_required_elements}
        assert "E1" not in missing
        assert "E2" not in missing
        assert "E4" not in missing
        assert "E5" not in missing

    def test_duration_recorded(self) -> None:
        uc = ReviewDrillUseCase()
        card = uc.execute(_c01(), [TraineeTurn(1, "有的。")],
                          trainee_id="T006", duration_sec=95)
        assert card.duration_sec == 95

    def test_ids_recorded(self) -> None:
        uc = ReviewDrillUseCase()
        card = uc.execute(_c01(), [TraineeTurn(1, "有的。")],
                          trainee_id="T007")
        assert card.trainee_id == "T007"
        assert card.case_id == "C01"


# ===================================================================
# 6. 展示结构
# ===================================================================
class TestDisplay:
    def test_four_sections(self) -> None:
        """P5 要求「四区块齐全」。"""
        uc = ReviewDrillUseCase()
        card = uc.execute(_c01(), [TraineeTurn(1, BAD_ANSWER)],
                          trainee_id="T002")
        d = card.to_display_dict()

        assert set(d.keys()) == {"基础信息", "风险提示", "要素覆盖", "待人工确认"}

    def test_render_text_readable(self) -> None:
        """文本渲染要能直接贴给学员看。"""
        uc = ReviewDrillUseCase()
        card = uc.execute(_c01(), [TraineeTurn(1, BAD_ANSWER)],
                          trainee_id="T002")
        text = uc.render_text(card)

        assert "【风险提示】" in text
        assert "【要素覆盖】" in text
        assert "【待人工确认】" in text
        assert "第1轮" in text

    def test_render_no_score_words(self) -> None:
        """渲染文本里不能出现「得分」「分数」这类词。"""
        uc = ReviewDrillUseCase()
        card = uc.execute(_c01(), [TraineeTurn(1, BAD_ANSWER)],
                          trainee_id="T002")
        text = uc.render_text(card)

        assert "得分" not in text
        assert "总分" not in text
        assert "评分" not in text
