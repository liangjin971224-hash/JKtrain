"""月度报告聚合与评分的测试。

★ 全部是纯函数测试，不需要 LLM / 数据库 / 网络，毫秒级。

覆盖三块：
  1. 能力项归并（14 种提问类型 → 6 个能力项）
  2. 单次得分公式（含 critical 一票否决、扣分取低者）
  3. 月度聚合（样本不足、红线、降级、薄弱判定）
"""
from __future__ import annotations

import pytest

from domain.services.monthly_report import (
    CAPABILITY_UNKNOWN,
    LEVEL_FAIR,
    LEVEL_INSUFFICIENT,
    LEVEL_RED_LINE,
    LEVEL_WEAK,
    MIN_SAMPLES,
    PENALTY_HIGH,
    PENALTY_MEDIUM,
    THRESHOLD_WEAK,
    build_report,
    capability_of,
    extract_question_type_key,
    score_one,
)


class _Review:
    """最小化的 DrillReviewRow 替身（只用聚合需要的字段）。"""

    def __init__(self, drill_id, case_id, case_type, *,
                 element_hit=4, element_total=4, duration=60,
                 high=0, medium=0, critical=0, created_at="2026-10-01",
                 risks=()):
        self.drill_id = drill_id
        self.case_id = case_id
        self.case_type = case_type
        self.trainee_id = "E001"
        self.duration_sec = duration
        self.element_hit = element_hit
        self.element_total = element_total
        self.element_hit_all = element_hit
        self.element_total_all = element_total
        self.risk_count = high + medium + critical
        self.high_count = high
        self.medium_count = medium
        self.critical_count = critical
        self.created_at = created_at
        self.risk_items = tuple(risks)


# ===================================================================
# 一、提问类型提取
# ===================================================================
class TestExtractKey:
    @pytest.mark.parametrize("raw,expected", [
        ("禁忌询问（contraindication）", "contraindication"),
        ("产品信息询问（product_existence）", "product_existence"),
        ("功效导向询问（efficacy_inquiry）", "efficacy_inquiry"),
        ("禁忌询问(contraindication)", "contraindication"),   # 半角括号
        ("contraindication", "contraindication"),             # 只有英文
    ])
    def test_extract(self, raw, expected) -> None:
        assert extract_question_type_key(raw) == expected

    @pytest.mark.parametrize("raw", [
        "", "   ", "无法识别的类型", "某某（unknown_key）",
    ])
    def test_unknown_returns_empty(self, raw) -> None:
        """识别不出来就返回空串 —— 不猜。"""
        assert extract_question_type_key(raw) == ""


# ===================================================================
# 二、能力项归并（14 种类型全覆盖）
# ===================================================================
class TestCapability:
    @pytest.mark.parametrize("raw,expected", [
        # A 产品信息
        ("产品信息询问（product_existence）", "A_product_info"),
        ("产品信息询问（product_ingredient）", "A_product_info"),
        # B 人群适配
        ("人群适配（target_group）", "B_population_fit"),
        ("特殊人群（special_population）", "B_population_fit"),
        # C 禁忌与配伍
        ("禁忌询问（contraindication）", "C_contraindication"),
        ("配伍询问（incompatibility）", "C_contraindication"),
        # D 功效合规
        ("功效导向询问（efficacy_inquiry）", "D_efficacy_compliance"),
        # E 异议处理（5 种）
        ("疗效质疑（efficacy_challenge）", "E_objection"),
        ("价格异议（price_objection）", "E_objection"),
        ("信任质疑（trust_challenge）", "E_objection"),
        ("他人经验（others_experience）", "E_objection"),
        ("竞品比较（competitor_compare）", "E_objection"),
        # F 红线
        ("替代药物询问（drug_replacement）", "F_red_line"),
        # G 扩展
        ("扩展询问（more_options）", "G_extension"),
    ])
    def test_all_14_types_mapped(self, raw, expected) -> None:
        """14 种提问类型必须全部有归属 —— 一个都不能漏。"""
        assert capability_of(raw) == expected

    def test_unknown_type(self) -> None:
        assert capability_of("完全不认识的类型") == CAPABILITY_UNKNOWN
        assert capability_of("") == CAPABILITY_UNKNOWN


# ===================================================================
# 三、单次得分
# ===================================================================
class TestScoreOne:
    def test_perfect(self) -> None:
        s, note = score_one(element_hit=4, element_total=4)
        assert s == 60.0
        assert "无违规" in note

    def test_half_elements(self) -> None:
        s, _ = score_one(element_hit=2, element_total=4)
        assert s == 30.0

    def test_high_penalty(self) -> None:
        """1 条 high：扣 15 → 上限 45；要素分 60 → 取 45。"""
        s, note = score_one(element_hit=4, element_total=4, high_count=1)
        assert s == 45.0
        assert "扣 15" in note

    def test_medium_penalty(self) -> None:
        """1 条 medium：扣 5 → 上限 55；要素分 60 → 取 55。"""
        s, _ = score_one(element_hit=4, element_total=4, medium_count=1)
        assert s == 55.0

    def test_takes_lower_of_two(self) -> None:
        """要素分低于扣分上限时，取要素分（不能因为扣分反而涨分）。"""
        # 要素 2/3 = 40 分；1 条 high → 上限 45；取 40
        s, _ = score_one(element_hit=2, element_total=3, high_count=1)
        assert s == 40.0

    def test_critical_is_zero(self) -> None:
        """红线一票否决，不管要素答得多好。"""
        s, note = score_one(element_hit=4, element_total=4, critical_count=1)
        assert s == 0.0
        assert "一票否决" in note

    def test_critical_overrides_everything(self) -> None:
        s, _ = score_one(element_hit=10, element_total=10, critical_count=2)
        assert s == 0.0

    def test_zero_elements_no_crash(self) -> None:
        """必答要素为 0 不能除零崩溃。"""
        s, note = score_one(element_hit=0, element_total=0)
        assert s == 0.0
        assert "数据异常" in note

    def test_penalty_cannot_go_negative(self) -> None:
        """扣分超过 60 时分数最低 0，不出现负数。"""
        s, _ = score_one(element_hit=4, element_total=4,
                         high_count=5, medium_count=5)
        assert s == 0.0

    def test_formula_constants(self) -> None:
        """常量必须与文档一致（改常量时这条会提醒同步文档）。"""
        from domain.services.monthly_report import ELEMENT_FULL_SCORE

        assert ELEMENT_FULL_SCORE == 60
        assert PENALTY_HIGH == 15
        assert PENALTY_MEDIUM == 5
        # ★ 分界线 = 满分 - 1 条 high（「1 条 high 就进薄弱」）
        assert THRESHOLD_WEAK == 45
        assert MIN_SAMPLES == 2


# ===================================================================
# 四、月度聚合
# ===================================================================
class TestBuildReport:
    def test_empty_reviews(self) -> None:
        r = build_report(trainee_id="E001", period="2026-10", reviews=[])
        assert r.total_drills == 0
        assert r.capabilities == ()
        assert r.has_weak is False

    def test_single_capability_avg(self) -> None:
        """同一能力项两次演练 → 算术平均。"""
        reviews = [
            _Review("d1", "C10", "价格异议（price_objection）",
                    element_hit=3, element_total=4, high=1),   # 45
            _Review("d2", "C09", "疗效质疑（efficacy_challenge）",
                    element_hit=4, element_total=4),            # 60
        ]
        r = build_report(trainee_id="E001", period="2026-10",
                         reviews=reviews)
        cap = r.capabilities[0]
        assert cap.capability == "E_objection"
        assert cap.samples == 2
        assert cap.score == 52.5
        assert cap.level == LEVEL_WEAK
        assert "E_objection" in r.weak_capabilities

    def test_insufficient_samples(self) -> None:
        """只有 1 次 → 不可评，且不参与薄弱判定。"""
        reviews = [
            _Review("d1", "C07", "禁忌询问（contraindication）",
                    element_hit=1, element_total=4),   # 要素分 15，很低
        ]
        r = build_report(trainee_id="E001", period="2026-10",
                         reviews=reviews)
        cap = r.capabilities[0]
        assert cap.samples == 1
        assert cap.score is None
        assert cap.level == LEVEL_INSUFFICIENT
        assert "样本不足" in cap.note
        # 不可评不该被算成薄弱
        assert r.weak_capabilities == ()

    def test_red_line_capability(self) -> None:
        """有红线 → 直接红线等级，即使样本只有 1 次。"""
        reviews = [
            _Review("d1", "C04", "替代药物询问（drug_replacement）",
                    element_hit=4, element_total=4, critical=1),
        ]
        r = build_report(trainee_id="E001", period="2026-10",
                         reviews=reviews)
        cap = r.capabilities[0]
        assert cap.level == LEVEL_RED_LINE
        assert cap.critical_count == 1
        assert "F_red_line" in r.weak_capabilities
        assert r.has_any_critical is True
        assert r.total_critical == 1

    def test_no_issue_level(self) -> None:
        """要素全对、零违规 → 「未见明显问题」。"""
        reviews = [
            _Review("d1", "C01", "产品信息询问（product_existence）",
                    element_hit=4, element_total=4),
            _Review("d2", "C02", "产品信息询问（product_ingredient）",
                    element_hit=4, element_total=4),
        ]
        r = build_report(trainee_id="E001", period="2026-10",
                         reviews=reviews)
        cap = r.capabilities[0]
        assert cap.score == 60.0
        assert cap.level == LEVEL_FAIR
        assert cap.level_label == "未见明显问题"

    def test_capability_order_stable(self) -> None:
        """能力项按固定顺序输出（报告排版稳定）。"""
        reviews = [
            _Review("d1", "C04", "替代药物询问（drug_replacement）",
                    critical=1),
            _Review("d2", "C01", "产品信息询问（product_existence）"),
        ]
        r = build_report(trainee_id="E001", period="2026-10",
                         reviews=reviews)
        keys = [c.capability for c in r.capabilities]
        assert keys == ["A_product_info", "F_red_line"]

    def test_degrade_when_many_unknown(self) -> None:
        """未分类超过 30% → 降级标记。"""
        reviews = [
            _Review("d1", "C01", "产品信息询问（product_existence）"),
            _Review("d2", "C99", "无法识别"),
        ]
        r = build_report(trainee_id="E001", period="2026-10",
                         reviews=reviews)
        assert r.degraded is True
        assert "未分类" in r.degrade_reason

    def test_no_degrade_when_few_unknown(self) -> None:
        """少量未分类不降级。"""
        reviews = [
            _Review(f"d{i}", "C01", "产品信息询问（product_existence）")
            for i in range(9)
        ] + [_Review("d9", "C99", "无法识别")]
        r = build_report(trainee_id="E001", period="2026-10",
                         reviews=reviews)
        assert r.degraded is False

    def test_outcomes_carry_risk_detail(self) -> None:
        """第 3 层明细必须带出来（第几轮第几句）。"""
        class _Risk:
            turn_index = 2
            sentence_index = 3
            level = "critical"
            rule_type = "功效声称"
            matched = "能降血脂"
            reason = "不得声称治疗功效"

        reviews = [
            _Review("d1", "C07", "禁忌询问（contraindication）",
                    critical=1, risks=[_Risk()]),
        ]
        r = build_report(trainee_id="E001", period="2026-10",
                         reviews=reviews)
        o = r.outcomes[0]
        assert len(o.risks) == 1
        assert o.risks[0]["location"] == "第2轮第3句"
        assert o.risks[0]["level"] == "critical"

    def test_totals(self) -> None:
        reviews = [
            _Review("d1", "C01", "产品信息询问（product_existence）",
                    duration=100),
            _Review("d2", "C01", "产品信息询问（product_existence）",
                    duration=200),
        ]
        r = build_report(trainee_id="E001", period="2026-10",
                         reviews=reviews)
        assert r.total_drills == 2
        assert r.total_duration_sec == 300
        assert r.distinct_cases == 1

    def test_case_title_fallback(self) -> None:
        """没给标题时用 case_id 兜底，不能空着。"""
        reviews = [_Review("d1", "C01", "产品信息询问（product_existence）")]
        r = build_report(trainee_id="E001", period="2026-10",
                         reviews=reviews)
        assert r.outcomes[0].case_title == "C01"

    def test_case_title_from_map(self) -> None:
        reviews = [_Review("d1", "C01", "产品信息询问（product_existence）")]
        r = build_report(trainee_id="E001", period="2026-10",
                         reviews=reviews,
                         case_titles={"C01": "客户询问是否有鱼油"})
        assert r.outcomes[0].case_title == "客户询问是否有鱼油"


class TestScoreSemantics:
    """★ 评分语义（2026-10-31 用户决策：方案 C，去掉「达标」档）。

    背景：原设计 §3.3.1 满分 60、§3.3.4 达标线 80，**结构上不自洽** ——
    全对零违规也只有 60 分，「达标」永远显示不出来。

    决策：不调阈值让它"能出现"，而是**承认公式分不出「优秀」**。
    要素覆盖 + 违规扣分只能回答「有没有错」，回答不了「做得多好」。

    现在的两档：
        ≥ 45 分 且 无 high 违规 → 未见明显问题
        其余                    → 薄弱（红线 / 样本不足另算）
    """

    def test_no_good_level_constant(self) -> None:
        """`LEVEL_GOOD` 必须不存在 —— 防止有人手滑加回来。"""
        import domain.services.monthly_report as m

        assert not hasattr(m, "LEVEL_GOOD"), (
            "LEVEL_GOOD 已被移除（见方案 C 决策），不该再出现")

    def test_threshold_derived_from_penalty(self) -> None:
        """分界线 = 满分 - 1 条 high（45）。

        ★ 这条关系必须成立：它表达「1 条 high 就进薄弱」。
          改扣分标准时这里会失败，提醒复核分界线。
        """
        from domain.services.monthly_report import (
            ELEMENT_FULL_SCORE,
            PENALTY_HIGH,
        )

        assert THRESHOLD_WEAK == ELEMENT_FULL_SCORE - PENALTY_HIGH

    def test_perfect_is_fair_not_better(self) -> None:
        """满分也只有「未见明显问题」，不存在更高档。"""
        reviews = [
            _Review("a", "C01", "产品信息询问（product_existence）",
                    element_hit=4, element_total=4),
            _Review("b", "C02", "产品信息询问（product_ingredient）",
                    element_hit=4, element_total=4),
        ]
        cap = build_report(trainee_id="E", period="2026-10",
                           reviews=reviews).capabilities[0]
        assert cap.score == 60.0
        assert cap.level == LEVEL_FAIR
        assert cap.level_label == "未见明显问题"

    def test_medium_only_still_fair(self) -> None:
        """只有 medium（建议改）→ 仍算「未见明显问题」。

        ★ 这是分界线取 45 的核心理由：
          medium 是「建议改」，把它等同于「薄弱」会让人失去重点。
        """
        reviews = [
            _Review("a", "C01", "产品信息询问（product_existence）",
                    element_hit=4, element_total=4, medium=2),
            _Review("b", "C02", "产品信息询问（product_ingredient）",
                    element_hit=4, element_total=4, medium=2),
        ]
        cap = build_report(trainee_id="E", period="2026-10",
                           reviews=reviews).capabilities[0]
        assert cap.score == 50.0
        assert cap.level == LEVEL_FAIR
        assert "仅建议级问题" in cap.note

    def test_one_high_is_weak(self) -> None:
        """出现 1 条 high（必须改）→ 薄弱。"""
        reviews = [
            _Review("a", "C01", "产品信息询问（product_existence）",
                    element_hit=4, element_total=4, high=1),
            _Review("b", "C02", "产品信息询问（product_ingredient）",
                    element_hit=4, element_total=4, high=1),
        ]
        cap = build_report(trainee_id="E", period="2026-10",
                           reviews=reviews).capabilities[0]
        assert cap.score == 45.0
        assert cap.level == LEVEL_WEAK
        assert "必须改级违规" in cap.note

    def test_high_blocks_even_with_high_score(self) -> None:
        """★ 分数够但含 high → 仍判薄弱。

        这是「不只看分数」的体现：要素答得少也能凑出 45 分，
        但那种凑出来的 45 分不该叫「没问题」。
        """
        reviews = [
            _Review("a", "C01", "产品信息询问（product_existence）",
                    element_hit=4, element_total=4, high=1),
            _Review("b", "C02", "产品信息询问（product_ingredient）",
                    element_hit=4, element_total=4, high=1),
        ]
        cap = build_report(trainee_id="E", period="2026-10",
                           reviews=reviews).capabilities[0]
        # 45 分 >= 分界线 45，但因有 high 仍判薄弱
        assert cap.score >= THRESHOLD_WEAK
        assert cap.level == LEVEL_WEAK
