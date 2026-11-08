"""月度培训报告：能力项归并与评分。

纯领域服务：**零框架、零 IO、零 LLM**。可毫秒级全测。

────────────────────────────────────────────────────────────────────
它做什么
────────────────────────────────────────────────────────────────────
把「一次一次演练的评阅结果」聚合成「这个月这个人怎么样」：

    演练记录（多次） → 按案例聚合 → 按能力项归并 → 打分 → 等级

────────────────────────────────────────────────────────────────────
★★★ 与项目约束 #3 的关系（必须理解，否则会改错）
────────────────────────────────────────────────────────────────────
`SummaryCard` 明确规定「没有 score 字段、没有任何总分属性，判定由人下」。
本模块**不在在线链路里**——它只在月度报告（离线汇总）中算分。

两条线互不干扰：
    · 单次演练：`ReviewDrillUseCase` → `SummaryCard`（**无分**）
    · 月度报告：本模块 → `MonthlyReport`（**有分，标"参考分"**）

所以本模块算分不违反约束 #3，但也**绝不能把分数写回 SummaryCard**。

────────────────────────────────────────────────────────────────────
评分公式（全部可核对，改参数即可调）
────────────────────────────────────────────────────────────────────
单次得分 = min(要素分, 60 - 违规扣分)，critical 直接 0
要素分   = 命中必答要素 / 必答要素总数 × 60
违规扣分 = high × 15 + medium × 5（从 60 起扣）

能力项分 = 该能力项下所有演练得分的算术平均
等级     = ≥80 达标 / 60–79 待加强 / <60 薄弱 / 有 critical 红线 / 样本<2 不可评

★ 为什么算术平均而不是加权：
  按难度加权——`difficulty` 字段部分为 0（实测 C01/C04 曾为空，已修），
  且「难度高就该权重高」本身没有业务依据。
  按时间加权——没有依据说明「后来的演练更重要」。
  算术平均最简单、可解释、可复算。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ===================================================================
# 可调参数（改这里就能调整评分，不用改逻辑）
# ===================================================================

#: 要素分满分。留 40 分给违规扣分。
ELEMENT_FULL_SCORE = 60

#: high 级违规每条扣分。
PENALTY_HIGH = 15

#: medium 级违规每条扣分。
PENALTY_MEDIUM = 5

# ★★ 没有「达标」档，只有「未见明显问题 / 薄弱」两档
#:（2026-10-31 用户决策：方案 C）
#:
#: ── 为什么去掉「达标」 ──
#:   原设计：要素分满分 60、违规从 60 起扣，但等级阈值写的是「≥80 达标」。
#:   两个数字**结构上不自洽** —— 全对零违规也只有 60 分，
#:   「达标」这个等级永远不可能出现，是个永远显示不了的档位。
#:
#:   三种修法里选了「去掉该档」，不是「把阈值调低让它能出现」：
#:   ★ 这套公式本来就是**只衡量「有没有错」，衡量不了「做得有多好」**。
#:     要素覆盖 + 违规扣分能回答「有没有漏答、有没有违规」，
#:     回答不了「答得够不够专业、沟通够不够流畅」。
#:     硬造一个「达标」档，是给不出结论的事假装给出结论。
#:
#: ── 分界线为什么是 45 而不是满分 60 ──
#:   实测组合（要素全对前提下）：
#:       0 high + 0 medium = 60        ← 零瑕疵
#:       0 high + 1 medium = 55        ← 只有「建议改」级别
#:       0 high + 2 medium = 50
#:       1 high + 0 medium = 45        ← 出现了「必须改」级别
#:       1 high + 1 medium = 40
#:
#:   ★ 分界放在 45：**只要没出现 high（必须改）级违规，就不算薄弱。**
#:     理由：medium 的定义是「建议改」，把它等同于「薄弱」会让人失去重点
#:     —— 全是建议级问题的报告，和真的说错话的报告，不该显示成同一档。
#:
#:   ★ 45 这个数不是拍的：它正好是「1 条 high」的分数（60-15=45）。
#:     也就是说「0 条 high」与「≥1 条 high」在分数上天然以 45 分界，
#:     分界线取在两者之间即可，取 45 表示「1 条 high 就进薄弱」。

#: 薄弱分界。≥ 此分（且无 high 以上问题）→ 未见明显问题；< 此分 → 薄弱。
#:
#: ★ 与 `PENALTY_HIGH` 相关：分界 = 满分 - 1 条 high。
#:   改扣分标准时要同步复核这个值。
THRESHOLD_WEAK = ELEMENT_FULL_SCORE - PENALTY_HIGH

#: 样本量下限：少于这个数不打分（只标「不可评」）。
MIN_SAMPLES = 2

# ---------------- 等级 ----------------
#: ★ 没有 LEVEL_GOOD。见 `THRESHOLD_WEAK` 上方注释：
#:   这套评分公式分不出「优秀」，只分得出「有没有问题」。
LEVEL_FAIR = "fair"            # 待加强（要素答全、无违规）
LEVEL_WEAK = "weak"            # 薄弱
LEVEL_RED_LINE = "red_line"    # 出现红线
LEVEL_INSUFFICIENT = "insufficient"   # 样本不足
LEVEL_NOT_TRAINED = "not_trained"     # 本月未练

#: 等级的展示文案。
#:
#: ★ `fair` 的文案从「待加强」改成「未见明显问题」：
#:   它现在的含义不是「勉强及格」，而是「这一档查不出具体毛病」。
#:   用「待加强」会让主管以为有问题但说不清是什么，反而误导。
LEVEL_LABELS = {
    LEVEL_FAIR: "未见明显问题",
    LEVEL_WEAK: "薄弱",
    LEVEL_RED_LINE: "红线",
    LEVEL_INSUFFICIENT: "不可评",
    LEVEL_NOT_TRAINED: "本月未练",
}

# ===================================================================
# ★ 能力项归并表（14 种提问类型 → 6 个能力项）
# ===================================================================
#: 提问类型 → 能力项。
#:
#: ★ 依据：案例库「基本设定」的 `提问类型` 字段（16 个案例共 14 种类型）。
#:   每个归并都写了理由，见 `CAPABILITY_REASONS`。
#:
#: ★ 键用**英文类型名**而不是中文全称：
#:   原文形如「禁忌询问（contraindication）」，用中文全称做键太脆弱
#:   （改一个字就对不上）。英文标识更稳定。
_QUESTION_TYPE_TO_CAPABILITY: dict[str, str] = {
    # A 产品信息：客观陈述产品是什么，不问个人适配、不涉功效
    "product_existence": "A_product_info",
    "product_ingredient": "A_product_info",
    # B 人群适配：陈述适用人群范围，不判断个人能否吃
    "target_group": "B_population_fit",
    "special_population": "B_population_fit",
    # C 禁忌与配伍：涉及与药物/其他产品同用，风险最高
    "contraindication": "C_contraindication",
    "incompatibility": "C_contraindication",
    # D 功效合规：单独成项——判定标准是「一字不改复述批件原文」
    "efficacy_inquiry": "D_efficacy_compliance",
    # E 异议处理：客户不信任/不接受，要求不贬低、不承诺、不硬推
    "efficacy_challenge": "E_objection",
    "price_objection": "E_objection",
    "trust_challenge": "E_objection",
    "others_experience": "E_objection",
    "competitor_compare": "E_objection",
    # F 红线意识：唯一「碰了就违法」的类别，不能与其他并列
    "drug_replacement": "F_red_line",
    # G 其他：C16 扩展询问，暂归一类（数量少，单独归并意义不大）
    "more_options": "G_extension",
}

#: 能力项展示名与归并理由（报告与文档都要用）。
CAPABILITY_META: dict[str, dict[str, str]] = {
    "A_product_info": {
        "name": "产品信息",
        "reason": "客观陈述产品是什么，不问个人适配、不涉功效",
    },
    "B_population_fit": {
        "name": "人群适配",
        "reason": "都在问「某类人能不能吃」，判定动作相同：陈述范围，不判断个人",
    },
    "C_contraindication": {
        "name": "禁忌与配伍",
        "reason": "涉及与药物/其他产品同用，答错直接有健康风险，风险等级最高",
    },
    "D_efficacy_compliance": {
        "name": "功效合规",
        "reason": "判定标准是「能否一字不改复述批件原文」，与其他项都不同",
    },
    "E_objection": {
        "name": "异议处理",
        "reason": "都是「客户不信任/不接受」，共同要求是不贬低、不承诺、不硬推",
    },
    "F_red_line": {
        "name": "红线意识",
        "reason": "唯一「碰了就违法」的类别（替代药物），不能与其他并列",
    },
    "G_extension": {
        "name": "扩展询问",
        "reason": "客户问「还有别的吗」，考核点是给出替代方案而非回避",
    },
}

#: 能力项固定顺序（报告展示用）。
CAPABILITY_ORDER: tuple[str, ...] = (
    "A_product_info",
    "B_population_fit",
    "C_contraindication",
    "D_efficacy_compliance",
    "E_objection",
    "F_red_line",
    "G_extension",
)

#: 归不进任何能力项的兜底键。
CAPABILITY_UNKNOWN = "UNKNOWN"


def extract_question_type_key(raw: str) -> str:
    """从「禁忌询问（contraindication）」里取出 `contraindication`。

    ★ 提取失败返回空串，调用方归入 UNKNOWN。
      绝不猜——猜错会让整个能力项分数失真。
    """
    text = (raw or "").strip()
    if not text:
        return ""
    left = text.find("（")
    right = text.rfind("）")
    if left < 0 or right < left:
        left = text.find("(")
        right = text.rfind(")")
    if left < 0 or right < left:
        # 没有括号：可能原文就只写了英文
        return text if text in _QUESTION_TYPE_TO_CAPABILITY else ""
    key = text[left + 1:right].strip()
    return key if key in _QUESTION_TYPE_TO_CAPABILITY else ""


def capability_of(case_type: str) -> str:
    """案例的提问类型 → 能力项键。归不进去返回 `UNKNOWN`。"""
    key = extract_question_type_key(case_type)
    return _QUESTION_TYPE_TO_CAPABILITY.get(key, CAPABILITY_UNKNOWN)


# ===================================================================
# 一、单次演练得分
# ===================================================================
def score_one(
    *,
    element_hit: int,
    element_total: int,
    high_count: int = 0,
    medium_count: int = 0,
    critical_count: int = 0,
) -> tuple[float, str]:
    """算一次演练的得分。

    Returns:
        `(分数, 说明)`。说明用于报告里解释「为什么是这个分」。

    ★ critical 直接 0 分（一票否决），不做任何折算。
      理由：红线违规不存在「扣多少」的问题——它本身就是不可接受的。
    """
    if critical_count > 0:
        return 0.0, f"出现 {critical_count} 条红线违规，一票否决"

    # 要素分。总数 0 时按 0 算（防止除零），但这种案例不可评。
    if element_total <= 0:
        element_score = 0.0
        element_note = "无必答要素（数据异常）"
    else:
        element_score = element_hit / element_total * ELEMENT_FULL_SCORE
        element_note = f"必答要素 {element_hit}/{element_total}"

    # 违规扣分
    penalty = high_count * PENALTY_HIGH + medium_count * PENALTY_MEDIUM
    after_penalty = ELEMENT_FULL_SCORE - penalty

    final = min(element_score, after_penalty)
    final = max(0.0, final)

    if penalty > 0:
        note = (f"{element_note}（{element_score:.0f} 分），"
                f"扣 {penalty} 分后上限 {after_penalty:.0f}，取低者")
    else:
        note = f"{element_note}，无违规"

    return round(final, 1), note


# ===================================================================
# 二、数据结构
# ===================================================================
@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """一个案例的一次演练结果（报告第 2 层）。"""

    drill_id: str
    case_id: str
    case_title: str
    case_type: str
    capability: str
    duration_sec: int
    element_hit: int
    element_total: int
    risk_count: int
    high_count: int
    medium_count: int
    critical_count: int
    score: float
    score_note: str = ""
    created_at: str = ""
    #: 违规明细（第 3 层），`(第几轮, 第几句, 等级, 类型, 命中片段, 理由)`
    risks: tuple[dict, ...] = ()

    @property
    def has_critical(self) -> bool:
        return self.critical_count > 0


@dataclass(frozen=True, slots=True)
class CapabilityScore:
    """一个能力项的月度成绩（报告第 1 层）。"""

    capability: str
    name: str
    samples: int
    score: float | None          # None = 不可评
    level: str
    level_label: str
    #: 该能力项下出现的红线数
    critical_count: int = 0
    #: 参与计算的演练 id（便于追溯「这个分是怎么来的」）
    drill_ids: tuple[str, ...] = ()
    #: 解释「为什么是这个分 / 为什么不可评」
    note: str = ""

    @property
    def is_weak(self) -> bool:
        return self.level in (LEVEL_WEAK, LEVEL_RED_LINE)


@dataclass(frozen=True, slots=True)
class MonthlyReport:
    """一份月度报告。"""

    trainee_id: str
    period: str                       # 如 "2026-10"
    total_drills: int = 0
    total_duration_sec: int = 0
    distinct_cases: int = 0
    capabilities: tuple[CapabilityScore, ...] = ()
    outcomes: tuple[CaseOutcome, ...] = ()
    #: 改名超过 30% 时的降级说明（见 `build_report`）
    degraded: bool = False
    degrade_reason: str = ""
    #: 需要下钻的能力项（条件边依据）
    weak_capabilities: tuple[str, ...] = ()

    @property
    def has_weak(self) -> bool:
        return bool(self.weak_capabilities)

    @property
    def has_any_critical(self) -> bool:
        return any(o.has_critical for o in self.outcomes)

    @property
    def total_critical(self) -> int:
        return sum(o.critical_count for o in self.outcomes)


# ===================================================================
# 三、聚合与评分
# ===================================================================
def build_report(
    *,
    trainee_id: str,
    period: str,
    reviews: list,
    case_titles: dict[str, str] | None = None,
) -> MonthlyReport:
    """把一堆评阅结果聚合成月度报告。

    Args:
        trainee_id: 学员编号。
        period: 统计周期（如 `2026-10`），仅用于展示。
        reviews: `DrillReviewRow` 列表（已按月过滤）。
        case_titles: `case_id → 标题`，报告展示用。缺了就用 case_id。

    Returns:
        `MonthlyReport`。**永不抛异常**——报告生成失败不该让批次挂掉。
    """
    titles = case_titles or {}
    if not reviews:
        return MonthlyReport(trainee_id=trainee_id, period=period)

    # ---- 第 2 层：逐次演练 ----
    outcomes: list[CaseOutcome] = []
    for r in reviews:
        score, note = score_one(
            element_hit=r.element_hit,
            element_total=r.element_total,
            high_count=r.high_count,
            medium_count=r.medium_count,
            critical_count=r.critical_count,
        )
        risks = tuple(
            {
                "turn_index": it.turn_index,
                "sentence_index": it.sentence_index,
                "level": it.level,
                "rule_type": it.rule_type,
                "matched": it.matched,
                "reason": it.reason,
                "location": f"第{it.turn_index}轮第{it.sentence_index}句",
            }
            for it in (r.risk_items or ())
        )
        outcomes.append(CaseOutcome(
            drill_id=r.drill_id,
            case_id=r.case_id,
            case_title=titles.get(r.case_id, r.case_id),
            case_type=r.case_type,
            capability=capability_of(r.case_type),
            duration_sec=r.duration_sec,
            element_hit=r.element_hit,
            element_total=r.element_total,
            risk_count=r.risk_count,
            high_count=r.high_count,
            medium_count=r.medium_count,
            critical_count=r.critical_count,
            score=score,
            score_note=note,
            created_at=r.created_at,
            risks=risks,
        ))

    # ---- 第 1 层：按能力项归并 ----
    buckets: dict[str, list[CaseOutcome]] = {}
    for o in outcomes:
        buckets.setdefault(o.capability, []).append(o)

    unknown = len(buckets.get(CAPABILITY_UNKNOWN, []))
    total = len(outcomes)
    degraded = total > 0 and unknown / total > 0.3

    caps: list[CapabilityScore] = []
    for key in CAPABILITY_ORDER:
        items = buckets.get(key, [])
        if not items:
            continue
        caps.append(_score_capability(key, items))

    # 归并失败的能力项（只有降级时才展示）
    if unknown and degraded:
        caps.append(_score_capability(CAPABILITY_UNKNOWN, buckets[CAPABILITY_UNKNOWN]))

    # ---- 薄弱项判定（复合条件）----
    weak = tuple(
        c.capability for c in caps
        if c.level == LEVEL_RED_LINE
        or (c.level == LEVEL_WEAK and c.samples >= MIN_SAMPLES)
    )
    # 有红线违规时，即使分数没低于阈值，也要视为有薄弱项
    if any(o.has_critical for o in outcomes) and not weak:
        weak = tuple(c.capability for c in caps if c.critical_count > 0)

    return MonthlyReport(
        trainee_id=trainee_id,
        period=period,
        total_drills=len(outcomes),
        total_duration_sec=sum(o.duration_sec for o in outcomes),
        distinct_cases=len({o.case_id for o in outcomes}),
        capabilities=tuple(caps),
        outcomes=tuple(outcomes),
        degraded=degraded,
        degrade_reason=(
            f"未分类案例 {unknown}/{total} 超过 30%，能力项归并不可靠"
            if degraded else ""
        ),
        weak_capabilities=weak,
    )


def _score_capability(key: str, items: list[CaseOutcome]) -> CapabilityScore:
    """给一个能力项打分。"""
    meta = CAPABILITY_META.get(key, {"name": key, "reason": ""})
    name = meta["name"]
    n = len(items)
    drill_ids = tuple(o.drill_id for o in items)
    critical = sum(o.critical_count for o in items)

    # 有红线 → 直接红线等级，不看平均分
    if critical > 0:
        return CapabilityScore(
            capability=key, name=name, samples=n,
            score=0.0, level=LEVEL_RED_LINE,
            level_label=LEVEL_LABELS[LEVEL_RED_LINE],
            critical_count=critical, drill_ids=drill_ids,
            note=f"出现 {critical} 条红线违规，一票否决",
        )

    # 样本不足 → 不可评
    if n < MIN_SAMPLES:
        return CapabilityScore(
            capability=key, name=name, samples=n,
            score=None, level=LEVEL_INSUFFICIENT,
            level_label=LEVEL_LABELS[LEVEL_INSUFFICIENT],
            critical_count=0, drill_ids=drill_ids,
            note=f"仅 {n} 次演练，样本不足（需 ≥ {MIN_SAMPLES} 次）",
        )

    avg = round(sum(o.score for o in items) / n, 1)
    high_total = sum(o.high_count for o in items)
    element_hit_sum = sum(o.element_hit for o in items)
    element_total_sum = sum(o.element_total for o in items)

    # ★ 两档判定（无「达标」档，见 THRESHOLD_WEAK 上方注释）
    #   进「未见明显问题」要同时满足：平均分 ≥ 分界线，且**没有 high 级违规**。
    #   单看分数不够 —— 要素答得少也能凑出 45 分，那不该叫「没问题」。
    if avg >= THRESHOLD_WEAK and high_total == 0:
        level = LEVEL_FAIR
        detail = []
        if element_hit_sum == element_total_sum:
            detail.append("要素答全")
        if not any(o.medium_count for o in items):
            detail.append("无违规")
        elif element_hit_sum == element_total_sum:
            detail.append("仅建议级问题")
        suffix = f"（{'、'.join(detail)}）" if detail else ""
        note = f"{n} 次演练平均 {avg} 分{suffix}"
    else:
        reasons = [f"{n} 次演练平均 {avg} 分"]
        if high_total:
            reasons.append(f"含 {high_total} 条必须改级违规")
        level, note = LEVEL_WEAK, "，".join(reasons)

    return CapabilityScore(
        capability=key, name=name, samples=n, score=avg,
        level=level, level_label=LEVEL_LABELS[level],
        critical_count=0, drill_ids=drill_ids, note=note,
    )


__all__ = [
    "CAPABILITY_META",
    "CAPABILITY_ORDER",
    "CAPABILITY_UNKNOWN",
    "LEVEL_LABELS",
    "LEVEL_FAIR",
    "LEVEL_INSUFFICIENT",
    "LEVEL_NOT_TRAINED",
    "LEVEL_RED_LINE",
    "LEVEL_WEAK",
    "MIN_SAMPLES",
    "CapabilityScore",
    "CaseOutcome",
    "MonthlyReport",
    "build_report",
    "capability_of",
    "extract_question_type_key",
    "score_one",
]
