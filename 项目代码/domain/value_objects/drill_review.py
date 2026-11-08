"""演练评阅相关值对象。

★ 本文件的核心是 `SummaryCard` —— 它承载**约束 #3：不含分数、不含结论**。

────────────────────────────────────────────────────────────────────
为什么不能给分数（这不是技术选择，是业务选择）
────────────────────────────────────────────────────────────────────
需求原文：「摘要卡不含 score / verdict（系统只出卡不出分）」。

三条理由：
1. **学员会优化分数，而不是优化话术**
   系统给 45 分，学员会问「怎么从 45 提到 80」，而不是「我这个话术哪里说错了」。
   培训目标是后者。

2. **分数有争议时系统成了背锅方**
   而系统其实只做了正则匹配——它没有能力判断「这个学员是不是真的理解」。

3. **规则匹配无法覆盖语义**
   「客户问能不能吃，学员说『建议先咨询医生』」
   和「学员说『你自己看着办』」在关键词上可能都算「未承诺」，
   但实际专业度差得远。这种差异只能靠人判断。

所以摘要卡提供的是**判断依据**：
    哪里违规了（含精确定位）、哪些要素没说到、哪些需要人再看一眼。
────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class RiskLevel(StrEnum):
    """风险级别。

    ★ 与 `Severity` 的区别：
      `Severity` 用于产品禁忌（医学严重度），这里是**话术合规风险**。
      两者不共用枚举：它们是不同的业务概念，混用会让规则表变得混乱。
    """

    CRITICAL = "critical"    # 违法宣传风险，一票否决
    HIGH = "high"            # 明显不当，必须改
    MEDIUM = "medium"        # 建议改

    @property
    def is_blocking(self) -> bool:
        return self is RiskLevel.CRITICAL


# ===================================================================
# 案例规格
# ===================================================================
@dataclass(frozen=True, slots=True)
class ElementSpec:
    """一条必答要素的定义。

    来源：`06_正式内容/02_情景案例/` 的「四、必答要素」表格。
    """

    element_id: str          # E1 / E2 …
    name: str
    weight: int              # 原始权重（仅作展示，**不参与算分**）
    required: bool           # True=必须，False=建议
    judgement_note: str = "" # 判定说明（给人看的）

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"权重不能为负: {self.weight}")
        if not self.element_id:
            raise ValueError("element_id 不能为空")

    @property
    def is_required(self) -> bool:
        return self.required


@dataclass(frozen=True, slots=True)
class ForbiddenRule:
    """一条违禁表述规则。

    来源：`06_正式内容/02_情景案例/` 的「五、违禁表述检测」表格。
    ★ 规则是**正则**，逐字来自案例文档，不是模型判断。
    """

    rule_type: str           # 违规宣称 / 效果承诺 / 主动推销 …
    pattern: str             # 正则
    level: RiskLevel
    note: str = ""

    def __post_init__(self) -> None:
        if not self.pattern:
            raise ValueError("pattern 不能为空")
        # 提前编译，规则写错时立刻暴露而不是运行时才炸
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(
                f"非法正则 {self.pattern!r}: {exc}") from exc

    def find_all(self, text: str) -> list[str]:
        """返回文本中所有命中片段。"""
        try:
            return [m.group(0) for m in re.finditer(self.pattern, text)]
        except re.error:
            return []


@dataclass(frozen=True, slots=True)
class ScriptedFollowUp:
    """预设追问。

    来源：「三、对话脚本」的追问表格。
    """

    follow_up_id: str        # F1 / F2 …
    question: str
    trigger_note: str = ""
    focus: str = ""          # 考察点


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """一个演练案例的完整规格。"""

    case_id: str
    title: str
    opening_line: str                       # 客户开场白

    question_type: str = ""
    difficulty: int = 0
    stage: str = ""
    teaching_point: str = ""

    elements: tuple[ElementSpec, ...] = ()
    forbidden_rules: tuple[ForbiddenRule, ...] = ()
    follow_ups: tuple[ScriptedFollowUp, ...] = ()

    reference_answer: str = ""
    is_red_line: bool = False               # 红线案例（一票否决）

    @property
    def required_elements(self) -> tuple[ElementSpec, ...]:
        """必需要素。"""
        return tuple(e for e in self.elements if e.required)

    @property
    def optional_elements(self) -> tuple[ElementSpec, ...]:
        """建议要素。"""
        return tuple(e for e in self.elements if not e.required)

    @property
    def critical_rules(self) -> tuple[ForbiddenRule, ...]:
        return tuple(r for r in self.forbidden_rules
                     if r.level is RiskLevel.CRITICAL)


# ===================================================================
# 学员对话
# ===================================================================
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])")


@dataclass(frozen=True, slots=True)
class TraineeTurn:
    """学员的一轮发言。

    ★ 保留分句：风险要定位到「第几句」，
      所以必须能按句切开，不能只存整段文本。
    """

    turn_index: int          # 从 1 开始（对外展示用第 1 轮）
    text: str

    def __post_init__(self) -> None:
        if self.turn_index < 1:
            raise ValueError(f"轮次从 1 开始，当前 {self.turn_index}")

    @property
    def sentences(self) -> list[str]:
        """按句切分，去掉空白句。

        切分点：。！？；!?;
        不在逗号切——「第几句」对用户要可数、可指认。
        """
        if not self.text or not self.text.strip():
            return []
        parts = _SENTENCE_SPLIT.split(self.text)
        return [p.strip() for p in parts if p and p.strip()]


# ===================================================================
# 风险项
# ===================================================================
@dataclass(frozen=True, slots=True)
class RiskItem:
    """一条风险提示。

    ★ 必须能定位到「第几轮第几句」——这是 P5 的硬验收项。
      只给「有违规」而不给位置，学员无法改进。
    """

    level: RiskLevel
    rule_type: str
    matched: str
    turn_index: int          # 从 1 开始
    sentence_index: int      # 从 1 开始
    reason: str = ""

    @property
    def location(self) -> str:
        """人类可读的定位。"""
        if self.turn_index < 1 or self.sentence_index < 1:
            raise ValueError(
                f"定位坐标非法: 第{self.turn_index}轮第{self.sentence_index}句。"
                f"轮次与句序都从 1 开始。"
            )
        return f"第{self.turn_index}轮第{self.sentence_index}句"

    @property
    def is_critical(self) -> bool:
        return self.level is RiskLevel.CRITICAL


# ===================================================================
# 要素判定
# ===================================================================
@dataclass(frozen=True, slots=True)
class ElementCheck:
    """一条要素的覆盖判定。

    ★ 注意 `weight` 只是**沿用案例定义里的原始权重**用于展示，
      系统**不**拿它算总分。
    """

    element_id: str
    element_name: str
    weight: int
    required: bool
    covered: bool
    evidence: str = ""       # 命中的原文片段（便于人工复核）

    @property
    def missing_required(self) -> bool:
        """是「缺失的必答要素」——判定回避的依据。"""
        return self.required and not self.covered

    @property
    def status_mark(self) -> str:
        """展示用标记。"""
        if self.covered:
            return "✅"
        return "❌" if self.required else "⚠️"


# ===================================================================
# 摘要卡
# ===================================================================
@dataclass(frozen=True, slots=True)
class SummaryCard:
    """演练判定摘要卡。

    ★★★ 约束 #3：**没有 score 字段，没有 verdict 字段，没有任何总分属性。**
    只提供判断依据，判定由人下。

    四区块（P5 验收要求）：
        基础信息 / 风险提示 / 要素覆盖 / 待人工确认
    """

    trainee_id: str = ""
    case_id: str = ""
    duration_sec: int = 0

    risk_items: tuple[RiskItem, ...] = ()
    element_coverage: tuple[ElementCheck, ...] = ()
    pending_human: tuple[str, ...] = ()

    # ---------------- 派生的筛选视图（都不是分数） ----------------
    @property
    def critical_items(self) -> tuple[RiskItem, ...]:
        """critical 级风险，供人工重点关注与「红线一票否决」展示。"""
        return tuple(r for r in self.risk_items if r.is_critical)

    @property
    def has_critical_violation(self) -> bool:
        """是否有 critical 违规。

        布尔判断，不是分数。前端可用它显示「红线」标记，
        但**不给分**——判定仍由人下。
        """
        return bool(self.critical_items)

    @property
    def missing_required_elements(self) -> tuple[ElementCheck, ...]:
        """缺失的必答要素。"""
        return tuple(c for c in self.element_coverage if c.missing_required)

    @property
    def covered_elements(self) -> tuple[ElementCheck, ...]:
        return tuple(c for c in self.element_coverage if c.covered)

    # ---------------- 展示 ----------------
    def to_display_dict(self) -> dict[str, object]:
        """转成展示用结构（四区块）。

        ★ 也不能夹带 score / verdict 键——否则等于绕过 dataclass 的约束。
        """
        return {
            "基础信息": {
                "学员": self.trainee_id,
                "案例": self.case_id,
                "用时": f"{self.duration_sec} 秒",
            },
            "风险提示": [
                {
                    "级别": item.level.value,
                    "类型": item.rule_type,
                    "命中": item.matched,
                    "定位": item.location,
                    "说明": item.reason,
                }
                for item in self.risk_items
            ],
            "要素覆盖": [
                {
                    "ID": c.element_id,
                    "要素": c.element_name,
                    "必答": c.required,
                    "已覆盖": c.covered,
                    "依据": c.evidence,
                }
                for c in self.element_coverage
            ],
            "待人工确认": list(self.pending_human),
        }


__all__ = [
    "CaseSpec",
    "ElementCheck",
    "ElementSpec",
    "ForbiddenRule",
    "RiskItem",
    "RiskLevel",
    "ScriptedFollowUp",
    "SummaryCard",
    "TraineeTurn",
]
