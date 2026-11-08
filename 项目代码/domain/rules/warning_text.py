"""强制警示语检测。

────────────────────────────────────────────────────────────────────
规则来源
────────────────────────────────────────────────────────────────────
`06_正式内容/D004` 4.1「必须说的那句话」：

> **保健食品不是药物，不能代替药物治疗疾病**
>
> 这是国家强制要求所有保健食品必须标注的警示语，
> **不可省略、不可改写**。

`06_正式内容/D002` 第 5 条也确认这是「法规强制」。

────────────────────────────────────────────────────────────────────
设计要点
────────────────────────────────────────────────────────────────────
1. **只对保健食品要求**。普通食品本来就不该声称功效，
   给它加保健食品警示语反而奇怪（会让人误以为它是保健食品）。

2. **接受等价表述**。法规原文是「保健食品不是药物，不能代替药物治疗疾病」，
   但实际文案常写成「本品不能代替药物」「不能代替药物治疗疾病」。
   这些语义等价，都放行。只认逐字一致会导致大量误拦。

3. **检测的是「有没有说清楚不能代替药物」**，不是「有没有逐字复制原文」。
   所以用关键片段匹配而非整句匹配。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from domain.value_objects import ProductCategory

#: 法规原文（D004 4.1，不可改写）
WARNING_TEXT = "保健食品不是药物，不能代替药物治疗疾病"

#: 语义等价的警示片段。
#: ★ 只要命中任意一条即视为「已说明不能代替药物」。
#: 这样既能接受不同文案，又不会漏掉真正缺失警示语的回答。
_WARNING_FRAGMENTS = (
    "不能代替药物治疗疾病",
    "不能代替药物",
    "不能替代药物治疗疾病",
    "不能替代药物",
    "不是药物，不能代替",
    "保健食品不是药物",
    "本品不能代替药物",
)


class WarningStatus(StrEnum):
    """警示语状态。"""

    PRESENT = "present"          # 已包含
    MISSING = "missing"          # 保健食品但缺警示语
    NOT_REQUIRED = "not_required"  # 普通食品，不要求


@dataclass(frozen=True, slots=True)
class WarningCheckResult:
    """警示语检测结果。

    ★ 不含 score 字段（约束 #3）。
    """

    status: WarningStatus
    matched: str = ""

    @property
    def present(self) -> bool:
        return self.status is WarningStatus.PRESENT

    @property
    def blocked(self) -> bool:
        """是否必须拦截。只有「保健食品缺警示语」才拦。"""
        return self.status is WarningStatus.MISSING

    @property
    def reason(self) -> str:
        if self.status is WarningStatus.MISSING:
            return (
                f"保健食品的回答必须包含法定警示语「{WARNING_TEXT}」。"
                f"这是国家强制要求，不可省略。"
            )
        return ""


def check_warning_present(
    answer: str,
    category: ProductCategory | str | None,
) -> WarningCheckResult:
    """检查回答是否包含必要的警示语。

    Args:
        answer: 待检查的回答文本。
        category: 产品类别。

    Returns:
        检测结果。

    Note:
        **fail-safe**：类别未知时按保健食品处理（要求更严）。
        理由：漏放一条缺警示语的保健食品回答 = 违法宣传风险。
    """
    # 判定是否要求警示语：只有普通食品豁免
    cat_is_ordinary = False
    if isinstance(category, ProductCategory):
        cat_is_ordinary = category is ProductCategory.ORDINARY_FOOD
    elif category:
        try:
            cat_is_ordinary = (
                ProductCategory(str(category)) is ProductCategory.ORDINARY_FOOD)
        except ValueError:
            cat_is_ordinary = False      # 未知类别 → 按保健食品要求（fail-safe）
    else:
        cat_is_ordinary = False          # 类别缺失 → 按保健食品要求

    if cat_is_ordinary:
        return WarningCheckResult(status=WarningStatus.NOT_REQUIRED)

    text = answer or ""
    for fragment in _WARNING_FRAGMENTS:
        if fragment in text:
            return WarningCheckResult(
                status=WarningStatus.PRESENT, matched=fragment)

    return WarningCheckResult(status=WarningStatus.MISSING)
