"""严重度枚举。

用于禁忌、违规表述、评阅风险项的定级。
图谱里 `severity` 属性的取值就是这三个小写字符串，必须严格对齐。
"""
from __future__ import annotations

from enum import StrEnum


class Severity(StrEnum):
    """风险严重度。

    critical：会致害或构成违法宣传，**必须拦截**，无例外
    high：明显不当，必须提示
    medium：表述不严谨，建议改进
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"

    @property
    def must_block(self) -> bool:
        """是否必须阻断输出。

        只有 critical 无条件拦截。这是 P4 护栏的判定依据。
        """
        return self is Severity.CRITICAL

    @property
    def order(self) -> int:
        """排序权重，数值越大越严重。便于取「最严重的一项」。"""
        return {Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}[self]
