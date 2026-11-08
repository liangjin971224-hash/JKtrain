"""产品类别枚举。

仅有两个值：保健食品、普通食品。
用 StrEnum 而非 Enum，是为了能直接与图谱里的中文字符串比较——
Neo4j 的 `category` 属性存的就是「保健食品」/「普通食品」。
"""
from __future__ import annotations

from enum import StrEnum


class ProductCategory(StrEnum):
    """产品类别。

    这是全系统最关键的一个判断：
    普通食品**不得有任何功能声称**（约束 #2）。
    任何生成功效相关文本的路径，必须先判类别。
    """

    HEALTH_FOOD = "保健食品"
    ORDINARY_FOOD = "普通食品"

    @property
    def allows_efficacy_claim(self) -> bool:
        """该类别的产品是否允许做功能声称。

        只有保健食品可以，普通食品一律不行。
        """
        return self is ProductCategory.HEALTH_FOOD
