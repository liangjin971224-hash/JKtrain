"""产品完整事实画像。

字段与 Neo4j `jiankangshujuku` 的属性名**一一对应**（大小写敏感）。
权威来源：`08_数据入库/neo4j/verify_final.py` 的查询语句。
"""
from __future__ import annotations

from dataclasses import dataclass

from domain.value_objects.approval_claim import (
    ApprovalClaim,
    Contraindication,
    Usage,
)
from domain.value_objects.product_category import ProductCategory


@dataclass(frozen=True, slots=True)
class ProductFacts:
    """产品的完整事实画像。

    这是问答链路的核心输入：回答「这款产品能不能吃」必须基于它，
    而不是靠向量检索捞到的一段文本——文本可能过期或不完整。
    """

    product_id: str
    name: str
    category: str
    approval_no: str = ""
    dosage_form: str = ""
    spec: str = ""
    retail_price: int | None = None

    claims: tuple[ApprovalClaim, ...] = ()
    ingredients: tuple[str, ...] = ()
    usage: Usage | None = None
    suitable_for: tuple[str, ...] = ()
    contraindications: tuple[Contraindication, ...] = ()
    aliases: tuple[str, ...] = ()

    # 普通食品专用：合规说明（为什么不能说功效）
    compliance_note: str | None = None
    data_origin: str = "real"

    @property
    def category_enum(self) -> ProductCategory | None:
        """类别枚举。图谱里出现未知值（脏数据）时返回 None。"""
        try:
            return ProductCategory(self.category)
        except ValueError:
            return None

    @property
    def is_health_food(self) -> bool:
        """是否保健食品。

        ★ 任何功效表述前必须先判这个。
        未知类别一律按「不允许声称」处理（fail-safe），不能放行。
        """
        return self.category_enum is ProductCategory.HEALTH_FOOD

    @property
    def allows_efficacy_claim(self) -> bool:
        """是否允许做功能声称。

        普通食品与未知类别都不允许。命名比 is_health_food 更贴合调用意图。
        """
        return self.is_health_food

    @property
    def critical_contraindications(self) -> tuple[Contraindication, ...]:
        """critical 级禁忌。这些一旦命中必须明确告知用户。"""
        return tuple(c for c in self.contraindications if c.severity.must_block)
