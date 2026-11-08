"""批准功能与禁忌等基础值对象。

全部 frozen + slots：值对象不可变，防止被下游意外改写。
"""
from __future__ import annotations

from dataclasses import dataclass

from domain.value_objects.severity import Severity


@dataclass(frozen=True, slots=True)
class ApprovalClaim:
    """批准的保健功能。

    `text` 必须是目录原文，**一字不可改**。
    改写「有助于维持血脂健康水平」为「降血脂」即构成违法宣传。

    来源：《允许保健食品声称的保健功能目录 非营养素补充剂（2023年版）》
    """

    efficacy_id: str
    text: str
    catalog_no: int | None = None
    catalog_version: str = "2023"


@dataclass(frozen=True, slots=True)
class Contraindication:
    """一条禁忌（不适宜人群）。

    `reason` 来自图谱 CONTRAINDICATED_FOR 关系，**全部带值**（47/47），
    这是给用户解释「为什么不能吃」的依据，不能空。
    """

    contra_id: str
    name: str
    severity: Severity = Severity.MEDIUM
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ContraindicationHit:
    """禁忌命中结果。

    注意：这是**返回值**不是异常——「产品对用户禁忌」是正常业务结果，
    不是错误。只有「调用方绕过判定」才是编程错误。
    """

    product_id: str
    contra_id: str
    name: str
    severity: Severity
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Usage:
    """用法用量。

    ★ 字段与 Neo4j `Usage` 节点的真实属性对齐（2026-10-22 核对）：
      usage_id / dosage / daily_limit / note
      文档里曾写过 method / frequency / timing，实物没有，按实物为准。
    """

    dosage: str = ""          # 如「每日2次，每次2粒，随餐服用」
    daily_limit: str = ""     # 如「不超过4.0g（EPA+DHA不低于1.0g）」
    note: str = ""            # 补充说明


@dataclass(frozen=True, slots=True)
class EfficacyRef:
    """一个批准功能目录项（**不要求本店有产品声称它**）。

    ★ 为什么要和 `ApprovalClaim` 分开：
      `ApprovalClaim` 是「某个产品被批准的功能」，挂在产品上；
      `EfficacyRef` 是「目录里允许声称的功能」，与产品无关。

      两者差一层，但差这一层决定了回答对不对：
        问「改善体脂可以吃什么」，若只查产品 → 0 个，
        模型就会编成「目录里没有这个功能」。
        而事实是目录第 9 项「有助于控制体内脂肪」**在**目录里，
        只是本店没有产品声称它。

      合规培训场景里，「法规不允许」和「本店没有」是两句完全不同的话，
      教错一句就是教错业务。
    """

    efficacy_id: str
    name: str
    catalog_no: int | None = None
    catalog_version: str = "2023"
    #: 是否营养素补充剂类（补齐了「补充钙、维生素D」这类）
    is_nutrient: bool = False


@dataclass(frozen=True, slots=True)
class ProductRef:
    """产品引用（轻量）。用于列表类返回，避免整份画像过大。"""

    product_id: str
    name: str
    category: str
    approval_no: str = ""


@dataclass(frozen=True, slots=True)
class SimilarPair:
    """相似产品对。

    关键约束：`difference` 是**客观差异描述**，不含优劣评价。
    「鱼油是软胶囊，Q10 是硬胶囊」可以；「Q10 比鱼油好」不行。
    """

    product_id: str
    other_product_id: str
    other_name: str
    difference: str
    lesson_point: str = ""
