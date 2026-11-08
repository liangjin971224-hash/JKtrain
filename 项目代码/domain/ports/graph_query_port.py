"""图谱查询端口。

只声明「我需要什么能力」，不涉及 Cypher、不涉及 neo4j 类型。
所有返回类型均为领域对象。

数据源：Neo4j `jiankangshujuku`（156 节点 / 184 关系）
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from domain.value_objects import (
    ContraindicationHit,
    ProductFacts,
    ProductRef,
    SimilarPair,
)


class GraphQueryPort(ABC):
    """图谱查询端口。"""

    @abstractmethod
    def get_product_facts(self, product_id: str) -> ProductFacts | None:
        """按产品 ID 取完整事实画像。

        包含：基本信息、批准功能、成分、用法、适宜人群、不适宜人群、别名。
        不存在返回 None（不抛异常——查询不到是正常结果）。

        Args:
            product_id: 如 "P001"。

        Returns:
            完整画像；产品不存在时 None。
        """

    @abstractmethod
    def find_efficacy_by_keyword(self, keyword: str) -> list[EfficacyRef]:
        """在**批准功能目录**里按关键词查（不要求有产品声称它）。

        ★ 为什么必须与 `find_products_by_efficacy` 分开
          （2026-10-23 接多跳时发现的实际错误）：

            问「改善体脂可以吃什么保健品」，只查产品会得到「0 个产品」。
            此时模型必须能区分两种情况，因为**说法完全不同**：

              A. 目录里没有这个功能 → 「保健食品不允许声称这个功能」
              B. 目录里有，但本店没产品声称 → 「法规允许声称
                 『有助于控制体内脂肪』，但本店在售产品没有这项批准功能」

            实测踩坑：真实库中 `EFF_09「有助于控制体内脂肪」` 存在，
            但没有任何产品声称它。只查产品得到 0，模型便自行编成
            「目录中没有『改善体脂』等表述」—— **说错了法规**。
            合规培训场景下这种错误比不回答更严重。

        Args:
            keyword: 需求/功效的口语说法，如「体脂」「改善体脂」「睡眠」。

        Returns:
            目录里匹配的功能项。**空列表表示目录中没有**（不等于没有产品）。
        """

    @abstractmethod
    def list_efficacy_catalog(self) -> list[EfficacyRef]:
        """取整个批准功能目录快照（15 项左右，量很小）。

        ★ 为什么必须有这个方法，而不是用空关键词调上面那个：
          上面那个约定「空关键词返回空」（否则会被误当成全部匹配）。
          但口语匹配需要**目录全量**做对撞（「体脂」要跟
          「控制体内脂肪」「改善骨密度」等逐项比），没有全量就匹配不了。

        ★ 为什么不担心性能：
          目录只有十几项，且内容基本不变，调用方会缓存。
          用「量小 + 缓存」换取「不猜」——比在代码里硬编码目录安全得多
          （目录更新时硬编码不会跟着变，那才是真正的事故）。
        """

    @abstractmethod
    def find_products_by_efficacy(self, efficacy_id: str) -> list[ProductRef]:
        """按批准功能反查产品。

        只返回保健食品——普通食品没有批准功能，不该出现在这里。

        Args:
            efficacy_id: 功效 ID，如 "EFF_19"（血脂）。

        Returns:
            具备该批准功能的产品列表。无匹配返回空列表。
        """

    @abstractmethod
    def find_products_by_need(self, need_id: str) -> list[ProductRef]:
        """按客户需求反查产品（经 Efficacy 的 ADDRESSES 关系，多跳）。

        Args:
            need_id: 需求 ID，如 "NEED_SLEEP"。

        Returns:
            满足该需求的产品列表。
        """

    @abstractmethod
    def check_contraindication(
        self, product_id: str, contra_ids: list[str]
    ) -> list[ContraindicationHit]:
        """检查指定产品对给定人群/状况是否禁忌。

        这是图谱最核心的安全能力：向量检索做不了「排除法」，
        图谱一条边就能查。

        ★ 返回空列表表示「不禁忌」，这是正常业务结果，不是异常。
          禁忌命中同样用返回值表达，不用异常。

        Args:
            product_id: 产品 ID。
            contra_ids: 禁忌 ID 列表，如 ["CI_BLEED", "CI_RENAL"]。

        Returns:
            命中的禁忌列表（含 severity 与 reason）。
            产品不存在或无一命中时返回空列表。
        """

    @abstractmethod
    def find_products_excluding(self, contra_ids: list[str]) -> list[ProductRef]:
        """排除法：找出**不含有**指定禁忌的所有产品。

        用于「客户有高血压 + 肾功能不全，能吃什么」这类问题。
        这是图谱相对向量检索的核心优势。

        Args:
            contra_ids: 要排除的禁忌 ID 列表。

        Returns:
            不含任何一项指定禁忌的保健食品列表。
        """

    @abstractmethod
    def resolve_alias(self, alias: str) -> ProductRef | None:
        """把口语名映射到标准产品。

        客户不会说「康源牌鱼油软胶囊」，只会说「鱼油」。
        没有这一层，检索会大量漏召。

        Args:
            alias: 口语名，如 "鱼油"、"Q10"。

        Returns:
            对应产品；无映射返回 None。
        """

    @abstractmethod
    def find_similar_products(self, product_id: str) -> list[SimilarPair]:
        """找相似产品对（含客观差异说明）。

        注意：返回的差异是**客观描述**，不含优劣评价。

        Args:
            product_id: 产品 ID。

        Returns:
            相似产品对列表，含差异说明与教学要点。
        """
