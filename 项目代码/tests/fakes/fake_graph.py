"""图谱端口的测试替身。

内存实现，不连 Neo4j。
排除法与别名归一的逻辑在这里**也要真实现**——
否则用假实现测出来的应用层逻辑是假的。
"""
from __future__ import annotations

from domain.ports import GraphQueryPort
from domain.value_objects import (
    ContraindicationHit,
    EfficacyRef,
    ProductFacts,
    ProductRef,
    SimilarPair,
)


class FakeGraphQueryPort(GraphQueryPort):
    """内存图谱。语义与 Neo4jGraphAdapter 保持一致。"""

    def __init__(
        self,
        products: dict[str, ProductFacts] | None = None,
        *,
        fail_on_call: bool = False,
        catalog: tuple[EfficacyRef, ...] = (),
    ) -> None:
        self._products = products or {}
        self._fail_on_call = fail_on_call
        #: 目录项（可含「没有产品声称」的项，用来模拟真实数据里的空缺）
        self._catalog = catalog
        self.calls: list[str] = []          # 记录调用，便于断言"某方法没被调"

    def _guard(self, name: str) -> None:
        self.calls.append(name)
        if self._fail_on_call:
            raise AssertionError(f"FakeGraphQueryPort 不应被调用: {name}")

    # ---------------- 读 ----------------
    def get_product_facts(self, product_id: str) -> ProductFacts | None:
        self._guard("get_product_facts")
        return self._products.get(product_id)

    def find_products_by_efficacy(self, efficacy_id: str) -> list[ProductRef]:
        self._guard("find_products_by_efficacy")
        out: list[ProductRef] = []
        for p in self._products.values():
            # 只返回保健食品：普通食品没有批准功能
            if not p.is_health_food:
                continue
            if any(c.efficacy_id == efficacy_id for c in p.claims):
                out.append(self._to_ref(p))
        return out

    def find_efficacy_by_keyword(self, keyword: str) -> list[EfficacyRef]:
        """目录查询。

        ★ 假实现要能表达「目录里有、但没产品声称」这个状态，
          否则「法规不允许」和「本店没有」会被测成同一件事 ——
          那正是真实环境踩过的错（EFF_09 存在但 0 个产品声称）。
          这里从产品的 claims 反推目录，并额外支持注入「无人声称的目录项」。
        """
        self._guard("find_efficacy_by_keyword")
        kw = (keyword or "").strip()
        if not kw:
            return []

        out: list[EfficacyRef] = []
        seen: set[str] = set()

        # 1. 注入的目录项（包括没有产品声称的）
        for ref in self._catalog:
            if kw in ref.name and ref.efficacy_id not in seen:
                seen.add(ref.efficacy_id)
                out.append(ref)

        # 2. 从产品 claims 反推
        for p in self._products.values():
            for c in p.claims:
                if c.efficacy_id in seen:
                    continue
                if kw in c.text:
                    seen.add(c.efficacy_id)
                    out.append(EfficacyRef(
                        efficacy_id=c.efficacy_id,
                        name=c.text,
                        catalog_no=c.catalog_no,
                    ))
        return out

    def list_efficacy_catalog(self) -> list[EfficacyRef]:
        """目录全量：注入项 + 从产品 claims 反推。"""
        self._guard("list_efficacy_catalog")
        out: list[EfficacyRef] = list(self._catalog)
        seen = {r.efficacy_id for r in out}
        for p in self._products.values():
            for c in p.claims:
                if c.efficacy_id in seen:
                    continue
                seen.add(c.efficacy_id)
                out.append(EfficacyRef(
                    efficacy_id=c.efficacy_id, name=c.text,
                    catalog_no=c.catalog_no,
                ))
        return out

    def find_products_by_need(self, need_id: str) -> list[ProductRef]:
        """假实现：按需求 ID 匹配适宜人群名称（简化映射）。

        真实实现是多跳（Need ← Efficacy ← Product），
        这里只保留"能返回结果"的契约，不模拟跳数。
        """
        self._guard("find_products_by_need")
        return [
            self._to_ref(p) for p in self._products.values()
            if p.is_health_food
        ]

    def check_contraindication(
        self, product_id: str, contra_ids: list[str]
    ) -> list[ContraindicationHit]:
        self._guard("check_contraindication")
        product = self._products.get(product_id)
        if product is None:
            return []
        wanted = set(contra_ids)
        return [
            ContraindicationHit(
                product_id=product.product_id,
                contra_id=c.contra_id,
                name=c.name,
                severity=c.severity,
                reason=c.reason,
            )
            for c in product.contraindications
            if c.contra_id in wanted
        ]

    def find_products_excluding(self, contra_ids: list[str]) -> list[ProductRef]:
        """排除法：语义与 Cypher 的 WHERE NOT EXISTS 一致。"""
        self._guard("find_products_excluding")
        blocked = set(contra_ids)
        out: list[ProductRef] = []
        for p in self._products.values():
            if not p.is_health_food:
                continue
            own = {c.contra_id for c in p.contraindications}
            if not own & blocked:
                out.append(self._to_ref(p))
        return out

    def resolve_alias(self, alias: str) -> ProductRef | None:
        self._guard("resolve_alias")
        key = alias.strip()
        for p in self._products.values():
            if key in p.aliases or key == p.name:
                return self._to_ref(p)
        return None

    def find_similar_products(self, product_id: str) -> list[SimilarPair]:
        self._guard("find_similar_products")
        if product_id not in self._products:
            return []
        return [
            SimilarPair(
                product_id=product_id,
                other_product_id=o.product_id,
                other_name=o.name,
                difference="（测试替身无差异说明）",
            )
            for o in self._products.values()
            if o.product_id != product_id
        ][:1]

    # ---------------- 辅助 ----------------
    @staticmethod
    def _to_ref(p: ProductFacts) -> ProductRef:
        return ProductRef(
            product_id=p.product_id,
            name=p.name,
            category=p.category,
            approval_no=p.approval_no,
        )
