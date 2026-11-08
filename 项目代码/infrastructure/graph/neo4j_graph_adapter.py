"""Neo4j 图谱查询适配器。

实现 `GraphQueryPort` 的 7 个方法。

数据源：`jiankangshujuku`（156 节点 / 184 关系）
    Product 20 / Alias 35 / Usage 20 / Contraindication 18 / Efficacy 15
    Ingredient 14 / Rule 10 / TargetGroup 8 / Scenario 8 / Need 7 / WarningMark 1

────────────────────────────────────────────────────────────────────
设计约束（违反即架构失效）
────────────────────────────────────────────────────────────────────
1. 返回值全部是领域对象，**绝不让 neo4j.Record / neo4j.Node 穿越到上层**
   转换逻辑集中在文件底部的 `_to_*` 函数。
2. 库名从构造函数注入，**不硬编码**（门禁要求）。
3. 连接/查询失败抛 `GraphQueryError` 并带上下文（product_id / database）。
4. 「查不到」「不禁忌」是**正常结果**，返回 None / 空列表，不抛异常。

────────────────────────────────────────────────────────────────────
实测数据要点（2026-10-22 核对真实库）
────────────────────────────────────────────────────────────────────
- Product 属性：product_id / name / category / approval_no / dosage_form /
  spec / retail_price / status / primary_scenario / data_origin
  ★ 没有 compliance_note —— 契约文档写过，实物没有，以实物为准
- Usage 属性：usage_id / dosage / daily_limit / note
  ★ 没有 method / frequency / timing
- CONTRAINDICATED_FOR **关系上没有属性**，reason 在 Contraindication 节点上
- SIMILAR_TO 是**单向**的（P001→P002 有，P002→P001 无），查询必须双向
- Alias 节点上也有 product_id，但以关系指向为准（两者当前一致）
"""
from __future__ import annotations

import logging
from typing import Any

from neo4j import Driver
from neo4j.exceptions import Neo4jError

from domain.ports import GraphQueryPort
from domain.value_objects import (
    ApprovalClaim,
    Contraindication,
    ContraindicationHit,
    EfficacyRef,
    ProductFacts,
    ProductRef,
    Severity,
    SimilarPair,
    Usage,
)
from infrastructure.exceptions import GraphQueryError

logger = logging.getLogger(__name__)


# ===================================================================
# Cypher
# ===================================================================
# ★ OPTIONAL MATCH 无匹配时会返回一行 null，所以 collect 后必须滤掉 null，
#   否则会得到 [None] 这种结果，转换时报 KeyError/TypeError。
_CYPHER_PRODUCT_FACTS = """
MATCH (p:Product {product_id: $pid})
OPTIONAL MATCH (p)-[:CLAIMS]->(e:Efficacy)
OPTIONAL MATCH (p)-[:HAS_INGREDIENT]->(i:Ingredient)
OPTIONAL MATCH (p)-[:DOSED_AS]->(u:Usage)
OPTIONAL MATCH (p)-[:SUITABLE_FOR]->(t:TargetGroup)
OPTIONAL MATCH (p)-[:CONTRAINDICATED_FOR]->(c:Contraindication)
OPTIONAL MATCH (a:Alias)-[:ALIAS_OF]->(p)
RETURN p AS product,
       [x IN collect(DISTINCT e) WHERE x IS NOT NULL] AS efficacies,
       [x IN collect(DISTINCT i) WHERE x IS NOT NULL] AS ingredients,
       [x IN collect(DISTINCT u) WHERE x IS NOT NULL] AS usages,
       [x IN collect(DISTINCT t) WHERE x IS NOT NULL] AS targets,
       [x IN collect(DISTINCT c) WHERE x IS NOT NULL] AS contras,
       [x IN collect(DISTINCT a) WHERE x IS NOT NULL] AS aliases
"""

_CYPHER_BY_EFFICACY = """
MATCH (p:Product {category: '保健食品'})-[:CLAIMS]->(e:Efficacy {efficacy_id: $eid})
RETURN p.product_id AS product_id, p.name AS name,
       p.category AS category, p.approval_no AS approval_no
ORDER BY p.product_id
"""

# 多跳：Need <-[:ADDRESSES]- Efficacy <-[:CLAIMS]- Product
_CYPHER_BY_NEED = """
MATCH (p:Product {category: '保健食品'})-[:CLAIMS]->(e:Efficacy)-[:ADDRESSES]->(n:Need)
WHERE n.need_id = $nid
RETURN DISTINCT p.product_id AS product_id, p.name AS name,
       p.category AS category, p.approval_no AS approval_no
ORDER BY p.product_id
"""

# 目录查询：**不限产品**，用来区分「目录里没有」与「本店没产品声称」。
#
# ★ 为什么需要独立的 Cypher：
#   `_CYPHER_BY_EFFICACY` 从 Product 出发，本店没产品声称时必然 0 行。
#   而「目录里有没有这个功能」是**法规问题**，与店铺库存无关。
#   混用会把「法规不允许」和「本店没有」说成同一件事（合规上是大错）。
_CYPHER_CATALOG = """
MATCH (e:Efficacy)
WHERE e.name CONTAINS $kw
OPTIONAL MATCH (p:Product {category: '保健食品'})-[:CLAIMS]->(e)
RETURN e.efficacy_id AS efficacy_id,
       e.name AS name,
       e.catalog_no AS catalog_no,
       e.catalog_version AS catalog_version,
       e.type = '营养素补充剂' AS is_nutrient,
       count(p) AS claiming_products
ORDER BY coalesce(e.catalog_no, 999), e.efficacy_id
"""

#: 目录全量（十几项）。供口语匹配做对撞，调用方缓存。
_CYPHER_CATALOG_ALL = """
MATCH (e:Efficacy)
RETURN e.efficacy_id AS efficacy_id,
       e.name AS name,
       e.catalog_no AS catalog_no,
       e.catalog_version AS catalog_version,
       e.type = '营养素补充剂' AS is_nutrient
ORDER BY coalesce(e.catalog_no, 999), e.efficacy_id
"""

_CYPHER_CHECK_CONTRA = """
MATCH (p:Product {product_id: $pid})-[:CONTRAINDICATED_FOR]->(c:Contraindication)
WHERE c.contra_id IN $cids
RETURN c.contra_id AS contra_id, c.name AS name,
       c.severity AS severity, c.reason AS reason
ORDER BY c.contra_id
"""

# 排除法：WHERE NOT EXISTS 做否定匹配，这是图谱相对向量检索的核心优势
_CYPHER_EXCLUDING = """
MATCH (p:Product {category: '保健食品'})
WHERE NOT EXISTS {
    MATCH (p)-[:CONTRAINDICATED_FOR]->(c:Contraindication)
    WHERE c.contra_id IN $cids
}
RETURN p.product_id AS product_id, p.name AS name,
       p.category AS category, p.approval_no AS approval_no
ORDER BY p.product_id
"""

_CYPHER_RESOLVE_ALIAS = """
MATCH (a:Alias {name: $alias})-[:ALIAS_OF]->(p:Product)
RETURN p.product_id AS product_id, p.name AS name,
       p.category AS category, p.approval_no AS approval_no
LIMIT 1
"""

# 同名别名可能指向多个产品时，再用产品全名兜底
_CYPHER_RESOLVE_BY_NAME = """
MATCH (p:Product {name: $alias})
RETURN p.product_id AS product_id, p.name AS name,
       p.category AS category, p.approval_no AS approval_no
LIMIT 1
"""

# ★ SIMILAR_TO 单向存储，必须双向查，否则 B→C 有边而 C→B 查不到
_CYPHER_SIMILAR = """
MATCH (p:Product {product_id: $pid})-[r:SIMILAR_TO]-(o:Product)
RETURN o.product_id AS other_product_id, o.name AS other_name,
       r.difference AS difference, r.lesson_point AS lesson_point
ORDER BY o.product_id
"""


# ===================================================================
# 适配器
# ===================================================================
class Neo4jGraphAdapter(GraphQueryPort):
    """Neo4j 图谱查询实现。"""

    def __init__(self, driver: Driver, database: str) -> None:
        """
        Args:
            driver: neo4j 驱动（由装配根创建并负责关闭）。
            database: 库名，走配置传入，不硬编码。
        """
        self._driver = driver
        self._db = database

    # ---------------- 内部 ----------------
    def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """执行查询并返回字典列表。

        Raises:
            GraphQueryError: 连接失败或 Cypher 出错，异常里带 context。
        """
        try:
            with self._driver.session(database=self._db) as session:
                return [dict(rec) for rec in session.run(cypher, **params)]
        except Neo4jError as exc:
            raise GraphQueryError(
                f"Cypher 执行失败: {exc.__class__.__name__}",
                product_id=str(params.get("pid", "")),
                database=self._db,
            ) from exc
        except Exception as exc:                  # noqa: BLE001 连接层异常
            raise GraphQueryError(
                f"图谱连接失败: {exc.__class__.__name__}: {exc}",
                database=self._db,
            ) from exc

    # ---------------- 端口实现 ----------------
    def get_product_facts(self, product_id: str) -> ProductFacts | None:
        rows = self._run(_CYPHER_PRODUCT_FACTS, pid=product_id)
        if not rows:
            return None

        row = rows[0]
        node = row["product"]
        facts = _to_product_facts(
            node=node,
            efficacies=row["efficacies"],
            ingredients=row["ingredients"],
            usages=row["usages"],
            targets=row["targets"],
            contras=row["contras"],
            aliases=row["aliases"],
        )
        logger.debug(
            "产品画像: product_id=%s, claims=%d, contra=%d, aliases=%d",
            product_id, len(facts.claims), len(facts.contraindications),
            len(facts.aliases),
        )
        return facts

    def find_products_by_efficacy(self, efficacy_id: str) -> list[ProductRef]:
        rows = self._run(_CYPHER_BY_EFFICACY, eid=efficacy_id)
        return [_to_product_ref(r) for r in rows]

    def find_efficacy_by_keyword(self, keyword: str) -> list[EfficacyRef]:
        """在批准功能目录里按关键词查（不要求有产品声称）。

        ★ 空关键词直接返回空：不要退回「返回全部目录」，
          调用方拿到的「全部」会被当成「全都匹配」，结论就全错了。
        """
        kw = (keyword or "").strip()
        if not kw:
            return []
        rows = self._run(_CYPHER_CATALOG, kw=kw)
        return [
            EfficacyRef(
                efficacy_id=r["efficacy_id"],
                name=r["name"],
                catalog_no=r.get("catalog_no"),
                catalog_version=r.get("catalog_version") or "2023",
                is_nutrient=bool(r.get("is_nutrient")),
            )
            for r in rows
        ]

    def list_efficacy_catalog(self) -> list[EfficacyRef]:
        """整个目录快照（十几项）。"""
        rows = self._run(_CYPHER_CATALOG_ALL)
        return [
            EfficacyRef(
                efficacy_id=r["efficacy_id"],
                name=r["name"],
                catalog_no=r.get("catalog_no"),
                catalog_version=r.get("catalog_version") or "2023",
                is_nutrient=bool(r.get("is_nutrient")),
            )
            for r in rows
        ]

    def find_products_by_need(self, need_id: str) -> list[ProductRef]:
        rows = self._run(_CYPHER_BY_NEED, nid=need_id)
        return [_to_product_ref(r) for r in rows]

    def check_contraindication(
        self, product_id: str, contra_ids: list[str]
    ) -> list[ContraindicationHit]:
        # 空输入直接返回，省一次查询
        if not contra_ids:
            return []
        rows = self._run(
            _CYPHER_CHECK_CONTRA, pid=product_id, cids=list(contra_ids))
        return [
            ContraindicationHit(
                product_id=product_id,
                contra_id=r["contra_id"],
                name=r["name"] or "",
                severity=_to_severity(r["severity"]),
                reason=r["reason"] or "",
            )
            for r in rows
        ]

    def find_products_excluding(self, contra_ids: list[str]) -> list[ProductRef]:
        # ★ 空列表是有意义的语义：不排除任何禁忌 → 返回全部保健食品。
        #   不能在这里提前 return []，否则「无禁忌约束」会被误解为「无产品」。
        rows = self._run(_CYPHER_EXCLUDING, cids=list(contra_ids))
        logger.debug(
            "排除法: 排除 %s → 剩余 %d 个保健食品", contra_ids, len(rows))
        return [_to_product_ref(r) for r in rows]

    def resolve_alias(self, alias: str) -> ProductRef | None:
        key = (alias or "").strip()
        if not key:
            return None

        rows = self._run(_CYPHER_RESOLVE_ALIAS, alias=key)
        if rows:
            return _to_product_ref(rows[0])

        # 别名表没有时，尝试按产品全名匹配（客户有时会说全称）
        rows = self._run(_CYPHER_RESOLVE_BY_NAME, alias=key)
        return _to_product_ref(rows[0]) if rows else None

    def find_similar_products(self, product_id: str) -> list[SimilarPair]:
        rows = self._run(_CYPHER_SIMILAR, pid=product_id)
        return [
            SimilarPair(
                product_id=product_id,
                other_product_id=r["other_product_id"],
                other_name=r["other_name"] or "",
                difference=r["difference"] or "",
                lesson_point=r["lesson_point"] or "",
            )
            for r in rows
        ]


# ===================================================================
# 转换函数：neo4j 类型 → 领域对象
# ===================================================================
# 集中放这里，一眼能看出「哪里有外部类型」，也方便审阅边界。
def _to_severity(raw: Any) -> Severity:
    """把图谱里的 severity 字符串转成枚举。

    ★ 未知取值一律降级为 **CRITICAL**，不是 MEDIUM。

    为什么是 fail-safe 而不是 fail-open：
        本项目三条硬约束的第一条是「输出护栏是必经节点」。
        severity 决定 P4 护栏是否拦截（只有 critical 无条件拦）。
        若未知值降成 medium，脏数据或 schema 变更会让本应拦截的
        禁忌悄悄放行——例如向在服抗凝药的客户推荐鱼油。
        宁可误拦（人工复核），不可漏放（安全事故）。

    数据层事实：47/47 条禁忌全带 severity，取值只有
    critical / high / medium 三种。出现未知值只可能是脏数据，
    此时报警并从严处理。
    """
    if raw is None:
        logger.warning("severity 缺失，按 critical（fail-safe）处理")
        return Severity.CRITICAL

    try:
        return Severity(str(raw).strip().lower())
    except ValueError:
        logger.warning(
            "未知 severity 取值: %r，按 critical（fail-safe）处理", raw)
        return Severity.CRITICAL


def _to_usage(usages: list[Any]) -> Usage | None:
    """用法用量。

    实物字段只有 dosage / daily_limit / note，
    第一个 Usage 节点即代表该产品的用法（数据层每个产品恰好 1 个）。
    """
    if not usages:
        return None
    node = usages[0]
    return Usage(
        dosage=_get(node, "dosage"),
        daily_limit=_get(node, "daily_limit"),
        note=_get(node, "note"),
    )


def _get(node: Any, key: str, default: str = "") -> str:
    """从 neo4j Node 安全取值。

    neo4j Node 支持 .get()，但测试里也可能传 dict，这里统一兼容。
    """
    try:
        value = node.get(key)
    except AttributeError:
        value = None
    return default if value is None else str(value)


def _to_product_facts(
    *,
    node: Any,
    efficacies: list[Any],
    ingredients: list[Any],
    usages: list[Any],
    targets: list[Any],
    contras: list[Any],
    aliases: list[Any],
) -> ProductFacts:
    """把查询结果组装成完整画像。"""
    return ProductFacts(
        product_id=_get(node, "product_id"),
        name=_get(node, "name"),
        category=_get(node, "category"),
        approval_no=_get(node, "approval_no"),
        dosage_form=_get(node, "dosage_form"),
        spec=_get(node, "spec"),
        retail_price=_get_int(node, "retail_price"),
        claims=tuple(
            ApprovalClaim(
                efficacy_id=_get(e, "efficacy_id"),
                text=_get(e, "name"),
                catalog_no=_get_int(e, "catalog_no"),
                catalog_version=_get(e, "catalog_version", "2023"),
            )
            for e in efficacies
        ),
        ingredients=tuple(_get(i, "name") for i in ingredients),
        usage=_to_usage(usages),
        suitable_for=tuple(_get(t, "name") for t in targets),
        contraindications=tuple(
            Contraindication(
                contra_id=_get(c, "contra_id"),
                name=_get(c, "name"),
                severity=_to_severity(_get(c, "severity")),
                reason=_get(c, "reason"),
            )
            for c in contras
        ),
        aliases=tuple(_get(a, "name") for a in aliases),
        data_origin=_get(node, "data_origin", "real"),
    )


def _get_int(node: Any, key: str) -> int | None:
    """取整数属性。缺失或非数字返回 None（不抛）。"""
    try:
        value = node.get(key)
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_product_ref(row: dict[str, Any]) -> ProductRef:
    """字典行 → 轻量产品引用。"""
    return ProductRef(
        product_id=str(row.get("product_id", "")),
        name=str(row.get("name", "")),
        category=str(row.get("category", "")),
        approval_no=str(row.get("approval_no") or ""),
    )
