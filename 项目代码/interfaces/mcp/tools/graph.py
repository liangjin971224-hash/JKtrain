"""图谱查询工具：`graph_entity`。

════════════════════════════════════════════════════════════════════
为什么单独做一个图谱工具（而不是并入 kb_search）
════════════════════════════════════════════════════════════════════
图谱能给出**排除法**，向量给不出。举例：

    问：血脂高的人能吃什么？
    向量：返回一堆「血脂」相关的文档段落，可能包含不适宜人群没提全的旧内容
    图谱：能明确查出「哪些产品批准了 EFF_19（维持血脂）」
          + 「这些产品各自排除哪些人群」

后者是**集合运算**，前者是**文本相似**。门店场景里「谁能吃、谁不能吃」
必须靠集合运算，不能靠「看起来像」。

────────────────────────────────────────────────────────────────────
为什么只暴露 4 个关系
────────────────────────────────────────────────────────────────────
GraphQueryPort 有 10 个方法，全暴露会让 LLM 选不准（工具越多，选中率越低）。
这里挑的是「用户会直接问的 4 类」：

    ALIAS_OF            「鱼油是啥」→ 别名归一
    CONTRAINDICATED_FOR 「谁能吃」← 最核心
    CLAIMS              「有什么功能」→ 批准功能原文
    SUITABLE_FOR        「适合谁」

其余（目录查询、相似产品、批量排除）留给 S3 的进阶工具。
"""
from __future__ import annotations

import json
import logging

from config.assembly import Container
from domain.value_objects.product_facts import ProductFacts
from interfaces.mcp.identity import McpIdentity
from interfaces.mcp.policy import require_tool

logger = logging.getLogger(__name__)

#: 允许查询的关系类型（白名单）。
#: ★ 未登记的关系一律忽略——防止 LLM 传任意字符串过来。
ALLOWED_RELATIONS = frozenset({
    "ALIAS_OF",
    "CONTRAINDICATED_FOR",
    "CLAIMS",
    "SUITABLE_FOR",
})


def _facts_to_dict(f: ProductFacts) -> dict:
    """把产品事实转成对 AI 友好的结构。

    ★ 批准功能原文单独放在 `claims_text`，不加工、不改写。
      红线 G1：功能声称只能引用批准原文，不可自由填写。
      所以这里原样输出，让下游也没有改写空间。
    """
    return {
        "product_id": f.product_id,
        "name": f.name,
        "category": f.category,
        "approval_no": f.approval_no,
        "aliases": list(f.aliases),
        # 批准功能：原文照抄
        "claims_text": [c.text for c in f.claims],
        "suitable_for": list(f.suitable_for),
        # 不适宜人群：带严重度，让 AI 能判「严重」还是「一般」
        "contraindications": [
            {
                "name": c.name,
                "severity": str(c.severity),
                "reason": c.reason,
                # ★ 是否「必须拦」（critical 级别）
                "must_block": bool(c.severity.must_block),
            }
            for c in f.contraindications
        ],
        "usage": (
            {
                "dosage": f.usage.dosage,
                "daily_limit": f.usage.daily_limit,
            }
            if f.usage
            else None
        ),
    }


def graph_entity(
    container: Container,
    identity: McpIdentity,
    *,
    name: str,
    with_relations: list[str] | None = None,
) -> str:
    """按名称查产品实体与关系。

    Args:
        container: 容器。
        identity: 会话身份。
        name: 产品名或别名（如「鱼油」）。
        with_relations: 需要返回的关系类型，None 表示全部允许的。

    Returns:
        JSON 字符串：{found, product, relations, note}
    """
    require_tool(identity, "graph_entity")

    graph = container.graph
    if graph is None:
        return json.dumps(
            {"found": False, "error": "图谱端口未就绪"}, ensure_ascii=False)

    wanted = (
        set(with_relations or ALLOWED_RELATIONS) & ALLOWED_RELATIONS
    ) or set(ALLOWED_RELATIONS)

    # ---- 1. 先按别名/名称归一 ----
    ref = graph.resolve_alias(name)
    if ref is None:
        # 别名没命中，尝试当成 product_id 直接取
        facts = graph.get_product_facts(name)
        if facts is None:
            return json.dumps(
                {
                    "found": False,
                    "query": name,
                    "note": "图谱中未找到该产品。可能名称/别名不在库中，"
                            "请改用 products 的实际名称再试。",
                },
                ensure_ascii=False,
            )
        product_id = facts.product_id
    else:
        product_id = ref.product_id

    facts = graph.get_product_facts(product_id)
    if facts is None:
        return json.dumps(
            {"found": False, "query": name, "product_id": product_id,
             "note": "产品存在但事实画像读取失败"},
            ensure_ascii=False,
        )

    # ---- 2. 按需装配关系 ----
    relations: dict[str, object] = {}
    if "ALIAS_OF" in wanted:
        relations["ALIAS_OF"] = {"aliases": list(facts.aliases)}
    if "CLAIMS" in wanted:
        relations["CLAIMS"] = [
            {"text": c.text, "catalog_no": c.catalog_no,
             "catalog_version": c.catalog_version}
            for c in facts.claims
        ]
    if "SUITABLE_FOR" in wanted:
        relations["SUITABLE_FOR"] = list(facts.suitable_for)
    if "CONTRAINDICATED_FOR" in wanted:
        relations["CONTRAINDICATED_FOR"] = [
            {"name": c.name, "severity": str(c.severity),
             "reason": c.reason, "must_block": bool(c.severity.must_block)}
            for c in facts.contraindications
        ]

    payload = {
        "found": True,
        "query": name,
        "product": _facts_to_dict(facts),
        "relations": relations,
        "note": "批准功能（claims_text）为法规原文，引用时不得改写。"
                "contraindications 中 must_block=true 表示必须拒绝该人群。",
    }
    return json.dumps(payload, ensure_ascii=False)
