"""MCP 资源：产品档案。

════════════════════════════════════════════════════════════════════
Resource 与 Tool 的区别（面试常问）
════════════════════════════════════════════════════════════════════
    Tool      AI 主动「调用」，有参数、有副作用可能
    Resource  AI 按 URI「读取」，只读、可被客户端列为上下文

`product://P001` 适合做 Resource 而不做 Tool，因为：
    · 它是**纯读**，没有副作用
    · 客户端可以把它挂进上下文窗口（不用每次「调用」）
    · 多个产品可以并列挂载（product://P001, product://P002）

────────────────────────────────────────────────────────────────────
★ 2.x 的一个坑
────────────────────────────────────────────────────────────────────
带参数的 URI（`product://{product_id}`）注册为**模板资源**，
它出现在 `list_resource_templates()`，**不在 `list_resources()`**。
写测试时按「资源数」断言会误判为 0。
"""
from __future__ import annotations

import json
import logging

from config.assembly import Container
from interfaces.mcp.identity import McpIdentity
from interfaces.mcp.policy import require_resource

logger = logging.getLogger(__name__)


def read_product(container: Container, identity: McpIdentity, product_id: str) -> str:
    """读取产品档案（JSON 字符串）。

    Args:
        container: 容器。
        identity: 会话身份。
        product_id: 产品 ID，如 P001。

    Returns:
        产品档案 JSON。
    """
    require_resource(identity, "product")

    graph = container.graph
    if graph is None:
        return json.dumps({"error": "图谱端口未就绪"}, ensure_ascii=False)

    facts = graph.get_product_facts(product_id)
    if facts is None:
        return json.dumps(
            {"error": f"产品 {product_id} 不存在", "product_id": product_id},
            ensure_ascii=False,
        )

    payload = {
        "product_id": facts.product_id,
        "name": facts.name,
        "category": facts.category,
        "approval_no": facts.approval_no,
        "aliases": list(facts.aliases),
        # ★ 批准功能原文，引用时不得改写（红线 G1）
        "claims": [
            {"text": c.text, "catalog_no": c.catalog_no,
             "catalog_version": c.catalog_version}
            for c in facts.claims
        ],
        "ingredients": list(facts.ingredients),
        "suitable_for": list(facts.suitable_for),
        "contraindications": [
            {"name": c.name, "severity": str(c.severity),
             "reason": c.reason, "must_block": bool(c.severity.must_block)}
            for c in facts.contraindications
        ],
        "usage": (
            {"dosage": facts.usage.dosage, "daily_limit": facts.usage.daily_limit}
            if facts.usage else None
        ),
        "compliance_note": facts.compliance_note,
        "_meta": {
            "claims_are_regulatory_text": True,
            "note": "批准功能为法规原文，不得改写、不得扩展。",
        },
    }
    return json.dumps(payload, ensure_ascii=False)
