"""MCP 工具级授权。

════════════════════════════════════════════════════════════════════
三层权限，MCP 侧要分别落地
════════════════════════════════════════════════════════════════════
    FR-048 功能权限  能不能用这个工具      → 本模块 `require_tool`
    FR-049 数据权限  能看到谁的数据        → 会话绑定（工具签名里没有 employee_id）
    FR-050 检索权限  能检索哪个级别的知识  → Role.retrieval_levels → Milvus filter

────────────────────────────────────────────────────────────────────
为什么不直接用 Role.require(feature)
────────────────────────────────────────────────────────────────────
`_FEATURES` 白名单里登记的是业务功能（ask / drill / exam_build…），
不是 MCP 工具名。硬套会让两套命名纠缠在一起。

这里单独维护一张「工具 → 所需功能权限」的映射表，
好处是：新增 MCP 工具时必须显式登记，**漏登记就等于无权限**（fail-safe）。
"""
from __future__ import annotations

import logging

from domain.exceptions import PermissionDeniedError
from interfaces.mcp.identity import McpIdentity

logger = logging.getLogger(__name__)


class ToolDeniedError(Exception):
    """工具调用被拒。

    与 `McpAuthError` 分开：
      · McpAuthError   身份本身有问题（令牌无效）→ 整个会话不可用
      · ToolDeniedError 身份有效但无权用这个工具 → 只拒这一次调用
    """


#: 工具 → 所需功能权限（`Role._FEATURES` 里的标识）。
#:
#: ★ 白名单：没登记的工具一律拒绝。
#:   这样新增工具时忘记登记的结果是「用不了」（可发现），
#:   而不是「所有人能用」（事故）。
#:
#: 注：`build_paper` 首版不暴露（有副作用，写 MySQL），
#:     S3 若加，这里登记 `exam_build` 即可——Role 里已经定义了这个功能。
TOOL_FEATURES: dict[str, str] = {
    "kb_search": "ask",
    "graph_entity": "ask",
    "my_drill_reviews": "drill",
}

#: 资源 → 所需功能权限。
RESOURCE_FEATURES: dict[str, str] = {
    "product": "ask",
}

#: 提示 → 所需功能权限。提示不改数据，最低权限即可。
PROMPT_FEATURES: dict[str, str] = {
    "expression_polish": "ask",
}


def require_tool(identity: McpIdentity, tool_name: str) -> None:
    """检查当前身份能否调用某工具。

    Args:
        identity: 会话身份。
        tool_name: 工具名。

    Raises:
        ToolDeniedError: 工具未登记，或身份无对应功能权限。
    """
    feature = TOOL_FEATURES.get(tool_name)
    if feature is None:
        # ★ 未登记 = 拒绝，不是放行。
        #   「忘了登记」应该是「新工具用不了」，不应该是「所有人能用」。
        logger.error(
            "MCP 工具 %s 未在 TOOL_FEATURES 登记，默认拒绝 | 身份=%s",
            tool_name, identity.employee_no,
        )
        raise ToolDeniedError(f"工具 {tool_name} 未登记权限要求，拒绝调用")

    try:
        identity.role.require(feature)
    except PermissionDeniedError as exc:
        logger.info(
            "MCP 越权被拒: 工具=%s 身份=%s(%s) 需要=%s",
            tool_name, identity.employee_no, identity.role.label, feature,
        )
        raise ToolDeniedError(
            f"{identity.role.label}无权使用「{tool_name}」（需要 {feature} 权限）"
        ) from exc


def require_resource(identity: McpIdentity, resource_kind: str) -> None:
    """检查能否读某类资源。"""
    feature = RESOURCE_FEATURES.get(resource_kind)
    if feature is None:
        raise ToolDeniedError(f"资源 {resource_kind} 未登记权限要求，拒绝访问")
    try:
        identity.role.require(feature)
    except PermissionDeniedError as exc:
        raise ToolDeniedError(f"无权访问资源 {resource_kind}") from exc


def require_prompt(identity: McpIdentity, prompt_name: str) -> None:
    """检查能否用某个提示模板。"""
    feature = PROMPT_FEATURES.get(prompt_name)
    if feature is None:
        raise ToolDeniedError(f"提示 {prompt_name} 未登记权限要求，拒绝使用")
    try:
        identity.role.require(feature)
    except PermissionDeniedError as exc:
        raise ToolDeniedError(f"无权使用提示 {prompt_name}") from exc


def clamp_top_k(value: int, *, low: int = 1, high: int = 10) -> int:
    """把 top_k 夹到合法区间。

    ★ 为什么不让 LLM 决定：
      LLM 可能传 100（想「查全」），单次检索就会拖慢整个链路。
      参数上限是服务端约束，不该交给调用方。
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, n))
