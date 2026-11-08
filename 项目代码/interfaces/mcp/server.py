"""MCP Server：把培训系统能力暴露给外部 AI 助手。

════════════════════════════════════════════════════════════════════
这是什么
════════════════════════════════════════════════════════════════════
`interfaces/mcp` 是本项目的**第二个入站适配器**，与 `interfaces/http` 并列：

    HTTP   → 给前端页面用（34→31 个端点）
    MCP    → 给 AI 助手用（5 个能力）

MCP（Model Context Protocol）是 Anthropic 提出的开放协议，
让 AI 助手能用统一格式发现和调用外部能力。
WorkBuddy / Claude Desktop / 企业 IM 都支持。

────────────────────────────────────────────────────────────────────
★ 为什么要做这个（背景）
────────────────────────────────────────────────────────────────────
本项目 31 个接口**全部只服务前端页面，而项目没有前端**。
实际状态是：接口都做好了，但只有开发者能在 Swagger 里 curl。

MCP 补的是「谁来调」这个入口：
    学员在手机上打开 AI 助手 → 问「孕妇能吃鱼油吗」→ 助手调 kb_search
    主管在微信里 → 问「我上周练得怎么样」 → 助手调 my_drill_reviews

────────────────────────────────────────────────────────────────────
★ 首版为什么只有 5 个能力（不把 31 个端点都转成工具）
────────────────────────────────────────────────────────────────────
工具越多，LLM 选中率越低。LLM 要在几十个工具里挑，很容易挑错。
实测经验：5~8 个工具时选中率较好，超过 15 个明显下降。

所以按「用户意图」而不是「HTTP 端点」来设计：
    · 31 个端点 → 裁剪合并成 5 个能力
    · 6 类高危操作**故意不暴露**（改权限、批量导入、审计日志、
      演练答题、导出文件、登录）——详见 `policy.py` 与方案 6.4

────────────────────────────────────────────────────────────────────
★ 身份怎么传（最关键的约束）
────────────────────────────────────────────────────────────────────
工具签名里**没有任何身份参数**。身份在连接建立时确定，存在会话对象里。

    stdio 传输：环境变量 MCP_TOKEN
    SSE/HTTP  ：Authorization: Bearer <token>

如果写成 `my_drill_reviews(employee_id)`，Agent 传别人工号就能读别人的
183 行演练评语 —— 这是 MCP 与传统 REST 最大的差别，也是本模块的核心设计。

────────────────────────────────────────────────────────────────────
★ mcp 2.x 的 API 变更（如果不是 2.x 会直接报错）
────────────────────────────────────────────────────────────────────
    FastMCP                    → MCPServer
    mcp.server.fastmcp         → mcp.server.mcpserver
    tool.inputSchema           → tool.input_schema
    Windows 必须装 pywin32

启动方式：
    python -m interfaces.mcp.server              # stdio（本地 demo）
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from config.assembly import Container, build_container
from config.settings import get_settings
from interfaces.mcp.identity import (
    McpAuthError,
    McpIdentity,
    resolve_identity,
    token_from_env,
)
from interfaces.mcp.logging_ import log_call
from interfaces.mcp.policy import ToolDeniedError
from interfaces.mcp.prompts import SCENES, expression_polish_template
from interfaces.mcp.resources import read_product
from interfaces.mcp.tools.drill import my_drill_reviews
from interfaces.mcp.tools.graph import graph_entity
from interfaces.mcp.tools.kb import kb_search

logger = logging.getLogger(__name__)

SERVER_NAME = "health-training-mcp"
SERVER_VERSION = "0.1.0"


class McpRuntime:
    """MCP 运行时：持有容器与会话身份。

    ★ 为什么要有这个类而不是全局变量：
      测试需要注入不同的容器与身份，全局状态没法测。
      这里做成实例，`build_server(runtime)` 接受注入。
    """

    def __init__(self, container: Container, identity: McpIdentity) -> None:
        self.container = container
        self.identity = identity

    @classmethod
    def from_env(cls, container: Container | None = None) -> McpRuntime:
        """按环境变量装配（stdio 启动时用）。"""
        if container is None:
            container = build_container(get_settings())
        identity = resolve_identity(container, token=token_from_env())
        return cls(container, identity)


def build_server(runtime: McpRuntime):
    """构建 MCP Server 并注册 5 个能力。

    Args:
        runtime: 运行时（容器 + 会话身份）。

    Returns:
        `MCPServer` 实例。
    """
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name=SERVER_NAME, version=SERVER_VERSION)

    # ==================================================================
    # Tool 1: kb_search —— 知识检索
    # ==================================================================
    #
    # ★ 描述必须写「什么时候用」，不是「它做什么」。
    #   LLM 靠 description 决定选哪个工具，写「执行知识库检索」等于没写。
    #
    # ★ 显式指定 name：默认会用函数名（kb_search_tool），
    #   但工具名是给 LLM 看的标识，不该带实现细节后缀。
    @server.tool(name="kb_search")
    def kb_search_tool(query: str, top_k: int = 5) -> str:
        """查询产品知识库，返回可直接对客户说的一致性话术。

        当用户问产品成分、适用人群、禁忌、服用方式、功效宣称时使用。
        返回内容已过合规护栏，可以直接使用。

        不要用它查员工个人数据（演练评语请用 my_drill_reviews）。
        """
        t0 = time.perf_counter()
        try:
            out = kb_search(
                runtime.container, runtime.identity,
                query=query, top_k=top_k,
            )
            _log(runtime, "tool", "kb_search", t0, True,
                 {"query": query, "top_k": top_k}, out)
            return out
        except ToolDeniedError as exc:
            _log(runtime, "tool", "kb_search", t0, False,
                 {"query": query}, "", str(exc))
            return _error("permission_denied", str(exc))
        except Exception as exc:                       # noqa: BLE001
            logger.exception("kb_search 失败")
            _log(runtime, "tool", "kb_search", t0, False,
                 {"query": query}, "", f"{type(exc).__name__}: {exc}")
            return _error("internal_error", f"{type(exc).__name__}: {exc}")

    # ==================================================================
    # Tool 2: graph_entity —— 图谱实体
    # ==================================================================
    @server.tool(name="graph_entity")
    def graph_entity_tool(
        name: str,
        with_relations: list[str] | None = None,
    ) -> str:
        """查询产品实体及其关系（别名、批准功能、适用人群、禁忌人群）。

        当需要判断「某类人能不能用某产品」时优先用这个——
        它返回结构化的禁忌关系（含严重度），这是文本检索给不出的。

        输入需为库中的产品名或别名，如「鱼油」「康源牌鱼油软胶囊」。
        """
        t0 = time.perf_counter()
        try:
            out = graph_entity(
                runtime.container, runtime.identity,
                name=name, with_relations=with_relations,
            )
            _log(runtime, "tool", "graph_entity", t0, True,
                 {"name": name, "with_relations": with_relations}, out)
            return out
        except ToolDeniedError as exc:
            _log(runtime, "tool", "graph_entity", t0, False,
                 {"name": name}, "", str(exc))
            return _error("permission_denied", str(exc))
        except Exception as exc:                       # noqa: BLE001
            logger.exception("graph_entity 失败")
            _log(runtime, "tool", "graph_entity", t0, False,
                 {"name": name}, "", f"{type(exc).__name__}: {exc}")
            return _error("internal_error", f"{type(exc).__name__}: {exc}")

    # ==================================================================
    # Tool 3: my_drill_reviews —— 演练评语（★ 无 employee_id）
    # ==================================================================
    @server.tool(name="my_drill_reviews")
    def my_drill_reviews_tool(days: int = 7) -> str:
        """查询当前用户自己的演练评语与风险点。

        当用户说「我上周练得怎么样」「我错在哪」「我练了几次」时使用。
        只能查本人，无法查询他人——这是设计约束，不是权限限制。

        返回要素命中数与风险点计数，不给出分数或优劣判定。
        """
        t0 = time.perf_counter()
        try:
            out = my_drill_reviews(
                runtime.container, runtime.identity, days=days)
            _log(runtime, "tool", "my_drill_reviews", t0, True,
                 {"days": days}, out)
            return out
        except ToolDeniedError as exc:
            _log(runtime, "tool", "my_drill_reviews", t0, False,
                 {"days": days}, "", str(exc))
            return _error("permission_denied", str(exc))
        except Exception as exc:                       # noqa: BLE001
            logger.exception("my_drill_reviews 失败")
            _log(runtime, "tool", "my_drill_reviews", t0, False,
                 {"days": days}, "", f"{type(exc).__name__}: {exc}")
            return _error("internal_error", f"{type(exc).__name__}: {exc}")

    # ==================================================================
    # Resource: product://{product_id}
    # ==================================================================
    #
    # ★ 带参数的 URI 注册为「模板资源」，在 list_resource_templates() 里，
    #   不在 list_resources() —— 2.x 的行为，写测试时要注意。
    @server.resource("product://{product_id}")
    def product_resource(product_id: str) -> str:
        """产品档案（名称、批准功能原文、成分、适用人群、禁忌）。"""
        t0 = time.perf_counter()
        try:
            out = read_product(runtime.container, runtime.identity, product_id)
            _log(runtime, "resource", f"product://{product_id}", t0, True,
                 {"product_id": product_id}, out)
            return out
        except ToolDeniedError as exc:
            _log(runtime, "resource", f"product://{product_id}", t0, False,
                 {"product_id": product_id}, "", str(exc))
            return _error("permission_denied", str(exc))
        except Exception as exc:                       # noqa: BLE001
            logger.exception("product resource 失败")
            _log(runtime, "resource", f"product://{product_id}", t0, False,
                 {"product_id": product_id}, "", f"{type(exc).__name__}: {exc}")
            return _error("internal_error", f"{type(exc).__name__}: {exc}")

    # ==================================================================
    # Prompt: expression_polish
    # ==================================================================
    @server.prompt()
    def expression_polish(text: str, scene: str = "daily_summary") -> str:
        """润色表达。

        当用户要求改写月度报告、日报、演练评语，或说「帮我把这段话说得专业点」时使用。

        硬约束（写死在模板里）：
        - 不改数字：日期、次数、比例原样保留
        - 不加事实：原文没说的不补
        - 不加强语气：「建议咨询医生」不得改成「必须就医」
        """
        # ★ Prompt 不经过 policy（模板本身不含敏感数据），
        #   但要记日志——它可能被用来改写内部内容。
        require_scene = scene if scene in SCENES else "daily_summary"
        out = expression_polish_template(text, require_scene)
        log_call(
            kind="prompt", name="expression_polish",
            employee_no=runtime.identity.employee_no,
            role=runtime.identity.role.label,
            ok=True, elapsed_ms=0.0,
            args={"text": text, "scene": scene},
            result_bytes=len(out.encode("utf-8")),
        )
        return out

    return server


def _log(
    runtime: McpRuntime,
    kind: str,
    name: str,
    t0: float,
    ok: bool,
    args: dict[str, Any],
    result: str,
    error: str = "",
) -> None:
    """统一记录调用日志。"""
    log_call(
        kind=kind, name=name,
        employee_no=runtime.identity.employee_no,
        role=runtime.identity.role.label,
        ok=ok,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        args=args,
        result_bytes=len(result.encode("utf-8")) if result else 0,
        error=error,
    )


def _error(code: str, message: str) -> str:
    """统一的错误返回结构。

    ★ 为什么返回 JSON 而不是抛异常：
      MCP 的工具抛异常会让客户端看到一段 Python traceback，
      AI 助手无法理解也不会重试。返回结构化错误它能自己决定怎么办。
    """
    return json.dumps(
        {"error": code, "message": message}, ensure_ascii=False)


# ======================================================================
# 启动入口
# ======================================================================
def main() -> None:
    """stdio 启动（本地 demo / 宿主 spawn 子进程）。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("启动 %s v%s", SERVER_NAME, SERVER_VERSION)

    try:
        runtime = McpRuntime.from_env()
    except McpAuthError as exc:
        logger.error("身份解析失败，无法启动: %s", exc)
        raise SystemExit(1) from exc

    logger.info(
        "会话身份: %s (%s) | 来源=%s",
        runtime.identity.employee_no,
        runtime.identity.role.label,
        runtime.identity.source,
    )
    logger.info(
        "可检索级别: %s",
        [str(x) for x in runtime.identity.retrieval_levels],
    )

    server = build_server(runtime)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
