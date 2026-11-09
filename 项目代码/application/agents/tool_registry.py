"""Agent 工具注册与执行（权限 + 超时 + 错误兜底）。

工具由装配层注入 handler；本模块不 import 基础设施，也不允许 Agent
直接写 SQL/任意调仓储。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any

from application.agents.types import AgentIdentity, ToolResult
from domain.exceptions import PermissionDeniedError

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any], AgentIdentity], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    required_feature: str
    read_only: bool
    handler: ToolHandler
    timeout_sec: float = 15.0


class ToolRegistry:
    """按白名单注册工具；执行前校验身份与超时。"""

    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def describe(self, identity: AgentIdentity) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for tool in self._tools.values():
            try:
                identity.require(tool.required_feature)
            except PermissionDeniedError:
                continue
            out.append({
                "name": tool.name,
                "description": tool.description,
                "read_only": tool.read_only,
            })
        return out

    def run(
        self,
        name: str,
        args: dict[str, Any],
        identity: AgentIdentity,
    ) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(tool=name, ok=False, error=f"未知工具: {name}")

        try:
            identity.require(spec.required_feature)
        except PermissionDeniedError as exc:
            return ToolResult(tool=name, ok=False, denied=True,
                              error=f"无权限: {exc}")

        started = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(spec.handler, args or {}, identity)
            payload = future.result(timeout=spec.timeout_sec)
            elapsed = (time.perf_counter() - started) * 1000
            data = payload.get("data") if isinstance(payload, dict) else {}
            text = payload.get("text", "") if isinstance(payload, dict) else ""
            return ToolResult(
                tool=name,
                ok=True,
                text=str(text or ""),
                data=dict(data or {}),
                elapsed_ms=round(elapsed, 1),
            )
        except FutureTimeout:
            elapsed = (time.perf_counter() - started) * 1000
            return ToolResult(
                tool=name, ok=False,
                error=f"工具超时（>{spec.timeout_sec:.0f}s）",
                elapsed_ms=round(elapsed, 1),
            )
        except PermissionDeniedError as exc:
            elapsed = (time.perf_counter() - started) * 1000
            return ToolResult(tool=name, ok=False, denied=True,
                              error=f"无权限: {exc}",
                              elapsed_ms=round(elapsed, 1))
        except Exception as exc:                  # noqa: BLE001 单工具失败不拖垮任务
            elapsed = (time.perf_counter() - started) * 1000
            logger.warning("Agent 工具 %s 失败: %s", name, exc)
            return ToolResult(
                tool=name, ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=round(elapsed, 1),
            )
        finally:
            # 超时后不等待工作线程；只读工具不会留下写副作用。
            executor.shutdown(wait=False)
