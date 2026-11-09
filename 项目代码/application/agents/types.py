"""Agent 层的轻量值对象（不依赖接口层/基础设施）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.value_objects.role import Role
from domain.value_objects.search_result import PermissionLevel


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    """当前任务的身份。身份只来自 token，不能由工具参数声明。"""

    employee_no: str
    role: Role

    @property
    def is_manager(self) -> bool:
        return self.role.is_manager

    @property
    def retrieval_levels(self) -> tuple[PermissionLevel, ...]:
        return self.role.retrieval_levels

    def require(self, feature: str) -> None:
        self.role.require(feature)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """一次工具调用的结果。"""

    tool: str
    ok: bool
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    denied: bool = False
    elapsed_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class AgentStep:
    """对外展示的一步执行记录。"""

    tool: str
    ok: bool
    elapsed_ms: float
    summary: str
    denied: bool = False


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    """一次 Agent 任务的最终结果。"""

    answer: str
    refused: bool = False
    refusal_reason: str = ""
    plan: tuple[dict[str, Any], ...] = ()
    steps: tuple[AgentStep, ...] = ()
    citations: tuple[str, ...] = ()
