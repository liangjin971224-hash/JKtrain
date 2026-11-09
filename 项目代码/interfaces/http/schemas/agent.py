"""Agent 接口的请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    """一次 Agent 任务。"""

    question: str = Field(..., min_length=1, max_length=500)


class AgentPlanView(BaseModel):
    """规划出的一步。"""

    tool: str
    reason: str = ""


class AgentStepView(BaseModel):
    """实际执行的一步。"""

    tool: str
    ok: bool
    elapsed_ms: float = 0.0
    summary: str = ""
    denied: bool = False


class AgentChatResponse(BaseModel):
    """Agent 任务的最终结果。"""

    answer: str
    refused: bool = False
    refusal_reason: str = ""
    plan: list[AgentPlanView] = Field(default_factory=list)
    steps: list[AgentStepView] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
