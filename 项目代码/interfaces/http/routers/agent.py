"""培训助教 Agent 路由（只读版）。

    POST /api/agent/chat

身份从 token 取；工具权限在 ToolRegistry 内校验。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from application.agents.tools import build_agent_registry
from application.agents.training_agent import TrainingAgent
from application.agents.types import AgentIdentity
from application.use_cases.batch_report import BatchReportUseCase
from application.use_cases.build_paper import BuildPaperUseCase
from config.assembly import Container
from interfaces.http.auth_deps import Identity, current_identity
from interfaces.http.deps import get_container
from interfaces.http.routers.exam import load_question_bank
from interfaces.http.schemas.agent import (
    AgentChatRequest,
    AgentChatResponse,
    AgentPlanView,
    AgentStepView,
)
from interfaces.http.use_case_factory import build_answer_use_case

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Agent"])


def _build_agent(container: Container, identity: Identity) -> TrainingAgent:
    """按容器装配只读 Agent（测试可 monkeypatch 此函数）。"""
    use_case = None
    if container.llm is not None:
        try:
            use_case = build_answer_use_case(container)
        except Exception as exc:          # noqa: BLE001 KB 不可用不阻塞其他工具
            logger.warning("Agent 的 KB 用例装配失败: %s", exc)
    exam_factory = None
    if (container.repository is not None
            and getattr(container, "milvus_client", None) is not None):
        exam_factory = lambda: BuildPaperUseCase(
            repository=container.repository,
            bank_loader=lambda stage: load_question_bank(container, stage),
        )

    report_factory = None
    if (container.repository is not None
            and getattr(container, "task_queue", None) is not None):
        report_factory = lambda: BatchReportUseCase(
            repository=container.repository,
            queue=container.task_queue,
            max_attempts=container.settings.queue_max_attempts,
        )

    registry = build_agent_registry(
        answer_use_case_factory=(lambda: use_case) if use_case else None,
        graph=container.graph,
        repository=container.repository,
        case_catalog=container.case_catalog,
        exam_use_case_factory=exam_factory,
        exam_bank_loader=(
            (lambda stage: load_question_bank(container, stage))
            if exam_factory is not None else None
        ),
        report_use_case_factory=report_factory,
    )
    return TrainingAgent(registry)


@router.post("/agent/chat", response_model=AgentChatResponse,
             summary="培训助教 Agent（只读）")
def agent_chat(
    req: AgentChatRequest,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> AgentChatResponse:
    """执行一次只读 Agent 任务。"""
    if not getattr(container.settings, "agent_enabled", False):
        raise HTTPException(404, detail="Agent 未启用（AGENT_ENABLED=false）")

    agent = _build_agent(container, identity)
    try:
        result = agent.run(
            req.question,
            AgentIdentity(employee_no=identity.employee_no, role=identity.role),
        )
    except Exception as exc:              # noqa: BLE001 接口层兜底
        logger.error("Agent 执行失败: %s", exc, exc_info=True)
        raise HTTPException(500, detail="Agent 执行失败") from exc

    logger.info(
        "Agent task: employee=%s tools=%s refused=%s",
        identity.employee_no, [s.tool for s in result.steps], result.refused)
    return AgentChatResponse(
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        plan=[
            AgentPlanView(tool=str(p.get("tool", "")),
                          reason=str(p.get("reason", "")))
            for p in result.plan
        ],
        steps=[
            AgentStepView(
                tool=s.tool,
                ok=s.ok,
                elapsed_ms=s.elapsed_ms,
                summary=s.summary,
                denied=s.denied,
            )
            for s in result.steps
        ],
        citations=list(result.citations),
    )
