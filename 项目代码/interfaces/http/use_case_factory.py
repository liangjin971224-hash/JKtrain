"""HTTP 入口共用的问答用例装配。

ask 与 agent 两个路由都必须用同一套参数；放在这里避免「新入口漏透传」。
"""

from __future__ import annotations

from fastapi import HTTPException

from application.use_cases.answer_question import AnswerQuestionUseCase
from config.assembly import Container


def build_answer_use_case(container: Container) -> AnswerQuestionUseCase:
    """按当前容器装配问答用例（轻量对象，每请求新建即可）。"""
    if container.llm is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 未配置，问答功能不可用。请在 .env 填 QWEN_API_KEY 或 DEEPSEEK_API_KEY",
        )
    return AnswerQuestionUseCase(
        graph=container.graph,
        vector=container.vector,
        keyword=container.keyword,
        sparse=container.sparse,
        sparse_collections=frozenset(
            c.strip() for c in container.settings.sparse_recall_collections.split(",")
            if c.strip()),
        min_sparse_score=container.settings.min_sparse_score,
        path_head_collections=frozenset(
            c.strip() for c in container.settings.path_head_collections.split(",")
            if c.strip()),
        context_docs=container.settings.context_docs,
        prompt_variant=container.settings.prompt_variant,
        llm=container.llm,
        llm_max_tokens=container.settings.llm_max_tokens,
        answer_repair_enabled=container.settings.answer_repair_enabled,
        repository=container.repository,
        rerank=container.rerank,
        rerank_candidate_k=container.settings.rerank_candidate_k,
    )
