"""知识检索工具：`kb_search`。

════════════════════════════════════════════════════════════════════
为什么复用的是 AnswerQuestionUseCase 而不是 RetrieveUseCase
════════════════════════════════════════════════════════════════════
RetrieveUseCase 只返回「检索到什么」，AnswerQuestionUseCase 返回的是
**可以直接给门店员工看的话**——已经过了：
    · 三路 RRF 融合
    · 意图分类（决定权重）
    · 权限过滤（拒答红线）
    · 输出护栏（合规四类放行）
    · 输出清洗（剥 Markdown / 砍回吐）

MCP 的调用方是 AI 助手，它要的是**能回答用户的内容**，
不是「一堆 chunk 让它自己组织」——后者等于让 AI 去猜合规口径，
而这个项目的全部价值就在「口径是确定的」。

★ 所以 kb_search 返回的是「答案 + 出处 + 是否拒答」，
  不是「文档片段列表」。
"""
from __future__ import annotations

import json
import logging
import time

from application.use_cases.answer_question import AnswerQuestionUseCase
from config.assembly import Container
from interfaces.mcp.identity import McpIdentity
from interfaces.mcp.policy import clamp_top_k, require_tool

logger = logging.getLogger(__name__)


def build_answer_use_case(container: Container) -> AnswerQuestionUseCase:
    """按 HTTP 侧同样的方式组装用例。

    ★ 与 `interfaces/http/routers/ask.py:69` 保持一致的参数，
      避免「新入口漏透传可选参数」（历史上踩过 4 次）。
    """
    settings = getattr(container, "settings", None)
    return AnswerQuestionUseCase(
        graph=container.graph,
        vector=container.vector,
        keyword=container.keyword,
        llm=container.llm,
        repository=container.repository,
        reranker=container.rerank,
        rerank_candidates=getattr(settings, "rerank_candidates", 30),
        rerank_top_k=getattr(settings, "rerank_top_k", 5),
    )


def kb_search(
    container: Container,
    identity: McpIdentity,
    *,
    query: str,
    top_k: int = 5,
) -> str:
    """查询产品知识库，返回可直接使用的话术答案。

    Args:
        container: 容器。
        identity: 会话身份（决定可检索的知识级别）。
        query: 用户原话。
        top_k: 每路召回条数（1–10）。

    Returns:
        JSON 字符串：{answer, refused, refusal_reason, cited, intent}
    """
    require_tool(identity, "kb_search")

    k = clamp_top_k(top_k)
    uc = build_answer_use_case(container)

    # ★ 必须传 vector_collection，否则向量路会降级。
    #   实测后果（2026-11-06）：不传 → 日志出现
    #     "召回路 vector 失败，降级继续: ValueError: 未指定 vector_collection"
    #   → 三路变两路（graph + keyword），召回质量下降。
    #   这是本项目「新入口漏透传可选参数」的第 5 次复发
    #   （前 4 次：fact_hints / product_ids / history / 流式 done.meta）。
    settings = getattr(container, "settings", None)
    collection = getattr(settings, "coll_training", "")

    t0 = time.perf_counter()
    result = uc.execute(
        query,
        top_k=k,
        vector_collection=collection,
        # ★ 权限从会话推，绝不从参数读。
        #   这条元组最终拼进 Milvus 的 filter_expr —— 权限在检索阶段生效，
        #   不是「先全查再筛」（FR-050 明确要求）。
        allowed_permissions=identity.retrieval_levels,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    payload = {
        "answer": result.answer,
        "refused": result.refused,
        "refusal_reason": result.refusal_reason,
        "cited_chunk_ids": list(result.cited_chunk_ids),
        "product_ids": list(result.product_ids),
        "intent": (result.meta or {}).get("intent", ""),
        # ★ 把可检索级别回给调用方，便于排查「为什么这条查不到」
        "retrieval_levels": [str(x) for x in identity.retrieval_levels],
        "elapsed_ms": round(elapsed, 1),
    }
    if result.refused:
        logger.info(
            "kb_search 拒答 | 身份=%s | 原因=%s", identity.employee_no,
            result.refusal_reason or "未命中",
        )

    return json.dumps(payload, ensure_ascii=False)
