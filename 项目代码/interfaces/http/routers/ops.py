"""运维/调试路由（D2/D3）。

    POST /api/ops/retrieval   检索调试（只读，不调 LLM、不写库）
    GET  /api/ops/metrics     运行指标（LLM 主备、请求量、延迟）

两个接口都要求主管权限（复用 `miss_log_view`，与未命中清单一致）。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from application.services.ops_metrics import OpsMetrics
from application.use_cases.retrieve import RetrieveUseCase
from config.assembly import Container
from domain.services.intent_classifier import IntentClassifier
from interfaces.http.auth_deps import (
    Identity,
    current_identity,
    require_feature,
)
from interfaces.http.deps import get_container
from interfaces.http.schemas.ops import (
    FusedHitOut,
    RetrievalDebugRequest,
    RetrievalDebugResponse,
    RetrievalHit,
)

router = APIRouter(prefix="/api", tags=["运维"])


def _retriever(container: Container) -> RetrieveUseCase:
    """与 ask.py 同参数的检索用例（只读，不接 LLM）。"""
    s = container.settings
    return RetrieveUseCase(
        graph=container.graph,
        vector=container.vector,
        keyword=container.keyword,
        sparse=getattr(container, "sparse", None),
        sparse_collections=frozenset(
            c.strip() for c in s.sparse_recall_collections.split(",")
            if c.strip()),
        min_sparse_score=s.min_sparse_score,
        path_head_collections=frozenset(
            c.strip() for c in s.path_head_collections.split(",")
            if c.strip()),
        rerank=container.rerank,
        rerank_candidate_k=s.rerank_candidate_k,
    )


def _allowed_collections(container: Container) -> set[str]:
    s = container.settings
    return {s.coll_training, s.coll_case, s.coll_question, s.coll_self_test}


def _hits(hits) -> list[RetrievalHit]:
    out: list[RetrievalHit] = []
    for h in hits:
        c = getattr(h, "chunk", None)
        if c is None:
            continue
        out.append(RetrievalHit(
            chunk_id=c.chunk_id,
            score=round(float(h.score), 4),
            title=c.title,
            section=c.section,
        ))
    return out


@router.post("/ops/retrieval", response_model=RetrievalDebugResponse,
             dependencies=[Depends(require_feature("miss_log_view"))],
             summary="检索调试（只读）")
def debug_retrieval(
    req: RetrievalDebugRequest,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> RetrievalDebugResponse:
    """展示三路原始命中与融合窗口，便于排障「为什么这条排第一」。

    ★ 不调 LLM、不写库、不写未命中日志——只跑检索。
    """
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query 不能为空")

    collection = (req.collection or container.settings.coll_training).strip()
    if collection not in _allowed_collections(container):
        raise HTTPException(
            status_code=400,
            detail=f"collection 非法：{collection}；允许 "
                   f"{sorted(_allowed_collections(container))}",
        )
    if container.vector is None and container.keyword is None:
        raise HTTPException(status_code=503, detail="向量/关键词端口均未就绪")

    intent = IntentClassifier().classify(q).label
    res = _retriever(container).execute(
        q,
        intent=intent,
        vector_collection=collection,
        allowed_permissions=identity.role.retrieval_levels,
        top_k=req.top_k,
        fused_limit=10,
    )

    metrics = getattr(container, "metrics", None)
    if isinstance(metrics, OpsMetrics):
        metrics.record_retrieval(sum(res.elapsed_ms.values()))

    return RetrievalDebugResponse(
        query=q,
        collection=collection,
        intent=res.intent,
        weights=list(res.weights),
        elapsed_ms={k: round(v, 1) for k, v in res.elapsed_ms.items()},
        degraded=dict(res.degraded),
        paths={
            "vector": _hits(res.vector_hits),
            "keyword": _hits(res.keyword_hits),
            "sparse": _hits(res.sparse_hits),
        },
        fused=[
            FusedHitOut(chunk_id=h.chunk_id,
                        fused_score=round(float(h.fused_score), 5),
                        sources=list(h.sources))
            for h in res.fused
        ],
    )


@router.get("/ops/metrics", dependencies=[Depends(require_feature("miss_log_view"))],
            summary="运行指标（内存态）")
def read_metrics(container: Container = Depends(get_container)) -> dict:
    """LLM 主备统计 + 请求量/拒答/拦截/空答案 + 延迟分位 + 进程状态。"""
    metrics = getattr(container, "metrics", None)
    if isinstance(metrics, OpsMetrics):
        snapshot = metrics.snapshot()
    else:
        snapshot = {
            "counters": {"answer_total": 0, "refused": 0,
                         "guardrail_blocked": 0, "empty_answer": 0},
            "latency_ms": {"n": 0, "p50": None, "p95": None, "avg": None},
            "last_retrieval_ms": None,
            "uptime_sec": 0.0,
        }

    llm = getattr(container, "llm", None)
    llm_info = {
        "stats": llm.stats() if hasattr(llm, "stats") else {},
        "last_used": getattr(llm, "last_used", None),
        "last_error": getattr(llm, "last_error", None),
    }

    return {
        "metrics": snapshot,
        "llm": llm_info,
        "health": {
            "ready_ports": container.ready_ports(),
            "pending_ports": dict(container.pending),
        },
    }
