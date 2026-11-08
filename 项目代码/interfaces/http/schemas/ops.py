"""运维/调试接口的请求与响应模型（D2）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalDebugRequest(BaseModel):
    """检索调试请求。

    ★ 只读：不调 LLM、不写库、不写未命中日志。
    """

    query: str = Field(..., min_length=1, max_length=500,
                       description="要调试的查询")
    collection: str | None = Field(
        default=None,
        description="目标库；不传用默认训练文档库")
    top_k: int = Field(default=20, ge=1, le=50,
                       description="每路取多少条（1~50）")


class RetrievalHit(BaseModel):
    chunk_id: str
    score: float
    title: str = ""
    section: str = ""


class FusedHitOut(BaseModel):
    chunk_id: str
    fused_score: float
    sources: list[str] = Field(default_factory=list)


class RetrievalDebugResponse(BaseModel):
    query: str
    collection: str
    intent: str
    weights: list[float]
    elapsed_ms: dict[str, float]
    degraded: dict[str, str] = Field(default_factory=dict)
    paths: dict[str, list[RetrievalHit]] = Field(default_factory=dict)
    fused: list[FusedHitOut] = Field(default_factory=list)
