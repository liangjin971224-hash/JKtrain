"""运维接口测试（D2/D3）。"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from application.services.ops_metrics import OpsMetrics
from config.assembly import Container
from config.settings import Settings
from domain.value_objects import (
    ChunkMeta,
    KeywordHit,
    PermissionLevel,
    VectorHit,
)
from interfaces.http import deps
from interfaces.http.app import create_app
from tests.fakes import (
    FakeGraphQueryPort,
    FakeKeywordSearchPort,
    FakeLanguageModelPort,
    FakeVectorSearchPort,
)


def _chunk() -> ChunkMeta:
    return ChunkMeta(
        chunk_id="D005_c001", doc_id="D005",
        text="保健食品只能声称 24 项功能，须带法定警示语",
        title="功能声称规范", permission_level=PermissionLevel.PUBLIC,
    )


class _FixedVector(FakeVectorSearchPort):
    def __init__(self) -> None:
        super().__init__([_chunk()])

    def search(self, query, *, collection="", top_k=20,
               filter_expr=None, allowed_permissions=None):
        self.calls.append("search")
        return [VectorHit(chunk=_chunk(), score=0.72)][:top_k]


class _FixedKeyword(FakeKeywordSearchPort):
    def __init__(self) -> None:
        super().__init__([_chunk()])

    def search(self, query, *, top_k=20, allowed_permissions=None):
        self.calls.append("search")
        return [KeywordHit(chunk=_chunk(), score=20.0)][:top_k]


def _container() -> Container:
    return Container(
        settings=Settings(  # type: ignore[call-arg]
            _env_file=None, auth_required=False, multi_turn_enabled=False),
        graph=FakeGraphQueryPort(),
        vector=_FixedVector(),
        keyword=_FixedKeyword(),
        llm=FakeLanguageModelPort(
            reply="保健食品不是药物，不能代替药物治疗疾病。"),
        metrics=OpsMetrics(),
    )


@asynccontextmanager
async def _noop_lifespan(app):
    yield


@pytest.fixture()
def client():
    app = create_app()
    app.router.lifespan_context = _noop_lifespan
    deps.set_container(_container())
    try:
        with TestClient(app) as c:
            yield c
    finally:
        deps.set_container(None)


class TestRetrievalDebug:
    def test_manager_can_debug(self, client) -> None:
        r = client.post("/api/ops/retrieval", json={"query": "鱼油"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["collection"] == "health_training_docs"
        assert body["paths"]["vector"][0]["chunk_id"] == "D005_c001"
        assert body["paths"]["keyword"]
        assert body["fused"]

    def test_trainee_forbidden(self, client) -> None:
        r = client.post(
            "/api/ops/retrieval", json={"query": "鱼油"},
            headers={"X-Debug-Role": "trainee"})
        assert r.status_code == 403

    def test_empty_query_rejected(self, client) -> None:
        r = client.post("/api/ops/retrieval", json={"query": ""})
        assert r.status_code == 422

    def test_invalid_collection_rejected(self, client) -> None:
        r = client.post("/api/ops/retrieval",
                        json={"query": "鱼油", "collection": "nope"})
        assert r.status_code == 400


class TestMetrics:
    def test_metrics_shape(self, client) -> None:
        r = client.get("/api/ops/metrics")
        assert r.status_code == 200
        body = r.json()
        assert set(body["metrics"]["counters"]) == {
            "answer_total", "refused", "guardrail_blocked", "empty_answer"}
        assert "latency_ms" in body["metrics"]
        assert "health" in body

    def test_answer_increments_counter(self, client) -> None:
        client.post("/api/ask", json={"question": "鱼油有什么功能",
                                      "multi_hop": False})
        body = client.get("/api/ops/metrics").json()
        assert body["metrics"]["counters"]["answer_total"] == 1
        assert body["metrics"]["latency_ms"]["n"] == 1

    def test_trainee_forbidden(self, client) -> None:
        r = client.get("/api/ops/metrics",
                       headers={"X-Debug-Role": "trainee"})
        assert r.status_code == 403
