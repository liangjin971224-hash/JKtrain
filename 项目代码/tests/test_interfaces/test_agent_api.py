"""Agent API：鉴权、开关、响应结构。"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from application.agents.types import AgentAnswer, AgentStep
from config.assembly import Container
from config.settings import Settings
from domain.value_objects.account import EmployeeAccount
from infrastructure.auth.adapters import BcryptPasswordHasher, HmacTokenSigner
from interfaces.http import deps
from interfaces.http.app import create_app
from tests.fakes import FakeRepositoryPort

SECRET = "test-secret-key-32-bytes-long-xxxx"


@asynccontextmanager
async def _noop_lifespan(app):
    yield


class _Repo(FakeRepositoryPort):
    def __init__(self) -> None:
        super().__init__()
        hasher = BcryptPasswordHasher(rounds=4)
        self.accounts = {
            "M001": EmployeeAccount(
                employee_no="M001", role="manager",
                password_hash=hasher.hash("admin123"), active=True),
            "E001": EmployeeAccount(
                employee_no="E001", role="trainee",
                password_hash=hasher.hash("trainee123"), active=True),
        }

    def get_account(self, employee_no: str):
        return self.accounts.get(employee_no)

    def append_audit_log(self, entry) -> None:
        pass

    def touch_last_login(self, employee_no: str) -> None:
        pass


class _FakeAgent:
    def run(self, question, identity):
        return AgentAnswer(
            answer=f"agent:{question}",
            plan=({"tool": "kb_answer", "reason": "测试计划"},),
            steps=(AgentStep(
                tool="kb_answer", ok=True, elapsed_ms=1.5, summary="ok"),),
            citations=("D005_c001",),
        )


def _make_client(*, agent_enabled: bool) -> TestClient:
    repo = _Repo()
    app = create_app()
    app.router.lifespan_context = _noop_lifespan
    deps.set_container(Container(
        settings=Settings(
            _env_file=None,          # type: ignore[call-arg]
            auth_required=True,
            auth_secret=SECRET,
            agent_enabled=agent_enabled,
        ),
        repository=repo,
        password_hasher=BcryptPasswordHasher(rounds=4),
        token_signer=HmacTokenSigner(secret=SECRET),
    ))
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def client(monkeypatch):
    import interfaces.http.routers.agent as agent_router

    monkeypatch.setattr(
        agent_router, "_build_agent", lambda container, identity: _FakeAgent())
    c = _make_client(agent_enabled=True)
    try:
        yield c
    finally:
        deps.set_container(None)


def _login(client, no: str, pwd: str) -> dict:
    r = client.post("/api/auth/login",
                    json={"employee_no": no, "password": pwd})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


class TestAgentApi:
    def test_manager_can_run_task(self, client) -> None:
        headers = _login(client, "M001", "admin123")
        r = client.post("/api/agent/chat",
                        json={"question": "鱼油有什么功能"}, headers=headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["answer"] == "agent:鱼油有什么功能"
        assert body["plan"][0]["tool"] == "kb_answer"
        assert body["steps"][0]["ok"] is True
        assert body["citations"] == ["D005_c001"]

    def test_trainee_can_run_task(self, client) -> None:
        headers = _login(client, "E001", "trainee123")
        r = client.post("/api/agent/chat",
                        json={"question": "我上周练得怎么样"}, headers=headers)
        assert r.status_code == 200

    def test_requires_auth(self, client) -> None:
        r = client.post("/api/agent/chat", json={"question": "鱼油"})
        assert r.status_code == 401

    def test_empty_question_422(self, client) -> None:
        headers = _login(client, "M001", "admin123")
        r = client.post("/api/agent/chat", json={"question": ""},
                        headers=headers)
        assert r.status_code == 422

    def test_disabled_returns_404(self) -> None:
        c = _make_client(agent_enabled=False)
        try:
            headers = _login(c, "M001", "admin123")
            r = c.post("/api/agent/chat", json={"question": "鱼油"},
                       headers=headers)
            assert r.status_code == 404
            assert "未启用" in r.json()["detail"]
        finally:
            deps.set_container(None)
