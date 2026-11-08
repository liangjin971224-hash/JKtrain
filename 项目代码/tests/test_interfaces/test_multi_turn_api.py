"""多轮问答接口测试（B4）。

★ 沿用现有 TestClient 夹具套路：注入假容器 + 空 lifespan，
  不加载 bge-m3、不连 Neo4j/Milvus。

本文件守两条最容易回归的线：
1. 不带 `session_id`（或开关关闭）时，非流式 / 流式响应**一个键都不多**；
2. 带 `session_id` 时，改写后的独立问题真的进了生成链路，
   且会话按身份隔离、流式 done 后能续问。
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from application.services.conversation import ConversationStore
from config.assembly import Container
from config.settings import Settings
from domain.value_objects import (
    ChunkMeta,
    KeywordHit,
    PermissionLevel,
    ProductFacts,
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

SID = "mt_session_0001"

#: 改造前 `/api/ask` 的响应字段集合（多轮字段在单轮路径不得出现）。
ORIGINAL_ASK_KEYS = {
    "answer", "refused", "refusal_reason", "cited_chunk_ids",
    "product_ids", "meta", "hops", "annotation",
}
#: 改造前流式事件的字段集合。
ORIGINAL_STREAM_KEYS = {"kind", "text", "refused", "refusal_reason", "meta"}

COMPLIANT = ("这款鱼油是保健食品，批准功能是有助于维持血脂健康水平。"
             "保健食品不是药物，不能代替药物治疗疾病。")


# ===================================================================
# 假依赖
# ===================================================================
def _graph() -> FakeGraphQueryPort:
    return FakeGraphQueryPort({
        "P001": ProductFacts(
            product_id="P001", name="康源牌鱼油软胶囊",
            category="保健食品", approval_no="国食健注G20250012",
            aliases=("鱼油",)),
        "P002": ProductFacts(
            product_id="P002", name="康源牌辅酶Q10软胶囊",
            category="保健食品", aliases=("辅酶Q10", "Q10")),
    })


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


def _container(*, enabled: bool = True) -> Container:
    return Container(
        settings=Settings(                       # type: ignore[call-arg]
            _env_file=None, auth_required=False, multi_turn_enabled=enabled),
        graph=_graph(),
        vector=_FixedVector(),
        keyword=_FixedKeyword(),
        llm=FakeLanguageModelPort(reply=COMPLIANT),
        conversation_store=ConversationStore(
            max_sessions=200, max_turns=3, ttl_sec=1800),
    )


@asynccontextmanager
async def _noop_lifespan(app):
    """空 lifespan：不触发真实装配（否则每个用例多等 15s 加载模型）。"""
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


def _events(text: str) -> list[dict]:
    """解析 SSE 的 `data:` 行（忽略 [DONE]）。"""
    out = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload and payload != "[DONE]":
            out.append(json.loads(payload))
    return out


# ===================================================================
# 1. 单轮零回归
# ===================================================================
class TestSingleTurnUnchanged:
    def test_no_session_has_original_response_shape(self, client) -> None:
        r = client.post("/api/ask", json={
            "question": "鱼油有什么功能",
            "product_ids": ["P001"],
            "multi_hop": False,
        })
        assert r.status_code == 200
        body = r.json()
        assert set(body) == ORIGINAL_ASK_KEYS
        assert body["refused"] is False
        assert "有助于维持血脂" in body["answer"]
        assert "session_id" not in body

    def test_no_session_multi_hop_has_original_shape(self, client) -> None:
        r = client.post("/api/ask", json={"question": "鱼油有什么功能"})
        assert r.status_code == 200
        assert set(r.json()) == ORIGINAL_ASK_KEYS

    def test_no_session_stream_has_original_payload_shape(self, client) -> None:
        r = client.post("/api/ask/stream",
                        json={"question": "鱼油有什么功能"})
        assert r.status_code == 200
        events = _events(r.text)
        assert events and events[-1]["kind"] == "done"
        assert all(set(ev) == ORIGINAL_STREAM_KEYS for ev in events)

    def test_switch_off_ignores_session_id(self, client) -> None:
        deps.set_container(_container(enabled=False))
        for question in ("鱼油怎么吃？", "那孕妇能吃吗？"):
            r = client.post("/api/ask", json={
                "question": question, "session_id": SID, "multi_hop": False})
            assert r.status_code == 200
            body = r.json()
            assert set(body) == ORIGINAL_ASK_KEYS
            assert "turn" not in body
            assert "multi_turn" not in body

    @pytest.mark.parametrize("bad_sid", [
        "short",                 # 太短
        "bad/session/id",        # 非法字符
        "包含中文的会话编号",       # 非法字符
        "x" * 70,                # 超过 max_length（Pydantic 422，不是 500）
    ])
    def test_illegal_session_id_never_500(self, client, bad_sid) -> None:
        r = client.post("/api/ask", json={
            "question": "鱼油怎么吃？", "session_id": bad_sid,
            "multi_hop": False})
        assert r.status_code in (200, 422)
        if r.status_code == 200:
            assert set(r.json()) == ORIGINAL_ASK_KEYS


# ===================================================================
# 2. 两轮追问 / 产品切换
# ===================================================================
class TestFollowUp:
    def test_two_turn_followup(self, client) -> None:
        r1 = client.post("/api/ask", json={
            "question": "鱼油每天吃几粒？", "session_id": SID,
            "multi_hop": False})
        b1 = r1.json()
        assert b1["session_id"] == SID
        assert b1["turn"] == 1
        assert b1["multi_turn"] is True
        assert b1["rewritten_question"] == ""       # 首轮无历史，不改写
        assert b1["rewrite_source"] == "none"

        r2 = client.post("/api/ask", json={
            "question": "那孕妇能吃吗？", "session_id": SID,
            "multi_hop": False})
        b2 = r2.json()
        assert b2["turn"] == 2
        assert b2["rewritten_question"] == "康源牌鱼油软胶囊 孕妇能吃吗"
        assert b2["rewrite_source"] == "rule"
        assert b2["multi_turn"] is True
        assert b2["refused"] is False
        assert "有助于维持血脂" in b2["answer"]
        # ★ 改写后的问题真的进了生成链路，而不只是响应里好看
        llm = deps.get_container().llm
        assert "康源牌鱼油软胶囊 孕妇能吃吗" in llm.prompts[-1]

    def test_multi_hop_path_also_uses_rewritten_question(self, client) -> None:
        """默认 multi_hop=True 的路径同样要用改写后的问题。"""
        client.post("/api/ask", json={
            "question": "鱼油每天吃几粒？", "session_id": SID})
        r2 = client.post("/api/ask", json={
            "question": "那孕妇能吃吗？", "session_id": SID})
        b2 = r2.json()
        assert b2["rewritten_question"] == "康源牌鱼油软胶囊 孕妇能吃吗"
        assert b2["turn"] == 2
        llm = deps.get_container().llm
        assert "康源牌鱼油软胶囊 孕妇能吃吗" in llm.prompts[-1]

    def test_product_switch_replaces_product_id(self, client) -> None:
        client.post("/api/ask", json={
            "question": "鱼油多少钱？", "session_id": SID,
            "multi_hop": False})
        r2 = client.post("/api/ask", json={
            "question": "辅酶Q10 呢？", "session_id": SID,
            "multi_hop": False})
        b2 = r2.json()
        assert b2["turn"] == 2
        # 裸省略「辅酶Q10 呢？」→ 新产品 + 上一轮主题
        assert b2["rewritten_question"] == "康源牌辅酶Q10软胶囊 多少钱？"
        assert b2["rewrite_source"] == "rule"
        assert b2["product_ids"] == ["P002"]

    def test_correction_works_with_default_multi_hop(self, client) -> None:
        """纠正类问句不改写句子，新 ID 必须能透传进多跳的每一跳。"""
        client.post("/api/ask", json={
            "question": "P001 的禁忌？", "session_id": SID,
            "product_ids": ["P001"]})
        r2 = client.post("/api/ask", json={
            "question": "不是这款，是 P002", "session_id": SID})
        b2 = r2.json()
        assert b2["turn"] == 2
        assert b2["rewritten_question"] == ""      # 按设计不改句子
        assert b2["rewrite_source"] == "rule"
        assert b2["product_ids"] == ["P002"]

    def test_refusal_still_appends_turn(self, client) -> None:
        """拒答也要落轮次，否则下一轮接不上场景。"""
        from tests.fakes import FailOnCallLLM

        container = Container(
            settings=Settings(                   # type: ignore[call-arg]
                _env_file=None, auth_required=False, multi_turn_enabled=True),
            graph=_graph(),
            vector=_FixedVector(),
            keyword=_FixedKeyword(),
            llm=FailOnCallLLM(),
            conversation_store=ConversationStore(),
        )
        # 换成没有召回的容器：用空检索构造拒答，且不调用改写 LLM
        container.vector = _FixedVector()
        container.keyword = _FixedKeyword()
        container.vector.search = lambda *a, **kw: []      # type: ignore[method-assign]
        container.keyword.search = lambda *a, **kw: []     # type: ignore[method-assign]
        deps.set_container(container)
        r1 = client.post("/api/ask", json={
            "question": "客户吃了头晕怎么办？", "session_id": SID,
            "multi_hop": False})
        assert r1.json()["refused"] is True
        assert r1.json()["turn"] == 1

        r2 = client.post("/api/ask", json={
            "question": "那要退货吗？", "session_id": SID,
            "multi_hop": False})
        b2 = r2.json()
        assert b2["turn"] == 2
        assert "头晕" in b2["rewritten_question"]


# ===================================================================
# 3. 会话安全
# ===================================================================
class TestSessionIsolation:
    def test_cross_user_session_is_not_shared(self, client) -> None:
        client.post("/api/ask", json={
            "question": "鱼油每天吃几粒？", "session_id": SID,
            "multi_hop": False})
        # 换身份（主管 → 学员）用同一个 session_id
        r = client.post(
            "/api/ask",
            json={"question": "那孕妇能吃吗？", "session_id": SID,
                  "multi_hop": False},
            headers={"X-Debug-Role": "trainee"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["turn"] == 1                      # 新会话，读不到上文
        assert body["rewritten_question"] == ""
        assert body["rewrite_source"] == "none"


# ===================================================================
# 4. 流式 done 后能续问
# ===================================================================
class TestStreamContinuation:
    def test_stream_done_then_next_turn_can_follow_up(self, client) -> None:
        r = client.post("/api/ask/stream", json={
            "question": "鱼油每天吃几粒？", "session_id": SID})
        assert r.status_code == 200
        events = _events(r.text)
        done = [ev for ev in events if ev["kind"] == "done"]
        assert done, r.text
        last = done[-1]
        assert last["session_id"] == SID
        assert last["turn"] == 1
        assert last["multi_turn"] is True
        assert last["rewrite_source"] == "none"

        r2 = client.post("/api/ask", json={
            "question": "那孕妇能吃吗？", "session_id": SID,
            "multi_hop": False})
        b2 = r2.json()
        assert b2["turn"] == 2                        # 流式那一轮确实落库了
        assert b2["rewritten_question"] == "康源牌鱼油软胶囊 孕妇能吃吗"
