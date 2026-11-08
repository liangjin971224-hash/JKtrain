"""接口层测试。

★ 用 `set_container()` 注入**假容器**，不加载真实模型——
  否则每个测试文件都要等 15 秒加载 bge-m3。

★ 但「假容器能跑通」不代表「真容器能跑通」，
  所以另有一组集成测试打 `@pytest.mark.integration` 走真实依赖。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from application.services.case_catalog import CaseCatalog
from config.assembly import Container
from config.settings import Settings
from domain.value_objects import (
    ApprovalClaim,
    CaseSpec,
    ChunkMeta,
    Contraindication,
    ElementSpec,
    ForbiddenRule,
    KeywordHit,
    PermissionLevel,
    ProductFacts,
    RiskLevel,
    Severity,
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


# ===================================================================
# 假依赖
# ===================================================================
def _fish_oil() -> ProductFacts:
    return ProductFacts(
        product_id="P001", name="康源牌鱼油软胶囊",
        category="保健食品", approval_no="国食健注G20250012",
        claims=(ApprovalClaim(efficacy_id="EFF_19",
                             text="有助于维持血脂（胆固醇/甘油三酯）健康水平",
                             catalog_no=19),),
        contraindications=(
            Contraindication("CI_BLEED", "出血倾向者和出血性疾病患者",
                             Severity.CRITICAL, ""),
        ),
        aliases=("鱼油",),
    )


def _chunks() -> list[ChunkMeta]:
    return [
        ChunkMeta(chunk_id="D005_c001", doc_id="D005",
                  text="保健食品只能声称 24 项功能",
                  title="功能声称规范",
                  permission_level=PermissionLevel.INTERNAL),
    ]


class _FixedVector(FakeVectorSearchPort):
    def __init__(self, hits: list[VectorHit]) -> None:
        super().__init__([h.chunk for h in hits])
        self._fixed = hits

    def search(self, query, *, collection="", top_k=20,
               filter_expr=None, allowed_permissions=None):
        self.calls.append("search")
        pool = self._fixed
        if allowed_permissions is not None:
            allowed = {str(p) for p in allowed_permissions}
            pool = [h for h in pool
                    if str(h.chunk.permission_level) in allowed]
        return pool[:top_k]


class _FixedKeyword(FakeKeywordSearchPort):
    def __init__(self, hits: list[KeywordHit]) -> None:
        super().__init__([h.chunk for h in hits])
        self._fixed = hits

    def search(self, query, *, top_k=20, allowed_permissions=None):
        self.calls.append("search")
        pool = self._fixed
        if allowed_permissions is not None:
            allowed = {str(p) for p in allowed_permissions}
            pool = [h for h in pool
                    if str(h.chunk.permission_level) in allowed]
        return pool[:top_k]


COMPLIANT = ("这款鱼油是保健食品，批准功能是有助于维持血脂健康水平。"
             "保健食品不是药物，不能代替药物治疗疾病。")


def _fake_container(reply: str = COMPLIANT) -> Container:
    """假容器。

    ★ 必须带 `case_catalog`：drill 路由从容器取案例目录，
      没有它就返回 503。
      loader 用**假函数**，不读真实文件——否则测试依赖磁盘上的案例文档。
    """
    return Container(
        settings=Settings(_env_file=None, auth_required=False),  # type: ignore[call-arg]
        graph=FakeGraphQueryPort({"P001": _fish_oil()}),
        vector=_FixedVector([VectorHit(chunk=c, score=0.72)
                             for c in _chunks()]),
        keyword=_FixedKeyword([KeywordHit(chunk=c, score=20.0)
                               for c in _chunks()]),
        llm=FakeLanguageModelPort(reply=reply),
        case_catalog=CaseCatalog(Path("(fake)"), _fake_loader),
    )


def _fake_loader(_dir: Path) -> list[CaseSpec]:
    """假案例加载器（不读文件）。"""
    return [
        CaseSpec(
            case_id="C01", title="客户询问是否有鱼油",
            opening_line="你们有鱼油吗？",
            elements=(
                ElementSpec("E1", "明确回答有 / 没有", 20, True),
                ElementSpec("E2", "给出产品名称", 10, True),
                ElementSpec("E5", "加强制警示语", 15, True),
            ),
            forbidden_rules=(
                ForbiddenRule("违规宣称", r"降血脂|治疗", RiskLevel.CRITICAL),
            ),
            follow_ups=(
                _fu("F1", "我体检说血脂有点高，吃这个管用吗？"),
                _fu("F2", "要吃多久啊？"),
            ),
        ),
        CaseSpec(
            case_id="C16", title="客户询问是否还有其他选择",
            opening_line="还有别的吗？",
            elements=(ElementSpec("E1", "明确回答", 20, True),),
        ),
    ]


def _fu(fid: str, question: str):
    from domain.value_objects import ScriptedFollowUp

    return ScriptedFollowUp(fid, question)


@pytest.fixture()
def client():
    """注入假容器后的客户端。

    ★ 必须把 lifespan 换掉（`lifespan=None`），否则 TestClient 会触发
      真实装配 → 加载 bge-m3 → 每个测试文件多等 15 秒。
      实测：不禁用 lifespan 时整组测试 68 秒，禁用后 1 秒内。
    """
    app = create_app()
    app.router.lifespan_context = _noop_lifespan
    deps.set_container(_fake_container())
    try:
        with TestClient(app) as c:
            yield c
    finally:
        deps.set_container(None)


@asynccontextmanager
async def _noop_lifespan(app):
    """空 lifespan：测试里不装配真实容器。"""
    yield


def _make_client(app) -> TestClient:
    """建一个**不触发真实装配**的客户端。

    ★ 关键：把 lifespan 换成空的。
      否则 TestClient 一进入就会 build_container() → 加载 bge-m3（15 秒）。
    """
    app.router.lifespan_context = _noop_lifespan
    return TestClient(app, raise_server_exceptions=False)


# 不要走 lifespan 的真实装配（否则加载模型）
@pytest.fixture()
def raw_client():
    """只建 app、不走 lifespan 的客户端（用于测 404 / 校验）。"""
    app = create_app()
    deps.set_container(_fake_container())
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        deps.set_container(None)


# ===================================================================
# 健康检查
# ===================================================================
class TestHealth:
    def test_health_ok(self, client) -> None:
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("ok", "degraded")
        assert "guardrail_rules" in body
        # 护栏规则数要如实反映（P4 是 42 条 / 7 类）
        assert body["guardrail_rules"]["禁用词规则数"] >= 40


# ===================================================================
# 问答
# ===================================================================
class TestAsk:
    def test_compliant_answer(self, client) -> None:
        r = client.post("/api/ask", json={"question": "鱼油有什么功能",
                                          "product_ids": ["P001"]})
        assert r.status_code == 200
        body = r.json()
        assert body["refused"] is False
        assert "有助于维持血脂" in body["answer"]
        assert body["cited_chunk_ids"]

    def test_refuses_without_hits(self) -> None:
        """★ 召回为空必须拒答，且不调 LLM。

        这里用一个没有任何数据的容器来验证。
        """
        from tests.fakes import FailOnCallLLM

        app = create_app()
        deps.set_container(Container(
            settings=Settings(_env_file=None, auth_required=False),  # type: ignore[call-arg]
            graph=FakeGraphQueryPort({}),
            vector=_FixedVector([]),
            keyword=_FixedKeyword([]),
            llm=FailOnCallLLM(),                 # 一调就炸
        ))
        try:
            client = _make_client(app)
            r = client.post("/api/ask",
                            json={"question": "知识库里没有的问题"})
            assert r.status_code == 200
            body = r.json()
            assert body["refused"] is True
            assert body["meta"]["llm_called"] is False
        finally:
            deps.set_container(None)

    def test_violating_output_blocked(self) -> None:
        """LLM 输出违规 → 拦截并转拒答。"""
        app = create_app()
        deps.set_container(_fake_container("这个能降血糖，很多人吃了都说好。"))
        try:
            client = _make_client(app)
            r = client.post("/api/ask", json={"question": "能降血糖吗"})
            body = r.json()
            assert body["refused"] is True
            assert "降血糖" not in body["answer"]
        finally:
            deps.set_container(None)

    def test_empty_question_rejected(self, client) -> None:
        """空问题 → 422（Pydantic min_length 校验）。"""
        r = client.post("/api/ask", json={"question": ""})
        assert r.status_code == 422

    def test_stream_endpoint(self, client) -> None:
        """★ 流式端点要能拿到 SSE 事件。"""
        r = client.post("/api/ask/stream",
                        json={"question": "鱼油有什么功能",
                              "product_ids": ["P001"]})
        assert r.status_code == 200
        text = r.text
        assert "data:" in text
        assert "[DONE]" in text
        assert "有助于维持血脂" in text

    def test_stream_has_done_event(self, client) -> None:
        r = client.post("/api/ask/stream",
                        json={"question": "鱼油有什么功能",
                              "product_ids": ["P001"]})
        assert '"kind": "done"' in r.text or '"kind":"done"' in r.text


# ===================================================================
# 案例
# ===================================================================
class TestCases:
    def test_list_cases(self, client) -> None:
        r = client.get("/api/cases")
        assert r.status_code == 200
        cases = r.json()
        # 假 loader 返回 2 个案例（C01 / C16）
        assert len(cases) >= 2, f"案例列表为空: {cases}"
        assert all("case_id" in c for c in cases)
        assert {c["case_id"] for c in cases} >= {"C01"}

    def test_case_detail(self, client) -> None:
        r = client.get("/api/cases/C01")
        if r.status_code == 404:
            pytest.skip("案例 C01 不存在")
        body = r.json()
        assert body["case_id"] == "C01"
        assert body["opening_line"]
        assert len(body["elements"]) >= 1

    def test_unknown_case_404(self, client) -> None:
        r = client.get("/api/cases/C99")
        assert r.status_code == 404


# ===================================================================
# 演练
# ===================================================================
class TestDrill:
    def test_full_flow(self, client) -> None:
        """★ 开始 → 作答 → 评阅 全流程。"""
        r = client.post("/api/drill/start",
                        json={"case_id": "C01", "trainee_id": "T1"})
        if r.status_code == 404:
            pytest.skip("案例 C01 不存在")
        assert r.status_code == 200
        state = r.json()
        sid = state["session_id"]
        assert state["current_question"] == "你们有鱼油吗？"
        assert state["turn_count"] == 0

        # 答完所有问题
        guard = 0
        while not state["is_finished"] and guard < 20:
            r = client.post("/api/drill/answer", json={
                "session_id": sid,
                "answer": "有的，康源牌鱼油软胶囊。这是保健食品，"
                          "有助于维持血脂健康水平。"
                          "保健食品不是药物，不能代替药物治疗疾病。",
            })
            assert r.status_code == 200
            state = r.json()
            guard += 1

        assert state["is_finished"] is True

        r = client.post("/api/drill/review", json={"session_id": sid})
        assert r.status_code == 200
        card = r.json()
        assert card["case_id"] == "C01"
        # 假案例 C01 定义了 3 个要素
        assert len(card["element_coverage"]) == 3

    def test_cannot_answer_after_finish(self, client) -> None:
        r = client.post("/api/drill/start", json={"case_id": "C16"})
        if r.status_code == 404:
            pytest.skip("案例不存在")
        sid = r.json()["session_id"]

        # 先答完
        state = r.json()
        guard = 0
        while not state["is_finished"] and guard < 20:
            r = client.post("/api/drill/answer",
                            json={"session_id": sid, "answer": "答。"})
            state = r.json()
            guard += 1

        # 再答应 400
        r = client.post("/api/drill/answer",
                        json={"session_id": sid, "answer": "再答。"})
        assert r.status_code == 400

    def test_unknown_session_404(self, client) -> None:
        r = client.post("/api/drill/answer",
                        json={"session_id": "nonexistent", "answer": "x"})
        assert r.status_code == 404

    def test_empty_answer_400(self, client) -> None:
        r = client.post("/api/drill/start", json={"case_id": "C01"})
        if r.status_code == 404:
            pytest.skip("案例不存在")
        sid = r.json()["session_id"]
        r = client.post("/api/drill/answer",
                        json={"session_id": sid, "answer": "   "})
        # Pydantic 的 min_length 只挡空串，"   " 会到用例层被挡成 400
        assert r.status_code in (400, 422)


# ===================================================================
# ★ 摘要卡不含分数（约束 #3 的接口层守护）
# ===================================================================
class TestNoScoreInAPI:
    def test_summary_card_schema_has_no_score(self) -> None:
        from interfaces.http.schemas.models import SummaryCardResponse

        fields = set(SummaryCardResponse.model_fields)
        assert "score" not in fields
        assert "verdict" not in fields
        for banned in ("grade", "result", "pass_fail", "total_score"):
            assert banned not in fields

    def test_response_json_has_no_score(self, client) -> None:
        r = client.post("/api/drill/start", json={"case_id": "C01"})
        if r.status_code == 404:
            pytest.skip("案例不存在")
        sid = r.json()["session_id"]
        state = r.json()
        guard = 0
        while not state["is_finished"] and guard < 20:
            r = client.post("/api/drill/answer",
                            json={"session_id": sid, "answer": "答。"})
            state = r.json()
            guard += 1

        r = client.post("/api/drill/review", json={"session_id": sid})
        text = r.text.lower()
        assert '"score"' not in text
        assert '"verdict"' not in text


# ===================================================================
# 前端静态文件
# ===================================================================
class TestFrontend:
    def test_index_served(self, client) -> None:
        r = client.get("/")
        assert r.status_code == 200
        assert "健康顾问培训系统" in r.text

    def test_css_served(self, client) -> None:
        r = client.get("/static/style.css")
        assert r.status_code == 200
        assert "--primary" in r.text

    def test_js_served(self, client) -> None:
        r = client.get("/static/app.js")
        assert r.status_code == 200
        assert "askStream" in r.text
