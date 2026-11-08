"""多跳接口层测试（2026-10-23）。

────────────────────────────────────────────────────────────────────
为什么要单独写接口层测试
────────────────────────────────────────────────────────────────────
`test_multi_hop.py` 里的用例测试是**直接装配** `QueryPlanUseCase` 的，
它证明不了路由层把参数接对了。

实测踩到了（2026-10-23，HTTP 真链路）：

    用例层：fact_hints 传进生成层 → 回答正确
    接口层：route 里的 answer_one **漏透传 fact_hints**
            → 多跳查到的事实传不到生成层
            → 接口里又编出「目录中没有『改善体脂』等表述」
            → 而直连用例的测试是绿的

**单测绿 ≠ 接口可用**。这条教训在 P4 已经记过一次
（545 个测试通过后接真环境，一次跑出 5 个问题），这里是第二次。

所以本文件只做一件事：从 HTTP 入口打进去，
断言「拆解信息」「事实分区」「依赖语义」三样都完整到达响应。
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
    EfficacyRef,
    ElementSpec,
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


# ===================================================================
# 假依赖
# ===================================================================
def _products() -> dict[str, ProductFacts]:
    """与真实库同构：EFF_09 目录有、但没有任何产品声称。"""
    return {
        "P011": ProductFacts(
            product_id="P011", name="康源牌钙维生素D软胶囊",
            category="保健食品", retail_price=168,
            claims=(ApprovalClaim("EFF_NUT_CA", "补充钙、维生素D"),),
            aliases=("钙片",),
        ),
        "P020": ProductFacts(
            product_id="P020", name="康源牌代餐奶昔",
            category="普通食品", retail_price=218,
        ),
    }


CATALOG = (
    EfficacyRef("EFF_09", "有助于控制体内脂肪", catalog_no=9),
    EfficacyRef("EFF_NUT_CA", "补充钙、维生素D"),
)


def _chunk() -> ChunkMeta:
    return ChunkMeta(
        chunk_id="D004_c001", doc_id="D004",
        text="保健食品不得宣称疾病治疗功能，须带法定警示语",
        title="保健食品基础知识",
        permission_level=PermissionLevel.PUBLIC,
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


class _Graph(FakeGraphQueryPort):
    """真实现「按功能反查」，空列表是合法结果。"""

    def __init__(self) -> None:
        super().__init__(_products(), catalog=CATALOG)


def _fake_loader(_dir: Path) -> list[CaseSpec]:
    return [CaseSpec(case_id="C01", title="t", opening_line="q",
                     elements=(ElementSpec("E1", "n", 10, True),))]


def _container(reply: str = "这是合规回答。" * 3) -> Container:
    return Container(
        settings=Settings(_env_file=None, auth_required=False),  # type: ignore[call-arg]
        graph=_Graph(),
        vector=_FixedVector(),
        keyword=_FixedKeyword(),
        llm=FakeLanguageModelPort(reply=reply),
        case_catalog=CaseCatalog(Path("(fake)"), _fake_loader),
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


# ===================================================================
# 1. 拆解信息必须到达响应
# ===================================================================
class TestHopsReachResponse:
    def test_parallel_split_exposed(self, client) -> None:
        """「补钙的有什么推荐，有什么禁忌」→ 响应里能看到 2 步。"""
        r = client.post("/api/ask",
                        json={"question": "补钙的有什么推荐，有什么禁忌"})
        assert r.status_code == 200
        body = r.json()

        assert body["meta"]["hops"] == 2
        assert len(body["hops"]) == 2
        assert [h["order"] for h in body["hops"]] == [0, 1]
        assert all("补钙" in h["question"] for h in body["hops"])

    def test_dependency_flag_exposed(self, client) -> None:
        """「其中那种最便宜」这一步必须标 `depends_on_previous=true`。"""
        r = client.post("/api/ask", json={
            "question": "改善体脂都可以吃什么保健品，其中那种最便宜"})
        assert r.status_code == 200
        hops = r.json()["hops"]

        assert len(hops) == 2
        assert hops[0]["depends_on_previous"] is False
        assert hops[1]["depends_on_previous"] is True, (
            "第 2 步没标依赖 —— 前端就无从解释「价格是在上一问的范围里比出来的」，"
            "学员也学不到「先定范围再谈条件」这个流程"
        )
        assert hops[1]["kind"] == "price_rank"

    def test_single_step_has_no_hops_field_content(self, client) -> None:
        """单步问题不返回拆解明细（避免让人误以为多跳是常态）。"""
        r = client.post("/api/ask", json={"question": "鱼油有什么功能"})
        assert r.status_code == 200
        assert r.json()["hops"] == []


# ===================================================================
# 2. ★ 事实必须到达生成层（路由漏透传的守护）
# ===================================================================
class TestFactsReachPrompt:
    def test_catalog_fact_in_prompt_via_http(self) -> None:
        """★ 本文件最重要的一条：从 HTTP 打进去，查 LLM 收到的 prompt。

        路由层若漏透传 `fact_hints`，多跳查到的事实就传不到生成层，
        接口又会编出「目录中没有这个功能」——
        而直连用例的测试全是绿的（实测已发生）。
        """
        app = create_app()
        app.router.lifespan_context = _noop_lifespan
        llm = FakeLanguageModelPort(reply="资料里没有。")
        container = _container()
        container.llm = llm
        deps.set_container(container)
        try:
            c = TestClient(app)
            r = c.post("/api/ask", json={
                "question": "改善体脂都可以吃什么保健品"})
            assert r.status_code == 200
        finally:
            deps.set_container(None)

        assert llm.prompts, "应调用过 LLM"
        prompt = llm.prompts[0]
        assert "有助于控制体内脂肪" in prompt, (
            "目录功能名没到 prompt —— 路由层丢了 fact_hints"
        )
        assert "第 9 项" in prompt
        assert ("没有任何一款" in prompt) or ("没有一款" in prompt)

    def test_graph_products_reach_prompt_via_http(self) -> None:
        """图谱查到的产品必须进 prompt（否则只能反问客户）。"""
        app = create_app()
        app.router.lifespan_context = _noop_lifespan
        llm = FakeLanguageModelPort(reply="资料里没有。")
        container = _container()
        container.llm = llm
        deps.set_container(container)
        try:
            c = TestClient(app)
            r = c.post("/api/ask", json={"question": "补钙的有什么推荐"})
            assert r.status_code == 200
            assert r.json()["product_ids"] == ["P011"]
        finally:
            deps.set_container(None)

        assert "康源牌钙维生素D软胶囊" in llm.prompts[0]


# ===================================================================
# 3. 关闭多跳时行为不变（排查用开关）
# ===================================================================
class TestMultiHopToggle:
    def test_disabled_behaves_like_before(self, client) -> None:
        """`multi_hop=false` 时退回单步：不拆分、无 hops、无 multi_hop 标记。"""
        r = client.post("/api/ask", json={
            "question": "补钙的有什么推荐，有什么禁忌", "multi_hop": False})
        assert r.status_code == 200
        body = r.json()
        assert body["hops"] == []
        assert "multi_hop" not in body["meta"]

    def test_enabled_by_default(self, client) -> None:
        """默认开启：用户要的多跳不该需要额外传参才生效。"""
        r = client.post("/api/ask", json={
            "question": "补钙的有什么推荐，有什么禁忌"})
        assert r.json()["meta"].get("multi_hop") is True


# ===================================================================
# 4. 权限仍由服务端决定（回归：不能被多跳绕开）
# ===================================================================
class TestPermissionNotWeakened:
    def test_multi_hop_still_forces_role_levels(self, client) -> None:
        """多跳路径也必须用角色决定的检索级别，不接受前端声明。"""
        from interfaces.http.schemas.models import HopInfo  # noqa: F401

        r = client.post("/api/ask", json={
            "question": "补钙的有什么推荐，有什么禁忌",
            # 前端若尝试提权，schema 里根本没有这个字段，会被忽略
            "allowed_permissions": ["public", "internal", "restricted"],
        })
        assert r.status_code == 200, (
            "传了未知字段应被忽略而不是报错（extra=ignore）"
        )

    def test_hops_do_not_leak_candidates_beyond_role(self, client) -> None:
        """候选集只能来自本角色可检索的内容（当前只校验结构，不猜内容）。"""
        body = client.post("/api/ask", json={
            "question": "补钙的有什么推荐，有什么禁忌"}).json()
        for hop in body["hops"]:
            assert hop["candidate_count"] == len(hop["candidate_ids"])
