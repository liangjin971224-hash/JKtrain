"""越权扫描测试（P8 补，2026-10-23）。

★ 这个文件的来历

    用户测日报时发现「没保存」，排查出身份可伪造漏洞。
    修完后用户要求「把剩下所有路由再扫一遍」。

    扫出三处：

    ① `POST /api/ask` 的 `allowed_permissions` 来自请求体
       学员传 restricted 就能越权检索受限内容（违反 FR-050）

    ② `POST /api/drill/answer` / `review` **完全不校验会话归属**
       知道 session_id 就能替别人作答、看别人的摘要卡（IDOR）

    ③ `POST /api/drill/start` 的 `trainee_id` 来自请求体
       可以给别人建演练会话

    这个文件把「身份只能来自令牌」这条规则**逐路由固定下来**。

────────────────────────────────────────────────────────────────────
为什么这些漏洞之前没被发现
────────────────────────────────────────────────────────────────────
因为之前的测试**用传参模拟身份**：
    client.post("/api/drill/start", json={"trainee_id": "T1"})

那写法本身就假设「身份可以由请求声明」，所以永远测不出伪造。

正确写法：真实登录拿令牌，然后**故意传伪造的身份参数**，
断言它不生效。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from config.assembly import Container
from config.settings import Settings
from infrastructure.auth.adapters import BcryptPasswordHasher, HmacTokenSigner
from interfaces.http import deps
from interfaces.http.app import create_app
from tests.fakes import FakeRepositoryPort

SECRET = "test-secret-key-32-bytes-long-xxxx"


@asynccontextmanager
async def _noop_lifespan(app):
    yield


class ScanRepo(FakeRepositoryPort):
    def __init__(self) -> None:
        super().__init__()
        from domain.value_objects.account import EmployeeAccount

        self.accounts: dict[str, EmployeeAccount] = {}
        self.audits: list = []
        hasher = BcryptPasswordHasher(rounds=4)
        for no, pwd, role in (
            ("M001", "admin123", "manager"),
            ("E001", "trainee123", "trainee"),
            ("E002", "trainee456", "trainee"),
        ):
            self.accounts[no] = EmployeeAccount(
                employee_no=no, role=role,
                password_hash=hasher.hash(pwd), active=True)

    def get_account(self, employee_no: str):
        return self.accounts.get(employee_no)

    def touch_last_login(self, employee_no: str) -> None:
        pass

    def append_audit_log(self, entry) -> None:
        self.audits.append(entry)


@pytest.fixture()
def client():
    from application.services.case_catalog import CaseCatalog
    from domain.value_objects import CaseSpec, ElementSpec, ScriptedFollowUp

    def _fake_loader(_dir) -> list[CaseSpec]:
        return [CaseSpec(
            case_id="C01", title="测试案例", opening_line="你们有鱼油吗？",
            elements=(ElementSpec("E1", "明确回答", 20, True),),
            follow_ups=(ScriptedFollowUp("F1", "要吃多久？"),),
        )]

    repo = ScanRepo()
    app = create_app()
    app.router.lifespan_context = _noop_lifespan
    deps.set_container(Container(
        settings=Settings(
            _env_file=None,          # type: ignore[call-arg]
            auth_required=True,
            auth_secret=SECRET,
        ),
        repository=repo,
        password_hasher=BcryptPasswordHasher(rounds=4),
        token_signer=HmacTokenSigner(secret=SECRET),
        case_catalog=CaseCatalog(__import__("pathlib").Path("(fake)"),
                                 _fake_loader),
    ))
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        deps.set_container(None)


def _h(client, no="E001", pwd="trainee123") -> dict:
    r = client.post("/api/auth/login",
                    json={"employee_no": no, "password": pwd})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


# ===================================================================
# 1. ★★ 检索权限：不能由前端声明（FR-050）
# ===================================================================
class TestRetrievalPermissionNotDeclarable:
    """学员不能通过请求体提升检索级别。"""

    def test_schema_has_no_allowed_permissions(self) -> None:
        """★ 最彻底的验证：字段根本不存在。"""
        from interfaces.http.schemas.models import AskRequest

        fields = set(AskRequest.model_fields)
        assert "allowed_permissions" not in fields, (
            "AskRequest 又出现了 allowed_permissions —— "
            "学员可以传 restricted 越权检索"
        )

    def test_role_decides_retrieval_levels(self) -> None:
        """权限级别由角色决定，学员拿不到 restricted。"""
        from domain.value_objects import Role

        trainee = Role.TRAINEE.retrieval_levels
        assert "restricted" not in [str(x) for x in trainee]
        assert "confidential" not in [str(x) for x in trainee]

        manager = Role.MANAGER.retrieval_levels
        assert "restricted" in [str(x) for x in manager]


# ===================================================================
# 2. ★★ 演练会话归属（IDOR）
# ===================================================================
class TestDrillSessionOwnership:
    def test_start_uses_login_identity(self, client) -> None:
        """★ 传别人的工号也建不成别人的会话。"""
        h = _h(client)
        r = client.post("/api/drill/start", headers=h, json={
            "case_id": "C01",
            "trainee_id": "E002",       # 想建成别人的
        })
        assert r.status_code == 200
        assert r.json()["trainee_id"] == "E001", (
            "会话建到了别人名下 —— 身份可伪造"
        )

    def test_cannot_answer_others_session(self, client) -> None:
        """★★ 别人的会话不能作答（IDOR）。"""
        h1 = _h(client, "E001", "trainee123")
        h2 = _h(client, "E002", "trainee456")

        sid = client.post("/api/drill/start", headers=h1,
                          json={"case_id": "C01"}).json()["session_id"]

        # E002 拿着 E001 的 session_id 想作答
        r = client.post("/api/drill/answer", headers=h2, json={
            "session_id": sid, "answer": "我替他答",
        })
        assert r.status_code == 403, (
            f"能替别人的会话作答（HTTP {r.status_code}）"
        )

    def test_cannot_review_others_session(self, client) -> None:
        """★★ 别人的会话不能评阅（会看到别人的摘要卡）。"""
        h1 = _h(client, "E001", "trainee123")
        h2 = _h(client, "E002", "trainee456")

        sid = client.post("/api/drill/start", headers=h1,
                          json={"case_id": "C01"}).json()["session_id"]

        r = client.post("/api/drill/review", headers=h2,
                        json={"session_id": sid})
        assert r.status_code == 403, (
            f"能看别人的摘要卡（HTTP {r.status_code}）"
        )

    def test_own_session_works(self, client) -> None:
        """自己的会话正常可用（不能过度拦截）。"""
        h = _h(client)
        sid = client.post("/api/drill/start", headers=h,
                          json={"case_id": "C01"}).json()["session_id"]

        assert client.post("/api/drill/answer", headers=h, json={
            "session_id": sid, "answer": "有的。",
        }).status_code == 200

        assert client.post("/api/drill/review", headers=h,
                           json={"session_id": sid}).status_code == 200

    def test_manager_also_cannot_hijack(self, client) -> None:
        """★ 主管也不能替学员操作会话（他不是这个会话的演练人）。

        主管要看学员表现，应该看演练记录（P7 的 drill_record），
        而不是接管学员的实时会话。
        """
        h1 = _h(client, "E001", "trainee123")
        hm = _h(client, "M001", "admin123")

        sid = client.post("/api/drill/start", headers=h1,
                          json={"case_id": "C01"}).json()["session_id"]

        r = client.post("/api/drill/answer", headers=hm, json={
            "session_id": sid, "answer": "主管替答",
        })
        assert r.status_code == 403


# ===================================================================
# 3. ★ 组卷创建人
# ===================================================================
class TestExamCreator:
    def test_build_route_reads_identity(self) -> None:
        """★ 组卷的 created_by 必须来自 Identity，不是请求体。

        用源码级断言而不是发请求：这个容器没有 Milvus，
        组卷必然 503，测不出 created_by 的取值。
        直接检查路由实现更可靠。
        """
        import inspect

        from interfaces.http.routers import exam

        src = inspect.getsource(exam.build_exam)
        assert "identity.employee_no" in src, (
            "组卷没有用登录身份作为创建人"
        )
        assert "created_by=req.created_by" not in src, (
            "组卷仍在信任请求体里的 created_by"
        )


# ===================================================================
# 4. ★ 全量：所有写操作都要身份
# ===================================================================
class TestAllWriteEndpointsRequireAuth:
    """未登录访问所有 POST 端点必须 401。

    ★ 这是最粗但最有效的防线：
      只要有一个写接口漏挂身份，就能被匿名调用。
    """

    @pytest.mark.parametrize(("path", "body"), [
        ("/api/ask", {"question": "鱼油"}),
        ("/api/ask/stream", {"question": "鱼油"}),
        ("/api/drill/start", {"case_id": "C01"}),
        ("/api/drill/answer", {"session_id": "x", "answer": "y"}),
        ("/api/drill/review", {"session_id": "x"}),
        ("/api/exam/build", {"total": 3}),
        ("/api/employees/import", {"rows": []}),
        ("/api/summaries", {"content": "x"}),
        ("/api/summaries/review", {"trainee_no": "E001",
                                   "summary_date": "2026-10-23"}),
        ("/api/accounts/role", {"employee_no": "E001", "role": "manager"}),
        ("/api/accounts/active", {"employee_no": "E001", "active": False}),
    ])
    def test_post_requires_auth(self, client, path, body) -> None:
        r = client.post(path, json=body)
        assert r.status_code == 401, f"{path} 未登录也能调用"

    @pytest.mark.parametrize("path", [
        "/api/auth/me",
        "/api/employees",
        "/api/summaries",
        "/api/miss-logs",
        "/api/accounts",
        "/api/audit-logs",
        "/api/exam/papers",
        "/api/cases",
        "/api/cases/C01",
    ])
    def test_get_requires_auth(self, client, path) -> None:
        r = client.get(path)
        assert r.status_code == 401, f"{path} 未登录也能访问"


# ===================================================================
# 5. ★★ FR-050：评阅标准仅主管可见
# ===================================================================
class TestCaseDetailContentPermission:
    """案例详情里哪些字段该给学员。

    FR-050 原文：「参考答案、评阅标准应设为受限级别，仅主管可见」。

    实测数据（C01）：
        judgement_note = 「出现「有」+ 产品名，或「没有」。
                          不能是"好像有""我们产品挺多"」
        weight         = 20
        focus          = 「功效边界 + 不承诺」

    这些全是评阅侧信息：
      · judgement_note 是评分标准原文
      · weight 会引导学员只为高分答题
      · focus 等于提前告诉学员这道题考什么

    产品决策（2026-10-23 用户确认）：
        学员看到 name（不然不知道该说什么）
        不给 weight / judgement_note / focus / trigger_note
    """

    def test_trainee_cannot_see_judgement_note(self, client) -> None:
        """★★ 学员看不到判定说明（评分标准）。"""
        h = _h(client)
        r = client.get("/api/cases/C01", headers=h)
        assert r.status_code == 200
        body = r.json()

        for el in body["elements"]:
            assert "judgement_note" not in el, (
                f"学员看到了判定说明: {el}"
            )

    def test_trainee_cannot_see_weight(self, client) -> None:
        """★ 学员看不到分值（会只为高分答）。"""
        h = _h(client)
        body = client.get("/api/cases/C01", headers=h).json()
        for el in body["elements"]:
            assert "weight" not in el, f"学员看到了分值: {el}"

    def test_trainee_cannot_see_focus(self, client) -> None:
        """★ 学员看不到追问的考察点（等于给答案）。"""
        h = _h(client)
        body = client.get("/api/cases/C01", headers=h).json()
        for f in body["follow_ups"]:
            assert "focus" not in f, f"学员看到了考察点: {f}"

    def test_trainee_cannot_see_trigger_note(self, client) -> None:
        """学员看不到触发条件（流程控制信息）。"""
        h = _h(client)
        body = client.get("/api/cases/C01", headers=h).json()
        for f in body["follow_ups"]:
            assert "trigger_note" not in f, f"学员看到了触发条件: {f}"

    def test_trainee_can_see_name(self, client) -> None:
        """★ 但必须能看到要素名称 —— 不然不知道该说什么。"""
        h = _h(client)
        body = client.get("/api/cases/C01", headers=h).json()

        assert body["elements"], "学员看不到任何要素"
        for el in body["elements"]:
            assert el.get("name"), f"要素缺名称: {el}"
            assert "element_id" in el
            assert "required" in el

    def test_trainee_can_see_question(self, client) -> None:
        """学员必须能看到追问的问题（客户会问什么）。"""
        h = _h(client)
        body = client.get("/api/cases/C01", headers=h).json()

        assert body["follow_ups"], "学员看不到任何追问"
        for f in body["follow_ups"]:
            assert f.get("question"), f"追问缺问题文本: {f}"

    def test_manager_sees_everything(self, client) -> None:
        """★★ 主管要看完整信息（他要用这些做评阅）。"""
        hm = _h(client, "M001", "admin123")
        body = client.get("/api/cases/C01", headers=hm).json()

        for el in body["elements"]:
            assert "judgement_note" in el, "主管看不到判定说明"
            assert "weight" in el, "主管看不到分值"

        for f in body["follow_ups"]:
            assert "focus" in f, "主管看不到考察点"
            assert "trigger_note" in f, "主管看不到触发条件"

    def test_filter_is_server_side(self, client) -> None:
        """★★ 过滤必须在服务端做（前端隐藏等于没做）。

        只要学员的响应体里**根本没有**这些字段，
        抓包也拿不到 —— 这才是真的权限控制。
        """
        h = _h(client)
        raw = client.get("/api/cases/C01", headers=h).text

        # 判定说明的原文不该出现在响应里
        assert "judgement_note" not in raw
        # 权重字段名也不该有
        assert '"weight"' not in raw
