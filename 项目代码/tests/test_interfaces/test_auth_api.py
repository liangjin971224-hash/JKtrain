"""认证与权限接口测试（P8）。

★ 用真实 token 链路（假仓储 + 真密码/令牌端口），
  不打桩认证过程 —— 认证本身是这次要验的东西。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

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


class AuthRepo(FakeRepositoryPort):
    """带账号的假仓储。"""

    def __init__(self) -> None:
        super().__init__()
        self.accounts: dict[str, EmployeeAccount] = {}
        self.audits: list = []
        hasher = BcryptPasswordHasher(rounds=4)

        self.accounts["M001"] = EmployeeAccount(
            employee_no="M001", role="manager",
            password_hash=hasher.hash("admin123"), active=True)
        self.accounts["E001"] = EmployeeAccount(
            employee_no="E001", role="trainee",
            password_hash=hasher.hash("trainee123"), active=True)

    def get_account(self, employee_no: str):
        return self.accounts.get(employee_no)

    def save_account(self, account) -> None:
        self.accounts[account.employee_no] = account

    def set_account_role(self, employee_no: str, role: str) -> None:
        a = self.accounts.get(employee_no)
        if a is None:
            raise ValueError(f"账号不存在: {employee_no}")
        self.accounts[employee_no] = EmployeeAccount(
            employee_no=a.employee_no, role=role,
            password_hash=a.password_hash, active=a.active)

    def set_account_active(self, employee_no: str, active: bool) -> None:
        a = self.accounts.get(employee_no)
        if a is None:
            raise ValueError(f"账号不存在: {employee_no}")
        self.accounts[employee_no] = EmployeeAccount(
            employee_no=a.employee_no, role=a.role,
            password_hash=a.password_hash, active=active)

    def touch_last_login(self, employee_no: str) -> None:
        pass

    # ---- 试卷（这套测试只用到列表，给个最小实现）----
    def list_exam_papers(self, *, stage="", status="", limit=50) -> list:
        return []

    # ---- 案例（问答路由用不到，但 /api/cases 会调）----
    def list_question_usage(self, **kw) -> set:
        return set()

    def recent_used_question_ids(self, *, trainee_id="", days=30) -> set:
        return set()

    def list_accounts(self, *, limit: int = 500) -> list:
        return [
            {"employee_no": a.employee_no, "role": a.role,
             "active": a.active, "last_login_at": ""}
            for a in sorted(self.accounts.values(),
                            key=lambda x: x.employee_no)[:limit]
        ]

    def append_audit_log(self, entry) -> None:
        self.audits.append(entry)

    def list_audit_logs(self, *, actor="", action="", limit=100) -> list:
        out = []
        for e in reversed(self.audits):
            if actor and e.actor != actor:
                continue
            if action and e.action != action:
                continue
            out.append({
                "actor": e.actor, "role": e.role, "action": e.action,
                "target": e.target, "result": e.result,
                "detail": e.detail, "ip": e.ip, "created_at": "",
            })
        return out[:limit]


@pytest.fixture()
def repo():
    return AuthRepo()


@pytest.fixture()
def client(repo):
    """★ auth_required=True —— 这套测试就是验鉴权的。"""
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
    ))
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        deps.set_container(None)


def _login(client, no: str, pwd: str) -> str:
    r = client.post("/api/auth/login",
                    json={"employee_no": no, "password": pwd})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ===================================================================
# 1. 登录
# ===================================================================
class TestLogin:
    def test_ok(self, client) -> None:
        r = client.post("/api/auth/login",
                        json={"employee_no": "M001", "password": "admin123"})
        assert r.status_code == 200
        d = r.json()
        assert d["token"]
        assert d["role"] == "manager"
        assert d["role_label"] == "主管"

    def test_wrong_password_401(self, client) -> None:
        r = client.post("/api/auth/login",
                        json={"employee_no": "M001", "password": "bad"})
        assert r.status_code == 401

    def test_unknown_user_401(self, client) -> None:
        r = client.post("/api/auth/login",
                        json={"employee_no": "NOBODY", "password": "x"})
        assert r.status_code == 401

    def test_errors_indistinguishable(self, client) -> None:
        """★ 两种失败的信息必须一样，否则可枚举账号。"""
        r1 = client.post("/api/auth/login",
                         json={"employee_no": "M001", "password": "bad"})
        r2 = client.post("/api/auth/login",
                         json={"employee_no": "NOBODY", "password": "x"})
        assert r1.json()["detail"] == r2.json()["detail"]

    def test_inactive_cannot_login(self, client, repo) -> None:
        repo.set_account_active("E001", False)
        r = client.post("/api/auth/login",
                        json={"employee_no": "E001", "password": "trainee123"})
        assert r.status_code == 401


# ===================================================================
# 2. ★ 未登录一律 401
# ===================================================================
class TestUnauthenticated:
    @pytest.mark.parametrize(("method", "path"), [
        ("GET", "/api/exam/papers"),
        ("POST", "/api/exam/build"),
        ("GET", "/api/miss-logs"),
        ("GET", "/api/audit-logs"),
        ("GET", "/api/accounts"),
    ])
    def test_protected_401(self, client, method, path) -> None:
        r = client.request(method, path, json={} if method == "POST" else None)
        assert r.status_code == 401, f"{path} 未鉴权也放行了"

    def test_bad_token_401(self, client) -> None:
        r = client.get("/api/exam/papers",
                       headers={"Authorization": "Bearer garbage"})
        assert r.status_code == 401

    def test_token_from_other_secret_401(self, client) -> None:
        """★ 别的密钥签的 token 不能用（防止跨环境串用）。"""
        other = HmacTokenSigner(secret="a-totally-different-secret-32b")
        forged = other.issue(employee_no="M001", role="manager")
        r = client.get("/api/exam/papers", headers=_h(forged))
        assert r.status_code == 401


# ===================================================================
# 3. ★ 功能权限
# ===================================================================
class TestFeaturePermission:
    @pytest.mark.parametrize(("method", "path"), [
        ("GET", "/api/exam/papers"),
        ("GET", "/api/miss-logs"),
        ("GET", "/api/audit-logs"),
        ("GET", "/api/accounts"),
    ])
    def test_trainee_forbidden(self, client, method, path) -> None:
        """★ 学员不可访问组卷/试卷管理/日志/账号。"""
        t = _login(client, "E001", "trainee123")
        r = client.request(method, path, headers=_h(t))
        assert r.status_code == 403, f"{path} 学员也能访问"

    def test_manager_allowed(self, client) -> None:
        t = _login(client, "M001", "admin123")
        r = client.get("/api/exam/papers", headers=_h(t))
        assert r.status_code == 200

    def test_trainee_can_use_basic_features(self, client) -> None:
        """学员基础功能不受影响（问答、看自己身份）。

        ★ 这里刻意不测 `/api/cases` —— 它依赖真实案例目录，
          而这套测试的容器没注入 case_catalog。
          测一个不依赖外部内容的接口，才是干净的权限验证。
        """
        t = _login(client, "E001", "trainee123")
        r = client.get("/api/auth/me", headers=_h(t))
        assert r.status_code == 200

        # 日报是自己的功能，学员能用
        r2 = client.get("/api/summaries?requester=E001&role=trainee",
                        headers=_h(t))
        assert r2.status_code == 200

    def test_trainee_cannot_import_employees(self, client) -> None:
        t = _login(client, "E001", "trainee123")
        r = client.post("/api/employees/import",
                        json={"rows": [{"工号": "X1", "姓名": "测试"}]},
                        headers=_h(t))
        assert r.status_code == 403


# ===================================================================
# 4. ★★ 权限变更即时生效（FR-048）
# ===================================================================
class TestRoleChangeTakesEffect:
    def test_upgrade_effective_immediately(self, client) -> None:
        """★★ 学员升主管后，**同一个旧 token** 立即能访问。

        这是 FR-048「权限变更应即时生效」的核心验证。
        如果角色只从 token 读，这里会一直是 403 直到 token 过期。
        """
        et = _login(client, "E001", "trainee123")
        mt = _login(client, "M001", "admin123")

        # 升级前：403
        assert client.get("/api/exam/papers",
                          headers=_h(et)).status_code == 403

        # 升级
        r = client.post("/api/accounts/role",
                        json={"employee_no": "E001", "role": "manager"},
                        headers=_h(mt))
        assert r.status_code == 200

        # 用**旧 token** 再试 → 应立即可用
        assert client.get("/api/exam/papers",
                          headers=_h(et)).status_code == 200, (
            "升权没有即时生效 —— 说明角色是从 token 读的，不是回库取的"
        )

    def test_downgrade_effective_immediately(self, client) -> None:
        """★★ 降权也即时生效（更重要的安全场景）。

        主管降为学员后，**旧 token 必须立刻失效对应权限**。
        否则离职/降职当天他仍能导出试卷。
        """
        mt = _login(client, "M001", "admin123")
        assert client.get("/api/exam/papers",
                          headers=_h(mt)).status_code == 200

        r = client.post("/api/accounts/role",
                        json={"employee_no": "M001", "role": "trainee"},
                        headers=_h(mt))
        assert r.status_code == 200

        assert client.get("/api/exam/papers",
                          headers=_h(mt)).status_code == 403, (
            "降权没有即时生效 —— 旧 token 仍能访问"
        )

    def test_deactivate_blocks_immediately(self, client) -> None:
        """★ 停用账号后，旧 token 立即失效（401）。"""
        mt = _login(client, "M001", "admin123")
        # 用另一个主管来停用 M001（不能自己停自己）
        client.post("/api/accounts/role",
                    json={"employee_no": "E001", "role": "manager"},
                    headers=_h(mt))
        et = _login(client, "E001", "trainee123")

        r = client.post("/api/accounts/active",
                        json={"employee_no": "M001", "active": False},
                        headers=_h(et))
        assert r.status_code == 200

        # M001 的旧 token 立即不能用
        assert client.get("/api/exam/papers",
                          headers=_h(mt)).status_code == 401


# ===================================================================
# 5. 账号管理约束
# ===================================================================
class TestAccountRules:
    def test_cannot_deactivate_self(self, client) -> None:
        """★ 不能停用自己 —— 会把自己锁在系统外。"""
        mt = _login(client, "M001", "admin123")
        r = client.post("/api/accounts/active",
                        json={"employee_no": "M001", "active": False},
                        headers=_h(mt))
        assert r.status_code == 400

    def test_invalid_role_400(self, client) -> None:
        mt = _login(client, "M001", "admin123")
        r = client.post("/api/accounts/role",
                        json={"employee_no": "E001", "role": "god"},
                        headers=_h(mt))
        assert r.status_code == 400

    def test_account_list_no_password_hash(self, client) -> None:
        """★★ 账号列表不能带密码哈希。"""
        mt = _login(client, "M001", "admin123")
        r = client.get("/api/accounts", headers=_h(mt))
        text = r.text.lower()
        assert "password" not in text
        assert "hash" not in text


# ===================================================================
# 6. 审计日志
# ===================================================================
class TestAuditLog:
    def test_login_recorded(self, client, repo) -> None:
        _login(client, "M001", "admin123")
        assert any(e.action == "login" for e in repo.audits)

    def test_failed_login_recorded(self, client, repo) -> None:
        client.post("/api/auth/login",
                    json={"employee_no": "M001", "password": "bad"})
        assert any(e.action == "login_failed" for e in repo.audits)

    def test_role_change_recorded(self, client, repo) -> None:
        mt = _login(client, "M001", "admin123")
        client.post("/api/accounts/role",
                    json={"employee_no": "E001", "role": "manager"},
                    headers=_h(mt))
        assert any(e.action == "account_role" for e in repo.audits)

    def test_log_has_no_credentials(self, client, repo) -> None:
        """★★ 审计日志不能含密码/令牌。"""
        _login(client, "M001", "admin123")
        for e in repo.audits:
            blob = f"{e.actor}{e.role}{e.action}{e.target}{e.detail}"
            assert "admin123" not in blob
            assert "trainee123" not in blob
            assert SECRET not in blob

    def test_trainee_cannot_view_logs(self, client) -> None:
        et = _login(client, "E001", "trainee123")
        r = client.get("/api/audit-logs", headers=_h(et))
        assert r.status_code == 403


# ===================================================================
# 7. 当前身份
# ===================================================================
class TestMe:
    def test_me_ok(self, client) -> None:
        et = _login(client, "E001", "trainee123")
        r = client.get("/api/auth/me", headers=_h(et))
        assert r.status_code == 200
        d = r.json()
        assert d["employee_no"] == "E001"
        assert d["role"] == "trainee"
        assert d["is_manager"] is False

    def test_me_requires_token(self, client) -> None:
        assert client.get("/api/auth/me").status_code == 401


# ===================================================================
# 8. ★★ 身份不能由前端声明（2026-10-23 修的漏洞）
# ===================================================================
class TestIdentityCannotBeForged:
    """实测发现的安全漏洞：日报/员工接口信任前端传的 requester/role。

    后果：
      · 学员传 `requester=M001` 就能写主管的日报
      · 学员传 `role=manager` 就能看到全部人的日报

    这类漏洞的根源是「把身份当普通参数」。
    身份是**由令牌证明的**，不是由调用方声明的。

    修法：请求 schema 里直接删掉这些字段 + 路由从 Identity 取。
    字段不存在，前端连传都传不进来。
    """

    def test_save_summary_ignores_requester(self, client) -> None:
        """★★ 学员传别人的工号，写入的仍是自己。"""
        et = _login(client, "E001", "trainee123")

        r = client.post("/api/summaries", json={
            "requester": "M001",           # 想冒充主管
            "role": "manager",
            "summary_date": "2026-10-23",
            "content": "我冒充主管写的",
        }, headers=_h(et))

        assert r.status_code == 200
        assert r.json()["trainee_no"] == "E001", (
            f"写入到了 {r.json()['trainee_no']} 名下 —— 身份可伪造"
        )

    def test_save_summary_schema_rejects_requester(self, client) -> None:
        """★ 更彻底：schema 里没有 requester 字段，传了会被忽略。"""
        from interfaces.http.schemas.models import SaveSummaryRequest

        fields = set(SaveSummaryRequest.model_fields)
        assert "requester" not in fields
        assert "role" not in fields

    def test_list_summaries_ignores_role(self, client, repo) -> None:
        """★★ 学员传 role=manager，仍只能看到自己。"""
        et = _login(client, "E001", "trainee123")
        # 先让主管写一条，制造「有别人数据」的环境
        mt = _login(client, "M001", "admin123")
        client.post("/api/summaries", json={
            "summary_date": "2026-10-23", "content": "主管的日报",
        }, headers=_h(mt))

        r = client.get("/api/summaries?role=manager", headers=_h(et))
        assert r.status_code == 200
        rows = r.json()
        assert all(x["trainee_no"] == "E001" for x in rows), (
            f"学员看到了别人的日报: {[x['trainee_no'] for x in rows]}"
        )

    def test_list_summaries_trainee_cannot_filter_others(self, client) -> None:
        """学员传 trainee_no=M001 也无效。"""
        et = _login(client, "E001", "trainee123")
        r = client.get("/api/summaries?trainee_no=M001", headers=_h(et))
        rows = r.json()
        assert all(x["trainee_no"] == "E001" for x in rows)

    def test_review_requires_manager(self, client) -> None:
        """★★ 学员不能写评语（改用 token 判定后仍然要拦）。"""
        et = _login(client, "E001", "trainee123")
        r = client.post("/api/summaries/review", json={
            "trainee_no": "E001",
            "summary_date": "2026-10-23",
            "note": "自己给自己写评语",
        }, headers=_h(et))
        assert r.status_code == 403

    def test_employee_list_ignores_role(self, client, repo) -> None:
        """★★ 学员传 role=manager，仍只看到自己。"""
        from domain.value_objects import EmployeeProfile

        repo.save_employee(EmployeeProfile(
            employee_no="M001", name="李主管", hire_date="2026-10-01"))
        repo.save_employee(EmployeeProfile(
            employee_no="E001", name="张三", hire_date="2026-10-01"))

        et = _login(client, "E001", "trainee123")
        r = client.get("/api/employees?role=manager", headers=_h(et))
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 1
        assert rows[0]["employee_no"] == "E001"

    def test_import_ignores_operator_role(self, client) -> None:
        """★★ 学员传 operator_role=manager 也不能导入。"""
        et = _login(client, "E001", "trainee123")
        r = client.post("/api/employees/import", json={
            "rows": [{"工号": "X1", "姓名": "测试"}],
            "operator_role": "manager",
        }, headers=_h(et))
        assert r.status_code == 403
