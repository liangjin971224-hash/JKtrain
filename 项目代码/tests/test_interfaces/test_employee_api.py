"""员工与日报接口测试（P7）。

★ 用假仓储，不连真实 MySQL。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from config.assembly import Container
from config.settings import Settings
from interfaces.http import deps
from interfaces.http.app import create_app
from infrastructure.auth.adapters import BcryptPasswordHasher, HmacTokenSigner
from tests.fakes import FakeRepositoryPort

SECRET = "test-secret-key-32-bytes-long-xxxx"


@asynccontextmanager
async def _noop_lifespan(app):
    yield


class EmpRepo(FakeRepositoryPort):
    """带账号的假仓储（日报/员工接口需要真实登录）。"""

    def __init__(self) -> None:
        super().__init__()
        from domain.value_objects.account import EmployeeAccount

        self.accounts: dict[str, EmployeeAccount] = {}
        self.audits: list = []
        hasher = BcryptPasswordHasher(rounds=4)
        self.accounts["M001"] = EmployeeAccount(
            employee_no="M001", role="manager",
            password_hash=hasher.hash("admin123"), active=True)
        self.accounts["E001"] = EmployeeAccount(
            employee_no="E001", role="trainee",
            password_hash=hasher.hash("trainee123"), active=True)
        self.accounts["E002"] = EmployeeAccount(
            employee_no="E002", role="trainee",
            password_hash=hasher.hash("trainee456"), active=True)

    def get_account(self, employee_no: str):
        return self.accounts.get(employee_no)

    def append_audit_log(self, entry) -> None:
        self.audits.append(entry)

    def touch_last_login(self, employee_no: str) -> None:
        pass


@pytest.fixture()
def client():
    """★ auth_required=True：这套测试要验「身份来自令牌」。

    原先是 auth_required=False + 前端传 requester/role，
    那正好掩盖了身份可伪造的漏洞。
    """
    repo = EmpRepo()
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


def _login(client, no: str, pwd: str) -> dict:
    """登录并返回认证头。"""
    r = client.post("/api/auth/login",
                    json={"employee_no": no, "password": pwd})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


@pytest.fixture()
def mgr(client):
    """主管的请求头。"""
    return _login(client, "M001", "admin123")


@pytest.fixture()
def tra(client):
    """学员 E001 的请求头。"""
    return _login(client, "E001", "trainee123")


@pytest.fixture()
def tra2(client):
    """学员 E002 的请求头（用于验证「学员之间互相看不到」）。"""
    return _login(client, "E002", "trainee456")


# ===================================================================
# 1. 员工导入
# ===================================================================
class TestEmployeeImport:
    def test_import_ok(self, client, mgr) -> None:
        r = client.post("/api/employees/import", headers=mgr, json={
            "rows": [
                {"工号": "X001", "姓名": "张三", "入职日期": "2026-10-01",
                 "岗位": "健康顾问", "团队": "一组", "带教人": "李四"},
                {"工号": "X002", "姓名": "王五", "入职日期": "2026-10-02"},
            ],
        })
        assert r.status_code == 200
        d = r.json()
        assert d["imported"] == 2
        assert d["ok"] is True

    def test_bad_rows_reported(self, client, mgr) -> None:
        """★ 坏行跳过并报告，不让整批失败。"""
        r = client.post("/api/employees/import", headers=mgr, json={
            "rows": [
                {"工号": "X001", "姓名": "张三", "入职日期": "2026-10-01"},
                {"工号": "", "姓名": "缺工号", "入职日期": "2026-10-02"},
            ],
        })
        d = r.json()
        assert d["imported"] == 1
        assert d["errors"], "坏行应被报告"

    def test_trainee_cannot_import(self, client, tra) -> None:
        """★★ 学员不能导入（角色从 token 判定）。"""
        r = client.post("/api/employees/import", headers=tra, json={
            "rows": [{"工号": "X001", "姓名": "张三"}],
        })
        assert r.status_code == 403

    def test_list_after_import(self, client, mgr) -> None:
        client.post("/api/employees/import", headers=mgr, json={
            "rows": [{"工号": "X001", "姓名": "张三"}],
        })
        r = client.get("/api/employees", headers=mgr)
        assert r.status_code == 200
        assert len(r.json()) == 1
        assert r.json()[0]["name"] == "张三"

    def test_trainee_sees_only_self(self, client, mgr, tra) -> None:
        """★ 学员只能看到自己。"""
        client.post("/api/employees/import", headers=mgr, json={
            "rows": [
                {"工号": "E001", "姓名": "张三"},
                {"工号": "E002", "姓名": "王五"},
            ],
        })
        r = client.get("/api/employees", headers=tra)
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 1
        assert rows[0]["employee_no"] == "E001"


class TestEmployeeActive:
    def _import(self, client, mgr) -> None:
        r = client.post("/api/employees/import", headers=mgr, json={
            "rows": [
                {"工号": "E001", "姓名": "张三"},
                {"工号": "E002", "姓名": "王五"},
            ],
        })
        assert r.status_code == 200

    def test_manager_deactivates_and_filter_works(self, client, mgr) -> None:
        self._import(client, mgr)
        r = client.post("/api/employees/active", headers=mgr,
                        json={"employee_no": "E001", "active": False})
        assert r.status_code == 200, r.text
        assert r.json()["active"] is False

        active = client.get("/api/employees", headers=mgr).json()
        assert [e["employee_no"] for e in active] == ["E002"]

        all_rows = client.get("/api/employees?active_only=false",
                              headers=mgr).json()
        by_no = {e["employee_no"]: e for e in all_rows}
        assert by_no["E001"]["active"] is False
        assert by_no["E002"]["active"] is True

    def test_trainee_cannot_deactivate(self, client, tra) -> None:
        r = client.post("/api/employees/active", headers=tra,
                        json={"employee_no": "E001", "active": False})
        assert r.status_code == 403

    def test_unknown_employee_404(self, client, mgr) -> None:
        r = client.post("/api/employees/active", headers=mgr,
                        json={"employee_no": "X999", "active": False})
        assert r.status_code == 404

    def test_cannot_deactivate_self(self, client, mgr) -> None:
        client.post("/api/employees/import", headers=mgr, json={
            "rows": [{"工号": "M001", "姓名": "主管"}],
        })
        r = client.post("/api/employees/active", headers=mgr,
                        json={"employee_no": "M001", "active": False})
        assert r.status_code == 400


# ===================================================================
# 2. ★ 日报：草稿可改 / 提交后冻结
# ===================================================================
class TestDailySummary:
    def test_save_draft(self, client, tra) -> None:
        r = client.post("/api/summaries", headers=tra, json={
            "summary_date": "2026-10-23",
            "content": "今天学习了鱼油的功能边界",
        })
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "draft"
        assert d["can_edit"] is True
        assert d["trainee_no"] == "E001", "写入的应是自己的工号"

    def test_draft_editable_multiple_times(self, client, tra) -> None:
        for text in ("第一版", "第二版", "第三版"):
            r = client.post("/api/summaries", headers=tra, json={
                "summary_date": "2026-10-23", "content": text,
            })
            assert r.status_code == 200
            assert r.json()["content"] == text

    def test_submit_freezes(self, client, tra) -> None:
        """★★ 提交后不可修改（FR-045）。"""
        client.post("/api/summaries", headers=tra, json={
            "summary_date": "2026-10-23",
            "content": "内容", "submit": True,
        })
        r = client.get("/api/summaries?start_date=2026-10-23"
                       "&end_date=2026-10-23", headers=tra)
        assert r.json()[0]["status"] == "submitted"
        assert r.json()[0]["can_edit"] is False

    def test_cannot_edit_after_submit(self, client, tra) -> None:
        """★ 已提交再改 → 409（状态不允许，不是请求格式错）。"""
        client.post("/api/summaries", headers=tra, json={
            "summary_date": "2026-10-23",
            "content": "内容", "submit": True,
        })
        r = client.post("/api/summaries", headers=tra, json={
            "summary_date": "2026-10-23", "content": "想偷偷改",
        })
        assert r.status_code == 409

    def test_one_per_day(self, client, tra) -> None:
        """★ 每人每天只有一条：同一天多次保存不会产生多条。"""
        for text in ("A", "B", "C"):
            client.post("/api/summaries", headers=tra, json={
                "summary_date": "2026-10-23", "content": text,
            })
        r = client.get("/api/summaries?start_date=2026-10-23"
                       "&end_date=2026-10-23", headers=tra)
        assert len(r.json()) == 1, "同一天出现了多条日报"

    def test_different_day_separate(self, client, tra) -> None:
        for d in ("2026-10-22", "2026-10-23"):
            client.post("/api/summaries", headers=tra, json={
                "summary_date": d, "content": d,
            })
        r = client.get("/api/summaries", headers=tra)
        assert len(r.json()) == 2


# ===================================================================
# 3. ★ 数据权限（身份来自令牌）
# ===================================================================
class TestSummaryPermission:
    def _two_summaries(self, client, tra, tra2) -> None:
        for h, who in ((tra, "E001"), (tra2, "E002")):
            client.post("/api/summaries", headers=h, json={
                "summary_date": "2026-10-23", "content": f"{who} 的内容",
            })

    def test_trainee_sees_only_own(self, client, tra, tra2) -> None:
        """★ 学员只能看自己的日报。"""
        self._two_summaries(client, tra, tra2)
        r = client.get("/api/summaries", headers=tra)
        rows = r.json()
        assert len(rows) == 1
        assert rows[0]["trainee_no"] == "E001"

    def test_trainee_cannot_peek_by_param(self, client, tra, tra2) -> None:
        """★★ 学员即使在参数里指定别人，也看不到。

        关键：不靠前端自觉，服务端强制覆盖。
        """
        self._two_summaries(client, tra, tra2)
        r = client.get("/api/summaries?trainee_no=E002", headers=tra)
        rows = r.json()
        assert len(rows) == 1
        assert rows[0]["trainee_no"] == "E001", "学员通过参数看到了他人日报"

    def test_manager_sees_all(self, client, mgr, tra, tra2) -> None:
        self._two_summaries(client, tra, tra2)
        r = client.get("/api/summaries", headers=mgr)
        assert len(r.json()) >= 2


# ===================================================================
# 4. ★ 主管评语
# ===================================================================
class TestReview:
    def _submit_one(self, client, tra) -> None:
        client.post("/api/summaries", headers=tra, json={
            "summary_date": "2026-10-23",
            "content": "学员内容", "submit": True,
        })

    def test_manager_can_review(self, client, mgr, tra) -> None:
        self._submit_one(client, tra)
        r = client.post("/api/summaries/review", headers=mgr, json={
            "trainee_no": "E001", "summary_date": "2026-10-23",
            "note": "注意区分保健食品和普通食品",
        })
        assert r.status_code == 200
        assert "保健食品" in r.json()["review_note"]

    def test_review_modifiable(self, client, mgr, tra) -> None:
        """FR-046：评语可修改。"""
        self._submit_one(client, tra)
        for note in ("初版", "改后"):
            r = client.post("/api/summaries/review", headers=mgr, json={
                "trainee_no": "E001", "summary_date": "2026-10-23",
                "note": note,
            })
            assert r.json()["review_note"] == note

    def test_trainee_cannot_review(self, client, tra) -> None:
        """★★ 学员不能写评语（含给自己写）。"""
        self._submit_one(client, tra)
        r = client.post("/api/summaries/review", headers=tra, json={
            "trainee_no": "E001", "summary_date": "2026-10-23",
            "note": "自己给自己写",
        })
        assert r.status_code == 403

    def test_review_does_not_change_content(self, client, mgr, tra) -> None:
        """★★ 主管写评语不能改学员正文。"""
        self._submit_one(client, tra)
        client.post("/api/summaries/review", headers=mgr, json={
            "trainee_no": "E001", "summary_date": "2026-10-23",
            "note": "评语",
        })
        r = client.get("/api/summaries?trainee_no=E001", headers=mgr)
        assert r.json()[0]["content"] == "学员内容", "主管评语把学员正文改了"

    def test_review_missing_summary_404(self, client, mgr) -> None:
        r = client.post("/api/summaries/review", headers=mgr, json={
            "trainee_no": "E999", "summary_date": "2026-10-23",
            "note": "评语",
        })
        assert r.status_code == 404


# ===================================================================
# 5. ★ 未命中日志：不得用于绩效
# ===================================================================
class TestMissLogs:
    def test_trainee_forbidden(self, client, tra) -> None:
        r = client.get("/api/miss-logs", headers=tra)
        assert r.status_code == 403

    def test_response_has_no_trainee_dimension(self, client, mgr) -> None:
        """★★ FR-047：不得用于绩效评价。

        返回结构里**不能有按学员的聚合**，否则很容易被做成「谁问得多」。
        """
        r = client.get("/api/miss-logs", headers=mgr)
        assert r.status_code == 200
        d = r.json()

        for item in d["recent"]:
            assert "trainee_id" not in item, (
                f"未命中日志暴露了学员身份: {item}"
            )
        for item2 in d["top"]:
            assert "trainee_id" not in item2
            assert "query" in item2
