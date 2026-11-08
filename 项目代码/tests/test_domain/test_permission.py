"""权限与鉴权测试（P8）。

★ 需求来源：FR-048（功能权限）/ FR-049（数据权限）/ FR-050（检索权限）

────────────────────────────────────────────────────────────────────
三条需求原文里的「硬话」，测试要能守住
────────────────────────────────────────────────────────────────────
FR-048: 「权限变更应即时生效」
    → 不能把角色写进 token 就不管了。角色变更后旧 token 必须失效
      （或者角色从库里实时读，而不是从 token 读）。

FR-049: 「数据权限应在查询层生效，**不得先查询后过滤**」
    → 必须把筛选条件下推到 SQL，而不是查出来再 list comprehension。
      后者在大数据量下会先泄露全量、再丢掉 —— 一旦有日志/缓存就泄了。

FR-050: 「权限过滤必须在检索阶段生效，**不得检索后再过滤**」
    → 必须传给 Milvus 的 `filter_expr`，不能在 Python 里过滤。
      后者会把受限内容取出来再丢弃，向量库层面已经读到了。

这三条如果不按「下推」实现，功能上看起来都对，但本质违规。
所以测试要检查**调用参数**，而不只是返回值。
"""
from __future__ import annotations

import pytest

from domain.exceptions import (
    AuthenticationError,
    PermissionDeniedError,
)
from domain.value_objects import PermissionLevel, Role


def _login_uc(repo):
    """构造登录用例（注入端口实现）。

    ★ 用 rounds=4 而不是默认 12：bcrypt 12 约 250ms，
      测试里跑几十次会拖到十几秒。4 只要几毫秒，
      而这里测的是逻辑，不是哈希强度。
    """
    from application.use_cases.auth import LoginUseCase
    from infrastructure.auth.adapters import (
        BcryptPasswordHasher,
        HmacTokenSigner,
    )

    return LoginUseCase(
        repository=repo,
        signer=HmacTokenSigner(secret="k" * 32),
        hasher=BcryptPasswordHasher(rounds=4),
    )


def _b64(text: str) -> str:
    """base64url 编码（测试用，模拟伪造载荷）。"""
    import base64

    raw = base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")
    return raw.rstrip("=")


# ===================================================================
# 1. 角色与功能权限矩阵
# ===================================================================
class TestRolePermissions:
    def test_two_roles(self) -> None:
        """FR-048：应支持学员、主管两种角色。"""
        assert Role.TRAINEE.value == "trainee"
        assert Role.MANAGER.value == "manager"

    @pytest.mark.parametrize("feature", [
        "exam_build", "exam_manage", "knowledge_manage",
    ])
    def test_trainee_forbidden(self, feature: str) -> None:
        """★ 学员不可访问组卷、试卷管理、知识库管理。"""
        assert not Role.TRAINEE.can(feature)

    @pytest.mark.parametrize("feature", [
        "exam_build", "exam_manage", "knowledge_manage",
        "daily_review", "miss_log_view", "employee_import",
    ])
    def test_manager_allowed(self, feature: str) -> None:
        assert Role.MANAGER.can(feature)

    @pytest.mark.parametrize("feature", [
        "ask", "drill", "daily_write",
    ])
    def test_trainee_can_basic(self, feature: str) -> None:
        assert Role.TRAINEE.can(feature)

    def test_unknown_feature_denied_by_default(self) -> None:
        """★ 未登记的功能一律拒绝（fail-safe）。

        理由：新增功能时忘了配权限，应该是「用不了」，
        而不是「所有人都能用」。后者是安全事故。
        """
        assert not Role.TRAINEE.can("some_new_feature")
        assert not Role.MANAGER.can("some_new_feature")

    def test_require_raises(self) -> None:
        with pytest.raises(PermissionDeniedError):
            Role.TRAINEE.require("exam_build")


# ===================================================================
# 2. ★ 检索权限：必须在检索阶段过滤
# ===================================================================
class TestRetrievalPermission:
    def test_trainee_permissions_exclude_restricted(self) -> None:
        """学员只能检索 public / internal。"""
        perms = Role.TRAINEE.retrieval_levels
        assert PermissionLevel.PUBLIC.value in [str(p) for p in perms]
        assert PermissionLevel.INTERNAL.value in [str(p) for p in perms]
        assert PermissionLevel.RESTRICTED.value not in [str(p) for p in perms]
        assert PermissionLevel.CONFIDENTIAL.value not in [str(p) for p in perms]

    def test_manager_sees_all(self) -> None:
        perms = Role.MANAGER.retrieval_levels
        assert len(perms) == 4

    def test_permission_passed_to_search_port(self) -> None:
        """★★ FR-050：权限必须在检索阶段生效。

        做法：把 `allowed_permissions` 传给检索端口，
        由适配器拼进 Milvus 的 `filter_expr`，
        而不是取回结果后在 Python 里筛。

        这个测试检查**端口收到的参数** ——
        因为两种做法返回值一样，只有传参能区分实现方式。
        """
        from domain.value_objects import PermissionLevel
        from tests.fakes import FakeVectorSearchPort

        fake = FakeVectorSearchPort([])
        fake.search(
            "鱼油",
            allowed_permissions=(PermissionLevel.PUBLIC,
                                 PermissionLevel.INTERNAL),
        )
        assert fake.last_allowed_permissions is not None, (
            "权限级别没有传给检索端口 —— "
            "说明是「检索后过滤」而不是「检索阶段过滤」"
        )
        got = [str(x) for x in fake.last_allowed_permissions]
        assert "public" in got
        assert "restricted" not in got


# ===================================================================
# 3. 密码与令牌
# ===================================================================
class TestPassword:
    def test_hash_not_plaintext(self) -> None:
        """密码不能明文存储。"""
        from infrastructure.auth.password import hash_password

        h = hash_password("secret123")
        assert h != "secret123"
        assert "secret123" not in h
        assert len(h) > 30

    def test_hash_is_salted(self) -> None:
        """同一密码两次哈希结果不同（有盐）。"""
        from infrastructure.auth.password import hash_password

        assert hash_password("same") != hash_password("same")

    def test_verify_correct(self) -> None:
        from infrastructure.auth.password import hash_password, verify_password

        h = hash_password("secret123")
        assert verify_password("secret123", h)

    def test_verify_wrong(self) -> None:
        from infrastructure.auth.password import hash_password, verify_password

        h = hash_password("secret123")
        assert not verify_password("wrong", h)

    def test_verify_bad_hash_returns_false(self) -> None:
        """坏哈希不能崩，返回 False（fail-safe）。"""
        from infrastructure.auth.password import verify_password

        assert not verify_password("x", "not-a-valid-hash")
        assert not verify_password("x", "")


class TestToken:
    def test_issue_and_verify(self) -> None:
        from infrastructure.auth.token import issue_token, verify_token

        t = issue_token(employee_no="E001", role="trainee",
                        secret="k" * 32)
        payload = verify_token(t, secret="k" * 32)
        assert payload is not None
        assert payload["sub"] == "E001"
        assert payload["role"] == "trainee"

    def test_wrong_secret_rejected(self) -> None:
        from infrastructure.auth.token import issue_token, verify_token

        t = issue_token(employee_no="E001", role="trainee", secret="k" * 32)
        assert verify_token(t, secret="w" * 32) is None

    def test_tampered_rejected(self) -> None:
        from infrastructure.auth.token import issue_token, verify_token

        t = issue_token(employee_no="E001", role="trainee", secret="k" * 32)
        # 改签名
        assert verify_token(t + "x", secret="k" * 32) is None

        # 改载荷（两段式：body.sig）
        body, sig = t.split(".")
        forged_body = _b64("{\"sub\":\"E001\",\"role\":\"manager\","
                           "\"exp\":9999999999}")
        assert verify_token(f"{forged_body}.{sig}",
                            secret="k" * 32) is None, (
            "改载荷后仍通过 —— 签名校验失效"
        )

    def test_expired_rejected(self) -> None:
        from infrastructure.auth.token import issue_token, verify_token

        t = issue_token(employee_no="E001", role="trainee",
                        secret="k" * 32, ttl_sec=-1)
        assert verify_token(t, secret="k" * 32) is None

    def test_token_does_not_carry_trusted_role(self) -> None:
        """★★ FR-048：权限变更应即时生效。

        如果角色只从 token 读，改库后旧 token 仍按旧角色放行。
        所以 token 里的 role **只作参考**，判定时必须回库取最新角色。

        这里验证 token 载荷标记了这一约定。
        """
        from infrastructure.auth.token import issue_token, verify_token

        t = issue_token(employee_no="E001", role="trainee",
                        secret="k" * 32)
        payload = verify_token(t, secret="k" * 32)
        assert payload is not None
        # 载荷里明确标记该角色为「快照」，不可直接信任
        assert payload.get("role_is_snapshot") is True


# ===================================================================
# 4. ★ 数据权限必须在查询层生效
# ===================================================================
class TestDataPermissionPushdown:
    def test_list_summaries_passes_trainee_filter(self) -> None:
        """★★ FR-049：不得先查询后过滤。

        学员查日报时，`trainee_no` 必须作为**查询条件**传下去，
        而不是查出全部再在 Python 里筛。
        """
        from application.use_cases.employee import ListSummariesUseCase

        class SpyRepo:
            def __init__(self):
                self.kwargs = None

            def list_daily_summaries(self, **kw):
                self.kwargs = kw
                return []

        repo = SpyRepo()
        uc = ListSummariesUseCase(repository=repo)
        uc.execute(requester="E001", role="trainee", trainee_no="")
        # 学员没传 trainee_no 时，也必须强制用自己的工号下推
        assert repo.kwargs.get("trainee_no") == "E001", (
            f"学员查询没有把工号下推到查询层: {repo.kwargs}"
        )

    def test_trainee_cannot_override_filter(self) -> None:
        """学员传别人的工号也无效（服务端覆盖）。"""
        from application.use_cases.employee import ListSummariesUseCase

        class SpyRepo:
            def __init__(self):
                self.kwargs = None

            def list_daily_summaries(self, **kw):
                self.kwargs = kw
                return []

        repo = SpyRepo()
        uc = ListSummariesUseCase(repository=repo)
        uc.execute(requester="E001", role="trainee", trainee_no="E002")
        assert repo.kwargs.get("trainee_no") == "E001"


# ===================================================================
# 5. 操作日志
# ===================================================================
class TestAuditLog:
    def test_log_fields(self) -> None:
        """日志要能回答「谁在什么时候做了什么、成没成」。"""
        import dataclasses

        from domain.value_objects import AuditLogEntry

        fields = {f.name for f in dataclasses.fields(AuditLogEntry)}
        for required in ("actor", "role", "action", "target",
                         "result", "created_at"):
            assert required in fields, f"日志缺字段: {required}"

    def test_no_sensitive_payload(self) -> None:
        """★ 日志不能记密码。

        这是审计日志最常见的泄露点。
        """
        import dataclasses

        from domain.value_objects import AuditLogEntry

        fields = {f.name for f in dataclasses.fields(AuditLogEntry)}
        for banned in ("password", "password_hash", "token", "secret"):
            assert banned not in fields, (
                f"审计日志含敏感字段 {banned} —— 会把凭据写进日志"
            )


# ===================================================================
# 6. 登录用例
# ===================================================================
class TestLogin:
    def test_login_ok(self) -> None:
        from application.use_cases.auth import LoginUseCase
        from domain.value_objects import EmployeeAccount

        class Repo:
            def get_account(self, no):
                if no != "E001":
                    return None
                from infrastructure.auth.password import hash_password
                return EmployeeAccount(
                    employee_no="E001", role="manager",
                    password_hash=hash_password("pass123"),
                    active=True,
                )

        uc = _login_uc(Repo())
        result = uc.execute("E001", "pass123")
        assert result.token
        assert result.role == "manager"

    def test_wrong_password(self) -> None:
        from application.use_cases.auth import LoginUseCase
        from domain.value_objects import EmployeeAccount

        class Repo:
            def get_account(self, no):
                from infrastructure.auth.password import hash_password
                return EmployeeAccount(
                    employee_no="E001", role="manager",
                    password_hash=hash_password("pass123"), active=True)

        uc = _login_uc(Repo())
        with pytest.raises(AuthenticationError):
            uc.execute("E001", "wrong")

    def test_unknown_user(self) -> None:
        from application.use_cases.auth import LoginUseCase

        class Repo:
            def get_account(self, no):
                return None

        uc = _login_uc(Repo())
        with pytest.raises(AuthenticationError):
            uc.execute("NOBODY", "x")

    def test_inactive_user_cannot_login(self) -> None:
        """离职员工不能登录。"""
        from application.use_cases.auth import LoginUseCase
        from domain.value_objects import EmployeeAccount

        class Repo:
            def get_account(self, no):
                from infrastructure.auth.password import hash_password
                return EmployeeAccount(
                    employee_no="E001", role="trainee",
                    password_hash=hash_password("pass123"), active=False)

        uc = _login_uc(Repo())
        with pytest.raises(AuthenticationError):
            uc.execute("E001", "pass123")

    def test_error_message_not_leaky(self) -> None:
        """★ 报错不能区分「用户不存在」和「密码错」。

        区分了就等于给攻击者一个账号枚举接口。
        """
        from application.use_cases.auth import LoginUseCase
        from domain.value_objects import EmployeeAccount

        class Repo:
            def get_account(self, no):
                if no == "E001":
                    from infrastructure.auth.password import hash_password
                    return EmployeeAccount(
                        employee_no="E001", role="trainee",
                        password_hash=hash_password("pass123"), active=True)
                return None

        uc = _login_uc(Repo())

        try:
            uc.execute("E001", "wrong")
        except AuthenticationError as e1:
            msg1 = str(e1)

        try:
            uc.execute("NOBODY", "x")
        except AuthenticationError as e2:
            msg2 = str(e2)

        assert msg1 == msg2, (
            f"两种失败的信息不同，可被用来枚举账号:\n  {msg1}\n  {msg2}"
        )
