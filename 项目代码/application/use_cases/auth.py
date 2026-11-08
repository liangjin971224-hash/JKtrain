"""认证用例（P8）。

★ 一个安全细节：**报错不能区分「用户不存在」和「密码错误」**

    如果用户不存在时报「工号不存在」、密码错时报「密码不正确」，
    就等于提供了一个**账号枚举接口** —— 攻击者可以批量探测哪些工号有效。

    所以两种情况统一返回：「工号或密码不正确」。

★ 另一个细节：**用户不存在时也要跑一次哈希校验**

    否则「不存在」会立即返回，「密码错」要等 250ms（bcrypt），
    通过响应时间就能区分账号是否存在（时序侧信道）。

    做法：用户不存在时，对一个固定的假哈希跑一次 verify，
    让两条路径耗时接近。
"""
from __future__ import annotations

import logging

from domain.exceptions import AuthenticationError
from domain.value_objects.account import AuditAction, AuditLogEntry, AuditResult
from domain.ports.auth_port import PasswordHasherPort, TokenSignerPort
from domain.value_objects.role import Role

logger = logging.getLogger(__name__)

#: 统一的登录失败话术（不能泄露是哪一步错了）。
_LOGIN_FAILED_MSG = "工号或密码不正确"

#: 用户不存在时用来「陪跑」的假哈希。
#:
#: 目的是让「不存在」和「密码错」两条路径耗时接近，
#: 消除时序侧信道。这个哈希对应一个没人知道的随机密码。
_DUMMY_HASH = (
    "$2b$12$C6UzMDM.H6dfI/f/IKcEe.7RWYVfVQqM8Z1Z1Z1Z1Z1Z1Z1Z1Z1Z1"
)


class LoginResult:
    """登录结果。"""

    def __init__(self, *, token: str, employee_no: str, role: str) -> None:
        self.token = token
        self.employee_no = employee_no
        self.role = role

    @property
    def role_label(self) -> str:
        return Role.parse(self.role).label


class LoginUseCase:
    """登录。"""

    def __init__(
        self,
        *,
        repository,
        signer: TokenSignerPort,
        hasher: PasswordHasherPort | None = None,
    ) -> None:
        """
        Args:
            repository: 需实现 `get_account(employee_no)`。
            signer: 令牌签发器。
            hasher: 密码校验器。为空时用默认 bcrypt 实现。

        ★ 依赖端口而不是具体实现：用例层不该知道用的是 bcrypt 还是 argon2。
        """
        self._repo = repository
        self._signer = signer
        # ★ 不在这里 import 具体实现（门禁禁止用例层依赖 infrastructure）。
        #   由装配根注入；未注入时拒绝构造，避免悄悄用了默认实现。
        if hasher is None:
            raise ValueError(
                "必须注入 hasher（PasswordHasherPort）。"
                "默认实现由装配根提供。")
        self._hasher = hasher

    def execute(self, employee_no: str, password: str) -> LoginResult:
        """校验凭据并签发令牌。

        Args:
            employee_no: 工号。
            password: 明文密码。

        Returns:
            登录结果（含令牌）。

        Raises:
            AuthenticationError: 凭据错误 / 账号停用。
        """
        emp_no = (employee_no or "").strip()

        account = None
        if emp_no and hasattr(self._repo, "get_account"):
            try:
                account = self._repo.get_account(emp_no)
            except Exception as exc:              # noqa: BLE001
                logger.error("查询账号失败: %s", exc)
                account = None

        # ★ 用户不存在也要跑一次哈希校验（消除时序差异）
        if account is None:
            self._hasher.verify(password or "", _DUMMY_HASH)
            self._audit_failed(emp_no)
            raise AuthenticationError(_LOGIN_FAILED_MSG)

        if not self._hasher.verify(password or "", account.password_hash):
            self._audit_failed(emp_no)
            raise AuthenticationError(_LOGIN_FAILED_MSG)

        if not account.active:
            # ★ 停用账号也报同样的话术 —— 否则等于告诉对方
            #   「这个工号存在，只是被停用了」
            self._audit_failed(emp_no)
            raise AuthenticationError(_LOGIN_FAILED_MSG)

        role = account.role

        token = self._signer.issue(
            employee_no=account.employee_no,
            role=role,
        )

        # 审计：登录成功
        self._audit(AuditLogEntry(
            actor=account.employee_no,
            role=role,
            action=AuditAction.LOGIN.value,
            result=AuditResult.OK.value,
        ))

        logger.info("登录成功: %s (%s)", account.employee_no, role)
        return LoginResult(
            token=token,
            employee_no=account.employee_no,
            role=role,
        )

    # ---------------- 内部 ----------------
    def _audit_failed(self, emp_no: str) -> None:
        """
        ★ 失败日志的 actor 记**输入的工号**，但 role 留空：
          此时不知道对方真实身份，写个角色反而误导排查。
          同时这也是「有人在试这个工号」的线索。
        """
        self._audit(AuditLogEntry(
            actor=emp_no or "(空)",
            role="",
            action=AuditAction.LOGIN_FAILED.value,
            result=AuditResult.FAILED.value,
        ))

    def _audit(self, entry: AuditLogEntry) -> None:
        """写审计日志（失败不影响登录主流程）。"""
        if not hasattr(self._repo, "append_audit_log"):
            return
        try:
            self._repo.append_audit_log(entry)
        except Exception as exc:                  # noqa: BLE001
            logger.warning("审计日志写入失败（不影响登录）: %s", exc)


class ResolveIdentityUseCase:
    """解析请求身份（每个请求一次）。

    ★★ FR-048「权限变更应即时生效」的实现点。

      令牌里虽然有 role，但那是**快照**。
      这里每次都回库取最新角色 —— 改库后下一个请求立即生效，
      不用等令牌过期。

      `role_is_snapshot` 标记就是提醒：不要图省事直接用 token 里的 role。
    """

    def __init__(self, *, repository, signer: TokenSignerPort) -> None:
        self._repo = repository
        self._signer = signer

    def execute(self, token: str) -> tuple[str, Role] | None:
        """从令牌解析出 (工号, 角色)。

        Args:
            token: 令牌串。

        Returns:
            (工号, 角色)；令牌无效或账号不存在/停用时返回 None。
        """
        payload = self._signer.verify(token)
        if not payload:
            return None

        emp_no = payload.get("sub") or ""

        # ★ 回库取最新角色（不信 token 里的快照）
        if hasattr(self._repo, "get_account"):
            try:
                account = self._repo.get_account(emp_no)
            except Exception:                     # noqa: BLE001
                return None

            if account is None or not account.active:
                return None

            try:
                return account.employee_no, Role.parse(account.role)
            except Exception:                     # noqa: BLE001
                return None

        # 仓储不支持账号查询时，退化为信 token（仅测试环境）
        try:
            return emp_no, Role.parse(payload.get("role") or "")
        except Exception:                         # noqa: BLE001
            return None


__all__ = ["LoginResult", "LoginUseCase", "ResolveIdentityUseCase"]
