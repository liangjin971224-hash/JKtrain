"""认证与账号管理路由（P8）。

    POST /api/auth/login      登录，返回令牌
    GET  /api/auth/me         当前身份
    GET  /api/accounts        账号列表（主管）
    POST /api/accounts/role   改角色（主管）
    POST /api/accounts/active 启用/停用（主管）
    GET  /api/audit-logs      审计日志（主管）
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from application.use_cases.auth import LoginUseCase
from config.assembly import Container
from domain.exceptions import AuthenticationError, PermissionDeniedError
from domain.value_objects.account import (
    AuditAction,
    AuditLogEntry,
    AuditResult,
    EmployeeAccount,
)
from interfaces.http.auth_deps import (
    Identity,
    current_identity,
    require_feature,
)
from interfaces.http.deps import current_settings, get_container
from interfaces.http.schemas.models import (
    AccountView,
    AuditLogView,
    ChangePasswordRequest,
    ChangeRoleRequest,
    CreateAccountRequest,
    LoginRequest,
    LoginResponse,
    ResetPasswordRequest,
    SetActiveRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["认证"])


@router.post("/auth/login", response_model=LoginResponse, summary="登录")
def login(
    req: LoginRequest,
    container: Container = Depends(get_container),
) -> LoginResponse:
    """工号 + 密码登录，返回 Bearer 令牌。

    ★ 失败统一返回「工号或密码不正确」——
      不区分「工号不存在」和「密码错」，避免账号枚举。
    """
    if container.repository is None:
        raise HTTPException(503, detail="仓储未就绪，无法登录")

    signer = getattr(container, "token_signer", None)
    hasher = getattr(container, "password_hasher", None)
    if signer is None or hasher is None:
        raise HTTPException(503, detail="认证能力未装配")

    uc = LoginUseCase(
        repository=container.repository,
        signer=signer,
        hasher=hasher,
    )

    try:
        result = uc.execute(req.employee_no, req.password)
    except AuthenticationError as exc:
        raise HTTPException(401, detail=str(exc)) from exc

    # 记录最后登录时间（失败不影响登录）
    try:
        container.repository.touch_last_login(result.employee_no)
    except Exception:                             # noqa: BLE001
        pass

    return LoginResponse(
        token=result.token,
        employee_no=result.employee_no,
        role=result.role,
        role_label=result.role_label,
    )


@router.get("/auth/me", summary="当前身份")
def me(identity: Identity = Depends(current_identity)) -> dict:
    """返回当前登录身份（前端用来判断显示哪些功能）。"""
    return {
        "employee_no": identity.employee_no,
        "role": identity.role.value,
        "role_label": identity.role.label,
        "is_manager": identity.is_manager,
    }


# ===================================================================
# 账号管理（主管）
# ===================================================================
@router.get("/accounts", response_model=list[AccountView],
            summary="账号列表")
def list_accounts(
    limit: int = 500,
    identity: Identity = Depends(require_feature("employee_view")),
    container: Container = Depends(get_container),
) -> list[AccountView]:
    """列账号（**不含密码哈希**）。"""
    repo = _repo(container)
    if not hasattr(repo, "list_accounts"):
        raise HTTPException(503, detail="仓储不支持账号查询")
    return [AccountView(**a) for a in repo.list_accounts(limit=limit)]


@router.post("/accounts/role", response_model=AccountView, summary="改角色")
def change_role(
    req: ChangeRoleRequest,
    identity: Identity = Depends(require_feature("employee_view")),
    container: Container = Depends(get_container),
) -> AccountView:
    """改角色。

    ★ FR-048「权限变更应即时生效」：
      改完**下一个请求立即生效**，因为身份解析每请求都回库取最新角色。
      不用等令牌过期。
    """
    repo = _repo(container)

    if req.role not in ("trainee", "manager"):
        raise HTTPException(400, detail=f"角色非法: {req.role}")

    try:
        repo.set_account_role(req.employee_no, req.role)
    except ValueError as exc:
        raise HTTPException(404, detail=str(exc)) from exc

    _audit(container, identity, AuditAction.ACCOUNT_ROLE, req.employee_no,
           detail={"changed_role_to": req.role})

    logger.info("角色变更: %s → %s by %s",
                req.employee_no, req.role, identity.employee_no)
    return _account_view(repo, req.employee_no)


@router.post("/accounts/active", response_model=AccountView,
             summary="启用/停用账号")
def set_active(
    req: SetActiveRequest,
    identity: Identity = Depends(require_feature("employee_view")),
    container: Container = Depends(get_container),
) -> AccountView:
    """启用/停用账号。

    ★ 停用后该员工**下一个请求就登不上**（身份解析会查 active）。
      这是「离职当天立即断权」的实现方式。
    """
    repo = _repo(container)

    if req.employee_no == identity.employee_no and not req.active:
        raise HTTPException(
            400, detail="不能停用自己的账号（会把你自己锁在系统外）")

    try:
        repo.set_account_active(req.employee_no, req.active)
    except ValueError as exc:
        raise HTTPException(404, detail=str(exc)) from exc

    _audit(container, identity, AuditAction.ACCOUNT_ACTIVE, req.employee_no,
           detail={"active": req.active})

    logger.info("账号%s: %s by %s",
                "启用" if req.active else "停用",
                req.employee_no, identity.employee_no)
    return _account_view(repo, req.employee_no)


@router.post("/accounts/create", response_model=AccountView,
             summary="开通账号（主管）")
def create_account(
    req: CreateAccountRequest,
    identity: Identity = Depends(require_feature("employee_view")),
    container: Container = Depends(get_container),
) -> AccountView:
    """为已导入的员工开通登录账号。

    ★ 导入只建档案、不自动发默认密码；账号由主管在这里显式开通，
    避免「一批默认密码」的安全窗口期。
    """
    repo = _repo(container)
    if not (hasattr(repo, "save_account") and hasattr(repo, "get_account")
            and hasattr(repo, "get_employee")):
        raise HTTPException(503, detail="仓储不支持账号开通")
    hasher = getattr(container, "password_hasher", None)
    if hasher is None:
        raise HTTPException(503, detail="密码服务未就绪")

    if req.role not in ("trainee", "manager"):
        raise HTTPException(400, detail=f"角色非法: {req.role}")
    if repo.get_employee(req.employee_no) is None:
        raise HTTPException(404, detail=f"员工档案不存在，请先导入: {req.employee_no}")
    if repo.get_account(req.employee_no) is not None:
        raise HTTPException(409, detail=f"账号已存在: {req.employee_no}")

    password = req.password.strip()
    if len(password) < 6:
        raise HTTPException(400, detail="密码至少 6 位")

    repo.save_account(EmployeeAccount(
        employee_no=req.employee_no,
        role=req.role,
        password_hash=hasher.hash(password),
        active=True,
    ))
    _audit(container, identity, AuditAction.ACCOUNT_CREATE, req.employee_no,
           detail={"role": req.role})
    logger.info("开通账号: %s (%s) by %s",
                req.employee_no, req.role, identity.employee_no)
    return _account_view(repo, req.employee_no)


@router.post("/accounts/reset-password", response_model=AccountView,
             summary="重置密码（主管）")
def reset_password(
    req: ResetPasswordRequest,
    identity: Identity = Depends(require_feature("employee_view")),
    container: Container = Depends(get_container),
) -> AccountView:
    """主管重置某员工密码。

    ★ 不返回、不记录明文或哈希；审计只记「重置」这个动作。
    """
    repo = _repo(container)
    if not hasattr(repo, "set_account_password") or not hasattr(repo, "get_account"):
        raise HTTPException(503, detail="仓储不支持密码重置")
    hasher = getattr(container, "password_hasher", None)
    if hasher is None:
        raise HTTPException(503, detail="密码服务未就绪")

    account = repo.get_account(req.employee_no)
    if account is None:
        raise HTTPException(404, detail=f"账号不存在: {req.employee_no}")

    new_password = req.new_password.strip()
    if hasher.verify(new_password, account.password_hash):
        raise HTTPException(400, detail="新密码不能与旧密码相同")

    repo.set_account_password(req.employee_no, hasher.hash(new_password))
    _audit(container, identity, AuditAction.ACCOUNT_PASSWORD, req.employee_no,
           detail={"mode": "reset"})
    logger.info("重置密码: %s by %s", req.employee_no, identity.employee_no)
    return _account_view(repo, req.employee_no)


@router.post("/auth/change-password", summary="修改自己的密码")
def change_password(
    req: ChangePasswordRequest,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> dict:
    """员工修改自己的密码（需校验旧密码）。"""
    repo = _repo(container)
    if not hasattr(repo, "set_account_password") or not hasattr(repo, "get_account"):
        raise HTTPException(503, detail="仓储不支持密码修改")
    hasher = getattr(container, "password_hasher", None)
    if hasher is None:
        raise HTTPException(503, detail="密码服务未就绪")

    account = repo.get_account(identity.employee_no)
    if account is None:
        raise HTTPException(401, detail="账号不存在")

    if not hasher.verify(req.old_password, account.password_hash):
        raise HTTPException(400, detail="旧密码不正确")

    new_password = req.new_password.strip()
    if hasher.verify(new_password, account.password_hash):
        raise HTTPException(400, detail="新密码不能与旧密码相同")

    repo.set_account_password(identity.employee_no, hasher.hash(new_password))
    _audit(container, identity, AuditAction.ACCOUNT_PASSWORD, identity.employee_no,
           detail={"mode": "self"})
    logger.info("修改密码: %s", identity.employee_no)
    return {"ok": True}


# ===================================================================
# 审计日志
# ===================================================================
@router.get("/audit-logs", response_model=list[AuditLogView],
            summary="审计日志")
def list_audit_logs(
    actor: str = "",
    action: str = "",
    limit: int = 100,
    identity: Identity = Depends(require_feature("audit_log_view")),
    container: Container = Depends(get_container),
) -> list[AuditLogView]:
    """查审计日志（主管）。

    ★ 只有主管能看 —— 日志里含全部人的操作记录。
    """
    repo = _repo(container)
    if not hasattr(repo, "list_audit_logs"):
        raise HTTPException(503, detail="仓储不支持审计日志查询")

    return [AuditLogView(**x)
            for x in repo.list_audit_logs(
                actor=actor, action=action, limit=limit)]


# ===================================================================
# 内部
# ===================================================================
def _repo(container: Container):
    if container.repository is None:
        raise HTTPException(503, detail="仓储未就绪")
    return container.repository


def _account_view(repo, employee_no: str) -> AccountView:
    accounts = repo.list_accounts(limit=1000)
    for a in accounts:
        if a["employee_no"] == employee_no:
            return AccountView(**a)
    raise HTTPException(404, detail=f"账号不存在: {employee_no}")


def _audit(container: Container, identity: Identity,
           action: AuditAction, target: str, *, detail: dict | None = None) -> None:
    """写审计日志（失败不影响主流程）。"""
    repo = container.repository
    if repo is None or not hasattr(repo, "append_audit_log"):
        return
    try:
        repo.append_audit_log(AuditLogEntry(
            actor=identity.employee_no,
            role=identity.role.value,
            action=action.value,
            target=target,
            result=AuditResult.OK.value,
            detail=detail or {},
        ))
    except Exception as exc:                      # noqa: BLE001
        logger.warning("审计日志写入失败: %s", exc)
