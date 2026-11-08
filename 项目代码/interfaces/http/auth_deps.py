"""接口层鉴权依赖（P8）。

★ 用法：

    @router.post("/exam/build", dependencies=[Depends(require_feature("exam_build"))])
    def build(identity: Identity = Depends(current_identity)): ...

────────────────────────────────────────────────────────────────────
身份从哪来
────────────────────────────────────────────────────────────────────
    `Authorization: Bearer <token>`
    → 校验签名与过期
    → **回库取最新角色**（不信 token 里的快照，满足 FR-048「即时生效」）
    → 得到 `Identity(employee_no, role)`

────────────────────────────────────────────────────────────────────
为什么要「回库取最新角色」
────────────────────────────────────────────────────────────────────
如果直接用 token 里的 role：
    · 学员升主管 → 要等 8 小时 token 过期才生效
    · 主管降学员 → 这 8 小时内他**仍是主管**，能继续组卷、导出试卷

后者是真实的安全问题（离职/降职当天仍然有权限）。
所以每个请求回库查一次。代价是每请求一次 DB 查询，
对培训系统这点并发完全可接受。

────────────────────────────────────────────────────────────────────
★ 一个兼容过渡：允许无 token 的「开发模式」
────────────────────────────────────────────────────────────────────
`settings.auth_required` 为 False 时，未带 token 的请求按
`X-Debug-Role` 头（默认 manager）处理，便于本地调试和已有的接口测试。

生产环境必须把 `AUTH_REQUIRED=true`。
启动自检会提示当前模式。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request

from application.use_cases.auth import ResolveIdentityUseCase
from config.assembly import Container
from domain.exceptions import PermissionDeniedError
from domain.value_objects.role import Role
from interfaces.http.deps import current_settings, get_container

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Identity:
    """当前请求的身份。"""

    employee_no: str
    role: Role

    @property
    def is_manager(self) -> bool:
        return self.role.is_manager

    def require(self, feature: str) -> None:
        """检查功能权限。

        Raises:
            PermissionDeniedError: 无权限。
        """
        self.role.require(feature)


def _extract_token(authorization: str) -> str:
    """从 Authorization 头提取 token。"""
    if not authorization:
        return ""
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return authorization.strip()


def current_identity(
    authorization: str = Header(default=""),
    x_debug_role: str = Header(default=""),
    container: Container = Depends(get_container),
) -> Identity:
    """解析当前身份（依赖注入用）。

    Args:
        authorization: `Bearer <token>`。
        x_debug_role: 开发模式下的角色覆盖（仅 auth_required=False 时有效）。

    Returns:
        当前身份。

    Raises:
        HTTPException: 401 未登录；403 角色非法。
    """
    # ★ 配置从**容器**取，不用 current_settings()。
    #
    #   原因（实测踩坑）：current_settings() 每次都 new Settings()，
    #   读的是进程环境变量 —— 测试里注入的容器配置被完全忽略，
    #   导致 auth_required 永远是环境里的值（默认 True），
    #   所有接口测试都 401。
    #
    #   容器里的 settings 是装配时定下的，测试注入什么就用什么。
    settings = getattr(container, "settings", None) or current_settings()
    token = _extract_token(authorization)
    auth_required = getattr(settings, "auth_required", True)

    if not token:
        if auth_required:
            raise HTTPException(
                status_code=401,
                detail="未登录。请在 Authorization 头提供 Bearer 令牌。",
            )
        # 开发模式：按 X-Debug-Role 处理（默认主管，便于调试全部功能）
        try:
            role = Role.parse(x_debug_role or "manager")
        except PermissionDeniedError as exc:
            raise HTTPException(403, detail=str(exc)) from exc
        logger.debug("开发模式：未带令牌，按 %s 处理", role.label)
        return Identity(employee_no="(dev)", role=role)

    repo = container.repository

    if repo is None:
        raise HTTPException(503, detail="仓储未就绪，无法校验身份")

    signer = getattr(container, "token_signer", None)
    if signer is None:
        raise HTTPException(503, detail="认证能力未装配（token_signer）")

    uc = ResolveIdentityUseCase(repository=repo, signer=signer)
    resolved = uc.execute(token)

    if resolved is None:
        raise HTTPException(
            status_code=401,
            detail="登录已失效，请重新登录。",
        )

    emp_no, role = resolved
    return Identity(employee_no=emp_no, role=role)


def require_feature(feature: str):
    """生成一个「要求某功能权限」的依赖。

    Args:
        feature: 功能标识（见 `domain/value_objects/role.py` 的矩阵）。

    Returns:
        依赖函数。

    ★ 用工厂函数而不是在每个路由里手写 if：
      手写容易漏，而且漏了也不报错。
      挂成 `dependencies=[...]` 时，FastAPI 会在进入函数体前就拦下。
    """

    def _checker(identity: Identity = Depends(current_identity)) -> Identity:
        try:
            identity.require(feature)
        except PermissionDeniedError as exc:
            logger.warning(
                "权限不足: %s (%s) 尝试 %s",
                identity.employee_no, identity.role.value, feature)
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return identity

    return _checker


def require_manager():
    """要求主管身份（简写）。"""
    return require_feature("exam_build")


__all__ = [
    "Identity",
    "current_identity",
    "require_feature",
    "require_manager",
]
