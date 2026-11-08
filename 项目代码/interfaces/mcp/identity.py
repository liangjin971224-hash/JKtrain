"""MCP 会话身份。

════════════════════════════════════════════════════════════════════
为什么要单独做一个 identity 模块
════════════════════════════════════════════════════════════════════
HTTP 侧的身份来自 `Authorization: Bearer <token>`，每个请求都带，所以
`current_identity` 是个 FastAPI 依赖，请求结束就丢。

MCP 不是这样：**连接建立一次，之后所有工具调用共享这个连接**。
工具调用本身没有「当前用户」这个概念——这就是风险所在。

    如果工具签名写成  my_drill_reviews(employee_id="E002")
    Agent 只要传别人的工号，就能读到别人的演练评语。
    而 drill_review 表有 183 行真实评语，这是内部数据泄露。

所以设计成：
    ① 建连时解析 token → Identity，存在会话里
    ② 工具签名里**不出现任何身份参数**
    ③ 每次调用从会话取 Identity，绝不读参数

────────────────────────────────────────────────────────────────────
为什么复用 HTTP 侧的解析逻辑而不是重写
────────────────────────────────────────────────────────────────────
`ResolveIdentityUseCase` 已经做了三件对的事：
    1. 校验签名与过期
    2. **回库取最新角色**（不信 token 快照，满足 FR-048 即时生效）
    3. 返工号 + 角色

重写一遍只会让两边规则漂移——历史上项目已经吃过「同一套标准两份实现」的亏。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from application.use_cases.auth import ResolveIdentityUseCase
from config.assembly import Container
from domain.exceptions import PermissionDeniedError
from domain.value_objects.role import Role

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class McpIdentity:
    """MCP 会话身份。

    与 HTTP 侧的 `Identity` 结构一致（employee_no + role），
    但**刻意不 import http 层**——`interfaces/mcp` 与 `interfaces/http`
    是并列的两个入站适配器，不该互相依赖。
    """

    employee_no: str
    role: Role
    #: 身份来源（token / dev），仅用于日志与排障
    source: str = "token"

    @property
    def is_manager(self) -> bool:
        return self.role.is_manager

    @property
    def retrieval_levels(self) -> tuple:
        """可检索的知识级别（FR-050）。直接转发 Role 的实现。"""
        return self.role.retrieval_levels


class McpAuthError(Exception):
    """MCP 身份解析失败。

    单独一个异常类型（不复用 HTTPException）：
    MCP 不是 HTTP 协议，返回 401 没有意义，需要把失败原因转成
    MCP 的 error 内容给客户端看。
    """


def resolve_identity(
    container: Container,
    *,
    token: str = "",
    debug_role: str = "",
) -> McpIdentity:
    """从 token 解析会话身份。

    Args:
        container: 已装配的容器（提供 repository / token_signer / settings）。
        token: Bearer 令牌（已去掉 "Bearer " 前缀）。
        debug_role: 开发模式下的角色覆盖（仅 `AUTH_REQUIRED=false` 时生效）。

    Returns:
        会话身份。

    Raises:
        McpAuthError: 令牌无效 / 未登录 / 认证能力未装配 / 角色非法。
    """
    settings = getattr(container, "settings", None)
    auth_required = getattr(settings, "auth_required", True)

    token = (token or "").strip()

    # ---- 开发模式：允许无 token，按 debug_role 处理 ----
    #
    # ★ 与 HTTP 侧行为保持一致（那里未带 token 时按 X-Debug-Role 处理）。
    #   生产必须 AUTH_REQUIRED=true，否则任何人都能当主管。
    if not token:
        if auth_required:
            raise McpAuthError(
                "未提供身份令牌。MCP 连接必须携带 Bearer 令牌"
                "（stdio 用环境变量 MCP_TOKEN，SSE 用 Authorization 头）。"
            )
        try:
            role = Role.parse(debug_role or "manager")
        except PermissionDeniedError as exc:
            raise McpAuthError(str(exc)) from exc
        logger.warning(
            "MCP 开发模式：未带令牌，按 %s 处理（生产必须 AUTH_REQUIRED=true）",
            role.label,
        )
        return McpIdentity(employee_no="(dev)", role=role, source="dev")

    # ---- 正常模式：校验令牌 + 回库取最新角色 ----
    repo = getattr(container, "repository", None)
    if repo is None:
        raise McpAuthError("仓储未就绪，无法校验身份")

    signer = getattr(container, "token_signer", None)
    if signer is None:
        raise McpAuthError("认证能力未装配（token_signer）")

    uc = ResolveIdentityUseCase(repository=repo, signer=signer)
    resolved = uc.execute(token)
    if resolved is None:
        raise McpAuthError("登录已失效，请重新登录")

    emp_no, role = resolved
    return McpIdentity(employee_no=emp_no, role=role, source="token")


def token_from_env() -> str:
    """从环境变量读令牌（stdio 传输用）。

    ★ 为什么用环境变量：
      stdio 是「宿主进程 spawn 子进程」，没有 HTTP 头可以带令牌。
      MCP 官方推荐用环境变量传这种连接级凭据。
    """
    raw = os.environ.get("MCP_TOKEN", "").strip()
    if raw.lower().startswith("bearer "):
        return raw[7:].strip()
    return raw


def token_from_header(authorization: str) -> str:
    """从 `Authorization: Bearer xxx` 头里取令牌（SSE / HTTP 传输用）。"""
    if not authorization:
        return ""
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return authorization.strip()
