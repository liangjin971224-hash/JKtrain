"""MCP 越权测试（T1–T5）。

★ 这组测试是 MCP 方案的验收核心
  方案第 7.4 节列的 5 条，覆盖 MCP 特有的三类风险：
    · 参数伪造（T1）—— 工具签名里不该有身份参数
    · 知识越级（T2）—— 权限必须在检索阶段生效
    · 功能越权（T3）—— 白名单，未登记=拒绝
    · 令牌失效（T4）—— 回库校验，不信 token 快照
    · 角色降级（T5）—— FR-048 即时生效

★ 这些用假容器测，不接真库
  越权是「逻辑是否成立」的问题，接真库只会让测试变慢变脆。
  真库联调在 verify_mcp_tools.py 里单独做。
"""
from __future__ import annotations

import inspect
import json

import pytest

from domain.exceptions import PermissionDeniedError
from domain.value_objects.role import Role
from interfaces.mcp.identity import McpAuthError, McpIdentity, resolve_identity
from interfaces.mcp.policy import (
    TOOL_FEATURES,
    ToolDeniedError,
    require_prompt,
    require_resource,
    require_tool,
)
from interfaces.mcp.tools.drill import my_drill_reviews


# ======================================================================
# 假容器（只提供越权测试需要的字段）
# ======================================================================
class _FakeRepo:
    """假仓储：记录被查询的 trainee_id，用来断言「没读参数」。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def list_drill_reviews(
        self, *, trainee_id: str, since: str = "", until: str = "",
        with_items: bool = False,
    ) -> list:
        self.calls.append({"trainee_id": trainee_id, "since": since})
        return []


class _FakeContainer:
    def __init__(self, repo: _FakeRepo | None = None) -> None:
        self.repository = repo or _FakeRepo()
        self.settings = None
        self.graph = None
        self.vector = None
        self.keyword = None
        self.llm = None
        self.rerank = None
        self.token_signer = None


def _trainee(no: str = "E001") -> McpIdentity:
    return McpIdentity(employee_no=no, role=Role.TRAINEE)


def _manager(no: str = "M001") -> McpIdentity:
    return McpIdentity(employee_no=no, role=Role.MANAGER)


# ======================================================================
# T1：参数伪造 —— 工具签名里不能有身份参数
# ======================================================================
class TestT1ParameterForgery:
    """学员塞 employee_id=B，只能查到自己的。"""

    def test_signature_has_no_identity_param(self) -> None:
        """★ 最直接的防线：签名里**根本没有** employee_id 这个参数。

        调用方传了也没用——Python 会直接报 unexpected keyword argument，
        或者被 MCP SDK 当未知字段忽略。
        """
        params = set(inspect.signature(my_drill_reviews).parameters)
        assert "employee_id" not in params
        assert "trainee_id" not in params
        assert "employee_no" not in params
        # 只有 container / identity / days
        assert params == {"container", "identity", "days"}

    def test_uses_session_identity_not_param(self) -> None:
        """仓储收到的 trainee_id 必须是**会话身份**，不受任何外部参数影响。"""
        repo = _FakeRepo()
        c = _FakeContainer(repo)
        trainee = _trainee("E001")

        my_drill_reviews(c, trainee, days=7)

        assert len(repo.calls) == 1
        assert repo.calls[0]["trainee_id"] == "E001"

    def test_two_identities_query_different_rows(self) -> None:
        """不同会话拿到不同结果（证明确实按身份隔离）。"""
        repo = _FakeRepo()
        c = _FakeContainer(repo)

        my_drill_reviews(c, _trainee("E001"), days=7)
        my_drill_reviews(c, _trainee("E002"), days=7)

        assert [x["trainee_id"] for x in repo.calls] == ["E001", "E002"]


# ======================================================================
# T2：知识越级 —— 检索级别由角色决定
# ======================================================================
class TestT2KnowledgeLevel:
    """学员不能拿到 restricted / confidential 级别。"""

    def test_trainee_limited_levels(self) -> None:
        t = _trainee()
        levels = [str(x) for x in t.retrieval_levels]
        assert "restricted" not in levels
        assert "confidential" not in levels
        assert "public" in levels
        assert "internal" in levels

    def test_manager_full_levels(self) -> None:
        m = _manager()
        levels = [str(x) for x in m.retrieval_levels]
        assert "restricted" in levels
        assert "confidential" in levels

    def test_levels_pass_to_retrieval_not_filtered_after(self) -> None:
        """★ 关键：级别要**传进**用例（拼进 Milvus filter），不是查完再筛。

        这里通过源码断言：kb.py 里必须把 allowed_permissions 传给 execute。
        """
        import interfaces.mcp.tools.kb as kb_mod

        src = inspect.getsource(kb_mod.kb_search)
        assert "allowed_permissions=identity.retrieval_levels" in src, (
            "kb_search 必须把会话的检索级别传给用例，"
            "否则权限是在结果上过滤（FR-050 要求检索阶段生效）"
        )

    def test_vector_collection_forwarded(self) -> None:
        """★ 回归测试：必须透传 vector_collection。

        实测（2026-11-06）：漏传后日志出现
            "召回路 vector 失败，降级继续: ValueError: 未指定 vector_collection"
        三路召回降级为两路 —— **不报错、不抛异常，只是变差**，
        这类静默降级最难发现。

        这是本项目「新入口漏透传可选参数」的第 5 次复发
        （前 4 次：fact_hints / product_ids / history / 流式 done.meta），
        所以用源码扫描守住，而不是等下次实测才发现。
        """
        import interfaces.mcp.tools.kb as kb_mod

        src = inspect.getsource(kb_mod.kb_search)
        assert "vector_collection=" in src, (
            "kb_search 必须显式传 vector_collection，否则向量路静默降级"
        )


# ======================================================================
# T3：功能越权 —— 白名单，未登记即拒绝
# ======================================================================
class TestT3FeaturePermission:
    def test_trainee_can_use_kb_search(self) -> None:
        require_tool(_trainee(), "kb_search")      # 不抛异常即通过

    def test_trainee_can_use_own_drill(self) -> None:
        require_tool(_trainee(), "my_drill_reviews")

    def test_unregistered_tool_denied(self) -> None:
        """未登记的工具一律拒绝（fail-safe）。"""
        with pytest.raises(ToolDeniedError, match="未登记"):
            require_tool(_manager(), "build_paper")

    def test_unknown_tool_denied(self) -> None:
        with pytest.raises(ToolDeniedError, match="未登记"):
            require_tool(_manager(), "drop_database")

    def test_prompt_whitelist(self) -> None:
        require_prompt(_trainee(), "expression_polish")
        with pytest.raises(ToolDeniedError):
            require_prompt(_trainee(), "unknown_prompt")

    def test_resource_whitelist(self) -> None:
        require_resource(_trainee(), "product")
        with pytest.raises(ToolDeniedError):
            require_resource(_trainee(), "employee")


# ======================================================================
# T4：令牌失效 —— 回库校验
# ======================================================================
class TestT4TokenInvalid:
    def test_no_token_and_auth_required(self) -> None:
        """生产模式（auth_required=True）未带 token 必须拒绝。"""

        class S:
            auth_required = True

        c = _FakeContainer()
        c.settings = S()
        with pytest.raises(McpAuthError, match="未提供身份令牌"):
            resolve_identity(c, token="")

    def test_invalid_token_rejected(self) -> None:
        """令牌校验失败（回库查不到）必须拒绝。"""

        class S:
            auth_required = True

        class Signer:
            pass

        class Repo:
            pass

        c = _FakeContainer()
        c.settings = S()
        c.token_signer = Signer()
        c.repository = Repo()

        # ResolveIdentityUseCase 对无签名能力的假对象会抛异常，
        # 这里接受「抛 McpAuthError 或原始异常」，关键是**不返回身份**
        with pytest.raises(Exception):
            resolve_identity(c, token="bad.token.here")

    def test_dev_mode_allows_no_token(self) -> None:
        """开发模式（auth_required=False）允许无 token，按 debug_role。"""

        class S:
            auth_required = False

        c = _FakeContainer()
        c.settings = S()
        ident = resolve_identity(c, token="", debug_role="trainee")
        assert ident.employee_no == "(dev)"
        assert ident.role is Role.TRAINEE
        assert ident.source == "dev"


# ======================================================================
# T5：角色降级即时生效
# ======================================================================
class TestT5RoleDowngrade:
    """★ FR-048：降权必须立刻生效，不能等 token 过期。

    实现方式：每次解析都回库取最新角色，不信 token 里的快照。
    这里验证「同一个 token，库里角色变了，解析结果跟着变」。
    """

    def test_role_read_from_db_each_time(self) -> None:
        # 用一个可控的假 use case 替换，模拟「库里角色变了」
        import application.use_cases.auth as auth_mod
        from interfaces.mcp import identity as ident_mod

        calls = {"n": 0}
        roles = ["manager", "trainee"]

        class FakeUseCase:
            def __init__(self, **_kw) -> None:
                pass

            def execute(self, _token: str):
                idx = min(calls["n"], len(roles) - 1)
                calls["n"] += 1
                return ("M001", Role.parse(roles[idx]))

        class S:
            auth_required = True

        c = _FakeContainer()
        c.settings = S()
        c.token_signer = object()

        orig = ident_mod.ResolveIdentityUseCase
        ident_mod.ResolveIdentityUseCase = FakeUseCase       # type: ignore[misc]
        try:
            first = resolve_identity(c, token="t")
            second = resolve_identity(c, token="t")
        finally:
            ident_mod.ResolveIdentityUseCase = orig          # type: ignore[misc]

        # 同一个 token，第二次读到的是降级后的角色
        assert first.role is Role.MANAGER
        assert second.role is Role.TRAINEE, (
            "角色降级没有即时生效：说明用了 token 快照而不是回库")
