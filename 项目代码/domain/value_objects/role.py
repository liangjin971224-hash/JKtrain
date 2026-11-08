"""角色与权限（P8）。

★ 需求：FR-048（功能权限）/ FR-049（数据权限）/ FR-050（检索权限）

────────────────────────────────────────────────────────────────────
设计要点：白名单，不是黑名单
────────────────────────────────────────────────────────────────────
`_FEATURES` 里列的是**每个角色被允许**的功能。
未登记的功能一律拒绝。

为什么这样：新增功能时忘了配权限，结果是「用不了」（可发现、可修复），
而不是「所有人都能用」（安全事故）。fail-safe 方向要选对。

────────────────────────────────────────────────────────────────────
三个权限层次要分清
────────────────────────────────────────────────────────────────────
    FR-048 功能权限   能不能用这个功能        → `can()` / `require()`
    FR-049 数据权限   能看到谁的数据          → `domain/rules/data_permission.py`
    FR-050 检索权限   能检索哪个级别的知识    → `retrieval_levels`

后者两者都在**查询层**生效，不能先查后筛（需求原文明确要求）。
"""
from __future__ import annotations

from enum import StrEnum

from domain.exceptions import PermissionDeniedError
from domain.value_objects.search_result import PermissionLevel


class Role(StrEnum):
    """系统角色（FR-048：只支持学员、主管两种）。"""

    TRAINEE = "trainee"
    MANAGER = "manager"

    @property
    def label(self) -> str:
        return {"trainee": "学员", "manager": "主管"}[self.value]

    # ---------------- 功能权限 ----------------
    def can(self, feature: str) -> bool:
        """是否有权使用某功能。

        Args:
            feature: 功能标识（如 `exam_build`）。

        Returns:
            True 表示允许。**未登记的功能一律 False。**
        """
        return feature in _FEATURES.get(self, frozenset())

    def require(self, feature: str) -> None:
        """同 `can`，无权限时抛异常。

        Raises:
            PermissionDeniedError: 无权限。
        """
        if not self.can(feature):
            raise PermissionDeniedError(
                f"{self.label}没有「{feature}」的权限")

    # ---------------- 检索权限（FR-050） ----------------
    @property
    def retrieval_levels(self) -> tuple[PermissionLevel, ...]:
        """可检索的知识级别。

        ★ 这个元组最终会拼进 Milvus 的 `filter_expr`，
          所以权限在**检索阶段**就生效，不是检索后过滤。
        """
        if self is Role.MANAGER:
            return (
                PermissionLevel.PUBLIC,
                PermissionLevel.INTERNAL,
                PermissionLevel.RESTRICTED,
                PermissionLevel.CONFIDENTIAL,
            )
        return (PermissionLevel.PUBLIC, PermissionLevel.INTERNAL)

    def can_retrieve_level(self, level: PermissionLevel | str) -> bool:
        """能否检索某个级别的内容。"""
        target = str(level)
        return target in {str(x) for x in self.retrieval_levels}

    # ---------------- 便捷判断 ----------------
    @property
    def is_manager(self) -> bool:
        return self is Role.MANAGER

    @classmethod
    def parse(cls, value: str | Role) -> Role:
        """字符串 → 角色。

        Raises:
            PermissionDeniedError: 角色非法。

        ★ 非法角色拒绝，**不能默认成学员或主管**。
          默认成任何一方都是错的：默认学员会让人干不了活，
          默认主管是越权。
        """
        if isinstance(value, Role):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise PermissionDeniedError(
                f"未知角色: {value!r}。可用：{[r.value for r in cls]}") from exc


#: 功能权限矩阵（白名单）。
#:
#: ★ 只列「允许」的。没列的 = 不允许。
_FEATURES: dict[Role, frozenset[str]] = {
    Role.TRAINEE: frozenset({
        "ask",              # 知识问答
        "drill",            # 情景演练
        "daily_write",      # 写自己的日报
        "employee_self",    # 看自己的档案
    }),
    Role.MANAGER: frozenset({
        "ask",
        "drill",
        "daily_write",
        "daily_review",     # 看全部日报 + 写评语
        "employee_self",
        "employee_view",    # 看全部员工
        "employee_import",  # 批量导入
        "exam_build",       # 组卷
        "exam_manage",      # 试卷管理 / 导出
        "knowledge_manage",  # 知识库管理
        "miss_log_view",    # 未命中清单
        "audit_log_view",   # 审计日志
    }),
}


def all_features() -> frozenset[str]:
    """全部已登记功能（测试与文档用）。"""
    out: set[str] = set()
    for v in _FEATURES.values():
        out |= v
    return frozenset(out)


__all__ = ["Role", "all_features"]
