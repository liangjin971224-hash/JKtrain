"""数据权限判定（FR-049）。

★ 与功能权限（FR-048）分开，两者管的事不同：

    功能权限 = 「能不能用这个功能」
        学员不能组卷、不能维护知识库
    数据权限 = 「能看到/改到谁的数据」
        学员只能看自己的总结，主管能看所辖的

────────────────────────────────────────────────────────────────────
一条容易搞错的边界：主管能写评语，但**不能改学员正文**
────────────────────────────────────────────────────────────────────
    「改正文」和「写评语」看起来都是「编辑这条记录」，
    但性质完全不同：

        改正文   = 篡改学员的原始记录 → 日报失去「真实想法」的价值
        写评语   = 主管的独立表达     → 这是日报的增值部分

    所以 `can_edit_summary`（正文）只给学员本人，
    `can_review_summary`（评语）只给主管。

────────────────────────────────────────────────────────────────────
为什么用「角色字符串」而不是枚举
────────────────────────────────────────────────────────────────────
角色来自请求上下文（P8 才做鉴权），此时是字符串。
这里只做**判定逻辑**，不依赖具体的鉴权实现——
这样 P8 接真鉴权时，判定规则不用改。
"""
from __future__ import annotations

from domain.exceptions import DataPermissionError
from domain.value_objects.employee import DailySummary

#: 角色常量
ROLE_TRAINEE = "trainee"
ROLE_MANAGER = "manager"

#: 视为「主管」的角色（P8 可能有 head_teacher 等细分，先都算主管）
_MANAGER_ROLES = frozenset({ROLE_MANAGER, "head_teacher", "admin"})


def _is_manager(role: str) -> bool:
    return (role or "").strip().lower() in _MANAGER_ROLES


# ===================================================================
# 每日总结
# ===================================================================
def can_view_summary(
    *, requester: str, role: str, summary: DailySummary
) -> bool:
    """能否查看这条总结。

    ★ FR-046：「学员可看到自己的评语，不可看到他人评语」。

    Args:
        requester: 请求者工号。
        role: 请求者角色。
        summary: 目标总结。

    Returns:
        True 表示可以查看。
    """
    if _is_manager(role):
        return True
    # 学员只能看自己的
    return (requester or "").strip() == (summary.trainee_no or "").strip()


def require_view_summary(
    *, requester: str, role: str, summary: DailySummary
) -> None:
    """同 `can_view_summary`，无权时抛异常。

    Raises:
        DataPermissionError: 无权查看。
    """
    if not can_view_summary(requester=requester, role=role, summary=summary):
        raise DataPermissionError(
            f"无权查看该总结（工号 {summary.trainee_no} 的记录）")


def can_edit_summary(
    *, requester: str, role: str, summary: DailySummary
) -> bool:
    """能否编辑总结**正文**。

    ★★ 两条限制，缺一不可：
      ① 只有学员本人（主管也不能改正文）
      ② 必须还在草稿态（提交后冻结）

    主管改正文 = 篡改学员原始记录，日报就失去价值了。
    """
    if _is_manager(role):
        return False                     # ← 主管不因角色获得改正文的权限
    if (requester or "").strip() != (summary.trainee_no or "").strip():
        return False
    return summary.can_edit


def can_review_summary(*, requester: str, role: str) -> bool:
    """能否填写评语。

    ★ 只有主管能写评语。学员不能给自己写评语。
    """
    return _is_manager(role)


def require_review_permission(*, requester: str, role: str) -> None:
    """无权写评语时抛异常。

    Raises:
        DataPermissionError: 不是主管。
    """
    if not can_review_summary(requester=requester, role=role):
        raise DataPermissionError(
            "只有主管可以填写评语")


# ===================================================================
# 员工档案
# ===================================================================
def can_view_employee(*, requester: str, role: str,
                      employee_no: str) -> bool:
    """能否查看员工档案。

    学员只能看自己的档案；主管能看全部。
    """
    if _is_manager(role):
        return True
    return (requester or "").strip() == (employee_no or "").strip()


def can_import_employees(*, role: str) -> bool:
    """能否批量导入员工（只有主管）。"""
    return _is_manager(role)


# ===================================================================
# 未命中日志
# ===================================================================
def can_view_miss_logs(*, role: str) -> bool:
    """能否查看未命中日志（只有主管）。

    ★ 注意：这里只判「能不能看」。
      日志本身**不得用于绩效评价**（FR-047），
      所以返回的数据结构里没有评分字段（见 `MissLogEntry`）。
    """
    return _is_manager(role)


__all__ = [
    "ROLE_MANAGER",
    "ROLE_TRAINEE",
    "can_edit_summary",
    "can_import_employees",
    "can_review_summary",
    "can_view_employee",
    "can_view_miss_logs",
    "can_view_summary",
    "require_review_permission",
    "require_view_summary",
]
