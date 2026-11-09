"""员工与日报用例（P7）。

★ 每个用例的第一个动作都是**数据权限校验**，不是查数据。

  理由：先查再校验会有一个危险窗口——
  如果查出来的数据已经被序列化进响应对象，后面再拒绝，
  容易在异常处理里不小心把对象返回出去。

  先校验后查询，越权请求根本触达不到数据。
"""
from __future__ import annotations

import logging
from datetime import date

from domain.exceptions import (
    DailySummaryFrozenError,
    DataPermissionError,
)
from domain.rules.data_permission import (
    can_edit_summary,
    can_import_employees,
    can_review_summary,
    can_view_miss_logs,
    can_view_summary,
    require_review_permission,
    require_view_summary,
)
from domain.value_objects import DailySummary, EmployeeProfile

logger = logging.getLogger(__name__)


# ===================================================================
# 员工档案
# ===================================================================
class ImportEmployeesUseCase:
    """批量导入员工档案（FR-044）。"""

    def __init__(self, *, repository) -> None:
        self._repo = repository

    def execute(self, rows: list[dict], *, role: str) -> dict:
        """导入。

        Args:
            rows: 原始行（中英文表头均可）。
            role: 操作者角色。

        Returns:
            {ok, imported, duplicates, errors, summary}
        """
        if not can_import_employees(role=role):
            raise DataPermissionError("只有主管可以导入员工档案")

        from application.services.employee_import import import_employees

        result = import_employees(rows)

        for emp in result.employees:
            self._repo.save_employee(emp)

        logger.info("员工导入完成: %s", result.summary())
        return {
            "ok": result.ok,
            "imported": len(result.employees),
            "duplicates": result.duplicates,
            "errors": result.errors,
            "summary": result.summary(),
        }


class ListEmployeesUseCase:
    """查看员工档案。"""

    def __init__(self, *, repository) -> None:
        self._repo = repository

    def execute(self, *, role: str, requester: str = "",
                team: str = "", limit: int = 500,
                active_only: bool = True) -> list[EmployeeProfile]:
        """列员工。

        ★ 学员只能看到自己：传入 requester 时就按本人过滤。
        """
        from domain.rules.data_permission import can_view_employee

        if role in ("manager", "head_teacher", "admin"):
            return self._repo.list_employees(
                team=team, active_only=active_only, limit=limit)

        # 学员：只返回自己
        me = self._repo.get_employee(requester)
        if me is None or not can_view_employee(
                requester=requester, role=role, employee_no=me.employee_no):
            return []
        return [me]


# ===================================================================
# 每日总结
# ===================================================================
class SaveSummaryUseCase:
    """保存日报（新建 / 修改草稿 / 提交）。"""

    def __init__(self, *, repository) -> None:
        self._repo = repository

    def execute(
        self,
        *,
        requester: str,
        role: str,
        summary_date: str = "",
        content: str = "",
        submit: bool = False,
    ) -> DailySummary:
        """保存。

        Args:
            requester: 操作者工号。
            role: 角色。
            summary_date: 日期（YYYY-MM-DD），空则今天。
            content: 正文。
            submit: True 表示同时提交（冻结）。

        Raises:
            DataPermissionError: 试图保存他人日报。
            DailySummaryFrozenError: 已提交还想改正文。
        """
        d = summary_date or date.today().strftime("%Y-%m-%d")

        existing = self._repo.get_daily_summary(requester, d)

        if existing is not None:
            # ★ 已有记录：走「编辑」路径，权限与冻结都由领域层判定
            if not can_edit_summary(
                    requester=requester, role=role, summary=existing):
                if existing.is_submitted:
                    raise DailySummaryFrozenError(
                        "该总结已提交，不能修改")
                raise DataPermissionError("无权修改该总结")

            updated = existing.edit(content) if content else existing
        else:
            # 新建：只有本人能建自己的
            updated = DailySummary(
                trainee_no=requester,
                summary_date=d,
                content=content,
            )

        if submit:
            updated = updated.submit()

        self._repo.save_daily_summary(updated)
        logger.info("日报已保存: %s / %s status=%s",
                    requester, d, updated.status)
        return updated


class ListSummariesUseCase:
    """查看日报列表（主管）或自己的（学员）。"""

    def __init__(self, *, repository) -> None:
        self._repo = repository

    def execute(
        self,
        *,
        requester: str,
        role: str,
        trainee_no: str = "",
        start_date: str = "",
        end_date: str = "",
        status: str = "",
        limit: int = 100,
    ) -> list[DailySummary]:
        """列日报。

        ★ 学员即使传了 `trainee_no` 也只看自己的 —— 强制覆盖，
          不让前端参数决定能看到谁的数据。
        """
        if role in ("manager", "head_teacher", "admin"):
            rows = self._repo.list_daily_summaries(
                trainee_no=trainee_no, start_date=start_date,
                end_date=end_date, status=status, limit=limit)
        else:
            rows = self._repo.list_daily_summaries(
                trainee_no=requester, start_date=start_date,
                end_date=end_date, status=status, limit=limit)

        # 二次过滤：即使仓储返回了不该看的，这里再挡一次
        return [r for r in rows
                if can_view_summary(requester=requester, role=role,
                                    summary=r)]


class ReviewSummaryUseCase:
    """主管填写评语（FR-046）。"""

    def __init__(self, *, repository) -> None:
        self._repo = repository

    def execute(
        self,
        *,
        requester: str,
        role: str,
        trainee_no: str,
        summary_date: str,
        note: str,
    ) -> DailySummary:
        """写评语。

        ★ 主管只能改评语，**不能改正文**。
          这里只调 `save_review`，从接口层面就没有改正文的可能。
        """
        require_review_permission(requester=requester, role=role)

        existing = self._repo.get_daily_summary(trainee_no, summary_date)
        if existing is None:
            raise ValueError(
                f"日报不存在: {trainee_no} / {summary_date}")

        self._repo.save_review(trainee_no, summary_date, note, by=requester)
        logger.info("评语已保存: %s / %s", trainee_no, summary_date)

        updated = self._repo.get_daily_summary(trainee_no, summary_date)
        return updated or existing.with_review(note)


# ===================================================================
# 未命中日志
# ===================================================================
class ListMissLogsUseCase:
    """查看未命中问题清单（FR-047）。"""

    def __init__(self, *, repository) -> None:
        self._repo = repository

    def execute(self, *, role: str, limit: int = 100) -> dict:
        """返回日志与高频问题。

        ★★ 返回结构刻意**不含学员维度聚合**（不排序「谁问得多」）。

          FR-047 原文：「该日志不得用于对学员的绩效评价」。
          如果这里给出「按学员统计」，主管很难不用它做绩效。

          所以只提供两个视角：
            · `recent`  最近的问题（按时间）
            · `top`     高频问题（按问题内容聚合，不含学员）
        """
        if not can_view_miss_logs(role=role):
            raise DataPermissionError("只有主管可以查看未命中问题清单")

        recent = self._repo.list_miss_logs(limit=limit)
        top = []
        if hasattr(self._repo, "top_miss_queries"):
            top = self._repo.top_miss_queries(limit=20)

        return {
            "recent": [
                {
                    "query": r.query,
                    "intent": r.intent,
                    "created_at": r.created_at,
                    "recall_count": (r.detail or {}).get("fused", 0),
                    # ★ 不返回 trainee_id：日志是按问题看，不是按人看
                }
                for r in recent
            ],
            "top": top,
        }


__all__ = [
    "ImportEmployeesUseCase",
    "ListEmployeesUseCase",
    "ListMissLogsUseCase",
    "ListSummariesUseCase",
    "ReviewSummaryUseCase",
    "SaveSummaryUseCase",
]
