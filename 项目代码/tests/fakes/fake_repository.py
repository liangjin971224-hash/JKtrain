"""仓储端口的测试替身。内存字典实现。

★ 唯一约束必须真的生效：同人同天第二次写日报要抛异常。
   否则用它测 P7 的「日报唯一性」就是假绿。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from domain.exceptions import DuplicateDailySummaryError
from domain.ports import (
    DailySummaryRow,
    DrillRecordRow,
    MissLogRow,
    RepositoryPort,
)

from tests.fakes.fake_clock import now_iso


class FakeRepositoryPort(RepositoryPort):
    """内存仓储。"""

    def __init__(self) -> None:
        self.drills: dict[str, DrillRecordRow] = {}
        self.summaries: dict[tuple[str, str], DailySummaryRow] = {}
        self.papers: dict[str, dict[str, Any]] = {}
        self.miss_logs: list[MissLogRow] = []
        #: 演练评阅结果（drill_id → 行）。★ 月度报告的数据源
        self.drill_reviews: dict[str, Any] = {}
        #: 月度报告定稿（(trainee_id, period) → 行）
        self.monthly_reports: dict[tuple[str, str], Any] = {}
        #: 批量任务（(trainee_id, period) → dict）
        self.report_tasks: dict[tuple[str, str], dict] = {}
        #: P7：员工档案（工号 → 档案）
        self.employees: dict[str, Any] = {}
        self._questions: list[dict[str, Any]] = []
        self._seq = 0

    # ---------------- 演练 ----------------
    def save_drill_record(self, record: DrillRecordRow) -> None:
        self.drills[record.drill_id] = record

    def get_drill_record(self, drill_id: str) -> DrillRecordRow | None:
        return self.drills.get(drill_id)

    def list_drill_records_by_trainee(
        self, trainee_id: str, *, limit: int = 50
    ) -> list[DrillRecordRow]:
        rows = [r for r in self.drills.values() if r.trainee_id == trainee_id]
        rows.sort(key=lambda r: r.started_at, reverse=True)
        return rows[:limit]

    # ---------------- 每日总结 ----------------
    def save_daily_summary(self, row: DailySummaryRow) -> None:
        """保存日报。

        ★ 行为已与真实仓储对齐（2026-10-23 修正）：
          原先「已存在就抛 Duplicate」的语义是错的——
          草稿本来就要能反复覆盖保存（FR-045）。

          真实规则是：
            · 不存在     → 新建
            · 存在且草稿 → 覆盖内容（这就是「草稿可反复修改」）
            · 存在已提交 → 拒绝改正文（抛 DailySummaryFrozenError）
        """
        from domain.exceptions import DailySummaryFrozenError

        key = (row.trainee_no, row.summary_date)
        existing = self.summaries.get(key)

        if existing is not None and existing.status == "submitted":
            raise DailySummaryFrozenError("该总结已提交，不能修改")

        self.summaries[key] = row

    def get_daily_summary(
        self, trainee_no: str, summary_date: str
    ) -> DailySummaryRow | None:
        return self.summaries.get((trainee_no, summary_date))

    # ---------------- 题库 / 试卷 ----------------
    def seed_questions(self, questions: list[dict[str, Any]]) -> None:
        """测试辅助：注入题库。"""
        self._questions = list(questions)

    def list_questions(
        self,
        *,
        knowledge_point: str | None = None,
        difficulty: int | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        rows = self._questions
        if knowledge_point is not None:
            rows = [q for q in rows if q.get("knowledge_point") == knowledge_point]
        if difficulty is not None:
            rows = [q for q in rows if q.get("difficulty") == difficulty]
        return rows[:limit]

    def save_exam_paper(self, paper: dict[str, Any]) -> str:
        self._seq += 1
        paper_id = paper.get("paper_id") or f"E{self._seq:03d}"
        self.papers[paper_id] = {**paper, "paper_id": paper_id}
        return paper_id

    def get_exam_paper(self, paper_id: str) -> dict[str, Any] | None:
        return self.papers.get(paper_id)

    # ---------------- 日志 ----------------
    def append_miss_log(self, row: MissLogRow) -> None:
        self.miss_logs.append(
            row if row.created_at else MissLogRow(
                id=row.id, query=row.query, intent=row.intent,
                trainee_id=row.trainee_id, created_at=now_iso(),
                detail=row.detail,
            )
        )

    def list_miss_logs(self, *, limit: int = 100) -> list[MissLogRow]:
        return sorted(self.miss_logs, key=lambda r: r.created_at, reverse=True)[:limit]

    # ---------------- P7：员工档案 ----------------
    def save_employee(self, employee) -> None:
        self.employees[employee.employee_no] = employee

    def get_employee(self, employee_no: str):
        return self.employees.get(employee_no)

    def list_employees(self, *, team: str = "", active_only: bool = True,
                       limit: int = 500) -> list:
        out = []
        for e in self.employees.values():
            if team and e.team != team:
                continue
            if active_only and not getattr(e, "active", True):
                continue
            out.append(e)
        return sorted(out, key=lambda x: x.employee_no)[:limit]

    # ---------------- P7：日报 ----------------
    def list_daily_summaries(
        self,
        *,
        trainee_no: str = "",
        start_date: str = "",
        end_date: str = "",
        status: str = "",
        limit: int = 100,
    ) -> list:
        from domain.value_objects import DailySummary, SummaryStatus

        out = []
        for (tno, d), row in self.summaries.items():
            if trainee_no and tno != trainee_no:
                continue
            if start_date and d < start_date:
                continue
            if end_date and d > end_date:
                continue
            if status and str(row.status) != status:
                continue
            out.append(DailySummary(
                trainee_no=tno, summary_date=d,
                content=row.content, status=SummaryStatus(str(row.status)),
                review_note=row.review_note,
            ))
        out.sort(key=lambda s: (s.summary_date, s.trainee_no), reverse=True)
        return out[:limit]

    def save_review(self, trainee_no: str, summary_date: str,
                    note: str, *, by: str = "") -> None:
        key = (trainee_no, summary_date)
        if key not in self.summaries:
            raise ValueError(f"日报不存在: {trainee_no} / {summary_date}")
        row = self.summaries[key]
        self.summaries[key] = DailySummaryRow(
            trainee_no=row.trainee_no, summary_date=row.summary_date,
            content=row.content, review_note=note, status=row.status,
        )

    # ---------------- P7：未命中日志 ----------------
    def top_miss_queries(self, *, limit: int = 20) -> list[dict]:
        counts: dict[str, int] = {}
        for m in self.miss_logs:
            counts[m.query] = counts.get(m.query, 0) + 1
        items = sorted(counts.items(), key=lambda x: -x[1])[:limit]
        return [{"query": q, "times": n, "last_at": ""} for q, n in items]

    # ---------------- 演练评阅结果 ----------------
    def save_drill_review(
        self,
        review,
        risk_items=(),
    ) -> None:
        """内存实现。语义与 MySQL 版一致：**幂等覆盖**。

        ★ 必须与真实现保持同一语义：
          只覆盖主表不清明细，重试时会累积重复违规条目。
          替身若做错，「月度报告算得对」就是假绿。
        """
        self.drill_reviews[review.drill_id] = replace(
            review, risk_items=tuple(risk_items))

    def list_drill_reviews(
        self,
        *,
        trainee_id: str,
        since: str = "",
        until: str = "",
        with_items: bool = False,
    ) -> list:
        out = [r for r in self.drill_reviews.values()
               if r.trainee_id == trainee_id]
        if since:
            out = [r for r in out if (r.created_at or "") >= since]
        if until:
            out = [r for r in out if (r.created_at or "") < until]
        out.sort(key=lambda r: r.created_at or "")
        return out

    def get_drill_review(self, drill_id: str):
        return self.drill_reviews.get(drill_id)

    # ---------------- 批量任务（P3） ----------------
    def upsert_report_task(self, *, task_id, trainee_id, period,
                           status, error="", thread_id=""):
        """内存实现。冲突键 (trainee_id, period)，与 MySQL 唯一约束一致。"""
        key = (trainee_id, period)
        old = self.report_tasks.get(key) or {}
        attempts = int(old.get("attempts") or 0)
        if status == "failed":
            attempts += 1
        self.report_tasks[key] = {
            "task_id": old.get("task_id") or task_id,
            "trainee_id": trainee_id,
            "period": period,
            "thread_id": thread_id or old.get("thread_id", ""),
            "status": status,
            "attempts": attempts,
            "last_error": error,
        }

    def get_report_task(self, trainee_id, period):
        return self.report_tasks.get((trainee_id, period))

    def list_report_tasks(self, *, period="", status="", limit=500):
        out = list(self.report_tasks.values())
        if period:
            out = [r for r in out if r["period"] == period]
        if status:
            out = [r for r in out if r["status"] == status]
        return out[:limit]

    # ---------------- 月度报告定稿 ----------------
    def save_monthly_report(self, row) -> None:
        """内存实现。幂等键 = (trainee_id, period)，与 MySQL 唯一约束一致。"""
        self.monthly_reports[(row.trainee_id, row.period)] = row

    def get_monthly_report(self, trainee_id: str, period: str):
        return self.monthly_reports.get((trainee_id, period))

    def list_monthly_reports(
        self, *, period: str = "", status: str = "",
        trainee_id: str = "", limit: int = 200,
    ) -> list:
        out = list(self.monthly_reports.values())
        if period:
            out = [r for r in out if r.period == period]
        if status:
            out = [r for r in out if r.status == status]
        if trainee_id:
            out = [r for r in out if r.trainee_id == trainee_id]
        return out[:limit]
