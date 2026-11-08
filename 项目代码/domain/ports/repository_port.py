"""仓储端口。

覆盖演练、日报、试卷、答卷、题库、日志六类持久化需求。
不用 Repository 模式（不返回聚合根），只做「存取」这一件事。

★ 注意：本端口只承载**业务数据**。
   答卷扫描件这类「文件」走 FileStoragePort，不塞进这里。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DrillRecordRow:
    """演练记录（对应 MySQL drill_record 表）。"""

    drill_id: str
    trainee_id: str
    case_id: str
    started_at: str
    finished_at: str = ""
    duration_sec: int = 0
    status: str = "in_progress"


@dataclass(frozen=True, slots=True)
class DailySummaryRow:
    """每日总结（对应 daily_summary 表）。

    ★ 唯一约束：`uk_trainee_date (trainee_no, summary_date)` —— 每人每天一条。

    ★ `submitted_at` 是 FR-045 明确要求记录的字段（「日期、内容、状态、提交时间」）。
    """

    trainee_no: str
    summary_date: str
    content: str = ""
    review_note: str = ""
    status: str = "draft"
    #: 提交时间（空串表示还没提交）
    submitted_at: str = ""


@dataclass(frozen=True, slots=True)
class MissLogRow:
    """未命中日志（对应 miss_log 表）。

    召回为空时自动记录，供后续补内容。
    """

    id: str
    query: str
    intent: str = ""
    trainee_id: str = ""
    created_at: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DrillRiskItemRow:
    """一条演练违规明细（对应 drill_risk_item 表）。

    ★ `turn_index` / `sentence_index` 必须存：
      这是「能说出第 3 轮第 2 句把『维持血脂健康水平』说成了『降血脂』」
      这个能力落到数据库的唯一方式。不存，历史回顾时就定位不到原句。
    """

    drill_id: str
    turn_index: int          # 从 1 开始
    sentence_index: int      # 从 1 开始
    rule_type: str = ""
    matched: str = ""
    level: str = ""          # critical / high / medium
    reason: str = ""
    id: int = 0


@dataclass(frozen=True, slots=True)
class DrillReviewRow:
    """一次演练的评阅结果（对应 drill_review 表）。

    ★ 这张表存在的理由：
      原实现里 `/drill/review` 算完只返回前端，**一行不落库**，
      导致月度报告只能算「练了几次」，算不出「错在哪」。

    ★ `case_type` 存案例的提问类型原文（如 `contraindication`），
      是月度报告归并「能力项」的唯一依据。**没有它就无法归并。**
    """

    drill_id: str
    trainee_id: str = ""
    case_id: str = ""
    case_type: str = ""
    duration_sec: int = 0
    #: 必答要素命中 / 总数（算要素分用）
    element_hit: int = 0
    element_total: int = 0
    #: 全部要素命中 / 总数（含选答，备用）
    element_hit_all: int = 0
    element_total_all: int = 0
    risk_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    critical_count: int = 0
    created_at: str = ""
    #: 违规明细（读库时一并返回；写入时可为空）
    risk_items: tuple[DrillRiskItemRow, ...] = ()


@dataclass(frozen=True, slots=True)
class MonthlyReportRow:
    """月度报告定稿（对应 monthly_report 表）。

    ★ 这张表**只存定稿**：草稿（图跑到一半的状态）在 Redis checkpoint 里。
      理由见 P2 设计文档：中间态可丢、重算即可；定稿不可丢、要能查能审计。

    ★ `report_json` 存的是**快照**（冻结），不是实时算的：
      主管确认后即使学员又练了几次，这份报告内容也不该变。
    """

    trainee_id: str
    period: str
    thread_id: str = ""
    #: draft / pending_review / approved / rejected
    status: str = "draft"
    report_json: dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""
    suggestion_passed: bool = True
    suggestion_note: str = ""
    total_drills: int = 0
    total_critical: int = 0
    weak_capabilities: tuple[str, ...] = ()
    graph_steps: int = 0
    reviewed_by: str = ""
    reviewed_at: str = ""
    review_note: str = ""
    id: int = 0
    created_at: str = ""
    updated_at: str = ""


class RepositoryPort(ABC):
    """业务数据仓储端口。

    方法按「一个实体一组」组织；返回领域 dataclass，不返回 ORM 对象。
    """

    # ---------------- 演练 ----------------
    @abstractmethod
    def save_drill_record(self, record: DrillRecordRow) -> None:
        """写入/更新一条演练记录（按 drill_id upsert）。"""

    @abstractmethod
    def get_drill_record(self, drill_id: str) -> DrillRecordRow | None:
        """按 ID 取演练记录。不存在返回 None。"""

    @abstractmethod
    def list_drill_records_by_trainee(
        self, trainee_id: str, *, limit: int = 50
    ) -> list[DrillRecordRow]:
        """取某学员的演练记录（按时间倒序）。

        ★ 数据权限：学员只能看自己的，调用方必须传自己的 trainee_id。
        """

    # ---------------- 每日总结 ----------------
    @abstractmethod
    def save_daily_summary(self, row: DailySummaryRow) -> None:
        """写入每日总结。

        Raises:
            DuplicateDailySummaryError: 同人同天已存在（唯一约束冲突）。
        """

    @abstractmethod
    def get_daily_summary(
        self, trainee_no: str, summary_date: str
    ) -> DailySummaryRow | None:
        """取某人某天的总结。"""

    # ---------------- 题库 / 试卷 ----------------
    @abstractmethod
    def list_questions(
        self,
        *,
        knowledge_point: str | None = None,
        difficulty: int | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """按条件取题（组卷用）。

        Returns:
            题目字典列表，含 knowledge_point / question_type / difficulty。
        """

    @abstractmethod
    def save_exam_paper(self, paper: dict[str, Any]) -> str:
        """保存试卷（含**题目快照**）。

        快照是硬要求：改题库后历史试卷内容不能变。

        Returns:
            paper_id。
        """

    @abstractmethod
    def get_exam_paper(self, paper_id: str) -> dict[str, Any] | None:
        """取试卷（读快照，不重新关联题库）。"""

    # ---------------- 员工档案（P7） ----------------
    @abstractmethod
    def save_employee(self, employee: Any) -> None:
        """写入/更新员工档案（按工号 upsert）。"""

    @abstractmethod
    def get_employee(self, employee_no: str) -> Any:
        """按工号取档案。不存在返回 None。"""

    @abstractmethod
    def list_employees(
        self, *, team: str = "", active_only: bool = True, limit: int = 500
    ) -> list[Any]:
        """列出员工档案。"""

    # ---------------- 每日总结查询（P7） ----------------
    @abstractmethod
    def list_daily_summaries(
        self,
        *,
        trainee_no: str = "",
        start_date: str = "",
        end_date: str = "",
        status: str = "",
        limit: int = 100,
    ) -> list[Any]:
        """按条件列日报（主管用；学员查询时调用方传自己的 trainee_no）。

        ★ 数据权限在调用方（用例层）保证，这里只管查。
        """

    # ---------------- 未命中日志 ----------------
    @abstractmethod
    def append_miss_log(self, row: MissLogRow) -> None:
        """追加一条未命中日志。"""

    @abstractmethod
    def list_miss_logs(self, *, limit: int = 100) -> list[MissLogRow]:
        """列出未命中日志（按时间倒序，主管用）。"""

    # ---------------- 演练评阅结果 ----------------
    @abstractmethod
    def save_drill_review(
        self,
        review: DrillReviewRow,
        risk_items: tuple[DrillRiskItemRow, ...] = (),
    ) -> None:
        """保存一次演练的评阅结果（主表 + 违规明细）。

        ★ 必须是**幂等**的：同一 `drill_id` 重复保存时覆盖旧记录，
          而不是插入第二条。原因：前端可能因网络问题重试评阅请求，
          重复插入会让月度统计翻倍。

        ★ 实现时应**先删该 drill_id 的明细再插新的**，
          否则重试时会累积重复的违规条目。
        """

    @abstractmethod
    def list_drill_reviews(
        self,
        *,
        trainee_id: str,
        since: str = "",
        until: str = "",
        with_items: bool = False,
    ) -> list[DrillReviewRow]:
        """按月拉取某学员的评阅结果（月度报告的数据源）。

        Args:
            trainee_id: 学员编号。
            since: 起始时间（`YYYY-MM-DD`，含）。空串表示不限。
            until: 结束时间（`YYYY-MM-DD`，不含）。空串表示不限。
            with_items: 是否一并把违规明细读出来。
                报告下钻需要明细，但纯聚合时不需要（省一次查询）。

        Returns:
            按时间升序。
        """

    @abstractmethod
    def get_drill_review(self, drill_id: str) -> DrillReviewRow | None:
        """按演练 ID 取评阅结果（含违规明细）。不存在返回 None。"""

    # ---------------- 月度报告定稿 ----------------
    @abstractmethod
    def save_monthly_report(self, row: MonthlyReportRow) -> None:
        """保存报告定稿（幂等：同 trainee_id + period 覆盖）。

        ★ 唯一约束是 `(trainee_id, period)`：
          同一人同月只应有一份定稿，重跑覆盖而不是新增。
        """

    @abstractmethod
    def get_monthly_report(
        self, trainee_id: str, period: str,
    ) -> MonthlyReportRow | None:
        """取某人的某月定稿。没有返回 None。"""

    # ---------------- 批量任务（P3 队列） ----------------
    @abstractmethod
    def upsert_report_task(
        self,
        *,
        task_id: str,
        trainee_id: str,
        period: str,
        status: str,
        error: str = "",
        thread_id: str = "",
    ) -> None:
        """写入或更新任务状态。

        ★ 幂等键是 `(trainee_id, period)`（唯一约束），
          不是 `task_id` —— 同一人同月只该有一条任务记录。
          重复投递时更新同一行，而不是新增。
        """

    @abstractmethod
    def get_report_task(self, trainee_id: str, period: str) -> dict | None:
        """按 `(trainee, period)` 取任务。没有返回 None。"""

    @abstractmethod
    def list_report_tasks(
        self, *, period: str = "", status: str = "", limit: int = 500,
    ) -> list[dict]:
        """列出任务（进度统计用）。参数为空表示不过滤。"""

    @abstractmethod
    def list_monthly_reports(
        self,
        *,
        period: str = "",
        status: str = "",
        trainee_id: str = "",
        limit: int = 200,
    ) -> list[MonthlyReportRow]:
        """列出报告（主管用）。参数为空表示不过滤。"""
