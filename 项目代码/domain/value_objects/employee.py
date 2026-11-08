"""员工档案与每日总结（P7）。

★ 三个值对象都是 frozen —— 修改一律返回新实例。

为什么要这样：日报是「学员当时真实想法」的记录。
如果允许就地修改，序列化/缓存/并发时都可能出现
「读出来改了一半」的中间态。frozen + 返回新值，
让「哪些字段变了」在每个调用点都必须显式说明。

────────────────────────────────────────────────────────────────────
提交后冻结（FR-045）
────────────────────────────────────────────────────────────────────
    草稿 → 可反复修改
    已提交 → 不可修改（只允许主管加评语）

`can_edit` 一处判断，`edit()` 里强制校验。
不让调用方自己记得判断——那必然会漏。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import StrEnum

from domain.exceptions import DailySummaryFrozenError

#: 日期格式 YYYY-MM-DD
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ===================================================================
# 员工档案（FR-044）
# ===================================================================
@dataclass(frozen=True, slots=True)
class EmployeeProfile:
    """员工档案。

    需求要求的字段：姓名、工号、入职日期、岗位、所属团队、带教人。

    ★ 需求特别说明：「员工档案涉及雇佣关系，需保存真实姓名」。
      所以 `name` 不能为空——这不是昵称，是人事档案。
    """

    employee_no: str
    name: str
    hire_date: str = ""
    position: str = ""
    team: str = ""
    mentor: str = ""
    #: 是否在职（离职后保留档案，但不出现在导入名单里）
    active: bool = True

    def __post_init__(self) -> None:
        if not self.employee_no or not self.employee_no.strip():
            raise ValueError("工号不能为空")
        if not self.name or not self.name.strip():
            raise ValueError(
                "姓名不能为空（FR-044：员工档案需保存真实姓名）")
        if self.hire_date and not _DATE_RE.match(self.hire_date):
            raise ValueError(
                f"入职日期格式应为 YYYY-MM-DD，实际 {self.hire_date!r}")

    @property
    def display(self) -> str:
        """展示名：`张三（E001）`。"""
        return f"{self.name}（{self.employee_no}）"


# ===================================================================
# 每日总结（FR-045 / FR-046）
# ===================================================================
class SummaryStatus(StrEnum):
    """总结状态。"""

    DRAFT = "draft"
    SUBMITTED = "submitted"

    @property
    def label(self) -> str:
        return {"draft": "草稿", "submitted": "已提交"}[self.value]


@dataclass(frozen=True, slots=True)
class DailySummary:
    """学员每日总结。

    ★ 每人每天只能一条 —— 由数据库 `uk_trainee_date` 保证，
      这里不做唯一性判断（那是存储层的职责）。
    """

    trainee_no: str
    summary_date: str
    content: str = ""
    status: SummaryStatus = SummaryStatus.DRAFT
    submitted_at: str = ""
    #: 主管评语（可改，与正文不同——评语不冻结）
    review_note: str = ""
    reviewed_at: str = ""
    reviewed_by: str = ""

    def __post_init__(self) -> None:
        if not self.trainee_no or not self.trainee_no.strip():
            raise ValueError("学员工号不能为空")
        if not _DATE_RE.match(self.summary_date or ""):
            raise ValueError(
                f"日期格式应为 YYYY-MM-DD，实际 {self.summary_date!r}")
        # 空内容只允许在草稿态存在（已提交却空 → 数据异常）
        if self.status is SummaryStatus.SUBMITTED and not self.content.strip():
            raise ValueError("已提交的总结不能为空")

    # ---------------- 状态 ----------------
    @property
    def is_submitted(self) -> bool:
        return self.status is SummaryStatus.SUBMITTED

    @property
    def can_edit(self) -> bool:
        """★ 只有草稿可编辑（FR-045）。"""
        return self.status is SummaryStatus.DRAFT

    # ---------------- 编辑 ----------------
    def edit(self, content: str) -> DailySummary:
        """修改正文。

        Raises:
            DailySummaryFrozenError: 已提交，不可修改。

        ★ 校验放在方法内，不靠调用方记得先查 `can_edit`。
          少一个判断点，就少一次「忘了判断」的事故。
        """
        if not self.can_edit:
            raise DailySummaryFrozenError(
                "该总结已提交，不能修改。若需更正请联系主管。")
        if not content or not content.strip():
            raise ValueError("总结内容不能为空")
        return replace(self, content=content.strip())

    def submit(self, at: str = "") -> DailySummary:
        """提交（冻结）。

        ★ 重复提交是幂等的：不报错，但**提交时间不变**——
          否则学员可以靠重复提交改时间。
        """
        if self.is_submitted:
            return self
        if not self.content.strip():
            raise ValueError("内容为空不能提交")
        return replace(
            self,
            status=SummaryStatus.SUBMITTED,
            submitted_at=at or self.submitted_at,
        )

    # ---------------- 评语 ----------------
    def with_review(
        self,
        note: str,
        at: str = "",
        by: str = "",
    ) -> DailySummary:
        """填写/修改主管评语（FR-046：评语可修改，不冻结）。"""
        return replace(
            self,
            review_note=(note or "").strip(),
            reviewed_at=at if (note or "").strip() else "",
            reviewed_by=by if (note or "").strip() else "",
        )

    @property
    def has_review(self) -> bool:
        return bool(self.review_note.strip())


# ===================================================================
# 未命中日志（FR-047）
# ===================================================================
@dataclass(frozen=True, slots=True)
class MissLogEntry:
    """检索未命中记录。

    需求要求记录的字段：提问原文、提问学员、召回数量、提问时间。

    ★★ **刻意不含任何评分/评价字段**（FR-047 原文：
      「该日志不得用于对学员的绩效评价」）。

      这不是疏漏，是设计：把能力边界写进数据结构，
      比在文档里写一句「不要用于绩效」更可靠 ——
      字段不存在，就没人能拿它做绩效。

      这与「摘要卡不给分」是同一类约束。
    """

    query: str
    trainee_id: str = ""
    #: 本次召回到几条（0 表示完全没命中）
    recall_count: int = 0
    intent: str = ""
    created_at: str = ""
    #: 诊断信息（各路命中数等），只用于排查召回问题
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise ValueError("提问原文不能为空")
        if self.recall_count < 0:
            raise ValueError("召回数量不能为负")


__all__ = [
    "DailySummary",
    "EmployeeProfile",
    "MissLogEntry",
    "SummaryStatus",
]
