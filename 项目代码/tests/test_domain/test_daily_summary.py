"""员工与日报测试（P7）。

★ 需求来源：FR-044（员工档案）/ FR-045（每日总结）/ FR-046（主管评语）

────────────────────────────────────────────────────────────────────
四条硬规则（判错了就是业务事故）
────────────────────────────────────────────────────────────────────
1. **每人每天只允许一条总结**（FR-045）—— 重复提交会覆盖，不能产生两条
2. **草稿可反复修改，提交后不可修改**（FR-045）—— 这是「冻结」语义
3. **学员只能看自己的，学员看不到他人评语**（FR-045/046）
4. **未命中日志不得用于绩效评价**（FR-047）—— 数据模型上就要避免

第 4 条容易被忽略：需求原文明确写了
「该日志不得用于对学员的绩效评价」。
所以日志表**不能有评分字段**，也不该按学员排序展示。
这跟「摘要卡不给分」是同一类约束：**系统的能力边界写进数据结构**。
"""
from __future__ import annotations

import pytest

from domain.exceptions import (
    DailySummaryFrozenError,
    DataPermissionError,
    DuplicateEmployeeError,
)
from domain.value_objects import (
    DailySummary,
    EmployeeProfile,
    SummaryStatus,
)


# ===================================================================
# 1. 员工档案
# ===================================================================
class TestEmployeeProfile:
    def _emp(self, **kw) -> EmployeeProfile:
        base = dict(
            employee_no="E001",
            name="张三",
            hire_date="2026-10-01",
            position="健康顾问",
            team="一组",
            mentor="李四",
        )
        base.update(kw)
        return EmployeeProfile(**base)

    def test_basic_fields(self) -> None:
        """档案要含需求列的 6 个字段。"""
        e = self._emp()
        assert e.employee_no == "E001"
        assert e.name == "张三"
        assert e.hire_date == "2026-10-01"
        assert e.position == "健康顾问"
        assert e.team == "一组"
        assert e.mentor == "李四"

    def test_employee_no_required(self) -> None:
        with pytest.raises(ValueError, match="工号"):
            self._emp(employee_no="")

    def test_name_required(self) -> None:
        """★ 需求原文：「员工档案涉及雇佣关系，需保存真实姓名」。

        这不是普通的用户昵称，是人事档案，必须有真实姓名。
        """
        with pytest.raises(ValueError, match="姓名"):
            self._emp(name="")

    def test_hire_date_format(self) -> None:
        with pytest.raises(ValueError, match="入职日期"):
            self._emp(hire_date="2026/10/01")

    def test_optional_fields(self) -> None:
        """岗位/团队/带教人可以为空（实习生可能还没定组）。"""
        e = EmployeeProfile(employee_no="E002", name="王五",
                            hire_date="2026-10-02")
        assert e.position == ""
        assert e.team == ""


# ===================================================================
# 2. ★ 每日总结：每人每天一条 + 冻结语义
# ===================================================================
class TestDailySummary:
    def _sum(self, **kw) -> DailySummary:
        base = dict(
            trainee_no="E001",
            summary_date="2026-10-23",
            content="今天学习了鱼油的功能边界",
        )
        base.update(kw)
        return DailySummary(**base)

    def test_default_is_draft(self) -> None:
        s = self._sum()
        assert s.status is SummaryStatus.DRAFT
        assert not s.is_submitted

    def test_can_edit_draft(self) -> None:
        """草稿可反复修改。"""
        s = self._sum()
        assert s.can_edit

        s2 = s.edit("改后的内容")
        assert s2.content == "改后的内容"
        assert s2.can_edit

    def test_cannot_edit_after_submit(self) -> None:
        """★★ 提交后不可修改（FR-045 明确要求）。"""
        s = self._sum().submit()
        assert s.is_submitted
        assert not s.can_edit

        with pytest.raises(DailySummaryFrozenError):
            s.edit("想偷偷改")

    def test_submit_sets_time(self) -> None:
        s = self._sum().submit("2026-10-23 18:30:00")
        assert s.submitted_at == "2026-10-23 18:30:00"

    def test_submit_twice_idempotent(self) -> None:
        """重复提交不报错，但提交时间不变（防止改时间）。"""
        s = self._sum().submit("2026-10-23 18:30:00")
        s2 = s.submit("2026-10-23 20:00:00")
        assert s2.submitted_at == "2026-10-23 18:30:00"

    def test_empty_draft_allowed(self) -> None:
        """★ 空草稿是允许的 —— 学员可以先建一条今天的位置，中途再写。

        真正该拦的是「空内容提交」，见下面的测试。
        """
        s = self._sum(content="")
        assert s.status is SummaryStatus.DRAFT
        assert not s.is_submitted

    def test_empty_submit_rejected(self) -> None:
        """★ 空内容不能提交。"""
        with pytest.raises(ValueError, match="内容"):
            self._sum(content="").submit()

    def test_whitespace_submit_rejected(self) -> None:
        with pytest.raises(ValueError, match="内容"):
            self._sum(content="   \n  ").submit()

    def test_empty_edit_rejected(self) -> None:
        """草稿能存在但不能再改成空（避免误清空已有内容）。"""
        s = self._sum(content="已有内容")
        with pytest.raises(ValueError, match="内容"):
            s.edit("   ")

    def test_date_format(self) -> None:
        with pytest.raises(ValueError, match="日期"):
            self._sum(summary_date="2026/10/23")

    def test_review_note_visible_to_trainee(self) -> None:
        """学员能看到自己的评语。"""
        s = self._sum()
        s2 = s.with_review("注意区分保健食品和普通食品", "2026-10-24 09:00")
        assert "保健食品" in s2.review_note
        assert s2.reviewed_at == "2026-10-24 09:00"


# ===================================================================
# 3. ★ 主管评语
# ===================================================================
class TestReviewNote:
    def test_review_can_be_modified(self) -> None:
        """FR-046：评语可修改（与总结不同，评语不冻结）。"""
        s = DailySummary(trainee_no="E001", summary_date="2026-10-23",
                         content="内容").with_review("初版评语")
        s2 = s.with_review("改后的评语")
        assert s2.review_note == "改后的评语"

    def test_review_does_not_change_status(self) -> None:
        s = DailySummary(trainee_no="E001", summary_date="2026-10-23",
                         content="内容").submit()
        s2 = s.with_review("评语")
        assert s2.status is SummaryStatus.SUBMITTED

    def test_empty_review_clears(self) -> None:
        s = DailySummary(trainee_no="E001", summary_date="2026-10-23",
                         content="内容").with_review("评语")
        s2 = s.with_review("")
        assert s2.review_note == ""


# ===================================================================
# 4. ★ 数据权限
# ===================================================================
class TestDataPermission:
    """FR-049 数据权限：学员只能看自己的。"""

    def test_trainee_can_view_own(self) -> None:
        from domain.rules.data_permission import can_view_summary

        s = DailySummary(trainee_no="E001", summary_date="2026-10-23",
                         content="内容")
        assert can_view_summary(requester="E001", role="trainee",
                                summary=s)

    def test_trainee_cannot_view_others(self) -> None:
        from domain.rules.data_permission import can_view_summary

        s = DailySummary(trainee_no="E001", summary_date="2026-10-23",
                         content="内容")
        assert not can_view_summary(requester="E002", role="trainee",
                                    summary=s)

    def test_manager_can_view_all(self) -> None:
        from domain.rules.data_permission import can_view_summary

        s = DailySummary(trainee_no="E001", summary_date="2026-10-23",
                         content="内容")
        assert can_view_summary(requester="M001", role="manager",
                                summary=s)

    def test_trainee_cannot_edit_others(self) -> None:
        from domain.rules.data_permission import can_edit_summary

        s = DailySummary(trainee_no="E001", summary_date="2026-10-23",
                         content="内容")
        assert not can_edit_summary(requester="E002", role="trainee",
                                    summary=s)

    def test_manager_cannot_edit_trainee_content(self) -> None:
        """★ 主管能写评语，但**不能改学员正文**。

        这是重要的边界：改了正文就等于篡改学员的原始记录，
        日报就失去了「学员真实反馈」的价值。
        """
        from domain.rules.data_permission import can_edit_summary

        s = DailySummary(trainee_no="E001", summary_date="2026-10-23",
                         content="内容")
        assert not can_edit_summary(requester="M001", role="manager",
                                    summary=s)

    def test_manager_cannot_edit_after_submit(self) -> None:
        from domain.rules.data_permission import can_edit_summary

        s = DailySummary(trainee_no="E001", summary_date="2026-10-23",
                         content="内容").submit()
        assert not can_edit_summary(requester="E001", role="trainee",
                                    summary=s)

    def test_enforce_raises(self) -> None:
        """越权访问要抛异常而不是静默返回。"""
        from domain.rules.data_permission import require_view_summary

        s = DailySummary(trainee_no="E001", summary_date="2026-10-23",
                         content="内容")
        with pytest.raises(DataPermissionError):
            require_view_summary(requester="E002", role="trainee", summary=s)


# ===================================================================
# 5. ★ 未命中日志：不得用于绩效
# ===================================================================
class TestMissLogConstraint:
    def test_no_score_field(self) -> None:
        """★★ FR-047 原文：「该日志不得用于对学员的绩效评价」。

        所以日志结构里**不能有评分/评价类字段**。
        这是把能力边界写进数据结构，而不是靠自觉。
        """
        from domain.value_objects import MissLogEntry
        import dataclasses

        fields = {f.name for f in dataclasses.fields(MissLogEntry)}
        for banned in ("score", "grade", "rating", "evaluation",
                       "performance", "verdict"):
            assert banned not in fields, (
                f"未命中日志出现 {banned} 字段 —— "
                f"FR-047 明确禁止用于绩效评价"
            )

    def test_has_required_fields(self) -> None:
        """需求列的四个字段要有：提问原文 / 提问学员 / 召回数量 / 时间。"""
        from domain.value_objects import MissLogEntry
        import dataclasses

        fields = {f.name for f in dataclasses.fields(MissLogEntry)}
        assert "query" in fields
        assert "trainee_id" in fields
        assert "recall_count" in fields
        assert "created_at" in fields


# ===================================================================
# 6. 员工导入
# ===================================================================
class TestEmployeeImport:
    def test_parse_csv_rows(self) -> None:
        """批量导入要能解析（需求要求支持批量导入）。"""
        from application.services.employee_import import parse_employee_rows

        rows = [
            {"工号": "E001", "姓名": "张三", "入职日期": "2026-10-01",
             "岗位": "健康顾问", "团队": "一组", "带教人": "李四"},
            {"工号": "E002", "姓名": "王五", "入职日期": "2026-10-02",
             "岗位": "", "团队": "", "带教人": ""},
        ]
        emps = parse_employee_rows(rows)
        assert len(emps) == 2
        assert emps[0].name == "张三"
        assert emps[1].team == ""

    def test_skip_invalid_rows(self) -> None:
        """坏行跳过并报告，不能让整批导入失败。"""
        from application.services.employee_import import parse_employee_rows

        rows = [
            {"工号": "E001", "姓名": "张三", "入职日期": "2026-10-01"},
            {"工号": "", "姓名": "缺工号", "入职日期": "2026-10-02"},
            {"工号": "E003", "姓名": "", "入职日期": "2026-10-03"},
        ]
        emps = parse_employee_rows(rows)
        assert len(emps) == 1, "坏行应被跳过"

    def test_duplicate_in_batch_kept_once(self) -> None:
        from application.services.employee_import import parse_employee_rows

        rows = [
            {"工号": "E001", "姓名": "张三", "入职日期": "2026-10-01"},
            {"工号": "E001", "姓名": "张三", "入职日期": "2026-10-01"},
        ]
        emps = parse_employee_rows(rows)
        assert len(emps) == 1, "同批重复工号应去重"

    def test_english_headers(self) -> None:
        """也要支持英文表头（导出的 CSV 常见）。"""
        from application.services.employee_import import parse_employee_rows

        rows = [{"employee_no": "E001", "name": "张三",
                 "hire_date": "2026-10-01"}]
        emps = parse_employee_rows(rows)
        assert emps[0].employee_no == "E001"
