"""只读 Agent 工具的权限、参数与空数据回归。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from application.agents.tools import (
    build_agent_registry,
    build_readonly_registry,
    make_build_exam_tool,
)
from application.agents.types import AgentIdentity
from application.use_cases.batch_report import BatchResult
from domain.ports.repository_port import DrillReviewRow
from domain.value_objects.drill_review import CaseSpec
from domain.value_objects.role import Role


class _FakeGraph:
    def resolve_alias(self, alias: str):
        if alias == "鱼油":
            return SimpleNamespace(product_id="P001")
        return None

    def get_product_facts(self, product_id: str):
        if product_id != "P001":
            return None
        return SimpleNamespace(
            product_id="P001",
            name="康源牌鱼油软胶囊",
            category="保健食品",
            approval_no="国食健注G20250012",
            claims=(SimpleNamespace(text="有助于维持血脂健康水平"),),
            suitable_for=("血脂偏高者",),
            contraindications=(
                SimpleNamespace(name="少年儿童", severity=SimpleNamespace(must_block=False)),
                SimpleNamespace(name="出血倾向者", severity=SimpleNamespace(must_block=True)),
            ),
            usage=SimpleNamespace(dosage="每日2次，每次2粒", daily_limit="4g"),
        )


class _FakeRepo:
    def __init__(self) -> None:
        self.rows = [
            DrillReviewRow(
                drill_id="d1", trainee_id="E001", case_id="C01",
                case_type="contraindication", duration_sec=120,
                element_hit=0, element_total=2, risk_count=2,
                critical_count=1, created_at="2026-11-01"),
            DrillReviewRow(
                drill_id="d2", trainee_id="E001", case_id="C02",
                case_type="contraindication", duration_sec=90,
                element_hit=1, element_total=2, risk_count=1,
                critical_count=0, created_at="2026-11-02"),
        ]

    def list_drill_reviews(self, *, trainee_id, since="", until="", with_items=False):
        return [r for r in self.rows if r.trainee_id == trainee_id]


class _FakeCatalog:
    def list_all(self):
        return [
            CaseSpec(case_id="C01", title="禁忌案例", opening_line="你好",
                     question_type="contraindication", difficulty=2),
            CaseSpec(case_id="C02", title="价格案例", opening_line="你好",
                     question_type="price", difficulty=1),
            CaseSpec(case_id="C03", title="红线案例", opening_line="你好",
                     question_type="contraindication", difficulty=3,
                     is_red_line=True),
        ]


class _FakeAnswerUC:
    def execute(self, question, **kwargs):
        return SimpleNamespace(
            answer=f"回答：{question}",
            refused=False,
            refusal_reason="",
            cited_chunk_ids=("D005_c001",),
        )


def _registry(*, with_kb: bool = False):
    return build_readonly_registry(
        answer_use_case_factory=(lambda: _FakeAnswerUC()) if with_kb else None,
        graph=_FakeGraph(),
        repository=_FakeRepo(),
        case_catalog=_FakeCatalog(),
    )


class _FakeExamUC:
    def execute(self, spec, **kwargs):
        return SimpleNamespace(
            paper_no="S1-20261109-001",
            count=spec.total,
            status=SimpleNamespace(value="draft"),
            warnings=(),
        )


class _RecordingExamUC(_FakeExamUC):
    def __init__(self) -> None:
        self.spec = None
        self.bank = None

    def execute(self, spec, **kwargs):
        self.spec = spec
        self.bank = kwargs.get("bank")
        return super().execute(spec, **kwargs)


class _FakeReportUC:
    def enqueue_batch(self, *, period, trainee_ids, since="", until=""):
        return BatchResult(
            period=period, total=1, enqueued=1, skipped=0,
            task_ids=("task-001",))


def _write_registry():
    return build_agent_registry(
        answer_use_case_factory=None,
        graph=_FakeGraph(),
        repository=_FakeRepo(),
        case_catalog=_FakeCatalog(),
        exam_use_case_factory=lambda: _FakeExamUC(),
        report_use_case_factory=lambda: _FakeReportUC(),
    )


TRAINEE = AgentIdentity("E001", Role.TRAINEE)
MANAGER = AgentIdentity("M001", Role.MANAGER)


class TestToolRegistry:
    def test_trainee_sees_four_readonly_tools(self) -> None:
        names = {t["name"] for t in _registry().describe(TRAINEE)}
        assert names == {
            "kb_answer", "graph_facts", "my_drill_reviews", "weakness_summary",
        }

    def test_unknown_tool_is_rejected(self) -> None:
        result = _registry().run("drop_table", {}, TRAINEE)
        assert result.ok is False
        assert "未知工具" in result.error

    def test_timeout_is_reported(self) -> None:
        import time

        from application.agents.tool_registry import ToolRegistry, ToolSpec

        def slow(args, identity):
            time.sleep(1.0)
            return {"text": "too late"}

        registry = ToolRegistry([
            ToolSpec("slow", "slow", "ask", True, slow, timeout_sec=0.1),
        ])
        result = registry.run("slow", {}, TRAINEE)
        assert result.ok is False
        assert "超时" in result.error


class TestKbTool:
    def test_returns_answer_and_citations(self) -> None:
        result = _registry(with_kb=True).run(
            "kb_answer", {"question": "鱼油有什么功能"}, TRAINEE)
        assert result.ok is True
        assert "回答：鱼油有什么功能" in result.text
        assert result.data["citations"] == ["D005_c001"]

    def test_missing_question_is_error(self) -> None:
        result = _registry(with_kb=True).run("kb_answer", {}, TRAINEE)
        assert result.ok is False
        assert "question" in result.error


class TestGraphTool:
    def test_product_id_and_alias(self) -> None:
        registry = _registry()
        by_id = registry.run("graph_facts", {"product_id": "P001"}, TRAINEE)
        by_alias = registry.run("graph_facts", {"alias": "鱼油"}, TRAINEE)
        assert by_id.ok and by_alias.ok
        assert "康源牌鱼油软胶囊" in by_id.text
        assert "出血倾向者" in by_id.text
        assert "P001" in by_alias.text

    def test_missing_product_is_empty_not_error(self) -> None:
        result = _registry().run("graph_facts", {"product_id": "P999"}, TRAINEE)
        assert result.ok is True
        assert "未找到产品" in result.text


class TestDrillTools:
    def test_trainee_reads_self(self) -> None:
        result = _registry().run(
            "my_drill_reviews", {"days": 30}, TRAINEE)
        assert result.ok is True
        assert "共演练 2 次" in result.text

    def test_trainee_cannot_read_others(self) -> None:
        result = _registry().run(
            "my_drill_reviews", {"days": 30, "trainee_no": "E002"}, TRAINEE)
        assert result.ok is False
        assert result.denied is True

    def test_manager_can_read_others(self) -> None:
        result = _registry().run(
            "my_drill_reviews", {"days": 30, "trainee_no": "M001"}, MANAGER)
        assert result.ok is True
        assert "没有演练记录" in result.text

    def test_weakness_summary_recommends_cases(self) -> None:
        result = _registry().run(
            "weakness_summary", {"days": 30}, TRAINEE)
        assert result.ok is True
        assert result.data["weak_capabilities"]
        assert 1 <= len(result.data["recommendations"]) <= 3
        assert "建议练习" in result.text

    def test_empty_reviews_are_friendly(self) -> None:
        class _Empty(_FakeRepo):
            def list_drill_reviews(self, **kwargs):
                return []

        registry = build_readonly_registry(
            answer_use_case_factory=None, graph=_FakeGraph(),
            repository=_Empty(), case_catalog=_FakeCatalog())
        result = registry.run("weakness_summary", {}, TRAINEE)
        assert result.ok is True
        assert "没有演练记录" in result.text


class TestWriteTools:
    def test_manager_sees_six_tools(self) -> None:
        names = {t["name"] for t in _write_registry().describe(MANAGER)}
        assert names == {
            "kb_answer", "graph_facts", "my_drill_reviews", "weakness_summary",
            "build_exam", "submit_monthly_report",
        }

    def test_trainee_does_not_see_write_tools(self) -> None:
        names = {t["name"] for t in _write_registry().describe(TRAINEE)}
        assert "build_exam" not in names
        assert "submit_monthly_report" not in names

    def test_build_exam_returns_paper_no(self) -> None:
        result = _write_registry().run(
            "build_exam", {"stage": "S1", "total": 5}, MANAGER)
        assert result.ok is True
        assert "S1-20261109-001" in result.text
        assert result.data["total"] == 5

    def test_small_quiz_uses_knowledge_point_subset(self) -> None:
        uc = _RecordingExamUC()
        bank = [SimpleNamespace(knowledge_point=f"KP{i:02d}")
                for i in range(8)]
        registry = build_agent_registry(
            answer_use_case_factory=None, graph=_FakeGraph(),
            repository=_FakeRepo(), case_catalog=_FakeCatalog(),
            exam_use_case_factory=lambda: uc,
            exam_bank_loader=lambda stage: bank,
        )
        result = registry.run(
            "build_exam", {"stage": "S1", "total": 5}, MANAGER)
        assert result.ok is True
        assert len(uc.spec.knowledge_points) == 5
        assert uc.bank is bank
        assert "练习题模式" in result.text

    def test_trainee_cannot_build_exam(self) -> None:
        result = _write_registry().run(
            "build_exam", {"stage": "S1", "total": 5}, TRAINEE)
        assert result.ok is False
        assert result.denied is True

    def test_monthly_report_enqueues(self) -> None:
        result = _write_registry().run(
            "submit_monthly_report", {"period": "2026-11"}, MANAGER)
        assert result.ok is True
        assert "已投递" in result.text
        assert result.data["task_ids"] == ["task-001"]

    def test_trainee_cannot_submit_report(self) -> None:
        result = _write_registry().run(
            "submit_monthly_report", {}, TRAINEE)
        assert result.ok is False
        assert result.denied is True
