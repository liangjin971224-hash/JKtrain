"""只读 Agent 的规划、预算、护栏与权限回归。"""

from __future__ import annotations

import pytest

from application.agents.planner import plan_steps
from application.agents.tool_registry import ToolRegistry, ToolSpec
from application.agents.training_agent import (
    PERMISSION_REFUSAL,
    SAFE_REFUSAL,
    TrainingAgent,
)
from application.agents.types import AgentIdentity
from domain.exceptions import PermissionDeniedError
from domain.value_objects.role import Role

TRAINEE = AgentIdentity("E001", Role.TRAINEE)
MANAGER = AgentIdentity("M001", Role.MANAGER)


class _Tools:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def spec(self, name: str, text: str, *, feature: str = "ask") -> ToolSpec:
        def handler(args, identity):
            self.calls.append(name)
            return {"text": text, "data": {}}
        return ToolSpec(name, name, feature, True, handler, timeout_sec=2.0)

    def registry(self, *, kb_text: str = "鱼油回答") -> ToolRegistry:
        return ToolRegistry([
            self.spec("my_drill_reviews", "近7天演练2次", feature="drill"),
            self.spec("weakness_summary", "薄弱项：禁忌判断", feature="drill"),
            self.spec("graph_facts", "产品事实：鱼油"),
            self.spec("kb_answer", kb_text),
        ])

    def write_registry(self) -> ToolRegistry:
        return ToolRegistry([
            self.spec("build_exam", "已生成试卷 S1-20261109-001",
                      feature="exam_manage"),
            self.spec("submit_monthly_report", "已投递月报任务",
                      feature="daily_review"),
        ])


class TestPlanner:
    def test_drill_question_uses_structured_tools_only(self) -> None:
        tools = [s.tool for s in plan_steps("我上周练得怎么样？推荐案例")]
        assert tools == ["my_drill_reviews", "weakness_summary"]

    def test_product_question_uses_graph_then_kb(self) -> None:
        tools = [s.tool for s in plan_steps("鱼油有什么功能")]
        assert tools == ["graph_facts", "kb_answer"]

    def test_contraindication_question_extracts_alias(self) -> None:
        steps = plan_steps("客户在吃阿司匹林能吃鱼油吗")
        graph = next(s for s in steps if s.tool == "graph_facts")
        assert graph.args["alias"] == "鱼油"

    def test_unknown_question_falls_back_to_kb(self) -> None:
        tools = [s.tool for s in plan_steps("帮我看看这个怎么办")]
        assert tools == ["kb_answer"]

    def test_plan_is_capped_at_four_steps(self) -> None:
        steps = plan_steps("我上周练得怎么样，鱼油禁忌是什么")
        assert len(steps) <= 4
        assert [s.tool for s in steps] == [
            "my_drill_reviews", "weakness_summary", "graph_facts", "kb_answer",
        ]

    def test_exam_question_plans_build_exam(self) -> None:
        steps = plan_steps("出 5 道题")
        assert [s.tool for s in steps] == ["build_exam"]
        assert steps[0].args["total"] == 5

    def test_report_question_plans_submit(self) -> None:
        steps = plan_steps("生成本月月报")
        assert [s.tool for s in steps] == ["submit_monthly_report"]

    def test_weakness_to_exam_combines_tools(self) -> None:
        steps = plan_steps("根据薄弱项出 5 题")
        assert [s.tool for s in steps] == [
            "my_drill_reviews", "weakness_summary", "build_exam",
        ]


class TestTrainingAgent:
    def test_kb_and_graph_are_combined(self) -> None:
        tools = _Tools()
        agent = TrainingAgent(tools.registry())
        result = agent.run("鱼油有什么功能", TRAINEE)
        assert result.refused is False
        assert "鱼油回答" in result.answer
        assert "产品事实" in result.answer
        assert [s.tool for s in result.steps] == ["graph_facts", "kb_answer"]
        assert tools.calls == ["graph_facts", "kb_answer"]

    def test_pure_drill_question_skips_kb(self) -> None:
        tools = _Tools()
        agent = TrainingAgent(tools.registry())
        result = agent.run("我上周练得怎么样？推荐案例", TRAINEE)
        assert result.refused is False
        assert "薄弱项" in result.answer
        assert "鱼油回答" not in result.answer
        assert tools.calls == ["my_drill_reviews", "weakness_summary"]

    def test_duplicate_no_data_is_deduped(self) -> None:
        tools = _Tools()
        registry = ToolRegistry([
            tools.spec("my_drill_reviews", "E001 近7天没有演练记录。",
                       feature="drill"),
            tools.spec("weakness_summary",
                       "E001 近7天没有演练记录，无法判断薄弱项。",
                       feature="drill"),
        ])
        result = TrainingAgent(registry).run("我上周练得怎么样", TRAINEE)
        assert result.answer.count("没有演练记录") == 1

    def test_permission_denied_becomes_safe_refusal(self) -> None:
        def denied(args, identity):
            raise PermissionDeniedError("学员只能查看自己的演练记录")

        registry = ToolRegistry([
            ToolSpec("my_drill_reviews", "denied", "drill", True, denied, 2.0),
        ])
        # 纯演练问题只走这个工具
        result = TrainingAgent(registry).run("我上周练得怎么样", TRAINEE)
        assert result.refused is True
        assert result.answer == PERMISSION_REFUSAL
        assert result.steps[0].denied is True

    def test_all_tools_fail_becomes_safe_refusal(self) -> None:
        def boom(args, identity):
            raise RuntimeError("db down")

        registry = ToolRegistry([
            ToolSpec("kb_answer", "boom", "ask", True, boom, 2.0),
        ])
        result = TrainingAgent(registry).run("鱼油有什么功能", TRAINEE)
        assert result.refused is True
        assert result.answer == SAFE_REFUSAL
        assert "db down" in result.refusal_reason

    def test_guardrail_blocks_unsafe_kb_text(self) -> None:
        tools = _Tools()
        agent = TrainingAgent(tools.registry(kb_text="这个产品可以治疗糖尿病"))
        result = agent.run("鱼油有什么功能", TRAINEE)
        assert "治疗糖尿病" not in result.answer
        assert "部分内容被拦截" in result.refusal_reason
        # 保留的只能是结构化安全内容，不能把违规句放出去
        assert result.refused or "产品事实" in result.answer

    def test_plan_is_bounded(self) -> None:
        tools = _Tools()
        agent = TrainingAgent(tools.registry())
        result = agent.run("我上周练得怎么样，鱼油禁忌是什么", TRAINEE)
        assert len(result.plan) <= 4
        assert len(tools.calls) <= 4

    def test_manager_builds_exam(self) -> None:
        agent = TrainingAgent(_Tools().write_registry())
        result = agent.run("出 5 道题", MANAGER)
        assert result.refused is False
        assert "S1-20261109-001" in result.answer
        assert [s.tool for s in result.steps] == ["build_exam"]

    def test_trainee_exam_is_denied(self) -> None:
        agent = TrainingAgent(_Tools().write_registry())
        result = agent.run("出 5 道题", TRAINEE)
        assert result.refused is True
        assert result.answer == PERMISSION_REFUSAL
