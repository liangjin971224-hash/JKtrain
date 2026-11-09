"""培训助教 Agent（只读版）的 LangGraph 编排。

    plan → act → act …（最多 4 步）→ synthesize → guardrail → respond

- 规划是规则（见 planner.py），不调 LLM；
- 工具调用由 ToolRegistry 做权限/超时；
- 汇总用已护栏的 KB 答案 + 结构化模板，不额外调 LLM；
- 最终文本仍过一次 OutputGuardrail，护栏是必经节点。
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from application.agents.planner import plan_steps
from application.agents.tool_registry import ToolRegistry
from application.agents.types import AgentAnswer, AgentIdentity, AgentStep, ToolResult
from domain.services.output_guardrail import OutputGuardrail

logger = logging.getLogger(__name__)

SAFE_REFUSAL = (
    "抱歉，这次任务没有拿到足够的数据。建议换个说法，"
    "或到对应的问答/演练/日报页面查看。"
)
PERMISSION_REFUSAL = "抱歉，当前身份无权执行该任务。"
MAX_STEPS = 4


class AgentState(TypedDict, total=False):
    question: str
    identity: AgentIdentity
    plan: list[dict[str, Any]]
    cursor: int
    results: list[ToolResult]
    steps: list[AgentStep]
    answer: str
    refused: bool
    refusal_reason: str
    citations: tuple[str, ...]


class TrainingAgent:
    """只读版培训助教 Agent。"""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        guardrail: OutputGuardrail | None = None,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self._registry = registry
        self._guardrail = guardrail or OutputGuardrail()
        self._max_steps = max(1, min(int(max_steps), MAX_STEPS))
        self._graph = self._build_graph()

    # ---------------- 对外 ----------------
    def run(self, question: str, identity: AgentIdentity) -> AgentAnswer:
        final = self._graph.invoke({
            "question": (question or "").strip(),
            "identity": identity,
        })
        return AgentAnswer(
            answer=final.get("answer") or SAFE_REFUSAL,
            refused=bool(final.get("refused")),
            refusal_reason=final.get("refusal_reason", ""),
            plan=tuple(final.get("plan") or ()),
            steps=tuple(final.get("steps") or ()),
            citations=tuple(final.get("citations") or ()),
        )

    # ---------------- 图 ----------------
    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan_node)
        graph.add_node("act", self._act_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.add_node("guardrail", self._guardrail_node)
        graph.add_node("respond", lambda state: {})
        graph.add_edge(START, "plan")
        graph.add_conditional_edges(
            "plan",
            lambda state: "act" if state.get("plan") else "synthesize",
            {"act": "act", "synthesize": "synthesize"},
        )
        graph.add_conditional_edges(
            "act",
            lambda state: (
                "act"
                if state.get("cursor", 0) < len(state.get("plan") or ())
                else "synthesize"
            ),
            {"act": "act", "synthesize": "synthesize"},
        )
        graph.add_edge("synthesize", "guardrail")
        graph.add_edge("guardrail", "respond")
        graph.add_edge("respond", END)
        return graph.compile()

    def _plan_node(self, state: AgentState) -> dict[str, Any]:
        steps = plan_steps(state.get("question", ""), max_steps=self._max_steps)
        return {
            "plan": [
                {"tool": step.tool, "args": step.args, "reason": step.reason}
                for step in steps
            ],
            "cursor": 0,
            "results": [],
            "steps": [],
        }

    def _act_node(self, state: AgentState) -> dict[str, Any]:
        plan = state.get("plan") or []
        cursor = int(state.get("cursor", 0))
        if cursor >= len(plan):
            return {}
        item = plan[cursor]
        result = self._registry.run(
            str(item.get("tool", "")),
            dict(item.get("args") or {}),
            state["identity"],
        )
        results = list(state.get("results") or [])
        results.append(result)
        steps = list(state.get("steps") or [])
        steps.append(AgentStep(
            tool=result.tool,
            ok=result.ok,
            elapsed_ms=result.elapsed_ms,
            summary=(result.text or result.error or "")[:200],
            denied=result.denied,
        ))
        return {"results": results, "steps": steps, "cursor": cursor + 1}

    def _synthesize_node(self, state: AgentState) -> dict[str, Any]:
        answer, refused, reason, citations = self._synthesize(
            list(state.get("results") or ()))
        return {
            "answer": answer,
            "refused": refused,
            "refusal_reason": reason,
            "citations": citations,
        }

    def _guardrail_node(self, state: AgentState) -> dict[str, Any]:
        if state.get("refused"):
            return {}
        answer = state.get("answer", "")
        verdict = self._guardrail.check(answer, category=None)
        if not verdict.blocked:
            return {}
        annotation = self._guardrail.annotate(answer, category=None)
        if annotation.has_safe_content and annotation.safe_text.strip():
            return {
                "answer": annotation.safe_text,
                "refusal_reason": f"部分内容被拦截：{verdict.reason}",
            }
        return {
            "answer": SAFE_REFUSAL,
            "refused": True,
            "refusal_reason": f"输出护栏拦截: {verdict.reason}",
        }

    # ---------------- 汇总 ----------------
    @staticmethod
    def _synthesize(
        results: list[ToolResult],
    ) -> tuple[str, bool, str, tuple[str, ...]]:
        if any(r.denied for r in results):
            return PERMISSION_REFUSAL, True, "权限不足", ()

        kb = [r for r in results if r.tool == "kb_answer" and r.ok]
        base = kb[-1].text if kb else ""
        citations = tuple(kb[-1].data.get("citations") or ()) if kb else ()
        structured = TrainingAgent._dedupe_structured([
            r.text for r in results
            if r.tool != "kb_answer" and r.ok and r.text.strip()
        ])

        if base and structured:
            answer = f"{base}\n\n【补充数据】\n" + "\n\n".join(structured)
        elif base:
            answer = base
        elif structured:
            answer = "\n\n".join(structured)
        else:
            errors = "；".join(r.error for r in results if r.error) or "没有可用数据"
            return SAFE_REFUSAL, True, f"工具均未返回可用结果（{errors}）", ()

        if kb and kb[-1].data.get("refused") and not structured:
            return answer, True, str(
                kb[-1].data.get("refusal_reason") or "知识库拒答"), citations
        return answer, False, "", citations

    @staticmethod
    def _dedupe_structured(texts: list[str]) -> list[str]:
        """去掉互相包含的重复段落（如两个工具都说「没有演练记录」）。"""
        out: list[str] = []
        for text in texts:
            text = text.strip()
            if not text:
                continue
            normalized = text.rstrip("。；; ")
            for index, existing in enumerate(out):
                existing_norm = existing.rstrip("。；; ")
                if normalized in existing_norm:
                    break
                if existing_norm in normalized:
                    out[index] = text
                    break
            else:
                out.append(text)
        return out
