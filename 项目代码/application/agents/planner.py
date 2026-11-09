"""只读 Agent 的规则规划（v1 不调 LLM）。

规则优先的理由：可复现、零额外延迟、不会出现「模型编一个不存在的工具」。
未知问法直接落到 `kb_answer`，由知识库链路兜底。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_PRODUCT_RE = re.compile(r"P\d{3}", re.IGNORECASE)
_TRAINEE_RE = re.compile(r"[EM]\d{3}")
_PERIOD_RE = re.compile(r"\d{4}-\d{2}")
_ALIAS_RE = re.compile(
    r"(鱼油|辅酶Q10|蛋白粉|乳清蛋白|孢子粉|螺旋藻|钙维生素D|褪黑素|"
    r"益生菌|叶黄素|氨糖|蜂胶|胶原蛋白|维生素C|代餐奶昔|固体饮料)"
)
_DRILL_WORDS = (
    "练得", "演练", "评阅", "薄弱", "哪里不", "推荐案例", "练哪个",
    "上周", "本月练", "最近练",
)
_EXAM_WORDS = ("组卷", "出题", "小测", "试卷", "考一下", "几道题", "测验")
_EXAM_RE = re.compile(r"\d+\s*(?:道|题)")
_REPORT_WORDS = ("月报", "月度报告", "这个月报告", "生成报告", "月度培训报告")


@dataclass(frozen=True, slots=True)
class PlannedStep:
    tool: str
    args: dict[str, Any]
    reason: str


def plan_steps(question: str, *, max_steps: int = 4) -> list[PlannedStep]:
    """把问题映射成有界的工具序列。"""
    q = (question or "").strip()
    steps: list[PlannedStep] = []

    def add(tool: str, args: dict[str, Any], reason: str) -> None:
        if any(s.tool == tool for s in steps):
            return
        steps.append(PlannedStep(tool=tool, args=args, reason=reason))

    drill = any(word in q for word in _DRILL_WORDS)
    exam = any(word in q for word in _EXAM_WORDS) or bool(_EXAM_RE.search(q))
    report = any(word in q for word in _REPORT_WORDS)
    product_match = _PRODUCT_RE.search(q)
    alias_match = _ALIAS_RE.search(q)
    trainee_match = _TRAINEE_RE.search(q)
    trainee_no = trainee_match.group(0) if trainee_match else ""
    # 只有明确的产品 ID/别名才查图谱；「副作用/禁忌」这类泛词交给知识库。
    has_product = bool(product_match or alias_match)
    days = 7 if "上周" in q else 30
    action = exam or report

    if drill:
        args: dict[str, Any] = {"days": days}
        if trainee_no:
            args["trainee_no"] = trainee_no
        add("my_drill_reviews", args, "问题涉及近期演练记录")
        add("weakness_summary", dict(args), "需要薄弱项与案例推荐")

    if exam:
        total_match = re.search(r"(\d+)\s*(?:道|题)", q)
        total = int(total_match.group(1)) if total_match else 5
        stage = "S2" if ("S2" in q or "业务流程" in q) else "S1"
        args = {"stage": stage, "total": total}
        if trainee_no:
            args["trainee_no"] = trainee_no
        add("build_exam", args, "需要生成一份练习试卷（主管）")

    if report:
        period_match = _PERIOD_RE.search(q)
        period = (period_match.group(0) if period_match
                  else ("上个月" if "上个月" in q else ""))
        args = {"period": period}
        if trainee_no:
            args["trainee_no"] = trainee_no
        add("submit_monthly_report", args,
            "需要投递月度报告生成任务（主管）")

    if has_product and not action:
        args: dict[str, Any] = {}
        if product_match:
            args["product_id"] = product_match.group(0).upper()
        elif alias_match:
            args["alias"] = alias_match.group(0)
        add("graph_facts", args, "问题涉及产品事实，先查图谱")

    # 纯演练/纯写任务不补知识库；其余问题补一次知识库回答。
    if not action and (not drill or has_product):
        add("kb_answer", {"question": q}, "需要基于知识库给出完整回答")

    return steps[:max_steps]
