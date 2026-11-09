"""只读 Agent 工具：知识问答、图谱事实、演练评阅、薄弱项汇总。

工具 handler 只依赖端口/用例，不 import 基础设施；权限在 ToolRegistry
执行前校验，跨用户查询在 handler 内再次校验。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from application.agents.tool_registry import ToolRegistry, ToolSpec
from application.agents.types import AgentIdentity
from application.services.case_catalog import CaseCatalog
from application.use_cases.answer_question import AnswerQuestionUseCase
from domain.exceptions import ExamBuildError, PermissionDeniedError
from domain.ports import GraphQueryPort
from domain.services.monthly_report import build_report, capability_of
from domain.value_objects.exam import ExamSpec

logger = logging.getLogger(__name__)

AnswerFactory = Callable[[], AnswerQuestionUseCase]


def _clamp_days(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, min(90, value))


def _target_trainee(args: dict[str, Any], identity: AgentIdentity) -> str:
    requested = str(args.get("trainee_no") or "").strip()
    if not requested or requested == identity.employee_no:
        return identity.employee_no
    if not identity.is_manager:
        raise PermissionDeniedError("学员只能查看自己的演练记录")
    return requested


def _since(days: int) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


def make_kb_answer_tool(factory: AnswerFactory | None) -> ToolSpec:
    def handler(args: dict[str, Any], identity: AgentIdentity) -> dict[str, Any]:
        question = str(args.get("question") or "").strip()
        if not question:
            raise ValueError("question 不能为空")
        if factory is None:
            raise RuntimeError("LLM 未配置，知识问答不可用")
        uc = factory()
        result = uc.execute(
            question,
            vector_collection=str(
                args.get("collection") or "health_training_docs"),
            allowed_permissions=identity.retrieval_levels,
            top_k=20,
        )
        return {
            "text": result.answer or "暂无可用回答。",
            "data": {
                "refused": result.refused,
                "refusal_reason": result.refusal_reason,
                "citations": list(result.cited_chunk_ids)[:5],
            },
        }

    return ToolSpec(
        name="kb_answer",
        description="知识库问答，返回带出处与护栏的回答",
        required_feature="ask",
        read_only=True,
        handler=handler,
        timeout_sec=20.0,
    )


def make_graph_facts_tool(graph: GraphQueryPort | None) -> ToolSpec:
    def handler(args: dict[str, Any], identity: AgentIdentity) -> dict[str, Any]:
        if graph is None:
            raise RuntimeError("图谱未就绪")
        product_id = str(args.get("product_id") or "").strip().upper()
        alias = str(args.get("alias") or "").strip()
        if not product_id and alias:
            ref = graph.resolve_alias(alias)
            product_id = ref.product_id if ref else ""
        if not product_id:
            return {"text": f"未找到产品：{alias or '未提供'}", "data": {}}

        facts = graph.get_product_facts(product_id)
        if facts is None:
            return {"text": f"未找到产品：{product_id}", "data": {}}

        lines = [f"产品：{facts.name}（{facts.product_id}）",
                 f"类别：{facts.category}"]
        if facts.approval_no:
            lines.append(f"批准文号：{facts.approval_no}")
        if facts.claims:
            lines.append("批准功能：" + "；".join(
                f"「{c.text}」" for c in facts.claims))
        if facts.suitable_for:
            lines.append("适宜人群：" + "、".join(facts.suitable_for))
        if facts.contraindications:
            lines.append("不适宜人群：" + "、".join(
                c.name + ("（严重）" if c.severity.must_block else "")
                for c in facts.contraindications))
        if facts.usage and facts.usage.dosage:
            lines.append(f"用法用量：{facts.usage.dosage}")
        if facts.usage and facts.usage.daily_limit:
            lines.append(f"每日上限：{facts.usage.daily_limit}")

        return {
            "text": "\n".join(lines),
            "data": {
                "product_id": facts.product_id,
                "name": facts.name,
                "category": facts.category,
                "claims": [c.text for c in facts.claims],
                "contraindications": [c.name for c in facts.contraindications],
            },
        }

    return ToolSpec(
        name="graph_facts",
        description="按产品 ID 或别名查询批准功能、不适宜人群、用法",
        required_feature="ask",
        read_only=True,
        handler=handler,
        timeout_sec=5.0,
    )


def make_drill_reviews_tool(repository: Any) -> ToolSpec:
    def handler(args: dict[str, Any], identity: AgentIdentity) -> dict[str, Any]:
        if repository is None:
            raise RuntimeError("仓储未就绪")
        target = _target_trainee(args, identity)
        days = _clamp_days(args.get("days"), 7)
        reviews = repository.list_drill_reviews(
            trainee_id=target, since=_since(days), with_items=True)

        if not reviews:
            return {
                "text": f"{target} 近 {days} 天没有演练记录。",
                "data": {"trainee_id": target, "count": 0},
            }

        total_critical = sum(r.critical_count for r in reviews)
        total_risk = sum(r.risk_count for r in reviews)
        lines = [
            f"{target} 近 {days} 天共演练 {len(reviews)} 次，"
            f"critical {total_critical} 处，风险 {total_risk} 处。",
        ]
        for r in reviews[-10:]:
            lines.append(
                f"- {r.created_at or r.drill_id} 案例 {r.case_id}："
                f"要素 {r.element_hit}/{r.element_total}，"
                f"critical {r.critical_count}，风险 {r.risk_count}")
        return {
            "text": "\n".join(lines),
            "data": {
                "trainee_id": target,
                "count": len(reviews),
                "total_critical": total_critical,
                "reviews": [
                    {
                        "drill_id": r.drill_id,
                        "case_id": r.case_id,
                        "created_at": r.created_at,
                        "element_hit": r.element_hit,
                        "element_total": r.element_total,
                        "critical_count": r.critical_count,
                        "risk_count": r.risk_count,
                    }
                    for r in reviews
                ],
            },
        }

    return ToolSpec(
        name="my_drill_reviews",
        description="查看本人（主管可查指定学员）近 N 天演练评阅",
        required_feature="drill",
        read_only=True,
        handler=handler,
        timeout_sec=10.0,
    )


def make_weakness_summary_tool(
    repository: Any,
    case_catalog: CaseCatalog | None,
) -> ToolSpec:
    def handler(args: dict[str, Any], identity: AgentIdentity) -> dict[str, Any]:
        if repository is None:
            raise RuntimeError("仓储未就绪")
        target = _target_trainee(args, identity)
        days = _clamp_days(args.get("days"), 30)
        reviews = repository.list_drill_reviews(
            trainee_id=target, since=_since(days), with_items=True)

        if not reviews:
            return {
                "text": (
                    f"{target} 近 {days} 天没有演练记录，无法判断薄弱项。"
                    "建议先完成一次演练。"),
                "data": {"trainee_id": target, "count": 0},
            }

        cases = case_catalog.list_all() if case_catalog else []
        titles = {c.case_id: c.title for c in cases}
        report = build_report(
            trainee_id=target,
            period=date.today().strftime("%Y-%m"),
            reviews=reviews,
            case_titles=titles,
        )
        weak = list(report.weak_capabilities)
        capabilities = [
            {
                "capability": c.capability,
                "name": c.name,
                "samples": c.samples,
                "score": c.score,
                "level": c.level,
                "level_label": c.level_label,
                "critical_count": c.critical_count,
            }
            for c in report.capabilities
        ]

        practiced = {r.case_id for r in reviews}
        recommended = []
        if weak:
            for case in cases:
                if case.case_id in practiced:
                    continue
                if capability_of(case.question_type) in weak:
                    recommended.append(case)
        if len(recommended) < 3:
            seen = {c.case_id for c in recommended}
            for case in cases:
                if case.case_id in practiced or case.case_id in seen:
                    continue
                recommended.append(case)
                if len(recommended) >= 3:
                    break
        recommended = recommended[:3]

        lines = [
            f"{target} 近 {days} 天演练 {report.total_drills} 次，"
            f"critical {report.total_critical} 处。",
        ]
        if weak:
            lines.append("薄弱能力项：" + "、".join(weak))
        else:
            lines.append("暂无明显薄弱项（或样本不足）。")
        if recommended:
            lines.append("建议练习：" + "；".join(
                f"{c.case_id} {c.title}" + ("（红线）" if c.is_red_line else "")
                for c in recommended))

        return {
            "text": "\n".join(lines),
            "data": {
                "trainee_id": target,
                "weak_capabilities": weak,
                "capabilities": capabilities,
                "recommendations": [
                    {
                        "case_id": c.case_id,
                        "title": c.title,
                        "difficulty": c.difficulty,
                        "is_red_line": c.is_red_line,
                    }
                    for c in recommended
                ],
            },
        }

    return ToolSpec(
        name="weakness_summary",
        description="汇总薄弱能力项并推荐 3 个练习案例",
        required_feature="drill",
        read_only=True,
        handler=handler,
        timeout_sec=15.0,
    )


def _parse_quota(raw: Any) -> dict[int, int]:
    """解析难度配比：`1:3,2:9,3:8` 或 `3:9:8`。解析失败返回空（软约束）。"""
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        if "," in text and ":" in text:
            quota: dict[int, int] = {}
            for pair in text.split(","):
                key, value = pair.split(":", 1)
                difficulty = int(key)
                if difficulty not in (1, 2, 3):
                    return {}
                quota[difficulty] = max(0, int(value))
            return quota
        if text.count(":") == 2:
            values = [int(part) for part in text.split(":")]
            return {1: max(0, values[0]), 2: max(0, values[1]),
                    3: max(0, values[2])}
    except (TypeError, ValueError):
        return {}
    return {}


def make_build_exam_tool(
    use_case_factory: Callable[[], Any] | None,
    bank_loader: Callable[[str], list] | None = None,
) -> ToolSpec:
    def handler(args: dict[str, Any], identity: AgentIdentity) -> dict[str, Any]:
        if use_case_factory is None:
            raise RuntimeError("题库/仓储未就绪，无法组卷")
        stage = str(args.get("stage") or "S1").strip().upper()
        if stage not in ("S1", "S2"):
            stage = "S1"
        try:
            total = int(args.get("total") or 5)
        except (TypeError, ValueError):
            total = 5
        total = max(1, min(50, total))
        title = str(args.get("title") or f"{stage} Agent 小测").strip()
        trainee_no = str(args.get("trainee_no") or "").strip()
        persist = bool(args.get("persist", True))

        try:
            bank = bank_loader(stage) if bank_loader else None
        except Exception as exc:                  # noqa: BLE001 题库不可用要说清楚
            return {"text": f"组卷失败：题库不可用（{exc}）",
                    "data": {"error": str(exc)}}

        # 正式考试要求知识点全覆盖；Agent 的 5 题小测是练习题，
        # 题量小于知识点数时显式选一个子集，避免「5 题覆盖 8 个点」直接报错。
        knowledge_points: tuple[str, ...] = ()
        if bank:
            all_kps = list(dict.fromkeys(
                getattr(q, "knowledge_point", "")
                for q in bank if getattr(q, "knowledge_point", "")))
            if total < len(all_kps):
                knowledge_points = tuple(all_kps[:total])

        try:
            paper = use_case_factory().execute(
                ExamSpec(
                    total=total,
                    stage=stage,
                    knowledge_points=knowledge_points,
                    difficulty_quota=_parse_quota(args.get("difficulty")),
                ),
                title=title,
                created_by=identity.employee_no,
                trainee_id=trainee_no,
                persist=persist,
                bank=bank,
            )
        except ExamBuildError as exc:
            return {"text": f"组卷失败：{exc}", "data": {"error": str(exc)}}

        status = getattr(paper.status, "value", str(paper.status))
        warnings = list(getattr(paper, "warnings", ()) or ())
        lines = [
            f"已生成试卷 {paper.paper_no}（{paper.count} 题，状态 {status}）。",
            "请到「试卷」页定稿、导出或归档。",
        ]
        if knowledge_points:
            lines.append(
                f"练习题模式：本次覆盖 {len(knowledge_points)} 个知识点。")
        if warnings:
            lines.append("组卷提示：" + "；".join(warnings))
        return {
            "text": "\n".join(lines),
            "data": {
                "paper_no": paper.paper_no,
                "total": paper.count,
                "status": status,
                "knowledge_points": list(knowledge_points),
                "warnings": warnings,
            },
        }

    return ToolSpec(
        name="build_exam",
        description="按阶段/题量组一份试卷（主管）",
        required_feature="exam_manage",
        read_only=False,
        handler=handler,
        timeout_sec=20.0,
    )


def _report_period(raw: Any) -> str:
    text = str(raw or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return text
    today = date.today()
    if "上个月" in text:
        first = today.replace(day=1)
        return (first - timedelta(days=1)).strftime("%Y-%m")
    return today.strftime("%Y-%m")


def make_monthly_report_tool(
    use_case_factory: Callable[[], Any] | None,
) -> ToolSpec:
    def handler(args: dict[str, Any], identity: AgentIdentity) -> dict[str, Any]:
        if use_case_factory is None:
            raise RuntimeError("任务队列/仓储未就绪，无法生成月报")
        target = str(args.get("trainee_no") or identity.employee_no).strip()
        if target != identity.employee_no and not identity.is_manager:
            raise PermissionDeniedError("学员只能生成自己的月度报告")
        period = _report_period(args.get("period"))
        result = use_case_factory().enqueue_batch(
            period=period, trainee_ids=[target])
        if result.enqueued:
            text = (
                f"已投递 {period} 月度报告任务（学员 {target}），"
                "请到「月度报告」页查看并复核。")
        else:
            text = (
                f"{period} 月度报告任务已存在或正在处理（学员 {target}），"
                "未重复投递；请到「月度报告」页查看。")
        return {
            "text": text,
            "data": {
                "period": period,
                "trainee_id": target,
                "enqueued": result.enqueued,
                "skipped": result.skipped,
                "task_ids": list(result.task_ids),
            },
        }

    return ToolSpec(
        name="submit_monthly_report",
        description="投递月度报告生成任务，等待主管复核（主管）",
        required_feature="daily_review",
        read_only=False,
        handler=handler,
        timeout_sec=10.0,
    )


def build_agent_registry(
    *,
    answer_use_case_factory: AnswerFactory | None,
    graph: GraphQueryPort | None,
    repository: Any,
    case_catalog: CaseCatalog | None,
    exam_use_case_factory: Callable[[], Any] | None = None,
    exam_bank_loader: Callable[[str], list] | None = None,
    report_use_case_factory: Callable[[], Any] | None = None,
) -> ToolRegistry:
    tools = [
        make_kb_answer_tool(answer_use_case_factory),
        make_graph_facts_tool(graph),
        make_drill_reviews_tool(repository),
        make_weakness_summary_tool(repository, case_catalog),
    ]
    if exam_use_case_factory is not None:
        tools.append(make_build_exam_tool(
            exam_use_case_factory, exam_bank_loader))
    if report_use_case_factory is not None:
        tools.append(make_monthly_report_tool(report_use_case_factory))
    return ToolRegistry(tools)


#: 兼容只读阶段的旧名字。
build_readonly_registry = build_agent_registry
