"""月度培训报告：同步版用例（P1）。

────────────────────────────────────────────────────────────────────
这一版的定位：**不上任何框架，先验证「报告有没有人看」**
────────────────────────────────────────────────────────────────────
没有 LangGraph、没有 RabbitMQ。就是：
    拉数据 → 聚合（领域服务）→ LLM 生成建议 → 护栏校验 → 返回

★ 为什么先做这一版：
  如果主管不看这份报告，后面加图、加队列全是白做。
  这一步是**价值验证**，不是技术验证。

────────────────────────────────────────────────────────────────────
三条设计约束
────────────────────────────────────────────────────────────────────
1. **分数只在报告里**，绝不写回 `SummaryCard`（约束 #3）。
2. **建议必须过护栏**——避免建议里出现「你可以说降血脂」这种反向教错。
3. **建议必须绑定具体案例**——空泛建议（「要注意合规」）没有价值，
   提示词里强制要求引用案例编号与具体违规句。

────────────────────────────────────────────────────────────────────
LLM 调用成本
────────────────────────────────────────────────────────────────────
每次报告 **1 次**调用（只在有薄弱项时才调）。
没有薄弱项时不调——那种情况硬凑建议就是让模型编。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from domain.services.monthly_report import (
    LEVEL_LABELS,
    MonthlyReport,
    build_report,
)
from domain.services.report_guardrail import ReportGuardrail

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GeneratedSuggestion:
    """生成的改进建议及其校验结果。"""

    text: str
    #: 是否通过护栏
    passed: bool = True
    #: 未通过时的原因
    reason: str = ""
    #: 重写次数（0 表示一次通过）
    retries: int = 0
    #: 是否根本没有生成（无薄弱项）
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class MonthlyReportResult:
    """报告的最终交付物。"""

    report: MonthlyReport
    suggestion: GeneratedSuggestion
    #: 供前端/导出用的结构化摘要
    summary: dict[str, Any]


def build_suggestion_prompt(
    report: MonthlyReport,
    doc_catalog: dict[str, str] | None = None,
) -> str:
    """构造「改进建议」的提示词。

    ★ 三条硬要求写进提示词：
      1. 必须引用具体案例编号和具体违规位置
      2. 不许出现功效宣称（护栏会复核）
      3. 不许教「怎么说更好」的违规话术（只给练习方向）

    ★ 为什么把薄弱项明细塞进 prompt 而不是让模型自己总结：
      模型不知道哪个员工什么时候错在哪句，必须把事实喂进去。
      这也是「建议绑定案例」能落地的前提。

    ★★ 为什么要传 `doc_catalog`（2026-10-31 实测发现）：
      不给目录时，模型会**编造资料名**。实测它写了
      「《禁忌话术禁用词表（V3.2）》第4章」「《配伍禁忌应答四步法》第2章」
      「《健康顾问合规守则》第1.3条」——**这三份资料都不存在**。
      学员按建议去找根本找不到，建议就成了空话。

      解法与 `SemanticEfficacyMatcher` 一致：**给封闭词表**。
      只允许从真实文档清单里引用，模型就没有编造空间。

    Args:
        doc_catalog: `doc_id → 文档标题`（来自知识库真实数据）。
            空时退化为「不许提具体资料名」的保守模式。
    """
    lines: list[str] = []
    lines.append("你在为保健品门店的健康顾问写月度培训改进建议。")
    lines.append("")
    lines.append("【本月表现】")
    weak_caps = set(report.weak_capabilities)

    for cap in report.capabilities:
        score_text = (f"{cap.score} 分" if cap.score is not None
                      else cap.level_label)
        mark = "  ← 薄弱" if cap.capability in weak_caps else ""
        lines.append(f"- {cap.name}：{score_text}（{cap.samples} 次演练）{mark}")

    # 只把薄弱项的明细喂进去，避免提示词过长
    lines.append("")
    lines.append("【薄弱项的具体问题】")
    for o in report.outcomes:
        if o.capability not in weak_caps:
            continue
        lines.append(f"- 案例 {o.case_id}《{o.case_title}》："
                     f"必答要素 {o.element_hit}/{o.element_total}，"
                     f"违规 {o.risk_count} 条")
        for r in o.risks:
            lines.append(
                f"    · {r['location']}「{r['matched']}」"
                f"（{r['rule_type']}，{r['level']}）")

    lines.append("")
    lines.append("【写作要求】")
    lines.append("1. 每条建议必须引用具体案例编号和违规位置，例："
                 "「C07 第1轮第1句说『肯定有效果』属于功效声称」。")
    lines.append("2. 不许出现任何产品功效表述（如降血脂、增强免疫力）。")
    lines.append("3. 不要教怎么说才更打动人，只给练习方向"
                 "（该重练哪个案例、该复习哪一章）。")
    lines.append("4. 总字数 200–400 字，分 2–3 条，条与条之间空行。")
    lines.append("5. 直接输出建议正文，不要标题、不要解释你在做什么。")

    # ★ 封闭词表：只许引用真实存在的资料
    if doc_catalog:
        lines.append("")
        lines.append("【可引用的资料清单】（只能从这里选，不得编造其他资料名）")
        for doc_id in sorted(doc_catalog):
            lines.append(f"- {doc_id} {doc_catalog[doc_id]}")
        lines.append("推荐复习资料时必须用上面的名称，"
                     "不要写章号/条款号（你不知道章节结构）。")
    else:
        lines.append("")
        lines.append("【重要】不要提及任何具体资料名称、章号或条款号 ——"
                     "你不知道有哪些资料，编造会误导学员。"
                     "只说「复习相关章节」即可。")
    return "\n".join(lines)


class MonthlyReportUseCase:
    """生成一个人的月度培训报告（同步版）。"""

    def __init__(
        self,
        *,
        repository: Any,
        llm: Any = None,
        guardrail: ReportGuardrail | None = None,
        max_rewrites: int = 1,
        case_titles: dict[str, str] | None = None,
        doc_catalog: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            repository: 仓储（读评阅结果）。
            llm: 语言模型。None 时只出数据不出建议（便于无 LLM 环境测试）。
            guardrail: **报告专用**护栏（`ReportGuardrail`）。
                ★ 不能用 `OutputGuardrail`：报告必然要引述学员的违规话术
                  （「第1句说『降血脂』属于疾病治疗」），
                  普通护栏会把它判成违规拦下 —— 实测 3 条建议全被拦。
            max_rewrites: 建议被拦后的重写次数。默认 1（与问答链路一致）。
            case_titles: `case_id → 标题`。
            doc_catalog: `doc_id → 文档标题`（真实知识库清单）。
                ★ 用于约束建议里能引用的资料名，防止编造。
        """
        self._repo = repository
        self._llm = llm
        self._guardrail = guardrail or ReportGuardrail()
        self._max_rewrites = max(0, max_rewrites)
        self._case_titles = case_titles or {}
        self._doc_catalog = doc_catalog or {}

    # ---------------- 主入口 ----------------
    def execute(
        self,
        trainee_id: str,
        *,
        since: str = "",
        until: str = "",
        period: str = "",
    ) -> MonthlyReportResult:
        """生成报告。

        Args:
            trainee_id: 学员编号。
            since / until: 时间范围（`YYYY-MM-DD`）。
            period: 展示用的周期名（如 `2026-10`）。空则用 since 猜。

        Returns:
            `MonthlyReportResult`。**任何情况下都返回结构**，
            不会因为 LLM 挂了就抛异常——数据部分永远可用。
        """
        reviews = self._repo.list_drill_reviews(
            trainee_id=trainee_id,
            since=since,
            until=until,
            with_items=True,
        )

        report = build_report(
            trainee_id=trainee_id,
            period=period or (since[:7] if since else ""),
            reviews=reviews,
            case_titles=self._case_titles,
        )

        suggestion = self._make_suggestion(report)
        return MonthlyReportResult(
            report=report,
            suggestion=suggestion,
            summary=_to_summary(report, suggestion),
        )

    # ---------------- 建议生成 ----------------
    def _make_suggestion(self, report: MonthlyReport) -> GeneratedSuggestion:
        """生成改进建议。无薄弱项 / 无 LLM 时跳过。"""
        if not report.has_weak:
            # ★ 没有薄弱项时不生成建议。
            #   硬凑建议就是让模型编 —— 宁可不给。
            return GeneratedSuggestion(
                text="本月各能力项均无明显问题，建议增加案例覆盖广度。",
                passed=True, skipped=True)

        if self._llm is None:
            return GeneratedSuggestion(
                text="", passed=True, skipped=True,
                reason="未配置 LLM，仅输出数据部分")

        prompt = build_suggestion_prompt(report, self._doc_catalog)
        text = self._ask(prompt)
        if not text:
            return GeneratedSuggestion(
                text="", passed=False, reason="LLM 未返回内容")

        verdict = self._guardrail.check(text)
        retries = 0

        while verdict.blocked and retries < self._max_rewrites:
            retries += 1
            logger.info("建议被护栏拦截，重写第 %d 次: %s",
                        retries, verdict.reason[:80])
            retry_prompt = (
                prompt + "\n\n【上次的问题】\n"
                + f"你上一次的输出违反了下述规则：{verdict.reason}\n"
                + "请重写，务必不出现任何功效宣称或绝对化表述。"
            )
            text = self._ask(retry_prompt)
            if not text:
                break
            verdict = self._guardrail.check(text)

        return GeneratedSuggestion(
            text=text,
            passed=not verdict.blocked,
            reason=verdict.reason if verdict.blocked else "",
            retries=retries,
        )

    def _ask(self, prompt: str) -> str:
        try:
            res = self._llm.generate(
                prompt,
                system_prompt=(
                    "你是保健品门店的培训主管，负责写学员的月度改进建议。"
                    "只输出建议正文，不解释。严禁出现产品功效表述。"
                ),
                temperature=0.0,
                max_tokens=600,
            )
            return (getattr(res, "text", "") or "").strip()
        except Exception as exc:                       # noqa: BLE001
            logger.warning("生成建议失败: %s: %s", type(exc).__name__, exc)
            return ""


def _to_summary(report: MonthlyReport,
                suggestion: GeneratedSuggestion) -> dict[str, Any]:
    """转成前端/导出用的结构。

    ★ 三层结构完整保留：能力项（第1层）→ 案例（第2层）→ 违规（第3层）。
      前端靠 `capability` 字段做下钻关联。
    """
    return {
        "trainee_id": report.trainee_id,
        "period": report.period,
        "overview": {
            "total_drills": report.total_drills,
            "total_duration_sec": report.total_duration_sec,
            "distinct_cases": report.distinct_cases,
            "total_critical": report.total_critical,
        },
        "capabilities": [
            {
                "key": c.capability,
                "name": c.name,
                "samples": c.samples,
                "score": c.score,
                "level": c.level,
                "level_label": c.level_label,
                "critical_count": c.critical_count,
                "note": c.note,
                "is_weak": c.capability in report.weak_capabilities,
            }
            for c in report.capabilities
        ],
        "cases": [
            {
                "drill_id": o.drill_id,
                "case_id": o.case_id,
                "case_title": o.case_title,
                "capability": o.capability,
                "case_type": o.case_type,
                "score": o.score,
                "score_note": o.score_note,
                "element_hit": o.element_hit,
                "element_total": o.element_total,
                "risk_count": o.risk_count,
                "critical_count": o.critical_count,
                "created_at": o.created_at,
                "risks": list(o.risks),
            }
            for o in report.outcomes
        ],
        "weak_capabilities": list(report.weak_capabilities),
        "suggestion": {
            "text": suggestion.text,
            "passed": suggestion.passed,
            "reason": suggestion.reason,
            "skipped": suggestion.skipped,
            "retries": suggestion.retries,
        },
        "degraded": report.degraded,
        "degrade_reason": report.degrade_reason,
        "level_labels": LEVEL_LABELS,
    }


__all__ = [
    "GeneratedSuggestion",
    "MonthlyReportResult",
    "MonthlyReportUseCase",
    "build_suggestion_prompt",
]
