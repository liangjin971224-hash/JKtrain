"""演练评阅用例。

串起：风险定位 + 要素覆盖 → 摘要卡。

────────────────────────────────────────────────────────────────────
这个用例**不调 LLM**
────────────────────────────────────────────────────────────────────
评阅的判定标准全部写在案例文档里，是**规则**：

    要素判定   「出现『有』+ 产品名，或『没有』」
    违禁检测   正则表达式 + 级别

用 LLM 判会带来不确定性：同一份答卷今天说「说到警示语了」，
明天说「没说」。学员无法信服，也无法复核。

所以评阅是**纯确定性流程**，这也是 P5 在交付表里标注「是（确定性）」的原因。

LLM 在 P5 里的位置是另一个用途：**扮演客户角色追问**（演练对话本身），
那部分不在本用例里。

────────────────────────────────────────────────────────────────────
摘要卡不给分数
────────────────────────────────────────────────────────────────────
`SummaryCard` 里没有 score、没有 verdict。
本用例也**不计算任何形式的总分**——连「覆盖率百分比」都不给，
因为那会立刻被当成分数用（85% vs 90%，学员就会问怎么从 85 到 90）。
"""
from __future__ import annotations

import dataclasses
import json
import logging

from domain.rules.element_check import check_elements, is_fallback_judged
from domain.services.risk_locator import collect_turns_text, locate_risks
from domain.value_objects import (
    CaseSpec,
    ElementCheck,
    RiskItem,
    SummaryCard,
    TraineeTurn,
)

logger = logging.getLogger(__name__)


class ReviewDrillUseCase:
    """演练评阅用例。"""

    def __init__(self) -> None:
        # 无外部依赖：规则是纯函数
        pass

    # ---------------- 主流程 ----------------
    def execute(
        self,
        spec: CaseSpec,
        turns: list[TraineeTurn],
        *,
        trainee_id: str = "",
        duration_sec: int = 0,
        include_global_rules: bool = True,
    ) -> SummaryCard:
        """评阅一次演练。

        Args:
            spec: 案例规格。
            turns: 学员的各轮发言。
            trainee_id: 学员编号。
            duration_sec: 演练用时（秒）。
            include_global_rules: 是否叠加全局六类禁用词规则。

        Returns:
            判定摘要卡。**不含分数、不含结论**。

        Note:
            这里刻意不返回任何「通过 / 不通过」的判断。
            需求要求系统只提供依据，判定由人来下——
            即使有 critical 违规，卡里也只写「命中什么、在哪一句」。
        """
        # ---- 1. 风险定位 ----
        risks = locate_risks(spec, turns, include_global_rules=include_global_rules)

        # ---- 2. 要素覆盖 ----
        full_text = collect_turns_text(turns)
        checks = check_elements(spec.elements, full_text)

        # ---- 3. 待人工确认 ----
        pending = self._collect_pending(spec, checks, risks)

        card = SummaryCard(
            trainee_id=trainee_id,
            case_id=spec.case_id,
            duration_sec=duration_sec,
            risk_items=tuple(risks),
            element_coverage=tuple(checks),
            pending_human=tuple(pending),
        )

        logger.info(
            "评阅完成: case=%s trainee=%s 风险=%d(critical %d) "
            "缺必答=%d 待人工=%d",
            spec.case_id, trainee_id, len(risks),
            len(card.critical_items),
            len(card.missing_required_elements),
            len(pending),
        )
        return card

    # ---------------- 待人工确认 ----------------
    @staticmethod
    def _collect_pending(
        spec: CaseSpec,
        checks: list[ElementCheck],
        risks: list[RiskItem],
    ) -> list[str]:
        """收集需要人工再看一眼的事项。

        ★ 三类进待确认：
          ① 只能靠兜底判定的要素（宽松匹配，可能误判）
          ② 建议要素部分满足（E7 说了「孕妇」但没说「出血倾向者」这类）
          ③ 有 high 级（非 critical）风险——需人判断是否算问题

        ★ 为什么不把 critical 放进待确认：
          critical 是明确的违法宣传，不需要人工判断「算不算问题」，
          需要的是改。所以它直接进「风险提示」区。
        """
        pending: list[str] = []

        spec_by_id = {e.element_id: e for e in spec.elements}
        for check in checks:
            element_spec = spec_by_id.get(check.element_id)

            # ① 兜底判定
            if element_spec is not None and is_fallback_judged(element_spec):
                pending.append(
                    f"要素 {check.element_id}「{check.element_name}」"
                    f"无专门判定规则，结果为自动判断，请人工复核"
                )
                continue

            # ② 建议要素未覆盖
            if not check.required and not check.covered:
                pending.append(
                    f"建议要素 {check.element_id}「{check.element_name}」未提及，"
                    f"是否计为问题请人工判断"
                )

        # ③ high 级风险
        for risk in risks:
            if risk.level.is_blocking:
                continue
            pending.append(
                f"{risk.location}「{risk.matched}」命中{risk.rule_type}（"
                f"{risk.level.value}），严重程度请人工判断"
            )

        return pending

    # ---------------- 序列化 ----------------
    @staticmethod
    def to_dict(card: SummaryCard) -> dict:
        """转成普通 dict（四区块）。

        ★ 结果里不能出现 score / verdict —— 由测试守护。
        """
        return card.to_display_dict()

    @staticmethod
    def to_json(card: SummaryCard, *, indent: int = 2) -> str:
        """转成 JSON。

        注意 `ensure_ascii=False`：中文要能直接读，
        否则存档后翻查极其痛苦。
        """
        return json.dumps(
            card.to_display_dict(), ensure_ascii=False, indent=indent)

    @staticmethod
    def render_text(card: SummaryCard) -> str:
        """渲染成可读文本（贴给学员看的那种）。

        ★ 里面不能出现「得分 / 总分 / 评分」这类词——
          学员看到就会开始问怎么提分。
        """
        lines: list[str] = []

        lines.append(f"【基础信息】")
        lines.append(f"学员：{card.trainee_id or '未填'}　"
                     f"案例：{card.case_id}　"
                     f"用时：{card.duration_sec} 秒")
        lines.append("")

        lines.append("【风险提示】")
        if card.risk_items:
            for item in card.risk_items:
                mark = "🔴 critical" if item.is_critical else f"⚠️ {item.level.value}"
                lines.append(f"{mark}　「{item.matched}」— {item.rule_type}")
                lines.append(f"　　定位：{item.location}")
                if item.reason:
                    lines.append(f"　　说明：{item.reason}")
        else:
            lines.append("未发现违禁表述。")
        lines.append("")

        lines.append("【要素覆盖】")
        for check in card.element_coverage:
            tag = "必答" if check.required else "建议"
            lines.append(
                f"{check.status_mark} {check.element_id} "
                f"{check.element_name}（{tag}）"
            )
        lines.append("")

        lines.append("【待人工确认】")
        if card.pending_human:
            for p in card.pending_human:
                lines.append(f"· {p}")
        else:
            lines.append("无")

        return "\n".join(lines)

    @staticmethod
    def describe(card: SummaryCard) -> dict[str, object]:
        """给运维/调试用的结构化摘要（计数，不是评分）。"""
        counts: dict[str, int] = {}
        for r in card.risk_items:
            counts[r.level.value] = counts.get(r.level.value, 0) + 1

        return {
            "case_id": card.case_id,
            "risk_counts": counts,
            "missing_required": len(card.missing_required_elements),
            "covered": len(card.covered_elements),
            "elements_total": len(card.element_coverage),
            "pending_human": len(card.pending_human),
        }
