"""输出护栏。

★ 约束 #1 的实现点：**输出护栏是必经节点**，任何生成路径都必须过它。

────────────────────────────────────────────────────────────────────
三项检测
────────────────────────────────────────────────────────────────────
1. 禁用词检测      `domain/rules/forbidden_claims.py`   六类禁用表述
2. 功能声称准入    `domain/rules/product_category.py`   普通食品不得声称
3. 强制警示语      `domain/rules/warning_text.py`       保健食品必带

三者都通过才放行。任一项失败 → `blocked=True`。

────────────────────────────────────────────────────────────────────
为什么护栏必须是纯领域逻辑
────────────────────────────────────────────────────────────────────
如果用 LLM 判合规：
    · 结果不稳定（同输入两次可能不同）
    · 无法测试（要连模型、有随机性）
    · 无法解释（说不出命中了哪条规则）

而本项目的合规判定是**硬规则**，必须可复现、可追溯、可 100% 测试。
LLM 只负责「组织表达」，判断「这句话能不能说」由规则做。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from domain.rules.forbidden_claims import (
    ClaimCheckResult,
    check_forbidden_claims,
)
from domain.rules.product_category import ClaimVerdict, check_claim_allowed
from domain.rules.warning_text import (
    WarningCheckResult,
    WarningStatus,
    check_warning_present,
)
from domain.value_objects import ProductCategory

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GuardrailVerdict:
    """护栏判定结果。

    ★ 不含 score / verdict 分数字段（约束 #3）。
      `blocked` 是布尔判断，`reason` 是给业务看的解释。
    """

    blocked: bool
    reason: str = ""
    # 命中的禁用词违规（可多条）
    violations: tuple[str, ...] = ()
    # 命中规则的 id 列表，便于审计
    rule_ids: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.blocked


@dataclass(frozen=True, slots=True)
class AnswerSegment:
    """回答里的一句话（或一个自然段），带合规判定。

    ★ 供前端渲染：
        `ok=True`  → 正常显示
        `ok=False` → 标红显示 + 附上 `reason` 说明为什么不能说

    ★ 注意：`text` 里**含违规原文**。这段数据只应喂给前端展示，
      不应写进日志、不应被当作可对外复制的正文。
      对外安全版本在 `AnnotationResult.safe_text`。
    """

    text: str
    ok: bool = True
    #: 命中的规则类别（供前端做标签，如「疾病治疗」「替客户判断」）
    rule_types: tuple[str, ...] = ()
    #: 给学员看的原因说明（已去掉「命中「」」这类内部措辞）
    reason: str = ""


@dataclass(frozen=True, slots=True)
class AnnotationResult:
    """标注结果（双轨输出）。

    Attributes:
        safe_text: **剔除违规句后的合规版本**。可直接对外输出。
        segments: 逐句结构，含违规标注。只给前端渲染。
        reason: 无法产出安全版本时的说明（此时应整段拒答）。
        repaired_warning: 是否补回过法定警示语。
    """

    safe_text: str = ""
    segments: tuple[AnswerSegment, ...] = ()
    reason: str = ""
    repaired_warning: bool = False

    @property
    def has_safe_content(self) -> bool:
        """是否有可安全输出的内容。"""
        return bool(self.safe_text.strip())

    @property
    def has_violation(self) -> bool:
        """是否检出违规句。"""
        return any(not s.ok for s in self.segments)

    @property
    def violation_count(self) -> int:
        return sum(1 for s in self.segments if not s.ok)

    @property
    def removed_text(self) -> tuple[str, ...]:
        """被剔除的违规句（供审计/训练记录）。"""
        return tuple(s.text for s in self.segments if not s.ok)


#: 分句用的标点。
#:
#: ★ 不含顿号（同 `stream_guard` 的理由）：
#:   顿号是并列，「疾病预防、治疗」切开就断了语义。
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


def split_sentences(text: str) -> list[str]:
    """把文本切成句子（保留标点）。

    ★ 为什么不用 `stream_guard.SentenceBuffer`：
      那是给流式逐 token 喂入用的状态机（有缓冲长度上限、会提前切），
      这里只需要一次性切分，用正则更直接、更可预测。

    ★ 空行/纯空白片段会被丢弃，避免产出空段。

    Args:
        text: 待切分的文本。

    Returns:
        句子列表，按原顺序。至少返回一项（文本非空时）。
    """
    if not text or not text.strip():
        return []

    parts = _SENTENCE_SPLIT.split(text)
    out = [p.strip() for p in parts if p and p.strip()]
    return out or [text.strip()]


#: rule_id → 给学员看的类别标签。
#:
#: ★ 为什么不从 reason 里正则提 `[类别]`（实测踩坑）：
#:   只有「禁用词」那条路径的 reason 带 `[类别]` 前缀，
#:   「普通食品功能声称」和「缺警示语」都不带，
#:   于是前端会显示成笼统的「违规表述」。
#:   直接按 rule_id 映射，稳定且可读。
_RULE_ID_LABELS: dict[str, str] = {
    "CATEGORY-01": "普通食品不得声称功效",
    "WARN-01": "缺少法定警示语",
}


def _rule_labels(verdict: GuardrailVerdict) -> list[str]:
    """从判定结果里提取人类可读的规则类别。"""
    labels: list[str] = []

    # 1. 先按已知 rule_id 映射
    for rid in verdict.rule_ids:
        label = _RULE_ID_LABELS.get(rid)
        if label and label not in labels:
            labels.append(label)

    # 2. 禁用词类：reason 里带 `[类别]` 前缀，优先用它
    for m in re.finditer(r"\[([^\]]+)\]", verdict.reason or ""):
        label = m.group(1)
        if label not in labels:
            labels.append(label)

    return labels or ["违规表述"]


def _humanize_reason(reason: str) -> str:
    """把内部 reason 整理成给学员看的说明。

    原始 reason 形如：
        「命中禁用表述：[疾病治疗] 医疗用语：应为「有助于维持……」，
          不能用「降」（命中「降血脂」）」
        「普通食品不得作任何保健功能声称（违反《食品安全法》）。
          命中「乌发」。可改为客观描述配料、规格、食用方法。」

    处理：
        · 去掉「命中禁用表述：」前缀
        · 去掉「命中「xxx」。」——命中片段单独展示，重复说没意义
        · 去掉「[类别]」前缀（类别已单独成标签）
        · **去重**：多条规则同一原因时合并（实测「越界判断同服安全性」
          出现了两遍，因为 JD-02 和 JD-05 都命中同一句）
    """
    r = (reason or "").strip()
    r = re.sub(r"^命中禁用表述：", "", r)
    r = re.sub(r"（命中「[^」]*」）", "", r)
    r = re.sub(r"命中「[^」]*」。?", "", r)
    r = re.sub(r"\[[^\]]+\]", "", r)          # 去掉所有 [类别] 标签
    r = re.sub(r"\s{2,}", " ", r)

    # 按分号拆开去重（保留首次出现的顺序）
    parts = [p.strip(" ；;，,") for p in re.split(r"[；;]", r)]
    seen: list[str] = []
    for p in parts:
        if p and p not in seen:
            seen.append(p)

    return "；".join(seen) or "该表述不符合合规要求"


class OutputGuardrail:

    def check(
        self,
        answer: str,
        *,
        category: ProductCategory | str | None = None,
        check_warning: bool = True,
    ) -> GuardrailVerdict:
        """检查一段待输出文本。

        Args:
            answer: 待输出的回答。
            category: 涉及的产品类别。None 表示回答不涉及具体产品
                （此时只做禁用词检测，不做类别相关判定）。
            check_warning: 是否检查强制警示语。
                ★ **逐句检测时必须传 False**（实测踩坑）：
                  警示语是**整段**的属性，不是单句的属性。
                  「这是保健食品。」这一句本身不含警示语是正常的，
                  但对它单独跑警示语检查会判「缺警示语」→ 把合规句误判成违规。
                  所以按句标注时跳过这一项，最后对整段再验一次。

        Returns:
            判定结果。`blocked=True` 时**不得输出**。

        Note:
            三项检测的顺序有讲究：
              先查禁用词（最严重，涉及违法宣传）
              再查类别声称（约束 #2）
              最后查警示语（缺漏类问题）
            这样返回的 reason 是最该优先处理的那一条。
        """
        text = (answer or "").strip()
        if not text:
            # 空回答不算违规——是否输出由上层决定
            return GuardrailVerdict(blocked=False)

        # ---- 1. 禁用词 ----
        claim_result: ClaimCheckResult = check_forbidden_claims(text)
        if claim_result.has_critical:
            # 只取 critical 的（当前规则全部是 critical）
            crit = [v for v in claim_result.violations if v.severity.must_block]
            reason = "；".join(
                f"[{v.category}] {v.reason}（命中「{v.matched}」）"
                for v in crit[:3]
            )
            if len(crit) > 3:
                reason += f"；另有 {len(crit) - 3} 处"

            logger.warning(
                "护栏拦截: 禁用词 %d 处, rule_ids=%s",
                len(crit), [v.rule_id for v in crit],
            )
            return GuardrailVerdict(
                blocked=True,
                reason=f"命中禁用表述：{reason}",
                violations=tuple(v.matched for v in crit),
                rule_ids=tuple(v.rule_id for v in crit),
            )

        # ---- 2. 类别相关声称 ----
        if category is not None:
            verdict: ClaimVerdict = check_claim_allowed(category, text)
            if verdict.blocked:
                logger.warning(
                    "护栏拦截: 普通食品功能声称, matched=%s", verdict.matched)
                return GuardrailVerdict(
                    blocked=True,
                    reason=verdict.reason,
                    violations=(verdict.matched,),
                    rule_ids=("CATEGORY-01",),
                )

            # ---- 3. 强制警示语 ----
            if check_warning:
                warn: WarningCheckResult = check_warning_present(text, category)
                if warn.blocked:
                    logger.warning("护栏拦截: 保健食品缺警示语")
                    return GuardrailVerdict(
                        blocked=True,
                        reason=warn.reason,
                        rule_ids=("WARN-01",),
                    )

        return GuardrailVerdict(blocked=False)

    def annotate(
        self,
        answer: str,
        *,
        category: ProductCategory | str | None = None,
    ) -> AnnotationResult:
        """把回答按句切分，标出违规句并给出原因，同时产出「安全版」。

        ────────────────────────────────────────────────────────────
        为什么要这样设计（产品决策，2026-10-23）
        ────────────────────────────────────────────────────────────
        原策略是「整段拒答」：只要有一句违规，整篇都不给。
        问题是**培训场景下这是反效果**——学员问「黑芝麻丸有什么功能」，
        LLM 答的前两句（配料、规格）都对，只有最后一句「乌发养颜」违规。
        整段拒答等于连正确信息一起丢了，学员学不到东西。

        但直接展示违规内容有一个**真实的合规风险**：
        违规表述仍然出现在返回体里，被复制 / 被日志采集 / 将来开放给客户，
        就成了对外传播违法宣传，与「输出护栏是必经节点」冲突。

        ────────────────────────────────────────────────────────────
        ★ 双轨输出（这是本方案的核心）
        ────────────────────────────────────────────────────────────
            字段          内容                      谁用
            ----------------------------------------------------------
            safe_text   剔除违规句后的纯合规版        对外安全，任何
                                                      消费方拿到都合规
            segments    逐句结构，每句带             前端渲染标红
                        ok / violation / reason

        ★ 于是：
          - 前端可以标红展示（学员看得到「为什么不能说」）
          - 接口调用方、日志、复制粘贴拿到的是 safe_text（合规）
          - 二者从同一次分析产出，不会不一致

        ────────────────────────────────────────────────────────────
        ★ 三个必须守住的不变量
        ────────────────────────────────────────────────────────────
        ① **safe_text 必须整体通过护栏**。
           逐句删不等于删干净：拼回去可能出现新的跨句违规。
           所以最后必须再整体验一次。

        ② **不能因为剔除而引入新的违规**。
           最危险的形态是「否定依赖」：
               「这个不能代替药物，但能降血糖。」
           删掉前半句 → 「但能降血糖。」反而更像在宣称。
           发现这类情况时**放弃部分保留**，退回整段拒答。

        ③ **警示语被误删时要补回**。
           警示语若恰好和违规句在同一句里被删，
           safe_text 会变成「保健食品缺警示语」→ 二次检测不通过。
           此时补回法定警示语（这是**必须出现**的内容）。

        Args:
            answer: 待处理的回答（通常是重试用尽后仍被拦的文本）。
            category: 涉及的产品类别。

        Returns:
            `AnnotationResult`。
            `has_safe_content=True` 时 `safe_text` 可直接输出；
            否则应**整段拒答**。
        """
        text = (answer or "").strip()
        if not text:
            return AnnotationResult(safe_text="", segments=(),
                                    reason="内容为空")

        sentences = split_sentences(text)

        segments: list[AnswerSegment] = []
        kept: list[str] = []

        for sentence in sentences:
            # ★ check_warning=False：警示语是整段属性，单句不含是正常的
            verdict = self.check(sentence, category=category,
                                 check_warning=False)
            if verdict.blocked:
                segments.append(AnswerSegment(
                    text=sentence,
                    ok=False,
                    rule_types=tuple(_rule_labels(verdict)),
                    reason=_humanize_reason(verdict.reason),
                ))
            else:
                segments.append(AnswerSegment(text=sentence, ok=True))
                kept.append(sentence)

        # ── 逐句判定全通过 ──
        #
        # ★ 这里**不能直接返回**（实测踩到的严重漏洞）：
        #   逐句检测用的是 `check_warning=False`（警示语是整段属性，单句不含正常），
        #   所以「缺警示语」在逐句阶段**根本不会被发现**。
        #   若在这里直接 return，就会把一个缺法定警示语的保健食品回答
        #   判成「全部合规」—— 合规底线被绕过。
        #
        #   所以：逐句过了，还要再整段验一次。
        if not any(not seg.ok for seg in segments):
            whole = self.check(text, category=category)
            if not whole.blocked:
                return AnnotationResult(
                    safe_text=text, segments=tuple(segments))

            # 整段不通过（只可能是缺警示语）→ 补回警示语
            repaired = self._repair_warning(text, whole, category)
            if repaired is not None:
                return AnnotationResult(
                    safe_text=repaired, segments=tuple(segments),
                    repaired_warning=True)

            # 补不回（例如类别不是保健食品却触发了别的检查）
            return AnnotationResult(
                safe_text="", segments=tuple(segments),
                reason=f"整段检测未通过：{whole.reason}")

        # 全违规 → 无可输出内容
        if not kept:
            return AnnotationResult(
                safe_text="", segments=tuple(segments),
                reason="全部句子均违规，无可输出内容")

        # ★ 防「否定依赖」：剩余内容有无主语转折句 → 放弃部分保留
        if self._has_dangling_negation(kept):
            return AnnotationResult(
                safe_text="", segments=tuple(segments),
                reason="剔除后剩余内容出现无主语的转折句，语义不安全")

        safe_text = "\n\n".join(kept)

        # ★ 关键不变式：安全版必须整体通过
        final = self.check(safe_text, category=category)
        if final.blocked:
            repaired = self._repair_warning(safe_text, final, category)
            if repaired is not None:
                return AnnotationResult(
                    safe_text=repaired, segments=tuple(segments),
                    repaired_warning=True)
            return AnnotationResult(
                safe_text="", segments=tuple(segments),
                reason=f"剔除后仍不合格：{final.reason}")

        return AnnotationResult(safe_text=safe_text, segments=tuple(segments))

    # ---------------- 内部辅助 ----------------
    @staticmethod
    def _has_dangling_negation(sentences: list[str]) -> bool:
        """检测「无主语的转折句」。

        ★ 为什么必须检查（实测风险）：
            原文：「这个不能代替药物，但能降血糖。」
            若前半句被划为违规删掉，剩下「但能降血糖。」
            —— 反而变成更直白的功效宣称。

        Returns:
            True 表示剩余内容里有以转折词开头的句子。
        """
        dangling = re.compile(
            r"^(?:但|但是|不过|然而|可是|所以|因此|而且|并且|同时|另外)")
        return any(dangling.match(s.strip()) for s in sentences)

    def _repair_warning(
        self,
        text: str,
        verdict: GuardrailVerdict,
        category: ProductCategory | str | None,
    ) -> str | None:
        """剔除后缺警示语时补回法定警示语。

        Returns:
            修好的文本；若问题不是缺警示语则返回 None。
        """
        if "WARN-01" not in verdict.rule_ids:
            return None

        from domain.rules.warning_text import WARNING_TEXT

        candidate = f"{text}\n\n{WARNING_TEXT}。"
        if self.check(candidate, category=category).passed:
            logger.info("按句剔除后补回法定警示语")
            return candidate
        return None


    def describe_rules(self) -> dict[str, int]:
        """规则清单概览（供健康检查与文档生成）。"""
        from domain.rules.forbidden_claims import _RULES
        from domain.rules.product_category import _ORDINARY_FOOD_FORBIDDEN
        from domain.rules.warning_text import _WARNING_FRAGMENTS

        by_cat: dict[str, int] = {}
        for rule in _RULES:
            by_cat[rule.category] = by_cat.get(rule.category, 0) + 1

        return {
            "禁用词规则数": len(_RULES),
            "禁用词类别数": len(by_cat),
            "普通食品禁词数": len(_ORDINARY_FOOD_FORBIDDEN),
            "警示语等价片段数": len(_WARNING_FRAGMENTS),
        }
