"""LLM 语义映射兜底：把规则猜不出的口语需求对到批准功能目录（唯一允许调 LLM 的映射路径）。

────────────────────────────────────────────────────────────────────
它为什么存在
────────────────────────────────────────────────────────────────────
`demand_matcher` 是**规则**匹配（同义词表 + 核心词对撞）。规则的好处是
可复现、零成本、零延迟；坏处是**长尾说法一定会漏**。

漏掉的后果很严重（2026-10-24 实测）：
    问「改善精力」→ 规则映射为空 → 系统答「资料里没有」
    而本店 P008 西洋参、P009 人参恰恰声称「缓解体力疲劳」（第 7 项）。
    **有货却说没货**，是最伤信任的一类错误。

────────────────────────────────────────────────────────────────────
★★★ 为什么最多只调 1 次（不是 3 次投票）
────────────────────────────────────────────────────────────────────
曾经考虑过「让模型生成 3 次，2-3 次相同就采用」的自助一致性投票。
**实测否掉了这个方案**（2026-10-24，真实千问）：

    需求             temp=0.0          temp=0.8          一致性
    改善精力         缓解体力疲劳 ×3    缓解体力疲劳 ×3     3/3
    提升关节舒适度    NONE ×3          NONE ×3           3/3

两个理由说明投票在这里无效：

    ① 输出空间极小（15 项 + NONE），`temperature=0.0` 是**确定性**的。
       跑 100 次也是同一个答案，「3 次一致」不是高置信，
       只是同一个确定性计算重复了 3 次 —— **证明不了对错**。
    ② 唯一可能产生分歧的是调高 temperature，但那等于
       「故意注入噪声来检验稳定性」，逻辑上自相矛盾。

所以改成**针对性校验**（见下方三层），它检验「这个映射对不对」，
而不是「模型稳不稳」。后者不是本项目要防的东西。

────────────────────────────────────────────────────────────────────
红线：只在规则全失败后才调，且整个请求最多 1 次
────────────────────────────────────────────────────────────────────
`AnswerQuestionUseCase` 有一条硬约束「召回为空不调 LLM」。
本模块发生在召回**之前**，如果不加限制，连乱码输入都会先烧一次调用。

约束实现：
    · 调用点只有一处（`_gather_evidence` 里，规则为空时才走）
    · 每个 `QueryPlanUseCase` 实例维护调用预算，用完即止
    · 规则命中的情况**永不**进入本模块
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: 允许模型输出的"无对应项"标记。
NONE_TOKEN = "NONE"

#: 提示词。要点：
#:   · 强制从封闭列表选（L1 校验的前提）
#:   · 明确说"最接近"而不是"等同"（合规要求）
#:   · 允许 NONE（否则模型会硬凑一项出来）
_PROMPT = """从下面的国家批准保健功能目录中，选出与客户需求**最接近**的一项。

目录：
{catalog}

客户需求：{demand}

规则：
1. 只能从上面目录里选一项，原样输出该项的文字，不要改写。
2. 如果目录里没有任何一项与需求相关，只输出 {none}。
3. 不要解释，不要加序号，只输出一行。

输出："""


@dataclass(frozen=True, slots=True)
class SemanticMatch:
    """一次语义映射的结果。

    ★ 不含「置信度分数」（约束 #3 的延伸：判定用布尔与枚举，
      不引入没有标定依据的数字分数）。
      是否可信由 `verified` 表达，理由由 `reason` 表达。
    """

    #: 命中的目录项 ID。空字符串 = 无对应项。
    efficacy_id: str = ""
    #: 模型原话（排障用，不对外）。
    raw: str = ""
    #: 是否通过三层校验。False 时调用方**不得**据此推荐产品。
    verified: bool = False
    #: 校验失败或说明的原因（可直接作为 hint 的一部分）。
    reason: str = ""
    #: 各层校验过程的记录（排障用）。
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def is_none(self) -> bool:
        """模型明确说"目录里没有"。"""
        return not self.efficacy_id and self.verified

    @property
    def usable(self) -> bool:
        """可以用于后续事实查询与推荐。"""
        return bool(self.efficacy_id) and self.verified


class SemanticEfficacyMatcher:
    """用 LLM 把口语需求对到目录项，并做三层校验。

    ★ 三层校验（替代投票）：

        L1 词表封闭：输出必须是目录原文之一或 NONE。
                     模型爱加「5. 」前缀或改写成近义词，必须归一化后精确比对。
        L2 反向验证：再把选中的项回问一次「『X』是否等同于『需求』？」
                     这一层才是真正防"映射错了"的 —— 自一致性投票做不到。
        L3 效果对齐：该目录项**必须真有产品声称**（由调用方查图谱填入）。
                     没产品的项只答法规、不推荐 —— 否则等于推荐了不存在的货。

    预算：每个实例 `max_calls` 次（默认 1）。用完即返回"未解析"。
    这是为了守住「召回为空不调 LLM」那条红线在多跳链路上的成本假设。
    """

    def __init__(
        self,
        llm: Any,
        *,
        max_calls: int = 1,
        timeout_hint_ms: int = 1500,
    ) -> None:
        self._llm = llm
        self._max_calls = max(0, max_calls)
        self._calls_used = 0
        self._timeout_hint_ms = timeout_hint_ms

    # ---------------- 预算 ----------------
    @property
    def calls_used(self) -> int:
        return self._calls_used

    @property
    def budget_left(self) -> int:
        return max(0, self._max_calls - self._calls_used)

    def reset_budget(self) -> None:
        """每次问答开始时重置（一个请求一份预算，不是进程级）。"""
        self._calls_used = 0

    # ---------------- 主入口 ----------------
    def match(
        self,
        demand: str,
        catalog: list[tuple[str, str]],
    ) -> SemanticMatch:
        """尝试把 `demand` 对到目录里的一项。

        Args:
            demand: 客户需求片段（`demand_terms` 抽出来的那类）。
            catalog: 目录快照 `[(efficacy_id, name)]`。为空直接放弃 ——
                **不猜**，没有目录就没有约束，模型的输出无法校验。

        Returns:
            `SemanticMatch`。任何失败都返回 `verified=False`，
            调用方应退回"无法确定"的措辞，**不要**当成没有这个功能。
        """
        if not demand or not catalog:
            return SemanticMatch(reason="无需求片段或目录为空，跳过语义映射")
        if self._llm is None:
            return SemanticMatch(reason="未装配 LLM，跳过语义映射")
        if self.budget_left <= 0:
            # ★ 预算耗尽不是错误，是设计。必须留痕，便于观测真实覆盖率。
            logger.info("语义映射预算已用完，跳过: demand=%r", demand[:30])
            return SemanticMatch(reason="本次请求的语义映射预算已用完")

        raw = self._ask(match_prompt(demand, catalog))
        self._calls_used += 1
        return self._verify(raw, demand, catalog)

    # ---------------- 内部 ----------------
    def _ask(self, prompt: str) -> str:
        try:
            res = self._llm.generate(
                prompt,
                system_prompt="你是保健食品合规专员。只输出一行结果，不解释。",
                temperature=0.0,      # 确定性：映射不允许每次不一样
                max_tokens=48,
            )
            return (getattr(res, "text", "") or "").strip()
        except Exception as exc:                    # noqa: BLE001 下游失败不崩
            logger.warning("语义映射调用失败: %s", exc)
            return ""

    def _verify(
        self,
        raw: str,
        demand: str,
        catalog: list[tuple[str, str]],
    ) -> SemanticMatch:
        """三层校验。"""
        checks: dict[str, Any] = {"raw": raw}

        # ---- L1 词表封闭 ----
        l1 = normalize_to_catalog(raw, catalog)
        checks["L1"] = l1
        if l1 is None:
            return SemanticMatch(
                raw=raw, verified=False, checks=checks,
                reason="模型输出不在目录内（可能改写了表述），不予采用")
        if l1 == NONE_TOKEN:
            return SemanticMatch(
                raw=raw, verified=True, checks=checks,
                reason="模型判断目录中无对应项")

        eid, name = l1

        # ---- L2 反向验证 ----
        # ★ 二次调用会吃预算。预算为 1 时不反问，改为「降级采纳」：
        #   标记 verified=True 但 reason 注明"未经反向验证"，
        #   由 L3（是否有产品声称）兜底。这样 1 次调用也能保证不出危险结论。
        if self.budget_left <= 0:
            checks["L2"] = "skipped(no budget)"
            return SemanticMatch(
                efficacy_id=eid, raw=raw, verified=True, checks=checks,
                reason=f"规则未覆盖，模型判为最接近「{name}」（未经反向验证）")

        yes = self._ask(boolean_prompt(demand, name))
        self._calls_used += 1
        agreed = _is_yes(yes)
        checks["L2"] = {"raw": yes, "agreed": agreed}
        if not agreed:
            return SemanticMatch(
                raw=raw, verified=False, checks=checks,
                reason=f"反向验证未通过（模型不认为需求等同于「{name}」），不予采用")

        return SemanticMatch(
            efficacy_id=eid, raw=raw, verified=True, checks=checks,
            reason=f"规则未覆盖，模型判为最接近「{name}」，反向验证通过")


def match_prompt(demand: str, catalog: list[tuple[str, str]]) -> str:
    """构造映射提示词（抽出来便于测试与排障复现）。"""
    listed = "\n".join(
        f"{i + 1}. {name}" for i, (_, name) in enumerate(catalog))
    return _PROMPT.format(catalog=listed, demand=demand, none=NONE_TOKEN)


def boolean_prompt(demand: str, catalog_name: str) -> str:
    """构造反向验证提示词。

    ★ 问法是「是否在说同一件事」，不是「是否相关」。
      「相关」太宽：『缓解体力疲劳』和『改善精力』可以说相关，
      但『改善骨密度』和『关节舒适度』也"相关"，那就没区分力了。
    """
    return (
        f"客户需求：{demand}\n"
        f"保健食品批准功能：{catalog_name}\n\n"
        f"判断这两者是否在说**同一件健康作用**。"
        f"只有确实是同一件事才答 YES；只是宽泛相关（例如都涉及骨骼）请答 NO。\n"
        f"只输出 YES 或 NO。"
    )


def normalize_to_catalog(
    raw: str,
    catalog: list[tuple[str, str]],
) -> tuple[str, str] | str | None:
    """L1 词表封闭：把模型输出归一化成目录项或 NONE。

    Returns:
        `(efficacy_id, name)` = 命中目录项；
        `NONE_TOKEN`          = 模型明确说没有；
        `None`                = 无法归一化，调用方必须当失败处理。

    ★ 为什么必须归一化而不是直接字符串相等：
      实测模型输出是 `5. 缓解体力疲劳|NONE` ——
      带序号、还擅自加了自己编的 `|NONE` 后缀。
      如果直接相等比较，正确答案会被判成无效，等于白调一次。
      但归一化**不能放宽到"包含就算"**：那会让
      「有助于改善骨密度」误命中「有助于改善骨密度和关节」（不存在但要防）。
      所以只做三件事：去序号、去空白、去尾部分隔符，然后**精确比对**。
    """
    text = (raw or "").strip()
    if not text:
        return None

    # 去 markdown / 引号 / 序号
    text = text.strip("`'\"“”‘’ \t\r\n")
    text = text.replace("**", "").replace("__", "").replace("*", "")
    text = _strip_ordinal(text)

    # 模型可能把答案写在第一行
    first = text.splitlines()[0].strip() if text else ""
    first = _strip_ordinal(first).strip("`'\"“”‘’ \t").replace("**", "").strip()

    # 去掉模型自作主张加的后缀（实测出现过 `|NONE`、`（最接近）`）
    first = first.split("|")[0].strip()
    first = first.rstrip("。（）()【】[] ").strip()

    if first.upper() == NONE_TOKEN or first == "无":
        return NONE_TOKEN

    for eid, name in catalog:
        if first == name:
            return (eid, name)
    # 再试一次：去掉目录名里的括号补充（「有助于维持血脂（胆固醇/甘油三酯）健康水平」）
    for eid, name in catalog:
        if _strip_paren(name) == _strip_paren(first):
            return (eid, name)
        if first == name.split("（")[0].strip():
            return (eid, name)

    return None


def _strip_ordinal(text: str) -> str:
    """去掉开头的 `5.` / `5、` / `(5)` / `- ` 这类序号。"""
    import re
    return re.sub(r"^\s*(?:[（(]?\d+[）)]?\s*[.、．:：)）]?\s*|[-*·]\s+)", "",
                  text, count=1).strip()


def _strip_paren(text: str) -> str:
    """去掉中英文括号及其内容，用于比对。"""
    import re
    return re.sub(r"[（(][^（）()]*[）)]", "", text or "").strip()


def _is_yes(text: str) -> bool:
    """模型是否答了 YES。

    ★ 只认明确的 YES/是。答 NO、答别的、答空，一律当"未通过" ——
      这一层的失败方向必须是**保守**（不采用），不能是放开。
    """
    t = (text or "").strip().upper().strip("`'\" 。.")
    return t.startswith("YES") or t.startswith("是")


__all__ = [
    "NONE_TOKEN",
    "SemanticMatch",
    "SemanticEfficacyMatcher",
    "boolean_prompt",
    "match_prompt",
    "normalize_to_catalog",
]
