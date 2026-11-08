"""LLM 多跳规划：规则判不了时，让模型判断「拆几步、谁依赖谁」。

────────────────────────────────────────────────────────────────────
它解决什么（规则的三个盲区）
────────────────────────────────────────────────────────────────────
规则判依赖只认回指词白名单（`_ANAPHORA`）。实测漏掉的问法：

    规则失败                                    规则表现
    「改善体脂吃什么，里面哪款划算」            「里面哪款」不在词表 → 漏判依赖
    「补钙的推荐，孕妇能吃吗，贵不贵」          拆 3 步但主题补错
    「那种吃了对血脂好的，最便宜的多少钱」      无标点 → 单步（其实两步）

最后一行是真正的价值：**没有标点分隔、靠语义隐含的多步问题**，
规则完全无能为力。

────────────────────────────────────────────────────────────────────
★★★ 三条不能破的边界
────────────────────────────────────────────────────────────────────
1. **只在规则可疑时调**（`plan_looks_suspicious`）。
   80% 的问题是单步，为它调模型是纯浪费，且破坏了「规则路径完全可复现」。

2. **每请求最多 1 次**（`max_calls`，与 `SemanticEfficacyMatcher` 同一套预算机制）。
   `AnswerQuestionUseCase` 的红线是「召回为空不调 LLM」，
   规划器在链路最前面，不加预算就绕过了这条线。

3. **只决定「怎么拆」，不决定「查什么」**。
   输出的 `SubQuery` 只是「拆解方案」；候选集传递、空集处理、证据查询
   全部由执行层代码保证。模型没机会指定检索范围。

────────────────────────────────────────────────────────────────────
与「把规则换成 LLM」的区别
────────────────────────────────────────────────────────────────────
不是替换，是**叠加**：
    plan_queries()（规则，永远先跑）
        → 结果可疑？ → 是 → LLM 复核 → 校验 → 通过则采纳
                     → 否 → 直接用规则结果
                     → 校验不过 → 退回规则结果

所以 LLM 挂了、超时了、输出非法了，系统行为与改造前**完全一致**。
"""
from __future__ import annotations

import logging
from typing import Any

from application.services.query_plan_guard import (
    extract_json,
    plan_looks_suspicious,
    validate_plan,
)
from domain.services.query_planner import (
    HOP_PRICE_RANK,
    HOP_SIMPLE,
    MAX_STEPS,
    SubQuery,
)

logger = logging.getLogger(__name__)


def build_plan_prompt(question: str, *, max_steps: int = MAX_STEPS) -> str:
    """构造规划提示词。

    要点（都是踩过坑总结的）：
      · 给出**正向与反向判例** —— 只说「什么算依赖」，模型会过度标依赖
      · 明确「并列不算依赖」并举例，否则几乎每个问题都会被标依赖
      · 强制 JSON 且只输出 JSON（后面还有校验层兜底）
      · 说清 `kind` 只有两个取值
    """
    return f"""把用户的问题拆成有序的子问题。只输出 JSON，不要任何解释。

用户问题：{question}

规则：
1. 最多拆 {max_steps} 步。如果本来就是一个问题，只输出一步。
2. 子问题必须能独立表达完整意思（带主语，不能只剩「有什么禁忌」）。
3. depends_on_previous 只有在**后一步需要前一步的结果**时才为 true。
   判断依据：后一步里出现指代前面那批结果的词，如
   「其中」「里面」「这些」「上述」「那款」「哪款」「这几个」。
4. **并列的两个问题不算依赖**。例如「有什么推荐，有什么禁忌」——
   这是两件独立的事，两步都是 false。
5. kind 只有两个取值：
   - "simple"：普通问题
   - "price_rank"：要在候选里比价（最便宜/哪个划算/多少钱）
6. 第一步的 depends_on_previous 必须是 false。

示例一（依赖）：
问题：「改善体脂都可以吃什么保健品，其中那个最便宜」
输出：{{"steps":[{{"question":"改善体脂都可以吃什么保健品","kind":"simple","depends_on_previous":false}},{{"question":"这些里面哪个最便宜","kind":"price_rank","depends_on_previous":true}}]}}

示例二（并列，不依赖）：
问题：「补钙的有什么推荐，有什么禁忌」
输出：{{"steps":[{{"question":"补钙的有什么推荐","kind":"simple","depends_on_previous":false}},{{"question":"补钙的有什么禁忌","kind":"simple","depends_on_previous":false}}]}}

示例三（单步）：
问题：「鱼油有什么功能」
输出：{{"steps":[{{"question":"鱼油有什么功能","kind":"simple","depends_on_previous":false}}]}}

现在输出 JSON："""


class LLMQueryPlanner:
    """用 LLM 复核规则拆解。

    用法::

        planner = LLMQueryPlanner(llm, max_calls=1)
        steps = planner.plan(question, rule_steps=plan_queries(question))
        # steps 可能是规则结果（LLM 未触发/失败/校验不过），也可能是 LLM 结果

    ★ 调用方拿到的**永远是可用的步骤**，不会因为 LLM 出问题而拿到空。
    """

    def __init__(
        self,
        llm: Any,
        *,
        max_calls: int = 1,
        max_steps: int = MAX_STEPS,
        temperature: float = 0.0,
    ) -> None:
        """
        Args:
            llm: `LanguageModelPort`。
            max_calls: 每请求预算。默认 1 —— 与语义映射一致。
            max_steps: 步数上限。
            temperature: 默认 0.0。规划不允许每次不一样，
                但**不等价于完全确定**——同一输入仍可能有微小抖动，
                所以结果要落 meta 便于审计。
        """
        self._llm = llm
        self._max_calls = max(0, max_calls)
        self._calls_used = 0
        self._max_steps = max(1, min(max_steps, MAX_STEPS))
        self._temperature = temperature

        #: 最近一次决策的诊断信息（进 meta，便于「为什么这次拆成两步」）。
        self.last_decision: dict[str, Any] = {}

    # ---------------- 预算 ----------------
    @property
    def calls_used(self) -> int:
        return self._calls_used

    @property
    def budget_left(self) -> int:
        return max(0, self._max_calls - self._calls_used)

    def reset_budget(self) -> None:
        """每次问答开始时重置（一个请求一份预算）。"""
        self._calls_used = 0
        self.last_decision = {}

    # ---------------- 主入口 ----------------
    def plan(
        self,
        question: str,
        rule_steps: tuple[SubQuery, ...],
        *,
        dropped_fragments: tuple[str, ...] = (),
    ) -> tuple[SubQuery, ...]:
        """返回最终规划步骤。

        Args:
            question: 用户原句。
            rule_steps: 规则规划结果（已保证非空）。
            dropped_fragments: 规则阶段被丢弃的片段。

        Returns:
            采纳的步骤。**永不返回空**：任何异常都退回 `rule_steps`。
        """
        self.last_decision = {"source": "rule", "triggered": False, "reason": ""}

        suspicious, why = plan_looks_suspicious(
            question, rule_steps, dropped_fragments=dropped_fragments)
        if not suspicious:
            self.last_decision["reason"] = why or "规则结果无需复核"
            return rule_steps

        self.last_decision["triggered"] = True
        self.last_decision["trigger_reason"] = why

        # ★ 预算耗尽不是错误，是设计。必须留痕，便于观测真实覆盖率。
        if self.budget_left <= 0:
            self.last_decision["reason"] = "本次请求的规划预算已用完"
            logger.info("LLM 规划预算用完，沿用规则结果: %s", question[:30])
            return rule_steps

        if self._llm is None:
            self.last_decision["reason"] = "未装配 LLM"
            return rule_steps

        prompt = build_plan_prompt(question, max_steps=self._max_steps)
        raw = self._ask(prompt)
        self._calls_used += 1

        payload = extract_json(raw)
        self.last_decision["raw_output"] = _clip(raw, 300)

        result = validate_plan(
            payload, question=question, max_steps=self._max_steps)
        self.last_decision["validation_ok"] = result.ok
        self.last_decision["validation_reason"] = result.reason
        self.last_decision["repairs"] = list(result.repairs)

        if not result.ok:
            self.last_decision["reason"] = f"校验未通过，退回规则: {result.reason}"
            logger.info(
                "LLM 规划校验未通过，沿用规则: %s | 原因: %s",
                question[:30], result.reason)
            return rule_steps

        self.last_decision["source"] = "llm"
        self.last_decision["reason"] = "采纳 LLM 拆解"
        logger.info(
            "LLM 规划采纳: %d 步 %s（触发原因: %s）",
            len(result.steps),
            [(s.kind, s.depends_on_previous, s.question[:18])
             for s in result.steps],
            why,
        )
        return result.steps

    # ---------------- 内部 ----------------
    def _ask(self, prompt: str) -> str:
        """调模型。失败返回空串（后续校验会判失败并退回规则）。"""
        try:
            res = self._llm.generate(
                prompt,
                system_prompt=(
                    "你是查询规划器。只输出 JSON，不输出任何解释文字。"
                    "严格遵守：并列的问题不算依赖。"
                ),
                temperature=self._temperature,
                # 三步 JSON 大约需要 150 token，留余量到 320
                max_tokens=320,
            )
            return (getattr(res, "text", "") or "").strip()
        except Exception as exc:                       # noqa: BLE001
            logger.warning("LLM 规划调用失败，退回规则: %s", exc)
            return ""


def _clip(text: str, n: int) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t if len(t) <= n else t[:n] + "…"


class CachedQueryPlanner:
    """给 `LLMQueryPlanner` 套一层缓存（键 = 问题原文的哈希）。

    ★ 为什么按**原句**做键，而不是「原句 + 规则结果」：
      规则结果是原句的确定性函数，一一对应，加进去只会让键更长。

    ★ 为什么缓存值得做：培训场景里学员反复问相似问题，
      窗口内命中率很高，每次命中省掉一次 0.3–0.8s 的 LLM 调用。

    ★ 缓存值是「拆解方案」而非「答案」：
      它幂等 —— 同一问题任何进程算出来都一样，
      所以内存版缓存就够用，不必上 Redis。

    ★ 只缓存「LLM 真的参与过」的结果：
      规则直出本来就不贵，缓存它反而会在问题微变时读到旧拆解。
    """

    def __init__(self, planner: LLMQueryPlanner, cache: Any,
                 *, ttl_sec: int = 300) -> None:
        self._planner = planner
        self._cache = cache
        self._ttl = max(1, ttl_sec)
        #: 透传诊断信息（与内层规划器保持一致，便于上层读）。
        self.last_decision: dict[str, Any] = {}

    @property
    def calls_used(self) -> int:
        return self._planner.calls_used

    def reset_budget(self) -> None:
        self._planner.reset_budget()
        self.last_decision = {}

    def plan(
        self,
        question: str,
        rule_steps: tuple[SubQuery, ...],
        *,
        dropped_fragments: tuple[str, ...] = (),
    ) -> tuple[SubQuery, ...]:
        key = f"plan:v1:{_hash(question)}"

        cached = _cache_get(self._cache, key)
        if cached:
            steps = _deserialize(cached)
            if steps:
                self.last_decision = {
                    "source": "cache",
                    "triggered": True,
                    "reason": "命中规划缓存",
                }
                return steps

        steps = self._planner.plan(
            question, rule_steps, dropped_fragments=dropped_fragments)

        # 透出内层决策（含 source），上层据此写 meta
        self.last_decision = dict(self._planner.last_decision or {})

        if self.last_decision.get("source") == "llm":
            _cache_set(self._cache, key, _serialize(steps), self._ttl)

        return steps


# ---------------- 序列化（缓存值必须是 JSON 可序列化） ----------------
def _serialize(steps: tuple[SubQuery, ...]) -> list[dict[str, Any]]:
    return [
        {
            "question": s.question,
            "kind": s.kind,
            "depends_on_previous": s.depends_on_previous,
            "original_fragment": s.original_fragment,
            "order": s.order,
        }
        for s in steps
    ]


def _deserialize(raw: Any) -> tuple[SubQuery, ...]:
    """从缓存恢复。结构不符就返回空（当未命中）。"""
    if not isinstance(raw, list) or not raw:
        return ()
    out: list[SubQuery] = []
    try:
        for item in raw:
            if not isinstance(item, dict):
                return ()
            q = str(item.get("question") or "").strip()
            if not q:
                return ()
            kind = str(item.get("kind") or HOP_SIMPLE)
            if kind not in (HOP_SIMPLE, HOP_PRICE_RANK):
                kind = HOP_SIMPLE
            out.append(SubQuery(
                question=q,
                kind=kind,
                depends_on_previous=bool(item.get("depends_on_previous")),
                original_fragment=str(item.get("original_fragment") or q),
                order=int(item.get("order") or len(out)),
            ))
    except (TypeError, ValueError):
        return ()
    return tuple(out)


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha1((text or "").strip().encode("utf-8")).hexdigest()[:24]


def _cache_get(cache: Any, key: str) -> Any:
    try:
        return cache.get(key)
    except Exception:                                  # noqa: BLE001
        return None


def _cache_set(cache: Any, key: str, value: Any, ttl: int) -> None:
    try:
        cache.set(key, value, ttl_sec=ttl)
    except Exception:                                  # noqa: BLE001
        pass


__all__ = [
    "CachedQueryPlanner",
    "LLMQueryPlanner",
    "build_plan_prompt",
]
