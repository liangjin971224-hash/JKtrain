"""LLM 规划输出的校验层（纯函数，零 IO）。

────────────────────────────────────────────────────────────────────
为什么需要独立的校验层
────────────────────────────────────────────────────────────────────
LLM 输出 JSON 只是「一段可能对的文本」，不能直接当规划结果用。
实测过的问题（在语义映射那条路径上）：

  · 加了 ```json 包裹
  · 前面写一句「好的，分析如下：」
  · 给第一步也标 `depends_on_previous: true`（没有「前面」却标依赖）
  · 把原句原样拆成两段假装拆过
  · `kind` 写成 `price_rank` 以外的东西（如 `price`、`compare`）

这些都不能让它们流到执行层 —— 执行层一旦收到「第一步依赖前一步」，
就会去拿一个空候选集，结果要么答不出、要么答错。

所以这一层是**硬门槛**：不过就退回规则结果，而不是修正后勉强使用。

────────────────────────────────────────────────────────────────────
为什么单独成文件
────────────────────────────────────────────────────────────────────
校验是纯函数，**不依赖 LLM、不依赖端口**，因此：
  · 可以用几十条手工构造的非法输出做全量覆盖测试（毫秒级）
  · 修校验逻辑时不必启动模型

这是本项目一贯的做法：把「判断」和「副作用」分开，先让判断可测。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from domain.services.query_planner import (
    HOP_PRICE_RANK,
    HOP_SIMPLE,
    MAX_STEPS,
    SubQuery,
)

logger = logging.getLogger(__name__)

#: 允许的 kind 白名单。模型常见的错写（price / compare / cheapest）都归到 price_rank。
_KIND_ALIASES = {
    "simple": HOP_SIMPLE,
    "plain": HOP_SIMPLE,
    "normal": HOP_SIMPLE,
    "price_rank": HOP_PRICE_RANK,
    "price": HOP_PRICE_RANK,
    "compare": HOP_PRICE_RANK,
    "cheapest": HOP_PRICE_RANK,
    "rank": HOP_PRICE_RANK,
}

#: 从「一段可能带杂质的文本」里抠出 JSON 对象。
#: 先剥 markdown 代码围栏，再从第一个 `{` 开始做括号配对。
_FENCE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)```")


@dataclass(frozen=True, slots=True)
class PlanValidation:
    """校验结果。

    Attributes:
        steps: 通过的步骤；`ok=False` 时为空。
        ok: 是否可采纳。
        reason: 不可采纳的原因（可观测用，会进日志）。
        repairs: 做过的修补说明（例如「第1步的依赖标志被清除」）。
            修补不算失败 —— 有些错误可以安全纠正（如第一步标依赖）。
    """

    steps: tuple[SubQuery, ...] = ()
    ok: bool = False
    reason: str = ""
    repairs: tuple[str, ...] = field(default_factory=tuple)


# ===================================================================
# 一、解析：把文本变成 dict
# ===================================================================
def extract_json(raw: str) -> dict[str, Any] | None:
    """从模型输出里抠出 JSON 对象。失败返回 None。

    处理三种常见杂质：
      1. ```json ... ``` 围栏
      2. 前后有解释性文字
      3. 未闭合的 JSON（截断）→ 判失败，不猜
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # 1) 优先取围栏内的内容
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()

    # 2) 顶层若是数组，直接判失败（我们要的是 {"steps": [...]}）
    #    ★ 否则会把 [{"question":"q"}] 里的第一个对象抠出来当成合法 payload，
    #      接着因「缺少 steps」失败 —— 行为虽对，但失败原因会误导排障。
    stripped = text.lstrip()
    if stripped.startswith("["):
        return None

    # 3) 从第一个 { 开始做括号配对（要跳过字符串里的花括号）
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start:i + 1]
                try:
                    obj = json.loads(snippet)
                except (ValueError, TypeError):
                    return None
                return obj if isinstance(obj, dict) else None

    # 括号没闭合（输出被截断）→ 不猜
    return None


# ===================================================================
# 二、校验：dict → 可信的 SubQuery 列表
# ===================================================================
def validate_plan(
    payload: dict[str, Any] | None,
    *,
    question: str,
    max_steps: int = MAX_STEPS,
) -> PlanValidation:
    """校验模型输出，返回可用步骤或失败原因。

    Args:
        payload: `extract_json` 的结果。
        question: 用户原句，用于 C6「不许原句复读」判断。
        max_steps: 步数上限。

    Returns:
        `PlanValidation`。`ok=False` 时调用方应退回规则结果。
    """
    if not payload:
        return PlanValidation(reason="输出不是合法 JSON 对象")

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return PlanValidation(reason="缺少 steps 数组")
    if not raw_steps:
        return PlanValidation(reason="steps 为空")

    # C1 步数上限：截断而不是丢弃（多出来的部分是有效信息，只是超预算）
    limit = max(1, min(max_steps, MAX_STEPS))
    truncated = len(raw_steps) > limit
    raw_steps = raw_steps[:limit]

    repairs: list[str] = []
    steps: list[SubQuery] = []
    seen_questions: set[str] = set()
    normalized_original = _norm(question)

    for idx, item in enumerate(raw_steps):
        if not isinstance(item, dict):
            return PlanValidation(reason=f"第 {idx + 1} 步不是对象")

        # C2 每步必须有非空 question
        q = str(item.get("question") or "").strip()
        if not q:
            return PlanValidation(reason=f"第 {idx + 1} 步 question 为空")
        if len(q) > 200:
            # 超长大概率是模型把整段解释塞进来了
            return PlanValidation(reason=f"第 {idx + 1} 步 question 过长({len(q)}字)")

        # C6 不许原句复读（拆了等于没拆）
        if _norm(q) == normalized_original:
            if len(raw_steps) > 1:
                return PlanValidation(
                    reason=f"第 {idx + 1} 步与原句相同（假装拆解）")
            # 只有一步且等于原句：等价于「不用拆」，可以接受
            repairs.append("单步即原句，按不拆处理")

        # 重复子问题 → 冗余，直接失败（说明模型在凑步数）
        key = _norm(q)
        if key in seen_questions:
            return PlanValidation(reason=f"第 {idx + 1} 步与前面重复: {q[:20]}")
        seen_questions.add(key)

        # C3 第一步不许依赖前一步（没有「前面」）
        depends = bool(item.get("depends_on_previous", False))
        if idx == 0 and depends:
            depends = False
            repairs.append("第 1 步的依赖标志被清除（首步无前序）")

        # C4 kind 白名单
        kind_raw = str(item.get("kind") or HOP_SIMPLE).strip().lower()
        kind = _KIND_ALIASES.get(kind_raw)
        if kind is None:
            # 未知 kind 不丢弃这一步，降级为 simple —— 少拆比拆错安全
            kind = HOP_SIMPLE
            repairs.append(f"第 {idx + 1} 步 kind={kind_raw!r} 非法，降为 simple")

        steps.append(SubQuery(
            question=q,
            kind=kind,
            depends_on_previous=depends,
            original_fragment=str(item.get("original_fragment") or q),
            order=idx,
        ))

    if truncated:
        repairs.append(f"步数超过上限 {limit}，已截断")

    # C5 依赖步必须真有前一步产出 → 由执行层保证（前一步空则拒绝）
    #    这里只做一个静态检查：最后一步不能是唯一依赖步（前面全都不产生产品时无意义）
    #    这条不做强制，交给执行层 —— 规划时无法知道证据是否为空。

    return PlanValidation(steps=tuple(steps), ok=True, repairs=tuple(repairs))


def _norm(text: str) -> str:
    """归一化：去标点与空白，只留文字，用于「是否同一个问题」的比较。"""
    return re.sub(r"[\s，,。；;？?！!、\.·]+", "", text or "")


# ===================================================================
# 三、反向检查：规则结果是否「可疑」
# ===================================================================
def plan_looks_suspicious(
    question: str,
    rule_steps: tuple[SubQuery, ...],
    *,
    dropped_fragments: tuple[str, ...] = (),
) -> tuple[bool, str]:
    """规则结果值不值得让 LLM 复核。

    ★ 这是成本控制的关键：不是每个问题都调 LLM。
      只有这四个信号出现时才调 —— 它们表示「规则结果可能不完整/判错」。

    Args:
        question: 用户原句。
        rule_steps: 规则规划结果。
        dropped_fragments: 被 `_keep()` 丢掉的片段（可能是漏拆的线索）。

    Returns:
        `(是否可疑, 原因)`。
    """
    if not question or not question.strip():
        return False, ""

    # ★ T3 必须先判：单步 + 有片段被丢弃 = 最典型的「漏拆」
    #   （「改善体脂吃什么，给我讲讲那款」→ 丢弃了含回指词的有效第二步）
    #   若先判单步直接 return，这条永远走不到。
    if dropped_fragments:
        return True, f"有 {len(dropped_fragments)} 个片段被判为非问题而丢弃"

    # 单步问题：绝大多数，直接放行（不为它调 LLM）
    if len(rule_steps) <= 1:
        # 特例：单步但原文明显有两个问句（问号分隔的完整问句）
        if _has_multiple_question_marks(question):
            return True, "单步但原文含多个问号"
        return False, ""

    # T1 多步但没有任何一步被判依赖
    #
    # ★ 收紧条件：单纯的并列双问（「有什么推荐，有什么禁忌」）**不该触发** ——
    #   规则判对了，叫模型复核纯属浪费（实测触发率会从 44% 降到 22%）。
    #   只有出现「顺序性暗示」时才怀疑漏判依赖：
    #     · 条件式（不行的话 / 否则 / 那…呢）
    #     · 指代方位（这几样 / 上述 / 前面提到）
    #   这两类在规则里没有词表覆盖，才是 LLM 真正能补的地方。
    deps = [s for s in rule_steps if s.depends_on_previous]
    if not deps and _has_sequential_hint(question):
        return True, "多步无依赖且含顺序性暗示（可能漏判回指）"

    # T2 除第一步外全都是依赖（可能误判：把并列当依赖）
    #
    # ★ 只在 3 步及以上才判：2 步时「第二步依赖第一步」是**最常见的正常形态**
    #   （「改善体脂吃什么，其中那个最便宜」），把它当可疑会导致
    #   每个正常双跳都多调一次 LLM —— 成本失控且毫无收益。
    non_first = rule_steps[1:]
    if len(rule_steps) >= 3 and non_first and all(
            s.depends_on_previous for s in non_first):
        return True, "三步及以上、第二步起全部标为依赖（可能把并列误判为依赖）"

    # T4 原文问句数明显多于步数
    qmarks = _count_question_marks(question)
    if qmarks >= 2 and qmarks > len(rule_steps):
        return True, f"原文 {qmarks} 个问号但只拆出 {len(rule_steps)} 步"

    return False, ""


#: 顺序性暗示：出现这些词说明后一步可能依赖前一步的结果，
#: 但它们**不在** `_ANAPHORA` 里（规则判不出），所以值得让模型复核。
#:
#: ★ 为什么不直接加进 `_ANAPHORA`：
#:   那是领域层词表，改动会影响规则本身的行为（以及已有测试）。
#:   这里的定位是「触发复核的信号」，不是「判定依赖的依据」——
#:   两者职责不同，混在一起会让规则越来越难维护。
_SEQUENTIAL_HINT = re.compile(
    r"不行的话|如果不|否则|如果没|要不|那(?:就|么)|"
    r"这几样|那几样|上述|前面提到|刚说的|刚提到的|上面说的"
)


def _has_sequential_hint(text: str) -> bool:
    return bool(_SEQUENTIAL_HINT.search(text or ""))


def _count_question_marks(text: str) -> int:
    return len(re.findall(r"[？?]", text or ""))


def _has_multiple_question_marks(text: str) -> bool:
    return _count_question_marks(text) >= 2


__all__ = [
    "PlanValidation",
    "extract_json",
    "plan_looks_suspicious",
    "validate_plan",
]
