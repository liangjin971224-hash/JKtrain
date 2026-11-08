"""风险定位：把违禁表述定位到「第几轮第几句」。

★ 纯规则、纯函数、零依赖。

────────────────────────────────────────────────────────────────────
为什么定位精度重要
────────────────────────────────────────────────────────────────────
只告诉学员「你有违规」，他很难改进——一句一句回看十几轮对话去猜是哪句，
成本比重新练一遍还高。

定位到「第2轮第3句」后，学员可以直接看那一句、改那一句。

────────────────────────────────────────────────────────────────────
为什么 critical 必须 100% 检出
────────────────────────────────────────────────────────────────────
漏一条 critical，意味着学员说了违法宣传话术而系统没提示。
这条话术可能真的被他说给客户听。

所以：**只要规则匹配到，就必须报出来**。
不做「同一句只报最严重的一条」这类优化——那会掩盖其他违规。
只对「同一句 + 同一片段 + 同一规则」做去重，防止重复展示。
"""
from __future__ import annotations

import logging

from domain.rules.forbidden_claims import (
    _strip_compliant,
    _strip_quoted_negations,
    check_forbidden_claims,
)
from domain.value_objects import (
    CaseSpec,
    RiskItem,
    RiskLevel,
    TraineeTurn,
)

logger = logging.getLogger(__name__)


def collect_turns_text(turns: list[TraineeTurn]) -> str:
    """把多轮发言拼成一段文本（用于要素覆盖判定）。

    用换行分隔，便于正则里的 `[^。！？\\n]` 在轮次边界处自然断开。

    Args:
        turns: 学员的各轮发言。

    Returns:
        拼接后的文本。空输入返回空串。
    """
    return "\n".join(t.text for t in turns if t.text and t.text.strip())


def locate_risks(
    spec: CaseSpec, turns: list[TraineeTurn], *,
    include_global_rules: bool = True,
) -> list[RiskItem]:
    """扫描全部对话，定位风险。

    Args:
        spec: 案例规格（含违禁规则）。
        turns: 学员的各轮发言。
        include_global_rules: 是否叠加**全局禁用词规则**。

            ★ 为什么需要（实测发现）：
              案例文档的「五、违禁表述检测」表只列了该案例的重点规则。
              C01 的表里有「违规宣称 / 效果承诺 / 主动推销 / 回避推诿」，
              但**没有「他人经验背书」**这一栏。
              而 C01 的评分示例明确写「很多人吃了血脂都降下来了」是 critical。

              只用案例规则会漏检——学员说了「很多人吃了都说好」，
              系统不报，但他确实违规了（《广告法》）。

              所以评阅必须叠加 `forbidden_claims` 的六类全局规则，
              案例规则负责「这个案例的考察重点」，全局规则负责「兜住底线」。

    Returns:
        风险项列表，**按位置升序**排列。
        同一句里同一规则命中同一片段只报一次。

    Note:
        不同规则命中同一句的不同片段时**都报**——
        一句里既「降血脂」又「带两盒」是两件不同的事，
        合并成一条会让学员以为只需改一处。
    """
    if not turns:
        return []
    if not spec.forbidden_rules and not include_global_rules:
        return []

    risks: list[RiskItem] = []
    #: (轮次, 句序) → 该位置上已记录的 risks 下标列表。
    #: 用 list 而不是单个下标：同一句可能有多个**互不相关**的违规。
    seen: dict[tuple[int, int], list[int]] = {}

    for turn in turns:
        for s_idx, sentence in enumerate(turn.sentences, start=1):
            if not sentence.strip():
                continue

            # ★ 必须挖掉合规表述后再跑违禁规则（实测 bug）。
            #
            #   问题：法定警示语「保健食品不是药物，不能代替药物**治疗**疾病」
            #   含「治疗」，会被 C01 的 `治疗` 规则判为 critical 违规。
            #   结果是：学员把必须说的话说对了，系统反而报他不合规。
            #
            #   这比漏报更伤信任——漏报他不会知道，误报他会觉得系统不懂业务。
            #
            #   复用 P4 的白名单（`_strip_compliant` + `_strip_quoted_negations`），
            #   与输出护栏用同一套判定标准，避免两处规则漂移。
            probe = _strip_quoted_negations(_strip_compliant(sentence))

            # ① 案例自带规则
            for rule in spec.forbidden_rules:
                for matched in rule.find_all(probe):
                    _append(risks, seen, turn.turn_index, s_idx,
                            rule.rule_type, matched, rule.level, rule.note)

            # ② 全局禁用词规则（六类，与输出护栏同源）
            if include_global_rules:
                claim = check_forbidden_claims(probe)
                for v in claim.violations:
                    _append(
                        risks, seen, turn.turn_index, s_idx,
                        f"全局·{v.category}", v.matched,
                        RiskLevel.CRITICAL if v.severity.must_block
                        else RiskLevel.HIGH,
                        v.reason,
                    )

    # 按位置排序（轮次 → 句序），同级 critical 在前便于阅读
    risks.sort(key=lambda r: (
        r.turn_index, r.sentence_index,
        0 if r.is_critical else 1,
    ))

    crit_count = sum(1 for r in risks if r.is_critical)
    if crit_count:
        logger.info(
            "案例 %s 风险定位: 共 %d 条，其中 critical %d 条",
            spec.case_id, len(risks), crit_count,
        )
    return risks


def _append(
    risks: list[RiskItem],
    seen: dict[tuple[int, int], list[int]],
    turn_index: int,
    sentence_index: int,
    rule_type: str,
    matched: str,
    level: RiskLevel,
    note: str,
) -> None:
    """追加一条风险（去重）。

    ★ 去重规则（两个维度都要处理，实测踩过两次）：

      ① **同一片段被不同规则命中** → 只报一次
         「降血脂」既被案例规则「违规宣称」命中，也被全局规则
         「疾病治疗」命中。是同一件事，报两次是凑数。

      ② **同位置命中的片段互为子串** → 只保留最长的
         「可以的，一起吃没问题」同时被 JD-05 命中「可以的，一起吃没问题」
         和 JD-02 命中「一起吃没问题」。后者是前者的子串，
         报两次学员会疑惑。保留最长那条——信息最完整。

      ③ **同位置但片段不相关的**（如既说「降血脂」又说「带两盒」）
         → **两条都报**。那是两件不同的事，合并会让学员以为只需改一处。
    """
    key = (turn_index, sentence_index)
    idxs = seen.get(key, [])

    for idx in idxs:
        existing = risks[idx]

        # ① 完全相同 或 ② 互为子串
        if matched == existing.matched:
            return
        if matched in existing.matched:
            return                       # 已有更长，丢弃新的
        if existing.matched in matched:
            # 新的更长 → 替换旧条目
            risks[idx] = RiskItem(
                level=level,
                rule_type=rule_type,
                matched=matched,
                turn_index=turn_index,
                sentence_index=sentence_index,
                reason=note or f"命中{rule_type}",
            )
            return

    risks.append(RiskItem(
        level=level,
        rule_type=rule_type,
        matched=matched,
        turn_index=turn_index,
        sentence_index=sentence_index,
        reason=note or f"命中{rule_type}",
    ))
    seen.setdefault(key, []).append(len(risks) - 1)


def summarize_risk_levels(risks: list[RiskItem]) -> dict[str, int]:
    """按级别统计条数（供展示用，**不是评分**）。

    ★ 注意：这是「计数」，不是「得分」。
      摘要卡里不能出现任何形式的分数（约束 #3），
      但「critical 3 条 / high 1 条」这种计数是事实陈述，可以展示。
    """
    counts: dict[str, int] = {}
    for r in risks:
        counts[r.level.value] = counts.get(r.level.value, 0) + 1
    return counts


__all__ = [
    "collect_turns_text",
    "locate_risks",
    "summarize_risk_levels",
]
