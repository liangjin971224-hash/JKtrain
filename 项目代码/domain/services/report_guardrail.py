"""报告建议专用校验器。

────────────────────────────────────────────────────────────────────
为什么不能直接用 `OutputGuardrail`
────────────────────────────────────────────────────────────────────
`OutputGuardrail` 判的是「这段话是不是在**声称**功效」。
它有一条极窄的引述放行规则：**否定引导词 + 引号**
（如「不能说『降血脂』」）。

但月度报告的建议**天生就要引述违规话术**：
    「C07 第1轮第1句说『降血脂』属于疾病治疗，需重练」
这句话里没有否定引导词（「属于违规」不是），
于是被现有护栏判成「命中疾病治疗」→ 拦下。

实测（2026-10-31）：第一版直接用 `OutputGuardrail` 校验建议，
生成 3 条建议全部被拦，重写后仍然被拦，报告的建议部分无法产出。

────────────────────────────────────────────────────────────────────
两者的语义差别
────────────────────────────────────────────────────────────────────
| 场景 | 判断标准 |
|---|---|
| 学员回答 | 有没有**声称**功效（引号只在小范围内放行） |
| **评阅/报告** | 有没有**无引号**地声称功效；引号内的是**引述**，一律放行 |

关键洞察：**报告里出现违规话术是正常的，甚至是必需的**——
报告的作用就是指出「你说错了哪句」。
所以判断标准不是「有没有违规词」，而是「这个违规词是不是被当作反面例子引用」。

────────────────────────────────────────────────────────────────────
放行边界（要窄，否则真违规会漏过）
────────────────────────────────────────────────────────────────────
✅ 放行：「第1轮第1句说『降血脂』属于违规」
           —— 违规词在引号内
✅ 放行：「不能对客户说降血脂」
           —— 有否定引导词（复用现有护栏的这条规则）
❌ 拦住：「这款产品能降血脂，建议这么讲」
           —— 违规词在引号外，且无否定语境
❌ 拦住：「建议告诉客户可以降血脂」
           —— 同上

也就是说：**优先用现有护栏判；只有「违规词全部落在引号内」时才放行**。
这样真违规不会漏——因为真违规没必要给自己的禁用词加引号。
"""
from __future__ import annotations

import logging
import re

from domain.services.output_guardrail import (
    GuardrailVerdict,
    OutputGuardrail,
)

logger = logging.getLogger(__name__)

#: 成对引号。中英文都要，报告里两种都可能出现。
_QUOTE_CHARS = "「」『』\"'“”‘’《》"

#: 抓出所有「被引号包住」的片段。
#:
#: ★ 不用后向引用配对，而是按开闭引号字符逐个扫：
#:   中英文引号混用时（「…"…"…」）后向引用会错配。
_QUOTED = re.compile(r"[「『\"'“”‘’《]([^」』\"'“”‘’》\n]{1,60})[」』\"'“”‘’》]")


#: 报告里会出现的**规则类别名 / 违规类型名**，属于术语而非声称。
#:
#: ★ 实测踩坑（2026-10-31）：
#:   报告写「第1句说『降血脂』属于**疾病治疗**」，
#:   去掉引号后剩「属于疾病治疗」，护栏按「治疗」命中 → 误拦。
#:   但「疾病治疗」在这里是**违规类别名**，不是声称能治病。
#:
#: ★ 为什么用替换而不是加白名单跳过：
#:   如果整句跳过，「属于疾病治疗，建议说能降血脂」会漏过后半句。
#:   把类别名替换成中性占位符，**其余文字照常检测**，两边都不误。
_CATEGORY_TERMS = (
    "疾病治疗", "经验背书", "绝对化表述", "功效声称", "医疗用语",
    "医疗术语", "禁用表述", "违规宣称", "夸大宣传",
)

_PLACEHOLDER = "〇"


def strip_category_terms(text: str) -> str:
    """把「规则类别名」替换成占位符，避免被当成声称命中。

    只替换类别名本身，句子其余部分保留 —— 后续检测照常生效。
    """
    out = text or ""
    for term in _CATEGORY_TERMS:
        out = out.replace(term, _PLACEHOLDER)
    return out


def strip_quoted(text: str) -> str:
    """去掉所有引号片段，只留引号外的文字。

    用途：如果违规词**只出现在引号内**，去掉后就不该再命中；
    如果去掉后仍命中，说明引号外还有违规表述 → 该拦。
    """
    return _QUOTED.sub("", text or "")


class ReportGuardrail:
    """报告建议专用校验器。

    判定顺序：
        1. 先看**去掉引号片段后的文本**是否仍然违规
        2. 仍然违规 → 拦住（引号外有真实声称）
        3. 不再违规 → 再看原文（保留否定语境等现有放行规则）
        4. 原文也不违规 → 放行

    ★ 为什么不反过来（先原文后去引号）：
      原文里含引述，现有护栏大概率判违规；
      先判「去掉引述后是否还违规」才是报告场景的正确问法。
    """

    def __init__(self, base: OutputGuardrail | None = None) -> None:
        self._base = base or OutputGuardrail()

    def check(self, text: str) -> GuardrailVerdict:
        """检查一段报告建议。

        Returns:
            `GuardrailVerdict`。`blocked=True` 表示**引号外**存在违规声称。
        """
        if not text or not text.strip():
            return GuardrailVerdict(blocked=False, reason="空文本")

        # ① 去掉引号片段 + 类别术语后再判
        #   ★ 两步都要做：引号内是引述，类别名是术语，
        #     两者都不是「声称」，但都可能命中禁用词。
        bare = strip_category_terms(strip_quoted(text))
        verdict_bare = self._base.check(bare, category=None)
        if verdict_bare.blocked:
            return GuardrailVerdict(
                blocked=True,
                reason=("引号外存在违规表述：" + verdict_bare.reason),
                violations=verdict_bare.violations,
                rule_ids=verdict_bare.rule_ids,
            )

        # ② 引号外干净，再判原文（让否定语境的放行规则继续生效）
        verdict_full = self._base.check(text, category=None)
        if not verdict_full.blocked:
            return verdict_full

        # ③ 原文违规但引号外干净 → 判定为「纯引述」，放行
        logger.info(
            "报告建议中的违规词全部落在引号内，按引述放行: %s",
            verdict_full.reason[:60])
        return GuardrailVerdict(
            blocked=False,
            reason=f"引述放行（违规词均在引号内）: {verdict_full.reason}",
        )


__all__ = ["ReportGuardrail", "strip_category_terms", "strip_quoted"]
