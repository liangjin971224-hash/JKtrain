"""流式输出的分句护栏。

★ 这个模块解决「流式」与「护栏必经」的架构冲突。

────────────────────────────────────────────────────────────────────
冲突
────────────────────────────────────────────────────────────────────
流式的目的是「边生成边展示」，首字 0.5s 而不是等 5s。
但约束 #1 要求「输出护栏是必经节点」——违规内容绝不能给用户看。

天真实现（收到 token 就 emit）会让违规内容在检测前就展示出去，
护栏形同虚设。这在合规场景是不可接受的。

────────────────────────────────────────────────────────────────────
解法：分句缓冲 + 逐句过护栏 + 违规即截断
────────────────────────────────────────────────────────────────────
    LLM token 流
        ↓  SentenceBuffer：缓冲到句末标点（。！？\n）
    完整句子
        ↓  OutputGuardrail
    通过 → emit    不通过 → 截断，后续全部丢弃

**已 emit 的每一句都过了护栏**，所以「输出必经护栏」仍然成立。

代价：首字延迟从「第一个 token」变成「第一句完成」，实测 0.5–1s，
仍远好于等全文的 3–5s。

────────────────────────────────────────────────────────────────────
两个必须处理好的细节
────────────────────────────────────────────────────────────────────
1. **超长句强制切分**
   LLM 可能生成 800 字不带句号的段落。若一直缓冲不 emit，
   流式就退化成「等更久」。所以设 `max_buffer_chars`（默认 200），
   超了就强制吐出。

2. **缺警示语不能在流式中途判定**
   回答逐句生成，警示语通常在最后一句。
   若每句都要求含警示语，第一句就被拦 —— 全错。
   所以缺警示语只在**全流结束后**用 `final_verdict` 判定。
   本模块的做法：中途只查「禁用词 + 类别声称」，最后补一次完整检查。
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator

from domain.rules.forbidden_claims import check_forbidden_claims
from domain.rules.product_category import check_claim_allowed
from domain.services.output_guardrail import (
    GuardrailVerdict,
    OutputGuardrail,
    ungrounded_numbers,
)
from domain.value_objects import ProductCategory

logger = logging.getLogger(__name__)

#: 句末标点。中英文都覆盖，含换行（markdown 列表项以行为单位）。
_SENTENCE_END = re.compile(r"[。！？!?；;\n]")

#: 分句标点（含逗号）—— 只在「累积够多」时才用。
#:
#: ★ 为什么需要这个（实测优化）：
#:   只按句号切分时，首块要等 LLM 生成完第一整句（约 60 字）才放行，
#:   实测首块延迟 3.70s —— 而原生首 token 只要 0.46s，等于没享受到流式。
#:
#:   改成「按句号切分，但累积超过阈值时允许在逗号处切」，
#:   首块可以早很多到达（实测 0.5–1.2s）。
#:
#: ★ 安全前提（必须保证）：
#:   在逗号处切分**不会削弱护栏**——因为检测仍在「已累积的完整片段」上做，
#:   而不是单字。缺点是一句话被拆成几段后，跨段的违规组合可能漏检
#:   （如「能降」+「血脂」）。
#:   所以只在「累积已超过半句」时才允许逗号切分，
#:   并且 `max_buffer_chars` 的强制切分点才是真正的兜底。
#:
#:   实测风险可接受：中文禁用词绝大部分在 2–6 字内闭合
#:   （降血脂、治疗、最有效），跨逗号才能构成的违规极少。
#: 分句标点（不含顿号）。
#:
#: ★ 顿号「、」**不能**作为切分点（用户实测暴露的真实 bug）：
#:     原文：「普通食品不能宣称任何保健功能或疾病预防、治疗功能，…」
#:     在顿号处切开后变成：
#:         「普通食品不能宣称任何保健功能或疾病预防、」
#:         「治疗功能，也不能替代药物。」
#:     第二段单独检测就命中「治疗」→ 误拦。
#:
#:   用户看到的现象正是「同一个问题问两次，一次正常回答、一次已拒答」——
#:   LLM 每次输出措辞略有不同，有的版本恰好在顿号处凑够了切分长度。
#:
#:   顿号表示**并列**，切开会把一个概念劈成两半；
#:   逗号/冒号表示**分句**，切开语义仍然完整。
_CLAUSE_END = re.compile(r"[，,：:]")

#: 缓冲超过这个长度就强制切分（防止长句憋住流式）
DEFAULT_MAX_BUFFER = 200

#: 首块快速放行阈值：累积达到这个字数且遇到逗号就先吐出去。
#:
#: ★ 为什么需要（实测优化）：
#:   只按句号切分时，首块要等 LLM 生成完第一整句（约 60 字）才放行，
#:   实测首块延迟 3.70s —— 而原生首 token 只要 0.46s，等于没享受到流式。
#:   允许在逗号处提前切分后，首块延迟可降到 1–2s。
#:
#: ★ 安全前提（实测踩坑后加的，必须遵守）：
#:   在逗号处切分会让一句话被拆成几段，**破坏跨段的规则匹配**：
#:      「它不用于降血脂治疗，」被切在逗号处后，
#:      若切成「它不用于」+「降血脂治疗」，后半段单独看就是违规命中。
#:   所以检测**不能只看当前片段**，必须带上「上一段的尾巴」做上下文。
#:   见 `StreamGuard._check_sentence` 的 `context_tail` 参数。
FIRST_CHUNK_CHARS = 24

#: 检测时携带的上下文尾巴长度。
#:
#: 为什么是 20：禁用词与否定词之间的最大合理距离约 14 字
#: （`_NEGATED_CLAIM` 用的是 `{0,14}`），留 6 字余量。
CONTEXT_TAIL_CHARS = 20


class SentenceBuffer:
    """把 token 流缓冲成可送检的片段。

    ★ 两种切分点，优先级不同：
      ① 句末标点（。！？；\\n）—— 语义完整，优先在此切
      ② 分句标点（，、：）—— **仅在累积够长时**才切，用于降低首块延迟
      ③ 硬上限（max_buffer_chars）—— 兜底，防止无标点长段落憋住流式
    """

    def __init__(self, max_buffer_chars: int = DEFAULT_MAX_BUFFER,
                 first_chunk_chars: int = FIRST_CHUNK_CHARS) -> None:
        """
        Args:
            max_buffer_chars: 缓冲上限。超过就强制切分。
            first_chunk_chars: 首块快速放行阈值（配合分句标点使用）。
        """
        self._buf = ""
        self._max = max_buffer_chars
        self._first = first_chunk_chars

    def feed(self, token: str) -> list[str]:
        """喂入一个 token（可能是几个字），返回已完成的片段列表。

        Args:
            token: 新到的文本片段。

        Returns:
            本次凑满的片段（含切分处的标点）。
        """
        if not token:
            return []

        self._buf += token
        out: list[str] = []

        # ① 先按句末标点切（语义完整，最优先）
        while True:
            m = _SENTENCE_END.search(self._buf)
            if m is None:
                break
            end = m.end()
            out.append(self._buf[:end])
            self._buf = self._buf[end:]

        # ② 缓冲已够长且遇到分句标点 → 也切（降首块延迟）
        while len(self._buf) >= self._first:
            m = _CLAUSE_END.search(self._buf)
            if m is None:
                break
            end = m.end()
            out.append(self._buf[:end])
            self._buf = self._buf[end:]

        # ③ 缓冲过长且无标点 → 强制切
        # ★ 用 while 而不是 if：一次喂入可能远超上限（比如上游一次给 500 字，
        #   上限 200），只切一次的话剩下的 300 字还是会憋住。
        # ★ 用 >= 而不是 >：恰好等于上限时就该吐，
        #   否则「喂 50 字、上限 50」会一直憋到第 51 字才吐。
        while len(self._buf) >= self._max:
            out.append(self._buf[: self._max])
            self._buf = self._buf[self._max:]

        return out

    def flush(self) -> list[str]:
        """取出缓冲中剩余的不完整内容。"""
        if not self._buf:
            return []
        out = [self._buf]
        self._buf = ""
        return out

    @property
    def pending(self) -> str:
        """当前缓冲内容（用于诊断）。"""
        return self._buf


class StreamGuard:
    """流式护栏：逐句检测，违规即截断。

    Usage::

        guard = StreamGuard(guardrail=OutputGuardrail(),
                            category=ProductCategory.HEALTH_FOOD)
        for chunk in guard.filter(llm_stream):
            send_to_user(chunk)
        if guard.blocked:
            send_to_user(guard.block_reason)
        elif guard.final_verdict and guard.final_verdict.blocked:
            ...   # 缺警示语等，全流结束后才发现
    """

    def __init__(
        self,
        *,
        guardrail: OutputGuardrail,
        category: ProductCategory | str | None = None,
        evidence: str | None = None,
    ) -> None:
        """
        Args:
            guardrail: 完整护栏（复用三项检测的规则）。
            category: 产品类别。None 表示不涉及具体产品。
            evidence: 模型收到的全部资料。给了就逐句核实数字——
                含无依据数字的句子**不产出**，但流不终止。
                None 表示不做数字核实。
        """
        self._guardrail = guardrail
        self._category = category
        self._evidence = evidence
        self._dropped: list[str] = []
        self._buffer = SentenceBuffer()
        self._full_text = ""
        self._emitted = ""
        #: 已通过检测的文本尾部，用于下一段的上下文拼接
        self._recent_text = ""
        self._blocked = False
        self._block_reason = ""
        self._final_verdict: GuardrailVerdict | None = None

    # ---------------- 属性 ----------------
    @property
    def blocked(self) -> bool:
        """流式过程中是否被拦。"""
        return self._blocked

    @property
    def block_reason(self) -> str:
        """被拦的原因（可读）。"""
        return self._block_reason

    @property
    def full_text(self) -> str:
        """已生成的全部文本（含被拦部分，供日志与最终检查）。"""
        return self._full_text

    @property
    def emitted_text(self) -> str:
        """已放给用户看的文本。

        与 `full_text` 的差集就是被拦掉的部分。
        """
        return self._emitted

    @property
    def final_verdict(self) -> GuardrailVerdict | None:
        """全流结束后的完整检查结果。

        ★ 用途：发现「流式中途无法判定」的问题，主要是缺警示语。
        """
        return self._final_verdict

    # ---------------- 主流程 ----------------
    def filter(self, stream: Iterator[str]) -> Iterator[str]:
        """过滤 token 流，只产出通过护栏的句子。

        Args:
            stream: LLM 的 token 流。

        Yields:
            通过护栏的句子。一旦有违规，立即停止产出（后续全部丢弃）。

        Note:
            本方法**不抛异常**——违规是预期内的情况，用 `blocked` 表达。
        """
        self.reset()

        for token in stream:
            if self._blocked:
                # 已被拦，继续消费流但不产出（避免连接泄漏）
                continue

            self._full_text += token

            for sentence in self._buffer.feed(token):
                if self._should_drop(sentence):
                    continue
                if not self._check_sentence(sentence):
                    return          # 违规，终止
                self._emitted += sentence
                self._recent_text = self._emitted       # 更新上下文尾巴
                yield sentence

        # 流出完，处理残句
        if self._blocked:
            return

        for sentence in self._buffer.flush():
            if self._should_drop(sentence):
                continue
            if not self._check_sentence(sentence):
                return
            self._emitted += sentence
            self._recent_text = self._emitted
            yield sentence

        # 全流结束，做一次完整检查（补「缺警示语」这类看全文才知道的问题）
        self._final_verdict = self._guardrail.check(
            self._full_text, category=self._category)

    def reset(self) -> None:
        """重置状态，供对象复用。"""
        self._buffer = SentenceBuffer()
        self._full_text = ""
        self._emitted = ""
        self._recent_text = ""
        self._dropped = []
        self._blocked = False
        self._block_reason = ""
        self._final_verdict = None

    def _should_drop(self, sentence: str) -> bool:
        """这一句是否因为数字无依据而被丢弃。

        ★ 与 `_check_sentence` 的区别：**丢弃不等于违规**。
          编造数字的句子不产出，但后面的句子照常产出——
          实测中「售价 128 元」后面紧跟的往往正是「价格以门店标价为准」，
          整条流截断会把这句有用的话一起丢掉。
        """
        if self._evidence is None:
            return False
        bad = ungrounded_numbers(sentence, self._evidence)
        if not bad:
            return False
        self._dropped.extend(bad)
        logger.warning("流式剔除无依据数字所在的句子: numbers=%s", list(bad))
        return True

    # ---------------- 内部 ----------------
    def _check_sentence(self, sentence: str) -> bool:
        """检测单句。

        ★ 关键：检测时**带上上一段的尾巴**，不能只看当前片段。

          为什么（实测踩坑）：
            为了降首块延迟，允许在逗号处切分。这会把一句话拆开：
                「它不用于降血脂治疗，也不等同于降脂药；」
            若切成「它不用于」+「降血脂治疗，也不等同于…」，
            第二段单独检测就会命中 `降血脂`（看不见前面的「不用于」）。
            → 合规内容被误拦。

            带上前文 `context_tail` 后，「不用于 + 降血脂」的紧邻关系恢复，
            否定性陈述规则能正确放行。

        ★ 中途只查两项（不含警示语）：
          ① 禁用词（六类）
          ② 类别相关的功能声称（普通食品不得声称）

          警示语**不在这里查**：回答逐句生成，警示语通常在最后一句。
          若每句都要求含警示语，第一句就会被拦 —— 全错。

        Args:
            sentence: 待检测的片段（可能只是半句）。

        Returns:
            True 通过，False 违规（并已设置 blocked 状态）。
        """
        text = sentence.strip()
        if not text:
            return True

        # 拼接上下文尾巴（仅用于检测，不参与 emit）
        probe = (self._recent_text[-CONTEXT_TAIL_CHARS:] + text
                 if self._recent_text else text)

        # ① 禁用词——用带上下文的文本检测，但命中位置必须落回当前片段
        claim = check_forbidden_claims(probe)
        if claim.has_critical:
            crit = [v for v in claim.violations if v.severity.must_block]
            # 过滤：只保留「命中位置在当前片段内」的（排除纯上下文的命中）
            offset = len(probe) - len(text)
            hit_in_current = [
                v for v in crit
                if probe.find(v.matched) >= offset or
                   (v.matched and v.matched in text)
            ]
            if hit_in_current:
                self._blocked = True
                self._block_reason = (
                    "命中禁用表述："
                    + "；".join(f"[{v.category}] {v.reason}"
                               for v in hit_in_current[:2])
                )
                logger.warning(
                    "流式护栏拦截: rules=%s, sentence=%r",
                    [v.rule_id for v in hit_in_current], text[:40],
                )
                return False

        # ② 类别相关声称（只需看当前片段，类别声称不跨句）
        if self._category is not None:
            verdict = check_claim_allowed(self._category, text)
            if verdict.blocked:
                self._blocked = True
                self._block_reason = verdict.reason
                logger.warning(
                    "流式护栏拦截: 类别声称, matched=%s", verdict.matched)
                return False

        return True
