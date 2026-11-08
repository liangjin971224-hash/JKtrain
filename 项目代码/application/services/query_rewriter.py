"""多轮查询改写（docs/TDD_多轮问答_2026-11-08.md §3）。

────────────────────────────────────────────────────────────────────
为什么只改写、不注入历史
────────────────────────────────────────────────────────────────────
P1-b 已验证：把上下文塞进 prompt 会稀释当前问题、把答案带偏。
所以改写器只做一件事 —— 把「那孕妇能吃吗」变成能独立检索的
「康源牌鱼油软胶囊 孕妇能吃吗」，然后交给**原链路**。
检索、融合、窗口、生成、护栏、prompt 全部不用改。

────────────────────────────────────────────────────────────────────
优先级
────────────────────────────────────────────────────────────────────
1. 规则（<50ms，不调 LLM）：产品切换 / 纠正 / 指代 / 省略 / 场景继承；
2. LLM 兜底：仅规则未命中且注入了 llm 时，2s 超时、失败回退原问题；
3. 无历史 或 低置信度 → 原问题、`source=none`，单轮行为不变。
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass

from application.services.conversation import ConversationSession, ConversationTurn
from application.use_cases.retrieve import RetrieveUseCase

logger = logging.getLogger(__name__)

#: 指代词表（TDD §3.1）。命中且当前问句没有新主语时才继承上文。
_DEIXIS = (
    "这个产品", "这款", "该产品", "还有", "同样", "那", "这", "它", "呢", "也",
)
_DEIXIS_RE = re.compile("|".join(re.escape(w) for w in _DEIXIS))
_LEADING_DEIXIS_RE = re.compile(
    r"^(?:(?:这个产品|这款|该产品|还有|同样|那|这|它|也)\s*[，,、]?\s*)+"
)
_TRAILING_NE_RE = re.compile(r"[呢]+\s*[？?]?\s*$")

#: 场景关键词：只把**实体**拼进改写结果，不拼上一轮整个问句。
_SCENE_WORDS = (
    "头晕", "退货", "退款", "换货", "投诉", "过敏", "腹泻", "拉肚子",
    "肚子疼", "恶心", "停药", "孕妇", "哺乳", "儿童", "禁忌", "不良反应",
    "保质期", "有效期", "发票", "上报", "搭配", "超量", "开封",
    "改善视力", "视力", "降血糖", "血糖", "赔偿",
)
#: 明确主题词：与产品名/文号一起构成「改写结果必须含实体」的判据。
_TOPIC_WORDS = _SCENE_WORDS + (
    "适宜人群", "批准功能", "保健功能", "警示语", "用法用量",
    "规格", "批准文号", "成分", "价格",
)

#: 产品编号（「不是这款，是 P002」这类纠正靠它命中）。
_PRODUCT_ID_RE = re.compile(r"(?<![A-Za-z0-9])P\d{2,4}(?![A-Za-z0-9])", re.I)
_BRAND_RE = re.compile(r"康源牌")
_APPROVAL_RE = re.compile(r"国食健注\s*[A-Za-z0-9]+", re.I)

#: 省略型追问的判据：短、无实体、且确实是个追问（不是另起一个新话题）。
_ELLIPSIS_ASK_RE = re.compile(r"吗|呢|几[次粒片天]|多少|怎么|多久|能不能|要不要|[？?]")

_REWRITE_SYSTEM = "你是查询改写器。只输出改写后的独立问题本身，不要任何解释。"
_META_WORDS = ("根据上文", "上文", "改写", "以下是", "答案", "：", ":")
_CORRECTION_WORDS = ("不是", "换成", "改成", "纠正", "弄错")


def _is_correction(text: str) -> bool:
    """纠正型问句（「不是这款，是 P002」）不应套用主题继承。"""
    return any(w in (text or "") for w in _CORRECTION_WORDS)


def _has_product_entity(text: str) -> bool:
    """是否含产品名 / 品牌 / 文号 / 产品编号。

    ★ 品类词表直接复用 `RetrieveUseCase._extract_product_candidates`，
      不在改写器里另抄一份，避免两处词表漂移。
    """
    return bool(
        _BRAND_RE.search(text)
        or _APPROVAL_RE.search(text)
        or _PRODUCT_ID_RE.search(text)
        or RetrieveUseCase._extract_product_candidates(text)
    )


def _has_entity(text: str) -> bool:
    """改写结果是否含实体（产品名 / 文号 / 明确主题）。"""
    return _has_product_entity(text) or any(w in text for w in _TOPIC_WORDS)


@dataclass(frozen=True)
class RewriteResult:
    """改写结果。

    `source`: rule | llm | none；`confidence`: high | medium | low。
    `used_turn`: 本次改写参考了几轮历史（0 = 没用上文）。
    """

    question: str
    source: str = "none"
    confidence: str = "low"
    product_ids: tuple[str, ...] = ()
    used_turn: int = 0


class QueryRewriter:
    """把依赖上文的追问改写成独立问题。

    构造函数只接 Port，不碰具体适配器；`llm=None` 时完全走规则，
    规则未命中就原样返回（零额外延迟）。
    """

    def __init__(self, graph=None, llm=None, timeout_sec: float = 2.0) -> None:
        """
        Args:
            graph: 图谱端口，用于 `resolve_alias` 归一产品名/别名。
            llm: 语言模型端口。None 表示关闭 LLM 兜底。
            timeout_sec: LLM 改写超时（秒）。
        """
        self._graph = graph
        self._llm = llm
        self._timeout = max(0.01, float(timeout_sec))

    # ---------------- 主流程 ----------------
    def rewrite(
        self, question: str, session: ConversationSession | None
    ) -> RewriteResult:
        """改写问题。

        Args:
            question: 用户当前原话。
            session: 会话快照；None 或没有历史时原样返回。

        Returns:
            `RewriteResult`。任何异常都不得抛给调用方——改写是增益，
            失败必须退回原问题（TDD §6）。
        """
        q = (question or "").strip()
        turns = list(session.turns) if session is not None else []
        if not q or not turns:
            return self._none(q)

        try:
            return self._rewrite_with_history(q, turns)
        except Exception as exc:                # noqa: BLE001 改写失败降级单轮
            logger.warning("查询改写异常，回退原问题: %s: %s",
                           type(exc).__name__, exc)
            return self._none(q)

    def _rewrite_with_history(
        self, q: str, turns: list[ConversationTurn]
    ) -> RewriteResult:
        prev = turns[-1]

        # ---- 1. 产品切换 / 纠正：只换 product_ids；省略式切换继承上一轮主题 ----
        #   当前问句里出现新产品（或显式编号）说明主语已经变了，
        #   此时硬拼上一轮产品只会把两个产品混进同一个检索。
        product = self._detect_product(q)
        if product is not None:
            pid, name = product
            rewritten = q
            # 只有「新产品 + 裸省略（辅酶Q10 呢？）」才需要补上一轮主题；
            # 完整问句与纠正句（不是这款，是 P002）保持原句。
            if (pid not in set(prev.product_ids)
                    and not _is_correction(q)
                    and self._is_bare_product_ellipsis(q, name)):
                topic = self._previous_topic(prev)
                if topic:
                    rewritten = f"{name or pid} {topic}？"
            return RewriteResult(
                question=rewritten, source="rule", confidence="high",
                product_ids=(pid,), used_turn=1,
            )

        # 已有产品名/文号的完整问句，不该被指代规则误拼
        if _has_product_entity(q):
            return self._none(q)

        # ---- 2. 指代 / 省略 / 场景继承 ----
        if _DEIXIS_RE.search(q) or self._is_ellipsis(q):
            rewritten = self._join_context(q, prev)
            if rewritten and rewritten != q and _has_entity(rewritten):
                return RewriteResult(
                    question=rewritten, source="rule", confidence="high",
                    product_ids=tuple(prev.product_ids), used_turn=1,
                )

        # ---- 3. LLM 兜底 ----
        if self._llm is not None:
            return self._rewrite_by_llm(q, turns[-3:])
        return self._none(q)

    # ---------------- 规则 ----------------
    def _detect_product(self, q: str) -> tuple[str, str] | None:
        """识别当前问句里的新产品，返回 (product_id, 产品名)。

        先认显式编号（纠正场景），再用品类词表抽候选名交图谱解析。
        图谱查不到显式编号时也接受它——`P002` 本身就是无歧义实体。
        """
        m = _PRODUCT_ID_RE.search(q)
        if m is not None:
            pid = m.group(0).upper()
            name = self._product_name(pid)
            return (pid, name)

        for candidate in RetrieveUseCase._extract_product_candidates(q):
            try:
                ref = (self._graph.resolve_alias(candidate)
                       if self._graph is not None else None)
            except Exception as exc:            # noqa: BLE001 图谱异常不该拖垮改写
                logger.warning("别名解析失败: %s: %s", type(exc).__name__, exc)
                continue
            if ref is not None:
                return (str(ref.product_id), str(getattr(ref, "name", "") or ""))
        return None

    def _product_name(self, pid: str) -> str:
        """产品 ID → 产品名（拼接改写结果的是名字，不是编号）。"""
        try:
            facts = (self._graph.get_product_facts(pid)
                     if self._graph is not None else None)
            if facts is not None and getattr(facts, "name", ""):
                return str(facts.name)
            ref = (self._graph.resolve_alias(pid)
                   if self._graph is not None else None)
            if ref is not None and getattr(ref, "name", ""):
                return str(ref.name)
        except Exception as exc:                # noqa: BLE001
            logger.warning("产品名解析失败，回退用编号: %s: %s",
                           type(exc).__name__, exc)
        return ""

    @staticmethod
    def _is_ellipsis(q: str) -> bool:
        """「一天几次？」这类省略主语的短追问。

        ★ 必须限长且要求追问语气：否则「公司有什么培训制度」这种
          另起话题的完整问句会被误拼上一轮产品。
        """
        if _has_entity(q):
            return False
        core = q.strip("？? 。")
        return len(core) <= 8 and bool(_ELLIPSIS_ASK_RE.search(core))

    @staticmethod
    def _is_bare_product_ellipsis(q: str, product_name: str) -> bool:
        """「辅酶Q10 呢？」这类只换了产品的裸省略。

        去掉产品名/编号/品类词与语气词后，若几乎没有剩余内容，说明
        用户只说了新产品，需要把上一轮主题（多少钱/怎么吃）补上。
        """
        core = q
        if product_name:
            core = core.replace(product_name, "")
        for cand in RetrieveUseCase._extract_product_candidates(q):
            core = core.replace(cand, "")
        core = _PRODUCT_ID_RE.sub("", core)
        core = _BRAND_RE.sub("", core)
        core = _APPROVAL_RE.sub("", core)
        core = re.sub(r"[\s，,、？?。!！呢吗啊呀]+", "", core)
        return len(core) <= 2

    @staticmethod
    def _previous_topic(prev: ConversationTurn) -> str:
        """上一轮问题去掉产品实体后的主题（如「多少钱」「每天吃几粒」）。"""
        text = (prev.rewritten or prev.question or "").strip()
        if not text or _is_correction(text):
            return ""
        for cand in RetrieveUseCase._extract_product_candidates(text):
            text = text.replace(cand, "")
        text = _PRODUCT_ID_RE.sub("", text)
        text = _BRAND_RE.sub("", text)
        text = _APPROVAL_RE.sub("", text)
        text = text.strip(" 　的，,、？?。!！")
        return text[:40]

    def _join_context(self, q: str, prev: ConversationTurn) -> str:
        """把上一轮实体拼到当前问句前面，返回独立问题。"""
        prefix = self._product_prefix(prev)
        if not prefix:
            prefix = self._scene_prefix(q, prev)
        if not prefix:
            return ""

        core = _LEADING_DEIXIS_RE.sub("", q).strip()
        core = _TRAILING_NE_RE.sub("", core).strip()
        core = core.strip("，,、？? 　")
        if not core:
            return ""
        return f"{prefix} {core}"

    def _product_prefix(self, prev: ConversationTurn) -> str:
        """优先继承上一轮的产品名。"""
        for pid in prev.product_ids:
            return self._product_name(str(pid)) or str(pid)
        return ""

    @staticmethod
    def _scene_prefix(q: str, prev: ConversationTurn) -> str:
        """上一轮没产品时，继承「头晕 / 退货」这类场景关键词（最多 2 个）。

        ★ 只拼实体、不拼整句：把上一轮整句塞进检索会稀释当前问题。
        """
        words: list[str] = []
        for source in (prev.question, prev.rewritten):
            for word in _SCENE_WORDS:
                if word in source and word not in q and word not in words:
                    # 避免「改善视力 + 视力」这类包含关系重复
                    if any(word in w or w in word for w in words):
                        continue
                    words.append(word)
            if len(words) >= 2:
                break
        return " ".join(words[:2])

    # ---------------- LLM 兜底 ----------------
    def _rewrite_by_llm(
        self, q: str, history: list[ConversationTurn]
    ) -> RewriteResult:
        lines = []
        for i, turn in enumerate(history, start=1):
            question = (turn.question or "").replace("\n", " ")[:80]
            answer = (turn.answer or "").replace("\n", " ")[:80]
            lines.append(f"第{i}轮 问：{question} 答：{answer}")

        prompt = (
            "你在做多轮问答的查询改写。请把最后这个追问改写成"
            "不依赖上文、可以独立检索的完整问题。\n"
            "只输出改写后的问题本身，不要解释、不要引号、不要编号、"
            "不要出现「根据上文」等元话术。\n\n"
            "最近对话（问答摘要，每条不超过 80 字）：\n"
            + "\n".join(lines)
            + f"\n\n当前问题：{q}\n改写后的独立问题："
        )

        try:
            raw = self._call_llm(prompt)
        except Exception as exc:                # noqa: BLE001 超时/异常一律回退
            logger.warning("改写 LLM 调用失败，回退原问题: %s: %s",
                           type(exc).__name__, exc)
            return self._none(q)

        cleaned = self._clean_llm_output(raw)
        if not cleaned or not _has_entity(cleaned):
            logger.info("改写 LLM 输出为空或无实体，回退原问题")
            return self._none(q)                # low 置信度回退
        return RewriteResult(
            question=cleaned, source="llm", confidence="medium",
            used_turn=len(history),
        )

    def _call_llm(self, prompt: str) -> str:
        """带硬超时的 `generate`。

        ★ 不新增依赖：用标准库线程池做超时。`shutdown(wait=False)` 是关键，
          否则超时后退出 `with` 仍会等真实请求（30s）结束，超时形同虚设。
        """
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(
                self._llm.generate,
                prompt,
                system_prompt=_REWRITE_SYSTEM,
                temperature=0.0,
                max_tokens=200,
            )
            try:
                result = future.result(timeout=self._timeout)
            except FutureTimeoutError as exc:
                future.cancel()
                raise TimeoutError(f"改写超过 {self._timeout}s") from exc
            return str(getattr(result, "text", "") or "")
        finally:
            pool.shutdown(wait=False)

    @staticmethod
    def _clean_llm_output(raw: str) -> str:
        """清洗 LLM 输出：去引号/前缀，挡掉解释性文字与多句输出。"""
        text = (raw or "").strip()
        if not text:
            return ""
        s = text.splitlines()[0].strip()
        s = re.sub(
            r"^(?:改写(?:后)?(?:的)?(?:独立)?问题|独立问题|问题|问|答|输出)"
            r"\s*[:：]\s*", "", s)
        s = s.strip("「」『』\"'`“”‘’ 　").rstrip("。.!！；;")
        if not (4 <= len(s) <= 120):
            return ""
        if "。" in s or "\n" in s or any(w in s for w in _META_WORDS):
            return ""
        return s

    # ---------------- 回退 ----------------
    @staticmethod
    def _none(question: str) -> RewriteResult:
        return RewriteResult(question=question, source="none", confidence="low")


__all__ = ["QueryRewriter", "RewriteResult"]
