"""查询改写单测（B3）。

★ 图谱用 `FakeGraphQueryPort`，与真实库同构（别名归一真实现），
  不连 Neo4j；LLM 用最小替身，可精确控制成功/异常/超时/低质量输出。
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from application.services.conversation import ConversationSession, ConversationTurn
from application.services.query_rewriter import QueryRewriter
from domain.value_objects import ProductFacts
from tests.fakes import FakeGraphQueryPort

FISH_OIL = "康源牌鱼油软胶囊"


def _graph() -> FakeGraphQueryPort:
    return FakeGraphQueryPort({
        "P001": ProductFacts(
            product_id="P001", name=FISH_OIL, category="保健食品",
            approval_no="国食健注G20250012", aliases=("鱼油",)),
        "P002": ProductFacts(
            product_id="P002", name="康源牌辅酶Q10软胶囊", category="保健食品",
            aliases=("辅酶Q10", "Q10")),
    })


def _turn(
    question: str,
    *,
    product_ids: tuple[str, ...] = (),
    answer: str = "（合规正文）",
    rewritten: str = "",
) -> ConversationTurn:
    return ConversationTurn(
        question=question, rewritten=rewritten or question, answer=answer,
        product_ids=product_ids, at=1000.0,
    )


def _session(*turns: ConversationTurn) -> ConversationSession:
    return ConversationSession(
        session_id="session01", user_key="u1:manager",
        turns=list(turns), created_at=1000.0, last_active=1000.0,
    )


class _StubLLM:
    """最小 LLM 替身：可控回复 / 异常 / 延迟，并记录 prompt。"""

    def __init__(self, *, reply: str = "", fail: bool = False,
                 delay: float = 0.0) -> None:
        self._reply = reply
        self._fail = fail
        self._delay = delay
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def generate(self, prompt, *, system_prompt="", temperature=0.0,
                 max_tokens=1024):
        self.prompts.append(prompt)
        self.kwargs.append({
            "temperature": temperature, "max_tokens": max_tokens})
        if self._delay:
            time.sleep(self._delay)
        if self._fail:
            raise RuntimeError("改写服务不可用")
        return SimpleNamespace(text=self._reply)


# ===================================================================
# 规则优先
# ===================================================================
class TestRules:
    def test_product_switch_inherits_previous_topic(self) -> None:
        rw = QueryRewriter(_graph())
        r = rw.rewrite(
            "辅酶Q10 呢？",
            _session(_turn("鱼油多少钱？", product_ids=("P001",))),
        )
        # 裸省略：用新产品替换主语，并继承上一轮主题
        assert r.question == "康源牌辅酶Q10软胶囊 多少钱？"
        assert r.product_ids == ("P002",)
        assert (r.source, r.confidence, r.used_turn) == ("rule", "high", 1)

    def test_same_product_keeps_product_ids(self) -> None:
        """同一产品再次显式提及：不改写，但必须保留 product_ids 给下一轮。"""
        rw = QueryRewriter(_graph())
        r = rw.rewrite(
            "P001 的禁忌？",
            _session(_turn("鱼油多少钱？", product_ids=("P001",))),
        )
        assert r.question == "P001 的禁忌？"
        assert r.product_ids == ("P001",)
        assert r.source == "rule"

    def test_correction_replaces_product(self) -> None:
        rw = QueryRewriter(_graph())
        r = rw.rewrite(
            "不是这款，是 P002",
            _session(_turn("P001 的禁忌？", product_ids=("P001",))),
        )
        assert r.question == "不是这款，是 P002"
        assert r.product_ids == ("P002",)
        assert r.source == "rule"

    def test_deixis_inherits_previous_product_name(self) -> None:
        rw = QueryRewriter(_graph())
        r = rw.rewrite(
            "那孕妇能吃吗？",
            _session(_turn("鱼油每天吃几粒？", product_ids=("P001",))),
        )
        assert r.question == "康源牌鱼油软胶囊 孕妇能吃吗"
        assert (r.source, r.confidence) == ("rule", "high")
        assert r.product_ids == ("P001",)

    def test_ellipsis_inherits_previous_product_name(self) -> None:
        rw = QueryRewriter(_graph())
        r = rw.rewrite(
            "一天几次",
            _session(_turn("鱼油怎么吃？", product_ids=("P001",))),
        )
        assert r.question == "康源牌鱼油软胶囊 一天几次"
        assert r.source == "rule"

    def test_scene_inheritance_without_product(self) -> None:
        rw = QueryRewriter(_graph())
        r = rw.rewrite(
            "那要退货吗？",
            _session(_turn("客户吃了头晕怎么办？")),
        )
        assert r.question == "头晕 要退货吗"
        assert r.product_ids == ()
        assert r.source == "rule"

    def test_same_product_is_not_rewritten_but_keeps_pids(self) -> None:
        rw = QueryRewriter(_graph())
        r = rw.rewrite(
            "鱼油 孕妇能吃吗",
            _session(_turn("鱼油怎么吃？", product_ids=("P001",))),
        )
        assert r.question == "鱼油 孕妇能吃吗"
        # 句子本身独立，不改写；但 product_ids 要留给下一轮继承
        assert r.source == "rule"
        assert r.product_ids == ("P001",)

    def test_new_standalone_topic_is_not_rewritten(self) -> None:
        rw = QueryRewriter(_graph())
        r = rw.rewrite(
            "公司有什么培训制度",
            _session(_turn("鱼油怎么吃？", product_ids=("P001",))),
        )
        # 已有完整主题、非追问 → 不拼上一轮产品（也不调 LLM）
        assert r.question == "公司有什么培训制度"
        assert r.source == "none"

    def test_rule_hit_does_not_call_llm(self) -> None:
        llm = _StubLLM(reply="不该被调用")
        rw = QueryRewriter(_graph(), llm=llm)
        r = rw.rewrite(
            "那孕妇能吃吗？",
            _session(_turn("鱼油怎么吃？", product_ids=("P001",))),
        )
        assert r.source == "rule"
        assert llm.prompts == []                # 规则命中 <50ms，零 LLM 调用


# ===================================================================
# 无历史 / LLM 兜底 / 回退
# ===================================================================
class TestFallback:
    #: 规则不命中、但有历史的问句（LLM 兜底入口）
    LOOSE_QUESTION = "请说明一下培训制度"

    def _history(self) -> ConversationSession:
        return _session(_turn("鱼油怎么吃？", product_ids=("P001",)))

    def test_no_history_returns_original_without_llm(self) -> None:
        llm = _StubLLM(reply="康源牌鱼油软胶囊 适宜人群")
        rw = QueryRewriter(_graph(), llm=llm)
        r = rw.rewrite("那孕妇能吃吗？", _session())
        assert (r.question, r.source, r.used_turn) == (
            "那孕妇能吃吗？", "none", 0)
        assert llm.prompts == []

    def test_none_session_returns_original(self) -> None:
        r = QueryRewriter(_graph()).rewrite("那孕妇能吃吗？", None)
        assert (r.question, r.source) == ("那孕妇能吃吗？", "none")

    def test_llm_success_is_medium_confidence(self) -> None:
        llm = _StubLLM(reply="康源牌鱼油软胶囊 适宜人群有哪些")
        rw = QueryRewriter(_graph(), llm=llm)
        r = rw.rewrite(self.LOOSE_QUESTION, self._history())
        assert r.question == "康源牌鱼油软胶囊 适宜人群有哪些"
        assert (r.source, r.confidence) == ("llm", "medium")
        assert r.used_turn == 1
        assert llm.kwargs[0] == {"temperature": 0.0, "max_tokens": 200}
        assert "鱼油怎么吃？" in llm.prompts[0]
        assert "当前问题：请说明一下培训制度" in llm.prompts[0]

    def test_llm_uses_at_most_three_turns(self) -> None:
        llm = _StubLLM(reply="康源牌鱼油软胶囊 适宜人群有哪些")
        rw = QueryRewriter(_graph(), llm=llm)
        history = _session(*(_turn(f"第{i}轮问题", product_ids=("P001",))
                             for i in range(1, 6)))
        r = rw.rewrite(self.LOOSE_QUESTION, history)
        assert r.used_turn == 3
        assert "第1轮问题" not in llm.prompts[0]
        assert "第5轮问题" in llm.prompts[0]

    def test_llm_failure_falls_back_to_original(self) -> None:
        rw = QueryRewriter(_graph(), llm=_StubLLM(fail=True))
        r = rw.rewrite(self.LOOSE_QUESTION, self._history())
        assert (r.question, r.source) == (self.LOOSE_QUESTION, "none")

    def test_llm_timeout_falls_back_to_original(self) -> None:
        rw = QueryRewriter(_graph(), llm=_StubLLM(
            reply="康源牌鱼油软胶囊 适宜人群有哪些", delay=0.3),
            timeout_sec=0.02)
        r = rw.rewrite(self.LOOSE_QUESTION, self._history())
        assert (r.question, r.source) == (self.LOOSE_QUESTION, "none")

    def test_llm_empty_output_falls_back(self) -> None:
        rw = QueryRewriter(_graph(), llm=_StubLLM(reply="   "))
        r = rw.rewrite(self.LOOSE_QUESTION, self._history())
        assert (r.question, r.source, r.confidence) == (
            self.LOOSE_QUESTION, "none", "low")

    def test_llm_output_without_entity_is_low_confidence_fallback(self) -> None:
        rw = QueryRewriter(_graph(), llm=_StubLLM(reply="这个怎么办"))
        r = rw.rewrite(self.LOOSE_QUESTION, self._history())
        assert (r.question, r.source, r.confidence) == (
            self.LOOSE_QUESTION, "none", "low")

    def test_llm_meta_talk_is_rejected(self) -> None:
        rw = QueryRewriter(_graph(), llm=_StubLLM(
            reply="根据上文，改写为：康源牌鱼油软胶囊 适宜人群"))
        r = rw.rewrite(self.LOOSE_QUESTION, self._history())
        assert (r.question, r.source) == (self.LOOSE_QUESTION, "none")

    def test_rewriter_exception_never_escapes(self) -> None:
        class _Boom:
            def resolve_alias(self, _alias):
                raise RuntimeError("图谱挂了")

        r = QueryRewriter(_Boom()).rewrite(
            "那孕妇能吃吗？",
            _session(_turn("鱼油怎么吃？", product_ids=("P001",))),
        )
        # 图谱不可用时退而用产品编号当实体，仍然产出可独立检索的问题，
        # 关键是异常不能冒到接口层。
        assert r.question == "P001 孕妇能吃吗"
        assert r.source == "rule"
