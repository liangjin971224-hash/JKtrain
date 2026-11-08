"""LLM 多跳规划：校验层测试。

★ 这一层不需要 LLM，纯函数，所以可以穷举大量非法输入。
  它是整个功能的**安全阀**：只要它在，模型输出再离谱也不会污染执行层。

测试分四块：
  1. extract_json   —— 从含杂质的文本里抠 JSON
  2. validate_plan  —— 六道校验
  3. plan_looks_suspicious —— 触发条件（**必须双向测**）
  4. LLMQueryPlanner —— 端到端编排（用假 LLM）
"""
from __future__ import annotations

import pytest

from application.services.query_plan_guard import (
    extract_json,
    plan_looks_suspicious,
    validate_plan,
)
from application.services.llm_query_planner import (
    CachedQueryPlanner,
    LLMQueryPlanner,
    build_plan_prompt,
)
from domain.services.query_planner import (
    HOP_PRICE_RANK,
    HOP_SIMPLE,
    SubQuery,
    plan_queries,
)
from tests.fakes import FakeCachePort


# ===================================================================
# 一、extract_json：从杂质里抠 JSON
# ===================================================================
class TestExtractJson:
    def test_plain_json(self) -> None:
        assert extract_json('{"steps": []}') == {"steps": []}

    def test_markdown_fence(self) -> None:
        raw = '```json\n{"steps":[{"question":"q"}]}\n```'
        assert extract_json(raw) == {"steps": [{"question": "q"}]}

    def test_fence_without_lang(self) -> None:
        raw = '```\n{"a": 1}\n```'
        assert extract_json(raw) == {"a": 1}

    def test_leading_explanation(self) -> None:
        """模型爱在 JSON 前写一句「好的，分析如下：」。"""
        raw = '好的，分析如下：\n{"steps": []}\n以上是结果。'
        assert extract_json(raw) == {"steps": []}

    def test_nested_braces_inside_string(self) -> None:
        """字符串里的花括号不能被当成结构括号。"""
        raw = '{"steps":[{"question":"用 {大括号} 举例"}]}'
        got = extract_json(raw)
        assert got == {"steps": [{"question": "用 {大括号} 举例"}]}

    def test_truncated_json_returns_none(self) -> None:
        """输出被 max_tokens 截断 → 不猜，直接判失败。"""
        assert extract_json('{"steps":[{"question":"半截') is None

    def test_no_json_at_all(self) -> None:
        assert extract_json("我觉得这个问题很简单，不用拆") is None

    def test_empty_input(self) -> None:
        assert extract_json("") is None
        assert extract_json("   ") is None

    def test_json_array_is_not_object(self) -> None:
        """顶层是数组也不算（我们要的是 {"steps": [...]}）。"""
        assert extract_json('[{"question":"q"}]') is None


# ===================================================================
# 二、validate_plan：六道校验
# ===================================================================
def _p(steps: list[dict]) -> dict:
    return {"steps": steps}


def _s(q: str, *, dep: bool = False, kind: str = "simple") -> dict:
    return {"question": q, "kind": kind, "depends_on_previous": dep}


class TestValidatePlan:
    def test_valid_two_steps(self) -> None:
        payload = _p([_s("改善体脂吃什么"),
                      _s("这些里面哪个最便宜", dep=True, kind="price_rank")])
        out = validate_plan(payload, question="改善体脂吃什么，其中那个最便宜")
        assert out.ok is True
        assert len(out.steps) == 2
        assert out.steps[1].depends_on_previous is True
        assert out.steps[1].kind == HOP_PRICE_RANK

    def test_empty_payload(self) -> None:
        assert validate_plan(None, question="q").ok is False

    def test_missing_steps_key(self) -> None:
        assert validate_plan({"foo": 1}, question="q").ok is False

    def test_steps_not_list(self) -> None:
        assert validate_plan({"steps": "abc"}, question="q").ok is False

    def test_empty_steps(self) -> None:
        assert validate_plan(_p([]), question="q").ok is False

    def test_step_not_dict(self) -> None:
        assert validate_plan(_p(["字符串"]), question="q").ok is False

    def test_empty_question(self) -> None:
        assert validate_plan(_p([_s("")]), question="q").ok is False

    def test_question_too_long(self) -> None:
        """超长大概率是把整段解释塞进来了。"""
        assert validate_plan(_p([_s("字" * 250)]), question="q").ok is False

    # ---- C1 步数上限 ----
    def test_too_many_steps_truncated(self) -> None:
        """超上限截断而不是失败（多出部分是有效信息，只是超预算）。"""
        payload = _p([_s(f"问题{i}") for i in range(6)])
        out = validate_plan(payload, question="六个问题", max_steps=3)
        assert out.ok is True
        assert len(out.steps) == 3
        assert any("截断" in r for r in out.repairs)

    # ---- C3 第一步不许依赖 ----
    def test_first_step_dependency_cleared(self) -> None:
        """模型常给第一步也标依赖（没有「前面」）。修补而非失败。"""
        payload = _p([_s("第一步", dep=True), _s("第二步", dep=True)])
        out = validate_plan(payload, question="q")
        assert out.ok is True
        assert out.steps[0].depends_on_previous is False
        assert out.steps[1].depends_on_previous is True
        assert any("首步无前序" in r for r in out.repairs)

    # ---- C4 kind 白名单 ----
    @pytest.mark.parametrize("raw_kind,expected", [
        ("simple", HOP_SIMPLE),
        ("price_rank", HOP_PRICE_RANK),
        ("price", HOP_PRICE_RANK),      # 别名
        ("compare", HOP_PRICE_RANK),    # 别名
        ("cheapest", HOP_PRICE_RANK),   # 别名
        ("完全瞎写", HOP_SIMPLE),        # 未知 → 降为 simple
    ])
    def test_kind_normalized(self, raw_kind: str, expected: str) -> None:
        out = validate_plan(_p([_s("问题", kind=raw_kind)]), question="q")
        assert out.ok is True
        assert out.steps[0].kind == expected

    def test_unknown_kind_records_repair(self) -> None:
        out = validate_plan(_p([_s("问题", kind="瞎写")]), question="q")
        assert any("非法" in r for r in out.repairs)

    # ---- C6 不许原句复读 ----
    def test_verbatim_repeat_rejected(self) -> None:
        """拆了等于没拆 —— 必须拒绝，否则执行层会跑两遍同样的召回。"""
        q = "改善体脂都可以吃什么"
        payload = _p([_s(q), _s("其中那个最便宜", dep=True)])
        out = validate_plan(payload, question=q)
        assert out.ok is False
        assert "假装拆解" in out.reason

    def test_single_step_equals_question_is_ok(self) -> None:
        """只有一步且等于原句：等价于「不用拆」，允许。"""
        q = "鱼油有什么功能"
        out = validate_plan(_p([_s(q)]), question=q)
        assert out.ok is True

    def test_verbatim_ignores_punctuation(self) -> None:
        """标点差异不算不同。"""
        q = "改善体脂都可以吃什么"
        payload = _p([_s("改善体脂都可以吃什么？"), _s("其中那个最便宜", dep=True)])
        out = validate_plan(payload, question=q)
        assert out.ok is False

    # ---- 重复子问题 ----
    def test_duplicate_steps_rejected(self) -> None:
        payload = _p([_s("鱼油有什么功能"), _s("鱼油有什么功能")])
        out = validate_plan(payload, question="鱼油有什么功能，鱼油有什么功能")
        assert out.ok is False
        assert "重复" in out.reason

    def test_order_is_sequential(self) -> None:
        payload = _p([_s("甲"), _s("乙", dep=True)])
        out = validate_plan(payload, question="甲，其中乙")
        assert [s.order for s in out.steps] == [0, 1]


# ===================================================================
# 三、触发条件：★ 必须双向测
# ===================================================================
class TestSuspicious:
    """「该触发时触发」和「不该触发时不触发」同等重要。

    ★ 只测前者会做出一个「永远调 LLM」的实现 —— 成本失控；
      只测后者会做出一个「永远不调」的实现 —— 等于没做。
    """

    # ---------- 不该触发 ----------
    def test_single_step_not_suspicious(self) -> None:
        """80% 的问题是单步，绝不该为它调 LLM。"""
        q = "鱼油有什么功能"
        trigger, why = plan_looks_suspicious(q, plan_queries(q))
        assert trigger is False, why

    def test_normal_two_step_with_dependency_not_suspicious(self) -> None:
        """规则能判出依赖 → 不需要模型复核。"""
        q = "改善体脂都可以吃什么保健品，其中那个最便宜"
        steps = plan_queries(q)
        assert len(steps) == 2
        assert steps[1].depends_on_previous is True
        trigger, why = plan_looks_suspicious(q, steps)
        assert trigger is False, why

    def test_parallel_two_steps_not_suspicious(self) -> None:
        """并列双问、都没依赖、无顺序暗示 → 规则判对了，不该触发。"""
        q = "补钙的有什么推荐，有什么禁忌"
        steps = plan_queries(q)
        trigger, why = plan_looks_suspicious(q, steps)
        assert trigger is False, why

    # ---------- 该触发 ----------
    def test_multi_step_without_dependency_triggers(self) -> None:
        q = "鱼油和阿司匹林能一起吃吗，不行的话有什么替代"
        steps = plan_queries(q)
        # 「里面哪款」不在回指词表 → 规则可能漏判
        trigger, _ = plan_looks_suspicious(q, steps)
        assert trigger is True

    def test_all_after_first_dependent_triggers(self) -> None:
        """第二步起全是依赖 → 可能把并列误判为依赖。"""
        steps = (
            SubQuery(question="第一步", order=0),
            SubQuery(question="第二步", order=1, depends_on_previous=True),
            SubQuery(question="第三步", order=2, depends_on_previous=True),
        )
        trigger, why = plan_looks_suspicious("甲，其中乙，其中丙", steps)
        assert trigger is True
        assert "全部标为依赖" in why

    def test_dropped_fragments_triggers(self) -> None:
        """被丢弃的片段可能是漏拆的有效子问题。

        实测：「改善体脂吃什么，给我讲讲那款」
        「给我讲讲那款」含回指词「那款」却被丢弃 → 明显漏拆。
        """
        q = "改善体脂吃什么，这几样给我挑个划算的"
        steps = plan_queries(q)
        trigger, why = plan_looks_suspicious(
            q, steps, dropped_fragments=("这几样给我挑个划算的",))
        assert trigger is True
        assert "丢弃" in why

    def test_multi_question_marks_triggers(self) -> None:
        """原文两个问号但只拆一步 → 拆漏了。"""
        q = "鱼油有什么功能？蛋白粉有什么功能？"
        steps = plan_queries(q)
        if len(steps) <= 1:
            trigger, _ = plan_looks_suspicious(q, steps)
            assert trigger is True

    def test_empty_question_not_suspicious(self) -> None:
        trigger, _ = plan_looks_suspicious("", ())
        assert trigger is False


# ===================================================================
# 四、LLMQueryPlanner：编排
# ===================================================================
class _FakeLLM:
    """返回固定文本的假 LLM，并记录被调用次数。"""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def generate(self, prompt, *, system_prompt="", temperature=0.0,
                 max_tokens=1024, **kw):
        self.calls += 1

        class R:
            text = self.reply

        return R()


class _BoomLLM:
    def generate(self, *a, **kw):
        raise RuntimeError("llm down")


_GOOD_JSON = '''{"steps":[
 {"question":"改善体脂吃什么","kind":"simple","depends_on_previous":false},
 {"question":"这些里面哪个最便宜","kind":"price_rank","depends_on_previous":true}
]}'''


class TestLLMPlanner:
    def _rule(self, q: str = "鱼油和阿司匹林能一起吃吗，不行的话有什么替代"):
        return plan_queries(q)

    def test_not_triggered_means_no_call(self) -> None:
        """单步问题不该调 LLM。"""
        llm = _FakeLLM(_GOOD_JSON)
        p = LLMQueryPlanner(llm)
        q = "鱼油有什么功能"
        out = p.plan(q, plan_queries(q))

        assert llm.calls == 0
        assert p.last_decision["source"] == "rule"
        assert len(out) == 1

    def test_triggered_and_accepted(self) -> None:
        llm = _FakeLLM(_GOOD_JSON)
        p = LLMQueryPlanner(llm)
        q = "鱼油和阿司匹林能一起吃吗，不行的话有什么替代"
        out = p.plan(q, plan_queries(q))

        assert llm.calls == 1
        assert p.last_decision["source"] == "llm"
        assert len(out) == 2
        assert out[1].depends_on_previous is True

    def test_invalid_json_falls_back(self) -> None:
        llm = _FakeLLM("我不会拆这个问题")
        p = LLMQueryPlanner(llm)
        q = "鱼油和阿司匹林能一起吃吗，不行的话有什么替代"
        rule = plan_queries(q)
        out = p.plan(q, rule)

        assert llm.calls == 1
        assert p.last_decision["source"] == "rule"
        assert out == rule

    def test_validation_failure_falls_back(self) -> None:
        """输出是合法 JSON 但内容非法（第一步标依赖、原句复读）。"""
        bad = ('{"steps":[{"question":"鱼油和阿司匹林能一起吃吗，不行的话有什么替代",'
               '"kind":"simple","depends_on_previous":true},'
               '{"question":"里面哪款划算","kind":"simple",'
               '"depends_on_previous":true}]}')
        llm = _FakeLLM(bad)
        p = LLMQueryPlanner(llm)
        q = "鱼油和阿司匹林能一起吃吗，不行的话有什么替代"
        rule = plan_queries(q)
        out = p.plan(q, rule)

        assert p.last_decision["source"] == "rule"
        assert out == rule

    def test_llm_exception_falls_back(self) -> None:
        p = LLMQueryPlanner(_BoomLLM())
        q = "鱼油和阿司匹林能一起吃吗，不行的话有什么替代"
        rule = plan_queries(q)
        out = p.plan(q, rule)

        assert out == rule
        assert p.last_decision["source"] == "rule"

    def test_budget_exhausted(self) -> None:
        """预算用完 → 不再调，直接退回规则。"""
        llm = _FakeLLM(_GOOD_JSON)
        p = LLMQueryPlanner(llm, max_calls=1)
        q = "鱼油和阿司匹林能一起吃吗，不行的话有什么替代"
        rule = plan_queries(q)

        p.plan(q, rule)                 # 第 1 次用掉预算
        assert llm.calls == 1
        p.plan(q, rule)                 # 第 2 次应被拦
        assert llm.calls == 1

    def test_reset_budget(self) -> None:
        llm = _FakeLLM(_GOOD_JSON)
        p = LLMQueryPlanner(llm, max_calls=1)
        q = "鱼油和阿司匹林能一起吃吗，不行的话有什么替代"
        rule = plan_queries(q)

        p.plan(q, rule)
        p.reset_budget()
        p.plan(q, rule)
        assert llm.calls == 2

    def test_never_returns_empty(self) -> None:
        """任何失败都不能返回空 —— 空会让执行层没有步骤可跑。"""
        for reply in ["", "不是 JSON", '{"steps":[]}', '{"steps":"x"}']:
            p = LLMQueryPlanner(_FakeLLM(reply))
            q = "鱼油和阿司匹林能一起吃吗，不行的话有什么替代"
            out = p.plan(q, plan_queries(q))
            assert out, f"空结果（reply={reply!r}）"

    def test_prompt_contains_examples(self) -> None:
        """提示词必须带正向与反向判例 —— 否则模型会过度标依赖。"""
        prompt = build_plan_prompt("测试问题")
        assert "并列" in prompt
        assert "不算依赖" in prompt
        assert "depends_on_previous" in prompt


# ===================================================================
# 五、缓存包装
# ===================================================================
class TestCachedPlanner:
    def test_cache_hit_avoids_second_call(self) -> None:
        llm = _FakeLLM(_GOOD_JSON)
        p = LLMQueryPlanner(llm)
        cached = CachedQueryPlanner(p, FakeCachePort(), ttl_sec=60)

        q = "鱼油和阿司匹林能一起吃吗，不行的话有什么替代"
        rule = plan_queries(q)
        cached.plan(q, rule)
        assert llm.calls == 1

        out2 = cached.plan(q, rule)
        assert llm.calls == 1, "第二次应命中缓存，不该再调 LLM"
        assert cached.last_decision["source"] == "cache"
        assert len(out2) == 2

    def test_rule_only_result_not_cached(self) -> None:
        """规则直出的结果不缓存（它本来就不贵，缓存反而会读到旧拆解）。"""
        llm = _FakeLLM(_GOOD_JSON)
        p = LLMQueryPlanner(llm)
        cache = FakeCachePort()
        cached = CachedQueryPlanner(p, cache, ttl_sec=60)

        q = "鱼油有什么功能"        # 单步，不触发
        cached.plan(q, plan_queries(q))
        assert cached.last_decision["source"] == "rule"

        # 再跑一次仍然不触发、也不该从缓存读到东西
        cached.plan(q, plan_queries(q))
        assert llm.calls == 0

    def test_broken_cache_does_not_break_planning(self) -> None:
        """缓存抛异常 → 规划照常。"""

        class BoomCache:
            def get(self, k):
                raise RuntimeError("cache down")

            def set(self, k, v, *, ttl_sec=None):
                raise RuntimeError("cache down")

        llm = _FakeLLM(_GOOD_JSON)
        cached = CachedQueryPlanner(LLMQueryPlanner(llm), BoomCache(), ttl_sec=60)
        q = "鱼油和阿司匹林能一起吃吗，不行的话有什么替代"
        out = cached.plan(q, plan_queries(q))
        assert out
        assert llm.calls == 1
