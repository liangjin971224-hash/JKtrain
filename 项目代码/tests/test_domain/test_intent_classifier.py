"""意图分类测试（红灯先行，2026-10-23）。

★ 为什么先写这个文件：
  P3 就把 13 套意图权重表建好了，但接口层从不传 `intent`，
  全部走「开放问答」(0.1, 0.8, 0.1) —— 问「在吃阿司匹林能吃鱼油吗」时
  图谱权重本该 0.6（最准的一路），实际是 0.1。**权重结构是反的**。

★ 本文件的判据不是「能返回一个字符串」，而是：
  1. 安全类问题（禁忌/停药/特殊人群/应急）**必须**落到安全类意图，一条都不许漏
  2. 常见业务问法要有 ≥85% 落到预期意图
  3. 永远返回权重表里存在的标签（否则 weight 查询会静默走兜底）
  4. **分类不依赖 LLM**（保护「召回为空不调 LLM」这条红线）
"""
from __future__ import annotations

import pytest

from domain.services.intent_classifier import (
    OPEN_INTENT,
    SAFETY_INTENTS,
    IntentClassifier,
)
from domain.services.rrf_fusion import INTENT_WEIGHTS
from tests.fakes import FailOnCallLLM, FakeLanguageModelPort


# ===================================================================
# 1. 安全类必须一条不漏（最重要）
# ===================================================================
class TestSafetyNeverMissed:
    """安全类意图判错的代价：图谱权重不上调，最准的一路被压到 0.1。

    所以这里**逐个单独断言**，不放进「整体准确率」里摊薄。
    """

    @pytest.mark.parametrize("question", [
        "在吃阿司匹林能吃鱼油吗",
        "客户在吃华法林问能不能吃鱼油",
        "吃了这个能停降压药吗",
        "客户问能不能停药怎么答",
        "孕妇能吃辅酶Q10吗",
        "哺乳期可以吃吗",
        "肾功能不好能吃蛋白粉吗",
        "小孩能吃褪黑素吗",
        "一次吃多了怎么办",
        "这个一天最多吃几粒",
        "客户在店里突然胸闷怎么办",
        "顾客晕倒了怎么处理",
    ])
    def test_safety_question_lands_in_safety_intent(self, question: str) -> None:
        result = IntentClassifier().classify(question)
        assert result.label in SAFETY_INTENTS or result.label == "应急处理", (
            f"「{question}」被判成「{result.label}」，安全类意图不该漏。"
            f"命中词: {result.matched}"
        )

    def test_stop_medication_is_not_open_intent(self) -> None:
        """★ 本条对应未完成事项清单里点名的那个例子。"""
        result = IntentClassifier().classify("吃了能停降压药吗")
        assert result.label == "要停药"
        assert INTENT_WEIGHTS[result.label][0] == pytest.approx(0.5)


# ===================================================================
# 2. 业务问法准确率
# ===================================================================
#: (问题, 可接受的意图集合)
#: 集合允许多于一个，是因为同一句话确实可以是两类（「肾不好能吃吗」
#: 既属禁忌也属特殊人群，两者权重相同）。**集合刻意保持很小**，
#: 不是用来掩盖判错的。
CASES: tuple[tuple[str, frozenset[str]], ...] = (
    # —— 禁忌 / 特殊人群 ——
    ("鱼油哪些人不能吃", frozenset({"禁忌咨询"})),
    ("这个和降压药一起吃有冲突吗", frozenset({"禁忌咨询"})),
    ("吃海鲜过敏的人能吃这个吗", frozenset({"禁忌咨询", "特殊人群"})),
    ("孕妇能吃吗", frozenset({"禁忌咨询", "特殊人群"})),
    ("糖尿病人可以吃蛋白粉吗", frozenset({"禁忌咨询", "特殊人群"})),
    # —— 越界意图 ——
    ("我是不是得了糖尿病", frozenset({"要诊断"})),
    ("你帮我看看这个指标算不算病", frozenset({"要诊断"})),
    ("能不能给我开点药", frozenset({"要处方"})),
    ("吃多了会怎么样", frozenset({"超量服用"})),
    # —— 应急 ——
    ("客户突然胸闷怎么处理", frozenset({"应急处理"})),
    ("应急处理的基本原则是什么", frozenset({"应急处理"})),
    # —— 文号 ——
    ("批准文号的格式有哪几种", frozenset({"文号查询"})),
    ("国食健注是什么意思", frozenset({"文号查询"})),
    # —— 设备 ——
    ("体脂仪测量前要做哪些准备", frozenset({"设备使用"})),
    ("骨密度的T值是什么意思", frozenset({"设备使用"})),
    ("骨密度仪测出来低值能不能说客户骨质疏松", frozenset({"设备使用"})),
    # —— 业务流程 ——
    ("客户进店第一步该做什么", frozenset({"业务流程"})),
    ("什么情况下可以退货", frozenset({"业务流程"})),
    ("客户档案里的自述要怎么标", frozenset({"业务流程"})),
    ("接待过程中有哪些禁止行为", frozenset({"业务流程"})),
    ("客户说太贵了怎么应对", frozenset({"业务流程"})),
    ("无效退款能不能承诺", frozenset({"业务流程"})),
    ("介绍产品时必须说哪几件事", frozenset({"业务流程"})),
    # —— 产品功效 ——
    ("鱼油有什么功能", frozenset({"产品功效"})),
    ("辅酶Q10对心脏好不好", frozenset({"产品功效"})),
    ("复合蛋白固体饮料能增强免疫力吗", frozenset({"产品功效"})),
    ("普通食品能不能宣传功效", frozenset({"产品功效"})),
    ("保健食品能声称哪些功能", frozenset({"产品功效"})),
    ("降血糖这个说法对不对", frozenset({"产品功效"})),
    # —— 产品名 ——
    ("你们有鱼油吗", frozenset({"产品名查询", "产品功效"})),
    ("褪黑素多少钱一盒", frozenset({"产品名查询"})),
    # —— 开放 ——
    ("怎么区分保健食品和普通食品", frozenset({OPEN_INTENT, "产品功效"})),
    ("老员工有什么处理经验", frozenset({OPEN_INTENT})),
)


class TestAccuracy:
    def test_overall_accuracy_at_least_85_percent(self) -> None:
        """P3 记录里给的目标是 ≥85%，这里如实统计。

        失败时把判错的问题全列出来 —— 只说「准确率不够」没法调。
        """
        wrong: list[str] = []
        for question, acceptable in CASES:
            got = IntentClassifier().classify(question).label
            if got not in acceptable:
                wrong.append(f"{question} → {got}（期望 {sorted(acceptable)}）")

        accuracy = 1 - len(wrong) / len(CASES)
        assert accuracy >= 0.85, (
            f"意图分类准确率 {accuracy:.1%}，低于目标 85%\n"
            + "\n".join(f"  {w}" for w in wrong)
        )

    def test_all_cases_are_in_weight_table(self) -> None:
        """分类结果必须能在权重表里查到，否则静默走兜底权重。"""
        for question, _ in CASES:
            got = IntentClassifier().classify(question).label
            assert got in INTENT_WEIGHTS, (
                f"「{question}」判成「{got}」，但 INTENT_WEIGHTS 里没有这个标签"
            )


# ===================================================================
# 3. 契约
# ===================================================================
class TestContract:
    def test_empty_question_falls_back(self) -> None:
        for q in ("", "   ", "\n"):
            assert IntentClassifier().classify(q).label == OPEN_INTENT

    def test_gibberish_falls_back(self) -> None:
        """乱码不能被硬塞进安全类意图（否则会莫名其妙上调图谱权重）。"""
        result = IntentClassifier().classify("zzzzqqqqxxxx")
        assert result.label == OPEN_INTENT
        assert result.confidence <= 0.5

    def test_matched_keywords_exposed(self) -> None:
        """要能说出「凭什么判成这个」——否则调不准时无从下手。"""
        result = IntentClassifier().classify("在吃阿司匹林能吃鱼油吗")
        assert result.matched, "命中词不能为空"
        assert any("阿司匹林" in m or "能吃" in m for m in result.matched), (
            f"命中词应包含实际匹配到的说法，实际: {result.matched}"
        )

    def test_confidence_increases_with_more_evidence(self) -> None:
        one = IntentClassifier().classify("鱼油有什么功能")
        many = IntentClassifier().classify("鱼油有什么功效和作用，效果好吗")
        assert many.confidence >= one.confidence

    def test_result_is_immutable(self) -> None:
        r = IntentClassifier().classify("鱼油有什么功能")
        with pytest.raises((AttributeError, TypeError)):
            r.label = "别的"        # type: ignore[misc]

    @pytest.mark.parametrize("question", [
        "鱼油有什么功能", "能不能停药", "孕妇能吃吗", "", "12345", "体脂仪怎么用",
    ])
    def test_never_raises(self, question: str) -> None:
        """分类器在召回链路的**最前面**，抛异常等于整个问答挂掉。"""
        assert IntentClassifier().classify(question).label


# ===================================================================
# 4. 不依赖 LLM（红线）
# ===================================================================
class TestNoLLMByDefault:
    def test_rules_only_does_not_touch_llm(self) -> None:
        """★ 默认配置下分类器**不许**碰 LLM。

        理由：`AnswerQuestionUseCase` 的红线是「召回为空不调 LLM」。
        若分类器先调一次 LLM，这条红线在链路上就被绕过了。
        这里用「一调就炸」的替身把它变成测试失败。
        """
        clf = IntentClassifier(llm=FailOnCallLLM())
        assert clf.classify("鱼油有什么功能").label == "产品功效"

    def test_llm_fallback_opt_in(self) -> None:
        """显式打开时才用 LLM 兜底。"""
        clf = IntentClassifier(
            llm=FakeLanguageModelPort(label="业务流程"),
            use_llm_fallback=True,
        )
        result = clf.classify("这句话的意图是什么")
        assert result.label == "业务流程"
        assert result.source == "llm"

    def test_llm_fallback_only_for_unmatched(self) -> None:
        """规则已经判出来时不再调 LLM（省 0.3s 且避免规则被覆盖）。"""
        llm = FakeLanguageModelPort(label="业务流程")
        clf = IntentClassifier(llm=llm, use_llm_fallback=True)
        result = clf.classify("鱼油有什么功能")
        assert result.label == "产品功效"
        assert "classify" not in llm.calls

    def test_llm_failure_degrades_to_open_intent(self) -> None:
        """LLM 兜底自己挂了也不能让问答挂掉。"""
        clf = IntentClassifier(llm=FailOnCallLLM(), use_llm_fallback=True)
        result = clf.classify("这句话的意图是什么")
        assert result.label == OPEN_INTENT

    def test_llm_unknown_keeps_open_intent(self) -> None:
        clf = IntentClassifier(
            llm=FakeLanguageModelPort(label="unknown"), use_llm_fallback=True)
        assert clf.classify("嗯嗯").label == OPEN_INTENT
