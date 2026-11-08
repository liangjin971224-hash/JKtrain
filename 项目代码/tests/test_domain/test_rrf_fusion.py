"""RRF 融合测试（P3 红灯先行）。

RRF 是纯领域服务，**不依赖任何外部组件**，所以这个文件必须能在
无网、无库、无模型的机器上跑绿（秒级）。

覆盖三件事：
1. 公式正确性（score = Σ w_i / (k + rank_i)）
2. 参数校验（权重与路数不符必须报错，不能静默错配）
3. 边界（空输入、单路、并列、重复文档）
"""
from __future__ import annotations

import pytest

from domain.services.rrf_fusion import (
    DEFAULT_K,
    INTENT_WEIGHTS,
    reciprocal_rank_fusion,
    weights_for_intent,
)


# ===================================================================
# 公式正确性
# ===================================================================
class TestFormula:
    def test_single_list_keeps_order(self) -> None:
        """单路输入时，输出顺序应与输入一致。"""
        fused = reciprocal_rank_fusion([["a", "b", "c"]])
        assert [doc for doc, _ in fused] == ["a", "b", "c"]

    def test_single_list_scores(self) -> None:
        """单路时 score = 1 / (k + rank + 1)。"""
        k = 60
        fused = reciprocal_rank_fusion([["a", "b"]], k=k)
        scores = dict(fused)
        assert scores["a"] == pytest.approx(1 / (k + 1))
        assert scores["b"] == pytest.approx(1 / (k + 2))

    def test_two_lists_accumulate(self) -> None:
        """同一文档出现在多路时分数累加。"""
        k = 60
        fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]], k=k)
        scores = dict(fused)
        # a 在两路都是第 1
        assert scores["a"] == pytest.approx(2 / (k + 1))
        assert scores["b"] == pytest.approx(1 / (k + 2))
        assert scores["c"] == pytest.approx(1 / (k + 2))

    def test_weights_applied(self) -> None:
        """加权：w_i 作为分子。"""
        k = 60
        fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[0.6, 0.4], k=k)
        scores = dict(fused)
        assert scores["a"] == pytest.approx(0.6 / (k + 1))
        assert scores["b"] == pytest.approx(0.4 / (k + 1))

    def test_higher_weight_wins_on_tie(self) -> None:
        """两路都排第 1 时，权重大的一方应排前。"""
        fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[0.7, 0.3])
        assert fused[0][0] == "a"

    def test_output_sorted_desc(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]])
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)


# ===================================================================
# 参数校验：错配必须报错，不能静默
# ===================================================================
class TestValidation:
    def test_weight_count_mismatch_raises(self) -> None:
        """权重数量与路数不符必须报错。

        静默按最短对齐会导致「某一路权重被丢掉」，
        而调用方完全看不出来——这是最难查的一类 bug。
        """
        with pytest.raises(ValueError, match="权重"):
            reciprocal_rank_fusion([["a"], ["b"], ["c"]], weights=[0.5, 0.5])

    def test_negative_weight_raises(self) -> None:
        """负权重会让排名反转，必须拒绝。"""
        with pytest.raises(ValueError, match="权重"):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, -0.5])

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="k"):
            reciprocal_rank_fusion([["a"]], k=0)

    def test_all_zero_weights_raises(self) -> None:
        """全零权重等于没有信号，应报错而不是返回空。"""
        with pytest.raises(ValueError, match="权重"):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[0.0, 0.0])


# ===================================================================
# 边界
# ===================================================================
class TestBoundary:
    def test_empty_input(self) -> None:
        assert reciprocal_rank_fusion([]) == []

    def test_all_empty_lists(self) -> None:
        """三路全空 → 空结果（P4 据此拒答）。"""
        assert reciprocal_rank_fusion([[], [], []]) == []

    def test_some_empty_lists(self) -> None:
        """部分路为空时，其余路仍正常参与。"""
        fused = reciprocal_rank_fusion([["a", "b"], [], ["c"]])
        docs = [d for d, _ in fused]
        assert set(docs) == {"a", "b", "c"}

    def test_duplicate_in_same_list_counted_once(self) -> None:
        """同一路内重复的文档只按首次出现计分。

        否则一路里重复 10 次就能把分数刷上去。

        注意 rank 的语义：**按原始列表位置计**，不做紧凑化。
        `["a","a","a","b"]` 里 b 在第 4 位（rank=4），所以是 1/(k+4)。
        紧凑化（把 b 当第 2 位）会让分数虚高，也不符合「排名」的原意——
        真实检索里不会有重复，出现重复本身就是数据问题，不该被奖励。
        """
        fused = reciprocal_rank_fusion([["a", "a", "a", "b"]], k=60)
        scores = dict(fused)
        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["b"] == pytest.approx(1 / 64)

    def test_default_k_is_60(self) -> None:
        """业界惯例 k=60。"""
        assert DEFAULT_K == 60

    def test_large_list(self) -> None:
        """100 条不丢结果。"""
        docs = [f"d{i}" for i in range(100)]
        fused = reciprocal_rank_fusion([docs])
        assert len(fused) == 100


# ===================================================================
# 动态权重表
# ===================================================================
class TestIntentWeights:
    """权重随意图调整（置信度：稳定惯例，来自 ports-and-adapters §4.3）。"""

    def test_all_weight_sets_sum_to_one(self) -> None:
        """每组权重应归一化，否则不同意图的分数不可比。"""
        for intent, weights in INTENT_WEIGHTS.items():
            total = sum(weights)
            assert abs(total - 1.0) < 1e-6, (
                f"意图「{intent}」权重和为 {total}，应为 1.0"
            )

    def test_each_set_has_three_paths(self) -> None:
        """三路：图谱 / 向量 / 关键词。"""
        for intent, weights in INTENT_WEIGHTS.items():
            assert len(weights) == 3, f"意图「{intent}」应有 3 个权重"

    def test_safety_intent_favors_graph(self) -> None:
        """安全相关查询必须提高图谱权重——宁可不精确召回也不能漏禁忌。"""
        weights = weights_for_intent("禁忌咨询")
        graph_w, _, _ = weights
        assert graph_w >= 0.5, f"安全类图谱权重应 ≥0.5，实际 {graph_w}"

    def test_process_intent_favors_vector(self) -> None:
        """业务流程咨询全靠文本，向量权重应最高。"""
        _, vector_w, _ = weights_for_intent("业务流程")
        assert vector_w >= 0.6

    def test_product_no_query_favors_keyword(self) -> None:
        """产品名/文号查询靠精确匹配。"""
        _, _, keyword_w = weights_for_intent("产品名查询")
        assert keyword_w >= 0.4

    def test_unknown_intent_falls_back(self) -> None:
        """未知意图必须有兜底，不能抛异常（意图分类可能出错）。"""
        weights = weights_for_intent("完全不认识的意图")
        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_fallback_is_open_intent(self) -> None:
        """★ 兜底必须与主链路一致（2026-10-23 修）。

        改前：`weights_for_intent("")` 返回「产品功效」(0.5,0.3,0.2)，
        而 `answer_question` 会先把空串转成「开放问答」(0.1,0.8,0.1)。
        两处兜底不同 → 看代码的人以为默认是图谱主导，实际是向量主导。

        ★ 为什么兜底该选「开放问答」：
          兜底意味着「不知道用户问什么」，此时应让语义（向量）主导，
          不该假设「问的是产品」。
        """
        from domain.services.rrf_fusion import DEFAULT_WEIGHTS

        assert DEFAULT_WEIGHTS == INTENT_WEIGHTS["开放问答"]
        assert weights_for_intent("") == INTENT_WEIGHTS["开放问答"]
        assert weights_for_intent("完全不认识的意图") == DEFAULT_WEIGHTS
