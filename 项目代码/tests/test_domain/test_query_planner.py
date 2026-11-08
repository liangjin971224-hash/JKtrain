"""查询规划（拆子查询）+ 多跳执行（红灯先行，2026-10-23）。

────────────────────────────────────────────────────────────────────
两件事，两个文件
────────────────────────────────────────────────────────────────────
本文件测 `domain/services/query_planner.py`（纯领域，零 IO）：
    把一句复合问题拆成有序的子查询，并标出**后续步骤依赖前面结果**。

`tests/test_application/test_multi_hop.py` 测 `application/use_cases/query_plan.py`：
    真正执行这些子查询、把前一步的结果送进后一步。

────────────────────────────────────────────────────────────────────
为什么「依赖」必须显式建模，而不是「顺手带上上文」
────────────────────────────────────────────────────────────────────
用户给的例子：
    「改善体脂都可以吃什么保健品，其中那种最便宜」

第 2 步「那种最便宜」的**候选集完全来自第 1 步**。
这不是「上下文顺延」，是**集合传参**：
    step1 → {P020}  →  step2 只在 {P020} 里比价

两种做法的区别是可验证的：
    · 集合传参：第 2 步的候选数 ≤ 第 1 步的结果数
    · 上下文顺延：第 2 步会重新全库检索，可能比出一个第 1 步里没有的产品

库里真有这个坑：全店最便宜的保健食品是「维生素C咀嚼片 98 元」，
但它和体脂毫无关系。若第 2 步重新检索，答案就是 98 元那个 —— 错得很自然。
"""
from __future__ import annotations

import pytest

from domain.services.query_planner import (
    HOP_PRICE_RANK,
    HOP_SIMPLE,
    SubQuery,
    plan_queries,
)


# ===================================================================
# 1. 不该拆的不要拆（拆错的代价比不拆大）
# ===================================================================
class TestNoOverSplit:
    @pytest.mark.parametrize("question", [
        "鱼油有什么功能",
        "在吃阿司匹林能吃鱼油吗",
        "客户进店第一步该做什么",
        "体脂仪测量前要做哪些准备",
        "批准文号的格式有哪几种",
        "怎么区分保健食品和普通食品",
        "客户说太贵了怎么应对",
    ])
    def test_single_question_is_one_step(self, question: str) -> None:
        """单个问题必须只有一步。

        ★ 为什么这条排第一：
          多拆一步就多一次召回、多一段 prompt，还要多判一次「怎么合并」。
          把「在吃阿司匹林能吃鱼油吗」拆成「阿司匹林」+「鱼油」，
          会让禁忌问题变成两个产品查询 —— 从安全类掉到普通查询。
        """
        plan = plan_queries(question)
        assert len(plan) == 1
        assert plan[0].kind == HOP_SIMPLE

    def test_empty_and_gibberish(self) -> None:
        for q in ("", "   ", "zzzzqqqqxxxx"):
            plan = plan_queries(q)
            assert len(plan) == 1
            assert plan[0].question.strip() == q.strip()


# ===================================================================
# 2. 并列多意图（清单里点名的那条）
# ===================================================================
class TestParallelSplit:
    def test_calcium_recommend_and_contraindication(self) -> None:
        """★ 未完成事项清单里点名的例子。

        「补钙的有什么推荐，有什么禁忌」= 两个**并列**问题，
        各自独立召回，互不依赖。
        """
        plan = plan_queries("补钙的有什么推荐，有什么禁忌")

        assert len(plan) == 2
        assert "推荐" in plan[0].question
        assert "禁忌" in plan[1].question
        # 并列：后面的步骤不依赖前面的结果
        assert all(not s.depends_on_previous for s in plan)
        # 两句都要保留主题「补钙」，否则第 2 步会变成「有什么禁忌」（全库）
        assert all("补钙" in s.question for s in plan)

    def test_parallel_parts_share_topic(self) -> None:
        """★ 拆开后每句都要能独立看懂。

        去掉「补钙」只留「有什么禁忌」，第 2 步的召回范围会从
        「补钙类产品」扩到「所有禁忌」，答案就散了。
        """
        plan = plan_queries("鱼油有什么功效，有什么禁忌")
        assert len(plan) == 2
        for s in plan:
            assert "鱼油" in s.question

    def test_two_topics_split(self) -> None:
        plan = plan_queries("鱼油有什么功能？蛋白粉有什么功能？")
        assert len(plan) == 2

    def test_three_parts(self) -> None:
        plan = plan_queries("补钙的有什么推荐，有什么禁忌，孕妇能吃吗")
        assert len(plan) == 3
        assert all("补钙" in s.question for s in plan)


# ===================================================================
# 3. 多跳：后一步以前一步结果为前提
# ===================================================================
class TestSequentialHop:
    def test_body_fat_then_cheapest(self) -> None:
        """★ 用户给的例子。

        「改善体脂都可以吃什么保健品，其中那种最便宜」
            第 1 步：改善体脂可以吃什么保健品
            第 2 步：在这些里面哪个最便宜 ← **依赖第 1 步的集合**
        """
        plan = plan_queries("改善体脂都可以吃什么保健品，其中那种最便宜")

        assert len(plan) == 2
        first, second = plan
        assert first.kind == HOP_SIMPLE
        assert not first.depends_on_previous

        assert second.kind == HOP_PRICE_RANK
        assert second.depends_on_previous is True, (
            "「其中」「里面」「这几个」这类指代必须标成依赖前一步，"
            "否则第二步会重新全库检索，比出一个与主题无关的便宜货"
        )

    @pytest.mark.parametrize("question", [
        "改善体脂都可以吃什么保健品，其中那种最便宜",
        "补钙的产品有哪些，其中最便宜的是哪个",
        "助眠的保健品推荐几个，里面哪个便宜点",
        "增强免疫力的有哪些，这几个里最便宜的呢",
        "控脂的吃什么，上述产品中价格最低的是哪个",
    ])
    def test_anaphora_marks_dependency(self, question: str) -> None:
        """指代词必须出现在**后一步**，且只出现在后一步。"""
        plan = plan_queries(question)
        assert len(plan) == 2, f"应拆成两步: {[s.question for s in plan]}"
        assert plan[0].depends_on_previous is False
        assert plan[1].depends_on_previous is True

    def test_dependency_not_triggered_by_plain_second_question(self) -> None:
        """★ 反过来也要测：没有指代就不能标依赖。

        只测「该标的时候标了」会做出一个「永远标依赖」的实现 ——
        那样「补钙有什么推荐，有什么禁忌」的第 2 步会被塞进第 1 步的产品集合，
        变成「在这些推荐里面找禁忌」，语义完全错了。
        """
        plan = plan_queries("补钙的有什么推荐，有什么禁忌")
        assert all(s.depends_on_previous is False for s in plan)


# ===================================================================
# 4. 结构约束
# ===================================================================
class TestContract:
    def test_each_step_keeps_original(self) -> None:
        """子查询要保留原始片段，便于排障时对照。"""
        plan = plan_queries("补钙的有什么推荐，有什么禁忌")
        for s in plan:
            assert s.original_fragment
            assert isinstance(s, SubQuery)

    def test_steps_are_immutable(self) -> None:
        s = plan_queries("鱼油有什么功能")[0]
        with pytest.raises((AttributeError, TypeError)):
            s.question = "别的"       # type: ignore[misc]

    def test_max_steps_bounded(self) -> None:
        """兜底：拆出太多步要收拢（否则一次问答变 10 次召回）。

        这里用一句刻意塞满分隔符的话验证上限。
        """
        q = "，".join(f"问题{i}有什么推荐" for i in range(10))
        plan = plan_queries(q)
        assert len(plan) <= 3

    def test_never_raises(self) -> None:
        """规划器在链路最前面，抛异常等于问答不可用。"""
        for q in ("，", "，，，", "其中", "最便宜", "？", "。", "  \n "):
            assert plan_queries(q)


# ===================================================================
# 5. 不再追问「哪个最便宜」时不能硬加价格步
# ===================================================================
class TestNoPhantomPriceStep:
    @pytest.mark.parametrize("question", [
        "鱼油有什么功能", "补钙的有什么推荐",
    ])
    def test_price_step_only_when_asked(self, question: str) -> None:
        """没问价格就不要加比价步。

        ★ 为什么单列一条：比价看着「有用」，但它是**答非所问**——
          用户问功效，你给价格，等于把提问换掉了。
        """
        plan = plan_queries(question)
        assert all(s.kind != HOP_PRICE_RANK for s in plan)
