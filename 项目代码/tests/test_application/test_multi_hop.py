"""多跳执行测试（红灯先行，2026-10-23）。

────────────────────────────────────────────────────────────────────
这里测的是「前一步的结果真的被当作后一步的候选集」
────────────────────────────────────────────────────────────────────
`tests/test_domain/test_query_planner.py` 只证明「拆得对」。
**拆得对 ≠ 跳得对**：规划器标了 `depends_on_previous=True`，
执行层完全可能照样全库检索，然后回答里假装是先查出来的。

所以本文件的核心断言是：
    第 2 步的召回**入参**里带着第 1 步的产品集合

断言入参而不是返回值，因为返回值可以「碰巧对」：
全店最便宜的保健食品是「维生素C咀嚼片 98 元」，与体脂无关。
如果第 2 步自由检索，它照样能返回一个「最便宜」的答案 —— 只是答非所问。
"""
from __future__ import annotations

import pytest

from application.use_cases.query_plan import (
    HopResult,
    MultiHopAnswer,
    QueryPlanUseCase,
)
from domain.services.query_planner import HOP_PRICE_RANK, plan_queries
from domain.value_objects import (
    ApprovalClaim,
    ChunkMeta,
    EfficacyRef,
    KeywordHit,
    PermissionLevel,
    ProductFacts,
    VectorHit,
)
from tests.fakes import (
    FakeGraphQueryPort,
    FakeKeywordSearchPort,
    FakeLanguageModelPort,
    FakeVectorSearchPort,
)

# ===================================================================
# 夹具：4 个产品，价格故意设成「最便宜的与主题无关」
# ===================================================================
#: ★ 这组数据是整个测试的关键。模仿真实图谱：
#:   P015 维生素C 98 元（全店最便宜的保健食品，但与体脂无关）
#:   P020 代餐奶昔 218 元（普通食品，primary_scenario=体重管理，没有批准功能）
#:   P014 膳食纤维 158 元（保健食品，声称润肠通便，不声称控脂）
#:   P011 钙维生素D 168 元（保健食品，声称补钙）
#: 真实图谱里 EFF_09「有助于控制体内脂肪」**没有任何产品声称**。
def _products() -> dict[str, ProductFacts]:
    return {
        "P015": ProductFacts(
            product_id="P015", name="康源牌维生素C咀嚼片",
            category="保健食品", retail_price=98,
            claims=(ApprovalClaim("EFF_NUT_VC", "补充维生素C"),),
        ),
        "P020": ProductFacts(
            product_id="P020", name="康源牌代餐奶昔",
            category="普通食品", retail_price=218,
            compliance_note="普通食品，禁止功能声称",
        ),
        "P014": ProductFacts(
            product_id="P014", name="康源牌膳食纤维粉",
            category="保健食品", retail_price=158,
            claims=(ApprovalClaim("EFF_17", "有助于润肠通便"),),
        ),
        "P011": ProductFacts(
            product_id="P011", name="康源牌钙维生素D软胶囊",
            category="保健食品", retail_price=168,
            claims=(ApprovalClaim("EFF_NUT_CA", "补充钙、维生素D"),),
        ),
    }


def _chunk(cid: str = "D008_c001") -> ChunkMeta:
    return ChunkMeta(
        chunk_id=cid, doc_id="D008",
        text="普通食品不得宣称任何保健功能",
        title="普通食品须知",
        permission_level=PermissionLevel.PUBLIC,
    )


class _RecordingGraph(FakeGraphQueryPort):
    """记录所有图谱调用，并真实模拟「需求反查」。

    ★ 为什么必须覆盖 `find_products_by_need`：
      父类的实现是「返回所有保健食品」——它只保住了方法契约，
      不模拟跳数。用它测多跳会得到一个**假通过**：

          第 1 跳候选 = 所有保健食品（含无关的维生素C）
          第 2 跳在这个集合里比价 → 比出维生素C 98 元
          测试断言「候选不能超出第 1 跳集合」→ 通过
          但第 1 跳集合本身就是错的

      假替身的语义错了，测出来的应用层逻辑也是假的。
    """

    def __init__(self, products, *, efficacy_map=None, need_map=None,
                 catalog=()) -> None:
        super().__init__(products, catalog=catalog)
        self._efficacy_map = efficacy_map or {}
        self._need_map = need_map or {}

    def find_products_by_efficacy(self, efficacy_id: str):
        """按功能反查产品。

        ★ 空列表是合法返回值（＝本店没有声称该功能的产品）。
          真实库里 `EFF_09` 就是这种情况（目录有、产品无），
          替身必须能表达它，否则「法规不允许」和「本店没有」会被测成同一件事。
        """
        self.calls.append("find_products_by_efficacy")
        ids = self._efficacy_map.get(efficacy_id, [])
        return [self._to_ref(self._products[i]) for i in ids if i in self._products]

    def find_products_by_need(self, need_id: str):
        """按需求反查：need_id → 产品集合。

        ★ 空列表是合法返回值（＝本店没有满足该需求的产品）。
          父类那种「返回全部」的做法正是多跳最危险的假通过来源。
        """
        self.calls.append("find_products_by_need")
        ids = self._need_map.get(need_id, [])
        return [self._to_ref(self._products[i]) for i in ids if i in self._products]


class _RecordingVector(FakeVectorSearchPort):
    """记录每次检索的 query，便于断言「第 2 步用的是不是新问题」。"""

    def __init__(self, hits: list[VectorHit]) -> None:
        super().__init__([h.chunk for h in hits])
        self._fixed = hits
        self.queries: list[str] = []

    def search(self, query, *, collection="", top_k=20,
               filter_expr=None, allowed_permissions=None):
        self.calls.append("search")
        self.queries.append(query)
        return self._fixed[:top_k]


class _RecordingKeyword(FakeKeywordSearchPort):
    def __init__(self, hits: list[KeywordHit]) -> None:
        super().__init__([h.chunk for h in hits])
        self._fixed = hits
        self.queries: list[str] = []

    def search(self, query, *, top_k=20, allowed_permissions=None):
        self.calls.append("search")
        self.queries.append(query)
        return self._fixed[:top_k]


def _build(*, graph=None, llm=None, uc_kwargs=None):
    """组装多跳用例。

    ★ 用**真实**的 `AnswerQuestionUseCase` 作为单跳实现，
      而不是再造一个假的：
      多跳最容易犯的错就是「绕开护栏」，
      用真实现才能证明护栏仍在这一条链路上（见 `test_guardrail_still_mandatory`）。
    """
    from application.use_cases.answer_question import AnswerQuestionUseCase

    g = graph or _RecordingGraph(_products())
    v = _RecordingVector([VectorHit(chunk=_chunk(), score=0.72)])
    k = _RecordingKeyword([KeywordHit(chunk=_chunk(), score=20.0)])
    one = AnswerQuestionUseCase(
        graph=g, vector=v, keyword=k,
        llm=llm or FakeLanguageModelPort(reply="合规回答。"),
    )

    def answer_one(question, *, allowed_permissions=None, fact_hints=None,
                   product_ids=()):
        return one.execute(
            question, intent="产品功效",
            vector_collection="training",      # ★ 不传会让向量路降级（实测）
            allowed_permissions=allowed_permissions
            or ("public", "internal"),
            fact_hints=fact_hints,
            product_ids=tuple(product_ids),
        )

    uc = QueryPlanUseCase(answer_one=answer_one, graph=g)
    return uc, g, v, k


# ===================================================================
# 1. 核心：依赖前一步，而不是重新全库检索
# ===================================================================
class TestDependencyIsReal:
    def test_step2_candidates_come_from_step1(self) -> None:
        """★ 本文件最重要的一条。

        第 2 步「其中那种最便宜」的候选集必须**只来自**第 1 步。
        用「全店最便宜但无关的维生素C（98 元）」当诱饵：
        如果第 2 步自由检索，它会胜出 —— 那就是答非所问。
        """
        # 第 1 步只能找到 P020 代餐奶昔（体重管理）
        graph = _RecordingGraph(
            _products(),
            # 无产品声称控体脂，与真实库一致（EFF_09 目录有、产品无）
            efficacy_map={"EFF_09": ["P020"]},
        )
        uc, g, v, k = _build(graph=graph)

        result = uc.execute("改善体脂都可以吃什么保健品，其中那种最便宜")

        assert len(result.hops) == 2
        step2 = result.hops[1]
        assert step2.candidates_source == "previous_hop", (
            "第 2 步必须标明候选集来自第 1 步"
        )
        # 关键断言：第 2 步的候选里不能出现「全店最便宜的无关产品」
        assert "P015" not in step2.candidate_ids, (
            "第 2 步的候选里混进了第 1 步没有的产品（P015 维生素C 98 元）—— "
            "说明它重新全库检索了，答案会答非所问"
        )

    def test_price_answer_within_step1_set(self) -> None:
        """比价结果必须落在第 1 步的集合内。"""
        graph = _RecordingGraph(
            _products(),
            efficacy_map={"EFF_09": ["P020", "P014"]},
        )
        uc, *_ = _build(graph=graph)
        result = uc.execute("改善体脂都可以吃什么保健品，其中那种最便宜")

        first_ids = set(result.hops[0].candidate_ids)
        assert first_ids, "第 1 步应该找到候选（否则这条测试没意义）"
        if result.cheapest is not None:
            assert result.cheapest.product_id in first_ids, (
                f"最便宜的 {result.cheapest.product_id} 不在第 1 步候选 {first_ids} 里"
            )

    def test_empty_previous_hop_skips_price_ranking(self) -> None:
        """★ 第 1 步什么都没找到时，第 2 步**不能**自己去全库找便宜的。

        这是「依赖」语义的边界条件，也是最容易写错的地方：
        候选集为空 → 比价无意义 → 必须明确说「前面没找到，无法比价」，
        而不是「那我随便找个最便宜的给你」。
        """
        graph = _RecordingGraph(
            _products(), efficacy_map={})
        uc, *_ = _build(graph=graph)
        result = uc.execute("改善体脂都可以吃什么保健品，其中那种最便宜")

        assert result.cheapest is None
        assert result.hops[1].candidate_ids == ()
        assert "没有" in result.answer or "未" in result.answer


# ===================================================================
# 2. 并列多意图：互不依赖，各自独立召回
# ===================================================================
class TestParallelHops:
    def test_two_independent_hops(self) -> None:
        """「补钙的有什么推荐，有什么禁忌」→ 两步，都不依赖前一步。"""
        uc, *_ = _build()
        result = uc.execute("补钙的有什么推荐，有什么禁忌")

        assert len(result.hops) == 2
        assert all(h.candidates_source != "previous_hop" for h in result.hops)

    def test_each_hop_recalled_with_own_question(self) -> None:
        """两步各自用自己的问句去召回（不能共用第一个问句）。"""
        uc, g, v, k = _build()
        uc.execute("补钙的有什么推荐，有什么禁忌")

        assert len(v.queries) == 2, f"应向向量路发出 2 次查询，实际: {v.queries}"
        assert "推荐" in v.queries[0]
        assert "禁忌" in v.queries[1]
        assert v.queries[0] != v.queries[1]

    def test_parallel_result_contains_both_parts(self) -> None:
        """两段都要出现在最终回答里，不能只答后一段。"""
        uc, *_ = _build()
        result = uc.execute("补钙的有什么推荐，有什么禁忌")
        assert len(result.hops) == 2
        for h in result.hops:
            assert h.answer, f"第 {h.order} 步没有产出正文"


# ===================================================================
# 3. 单步问题不能退化成多跳
# ===================================================================
class TestSingleHop:
    @pytest.mark.parametrize("question", [
        "鱼油有什么功能",
        "在吃阿司匹林能吃鱼油吗",
        "客户进店第一步该做什么",
    ])
    def test_single_step_calls_llm_once(self, question: str) -> None:
        """★ 单步问题只准调一次 LLM。

        多跳的代价是 N 次生成。若简单问题也走多跳路径，
        延迟和成本直接翻倍，P95 立刻恶化。
        """
        llm = FakeLanguageModelPort(reply="合规回答。保健食品不是药物，不能代替药物治疗疾病。")
        uc, *_ = _build(llm=llm)
        uc.execute(question)

        assert len(uc.last_hops) == 1
        assert llm.calls.count("generate") == 1, (
            f"单步问题调了 {llm.calls.count('generate')} 次 LLM"
        )

    def test_single_step_no_price_ranking(self) -> None:
        uc, *_ = _build()
        result = uc.execute("补钙的有什么推荐")
        assert result.cheapest is None
        assert len(result.hops) == 1


# ===================================================================
# 4. 合规底线：比价不能说成功效
# ===================================================================
class TestCompliance:
    def test_price_answer_does_not_imply_efficacy(self) -> None:
        """★ 比价回答里不得出现「改善体脂」被当作产品功效的表述。

        比价是**客观信息**（价格），功效是**法律许可的声称**，
        两者不能混。真实库里连声称「有助于控制体内脂肪」的产品都没有，
        所以正确回答是「本店没有该功能的产品」，而不是「这个能控脂且最便宜」。
        """
        graph = _RecordingGraph(
            _products(),
            efficacy_map={"EFF_09": ["P020"]},
            catalog=(EfficacyRef(
                "EFF_09", "有助于控制体内脂肪", catalog_no=9),))
        llm = FakeLanguageModelPort(
            reply="代餐奶昔最便宜，能改善体脂。"
                  "保健食品不是药物，不能代替药物治疗疾病。")
        uc, *_ = _build(graph=graph, llm=llm)
        result = uc.execute("改善体脂都可以吃什么保健品，其中那种最便宜")

        # 答案要么经护栏处理过（不含裸功效表述），要么明确说明没有该功能
        assert "能改善体脂" not in result.answer

    def test_ordinary_food_not_offered_as_efficacy_product(self) -> None:
        """普通食品不得被当成「有效功效的保健品」推荐（约束 #2）。"""
        graph = _RecordingGraph(
            _products(),
            efficacy_map={"EFF_09": ["P020"]},
            catalog=(EfficacyRef(
                "EFF_09", "有助于控制体内脂肪", catalog_no=9),))
        uc, *_ = _build(graph=graph)
        result = uc.execute("改善体脂都可以吃什么保健品")

        # P020 是普通食品。若进了「保健品推荐」列表，就必须带合规提示
        if "P020" in result.product_ids:
            assert any("普通食品" in n for n in result.notes), (
                "普通食品被推荐时必须标注「普通食品，不得宣称保健功能」"
            )

    def test_guardrail_still_mandatory(self) -> None:
        """多跳不能绕开护栏：每一跳的生成都要过。"""
        llm = FakeLanguageModelPort(reply="这个能降血糖，很多人吃了都说好。")
        uc, *_ = _build(llm=llm)
        result = uc.execute("补钙的有什么推荐，有什么禁忌")
        assert "降血糖" not in result.answer


# ===================================================================
# 6. ★ 前提必须进 prompt（不只是拿去比价）
# ===================================================================
class TestEvidenceReachesPrompt:
    """多跳最容易只做一半：候选集传给了比价，却没告诉生成层。

    实测后果（2026-10-23，真实三库）：
      · 「补钙的有什么推荐」→ 答成「先问客户几个问题」，
        而图谱已查到 P011
      · 「改善体脂」→ 模型编成「目录中没有这个功能」，
        而目录第 9 项恰恰是「有助于控制体内脂肪」
        —— 把法规教错了，比不回答严重
    """

    def test_catalog_fact_reaches_prompt(self) -> None:
        """「目录里有、本店没有」这个事实必须出现在 prompt 里。"""
        graph = _RecordingGraph(
            _products(),
            efficacy_map={},                    # 无产品声称控体脂
            catalog=(EfficacyRef(
                "EFF_09", "有助于控制体内脂肪", catalog_no=9),))
        llm = FakeLanguageModelPort(reply="资料里没有。")
        uc, *_ = _build(graph=graph, llm=llm)
        uc.execute("改善体脂都可以吃什么保健品")

        assert llm.prompts, "应该调用过 LLM"
        prompt = llm.prompts[0]
        assert "有助于控制体内脂肪" in prompt, (
            "目录里的功能名没进 prompt —— 模型会编成「目录里没有这个功能」"
        )
        assert "第 9 项" in prompt, "应带上目录编号，便于核对法规"
        assert "没有任何一款" in prompt or "没有一款" in prompt, (
            "必须明确「本店没有产品声称」，否则模型分不清"
            "「法规不允许」与「本店没有」"
        )

    def test_products_found_reach_prompt(self) -> None:
        """图谱查到的产品清单必须进 prompt（不能只用来比价）。"""
        graph = _RecordingGraph(
            _products(),
            efficacy_map={"EFF_NUT_CA": ["P011"]},
            catalog=(EfficacyRef(
                "EFF_NUT_CA", "补充钙、维生素D"),))
        llm = FakeLanguageModelPort(reply="资料里没有。")
        uc, *_ = _build(graph=graph, llm=llm)
        uc.execute("补钙的有什么推荐")

        prompt = llm.prompts[0]
        assert "康源牌钙维生素D软胶囊" in prompt, (
            "图谱查到的产品没进 prompt —— 模型只能反问客户，答不出推荐"
        )
        assert "补充钙、维生素D" in prompt

    def test_no_catalog_match_makes_no_legal_claim(self) -> None:
        """目录里真查不到时，只能说「没查到」，不能替法规下结论。"""
        graph = _RecordingGraph(_products(), efficacy_map={})
        llm = FakeLanguageModelPort(reply="资料里没有。")
        uc, *_ = _build(graph=graph, llm=llm)
        uc.execute("改善体脂都可以吃什么保健品")

        prompt = llm.prompts[0]
        assert "没有查到" in prompt
        # 不得出现替法规断言的强表述
        assert "法规不允许" not in prompt

    def test_evidence_skipped_for_unrelated_question(self) -> None:
        """无关问题不查证据（省一次图谱往返，也别污染 prompt）。"""
        graph = _RecordingGraph(_products())
        llm = FakeLanguageModelPort(reply="接待分五步。")
        uc, *_ = _build(graph=graph, llm=llm)
        uc.execute("客户进店第一步该做什么")

        prompt = llm.prompts[0]
        assert "已核实的事实" not in prompt


# ===================================================================
# 5. 契约与降级
# ===================================================================
class TestContract:
    def test_hops_recorded_with_order(self) -> None:
        uc, *_ = _build()
        result = uc.execute("补钙的有什么推荐，有什么禁忌")
        assert [h.order for h in result.hops] == [0, 1]
        for h in result.hops:
            assert isinstance(h, HopResult)
            assert h.question

    def test_returns_multi_hop_answer(self) -> None:
        uc, *_ = _build()
        result = uc.execute("补钙的有什么推荐，有什么禁忌")
        assert isinstance(result, MultiHopAnswer)
        assert result.answer

    def test_planner_failure_degrades_to_single(self) -> None:
        """规划器炸了也要能答（按整句单步处理）。"""
        class _BrokenPlanner:
            def __call__(self, question):
                raise RuntimeError("规划器炸了")

        uc, *_ = _build()
        uc._planner = _BrokenPlanner()
        result = uc.execute("补钙的有什么推荐，有什么禁忌")
        assert result.answer
        assert len(result.hops) == 1

    def test_one_hop_failure_does_not_kill_others(self) -> None:
        """★ 第 2 步挂掉，第 1 步的答案仍要给出。

        并列场景下两步互不依赖，没有理由因为一步失败就整段拒答 ——
        「只答了一半」比「什么都没答」有用得多，只要如实说明。
        """
        uc, *_ = _build()
        calls = {"n": 0}
        real = uc._answer_one

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("第二跳炸了")
            return real(*a, **kw)

        uc._answer_one = flaky
        result = uc.execute("补钙的有什么推荐，有什么禁忌")

        assert len(result.hops) >= 1
        assert result.hops[0].answer
        assert result.notes, "要如实说明有一跳失败"

    def test_max_steps_respected(self) -> None:
        uc, *_ = _build()
        result = uc.execute("补钙的有什么推荐，有什么禁忌，孕妇能吃吗")
        assert len(result.hops) <= 3

    def test_empty_question(self) -> None:
        uc, *_ = _build()
        result = uc.execute("")
        assert result.answer
        assert result.refused is True
