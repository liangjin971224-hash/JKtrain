"""意图接入用例的测试（2026-10-23）。

────────────────────────────────────────────────────────────────────
为什么要单独一个文件
────────────────────────────────────────────────────────────────────
`test_intent_classifier.py` 证明「分类器判得对」，
但**判得对 ≠ 接得上**。未完成事项清单里那条缺陷的本质是：

    分类器不存在 → 接口层不传 intent → 永远走开放问答权重

所以这里要证明三件事：
    1. 不传 `intent` 时，用例**自己**会分类（不再默认开放问答）
    2. 分类结果**真的变成了召回时的权重**（断言到 RetrieveUseCase 的入参）
    3. 显式传入的意图不会被覆盖；非法意图不会被静默吞掉
"""
from __future__ import annotations

import pytest

from application.use_cases.answer_question import AnswerQuestionUseCase
from domain.services.intent_classifier import OPEN_INTENT, IntentClassifier
from domain.services.rrf_fusion import INTENT_WEIGHTS
from domain.value_objects import (
    ChunkMeta,
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

COMPLIANT = ("康源牌鱼油软胶囊是保健食品，批准功能是有助于维持血脂健康水平。"
             "保健食品不是药物，不能代替药物治疗疾病。")


def _chunks() -> list[ChunkMeta]:
    return [
        ChunkMeta(chunk_id="D004_c016", doc_id="D004",
                  text="鱼油的不适宜人群：出血倾向者、孕妇",
                  title="产品禁忌",
                  permission_level=PermissionLevel.INTERNAL),
    ]


class _FixedVector(FakeVectorSearchPort):
    def __init__(self, hits: list[VectorHit]) -> None:
        super().__init__([h.chunk for h in hits])
        self._fixed = hits

    def search(self, query, *, collection="", top_k=20,
               filter_expr=None, allowed_permissions=None):
        self.calls.append("search")
        return self._fixed[:top_k]


class _FixedKeyword(FakeKeywordSearchPort):
    def __init__(self, hits: list[KeywordHit]) -> None:
        super().__init__([h.chunk for h in hits])
        self._fixed = hits

    def search(self, query, *, top_k=20, allowed_permissions=None):
        self.calls.append("search")
        return self._fixed[:top_k]


class _RecordingRetriever:
    """包一层 `RetrieveUseCase`，只记录**传进去的 intent**。

    ★ 为什么断言传参而不是断言返回值：
      返回值里的 weights 是「链路末端」的证据，中间若有人把 intent 换掉
      （比如接口层又传了个默认值），只看 weights 也能发现 ——
      但断言传参能直接指出「哪一步没传」，排障快得多。
    """

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[dict] = []

    def execute(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self._inner.execute(query, **kwargs)


def _build(*, intent: str = "", classifier=None, recording: bool = True):
    """组装用例，返回 (用例, 记录器或 None)。"""
    hits_v = [VectorHit(chunk=c, score=0.72) for c in _chunks()]
    hits_k = [KeywordHit(chunk=c, score=20.0) for c in _chunks()]
    uc = AnswerQuestionUseCase(
        graph=FakeGraphQueryPort({
            "P001": ProductFacts(product_id="P001", name="康源牌鱼油软胶囊",
                                 category="保健食品", aliases=("鱼油",)),
        }),
        vector=_FixedVector(hits_v),
        keyword=_FixedKeyword(hits_k),
        llm=FakeLanguageModelPort(reply=COMPLIANT),
        intent_classifier=classifier,
    )
    rec = _RecordingRetriever(uc._retriever) if recording else None
    if rec is not None:
        uc._retriever = rec
    return uc, rec


# ===================================================================
# 1. 自动分类（缺陷的直接修复点）
# ===================================================================
class TestAutoClassification:
    def test_contraindication_question_uses_graph_heavy_weights(self) -> None:
        """★ 未完成事项清单里点名的那个例子。

        「在吃阿司匹林能吃鱼油吗」必须走「禁忌咨询」(0.6, 0.2, 0.2)，
        而不是「开放问答」(0.1, 0.8, 0.1)。
        """
        uc, rec = _build()
        result = uc.execute("在吃阿司匹林能吃鱼油吗",
                            vector_collection="training")

        assert rec.calls[0]["intent"] == "禁忌咨询"
        assert result.meta["intent"] == "禁忌咨询"
        assert result.meta["weights"] == INTENT_WEIGHTS["禁忌咨询"]
        # 图谱权重必须是三路里最高的
        assert result.meta["weights"][0] == 0.6

    def test_empty_intent_no_longer_means_open(self) -> None:
        """不传 intent 不等于「开放问答」——这是改造前的行为。"""
        uc, rec = _build()
        result = uc.execute("吃了能停降压药吗", vector_collection="training")

        assert rec.calls[0]["intent"] != OPEN_INTENT
        assert rec.calls[0]["intent"] == "要停药"

    def test_meta_records_intent_source(self) -> None:
        """meta 要说清「意图是怎么来的」，否则判错时无从下手。"""
        uc, _ = _build()
        result = uc.execute("体脂仪测量前要做哪些准备",
                            vector_collection="training")

        assert result.meta["intent_source"] == "rule"
        assert result.meta["intent_confidence"] > 0
        assert result.meta["intent_matched"], "命中词要记录下来"

    def test_stream_also_classifies(self) -> None:
        """★ 流式与非流式必须走同一套意图逻辑。

        分开判会导致「同一个问题，流式和非流式召回的文档不一样」——
        这种不一致极难发现（用户只看到答案不同，看不出原因）。
        """
        uc, rec = _build()
        events = list(uc.execute_stream("客户在吃华法林能吃鱼油吗",
                                        vector_collection="training"))
        done = [e for e in events if e.kind == "done"][0]

        assert rec.calls[0]["intent"] == "禁忌咨询"
        assert done.meta["intent"] == "禁忌咨询"
        assert done.meta["intent_source"] == "rule"


# ===================================================================
# 2. 显式意图不能被覆盖
# ===================================================================
class TestExplicitIntent:
    def test_explicit_intent_wins(self) -> None:
        """调用方显式指定时，不跑分类器。

        演练侧/前端选类别时要用这条：它们比文本规则更清楚上下文。
        """
        uc, rec = _build()
        result = uc.execute("鱼油有什么功能", vector_collection="training",
                            intent="文号查询")

        assert rec.calls[0]["intent"] == "文号查询"
        assert result.meta["intent_source"] == "explicit"

    def test_invalid_explicit_intent_falls_back_to_classifier(self) -> None:
        """传了权重表里没有的标签 → 改用分类器，不能静默走兜底权重。

        ★ 静默兜底最难查：日志里看不到任何异常，
          但实际权重是「产品功效」(0.5,0.3,0.2)，与预期完全不同。
        """
        uc, rec = _build()
        uc.execute("鱼油有什么功能", vector_collection="training",
                   intent="这个标签不存在")

        assert rec.calls[0]["intent"] == "产品功效"

    def test_whitespace_intent_treated_as_empty(self) -> None:
        uc, rec = _build()
        uc.execute("鱼油有什么功能", vector_collection="training",
                   intent="   ")
        assert rec.calls[0]["intent"] == "产品功效"


# ===================================================================
# 3. 降级：分类器挂掉不能拖垮问答
# ===================================================================
class TestDegrade:
    def test_classifier_exception_degrades_to_open(self) -> None:
        """分类器在链路最前面，抛异常等于整个问答不可用。

        所以用例必须自己兜住，退到中性权重继续答。
        """

        class _Broken:
            def classify(self, question):       # noqa: ANN001, ANN202
                raise RuntimeError("分类器炸了")

        uc, rec = _build(classifier=_Broken())
        result = uc.execute("鱼油有什么功能", vector_collection="training")

        assert rec.calls[0]["intent"] == OPEN_INTENT
        assert result.answer, "降级后仍要能回答"
        assert result.refused is False

    def test_classifier_not_invoked_when_question_empty(self) -> None:
        """空问题在分类之前就拒答了，不该白跑一次分类。"""

        class _Counting:
            def __init__(self) -> None:
                self.n = 0

            def classify(self, question):       # noqa: ANN001, ANN202
                self.n += 1
                from domain.services.intent_classifier import IntentResult
                return IntentResult(label=OPEN_INTENT)

        clf = _Counting()
        uc, _ = _build(classifier=clf)
        assert uc.execute("", vector_collection="training").refused is True
        assert clf.n == 0


# ===================================================================
# 4. 默认装配
# ===================================================================
class TestDefaultAssembly:
    def test_default_classifier_is_rules_only(self) -> None:
        """不传分类器时用的是纯规则版（不碰 LLM，红线见分类器模块注释）。"""
        uc = AnswerQuestionUseCase()
        assert isinstance(uc._intent_classifier, IntentClassifier)
        assert uc._intent_classifier._use_llm_fallback is False

    @pytest.mark.parametrize("question,expected_graph_weight", [
        ("在吃阿司匹林能吃鱼油吗", 0.6),
        ("客户进店第一步该做什么", 0.1),
        ("批准文号的格式有哪几种", 0.2),
    ])
    def test_weight_structure_follows_intent(
        self, question: str, expected_graph_weight: float
    ) -> None:
        """验证「权重结构」而不只是「标签」—— 这才是缺陷的实质。"""
        uc, _ = _build()
        meta = uc.execute(question, vector_collection="training").meta
        assert meta["weights"][0] == pytest.approx(expected_graph_weight)
