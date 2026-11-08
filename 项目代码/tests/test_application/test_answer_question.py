"""知识问答用例测试（P4 核心红线）。

★ 本文件最重要的一条：`test_refuses_when_no_hits_without_llm`。

  宁可说「暂无相关资料」，也不能让 LLM 凭参数知识编造。
  这条约束用 `FailOnCallLLM`（一调就炸）来证明——不是看日志，
  而是让「调用了 LLM」这件事变成测试失败。

★ 第二重要：生成后必须过护栏。
  护栏是必经节点（约束 #1），不能因为「LLM 看起来说得挺规范」就跳过。
"""
from __future__ import annotations

import pytest

from application.use_cases.answer_question import AnswerQuestionUseCase
from domain.value_objects import (
    ApprovalClaim,
    ChunkMeta,
    Contraindication,
    KeywordHit,
    PermissionLevel,
    ProductCategory,
    ProductFacts,
    Severity,
    VectorHit,
)
from tests.fakes import (
    FailOnCallLLM,
    FakeGraphQueryPort,
    FakeKeywordSearchPort,
    FakeLanguageModelPort,
    FakeVectorSearchPort,
)


# ===================================================================
# 夹具
# ===================================================================
def _fish_oil() -> ProductFacts:
    return ProductFacts(
        product_id="P001",
        name="康源牌鱼油软胶囊",
        category="保健食品",
        approval_no="国食健注G20250012",
        claims=(ApprovalClaim(efficacy_id="EFF_19",
                             text="有助于维持血脂（胆固醇/甘油三酯）健康水平",
                             catalog_no=19),),
        contraindications=(
            Contraindication("CI_BLEED", "出血倾向者和出血性疾病患者",
                             Severity.CRITICAL, "鱼油可能影响凝血功能"),
            Contraindication("CI_PREG", "孕妇", Severity.HIGH, ""),
        ),
        aliases=("鱼油",),
    )


def _ordinary_food() -> ProductFacts:
    return ProductFacts(
        product_id="P016",
        name="康源牌复合蛋白固体饮料",
        category="普通食品",
    )


def _chunks() -> list[ChunkMeta]:
    return [
        ChunkMeta(chunk_id="D005_c001", doc_id="D005",
                  text="保健食品只能声称 24 项功能，必须带「有助于」限定词",
                  title="功能声称规范",
                  permission_level=PermissionLevel.INTERNAL),
        ChunkMeta(chunk_id="D005_c002", doc_id="D005",
                  text="法定警示语：保健食品不是药物，不能代替药物治疗疾病",
                  title="警示语",
                  permission_level=PermissionLevel.PUBLIC),
    ]


def _vector(hits: list[VectorHit] | None = None) -> FakeVectorSearchPort:
    if hits is None:
        hits = [VectorHit(chunk=c, score=0.72) for c in _chunks()]
    return FakeVectorSearchPort([h.chunk for h in hits])


class _FixedVector(FakeVectorSearchPort):
    """返回固定分数的向量替身。"""

    def __init__(self, hits: list[VectorHit]) -> None:
        super().__init__([h.chunk for h in hits])
        self._fixed = hits

    def search(self, query, *, collection="", top_k=20,
               filter_expr=None, allowed_permissions=None):
        self.calls.append("search")
        pool = self._fixed
        if allowed_permissions is not None:
            allowed = {str(p) for p in allowed_permissions}
            pool = [h for h in pool
                    if str(h.chunk.permission_level) in allowed]
        return pool[:top_k]


class _FixedKeyword(FakeKeywordSearchPort):
    """返回固定高分的假关键词检索。

    ★ 分数必须高于 `MIN_KEYWORD_SCORE`（14.0，实测标定）。
      用 `FakeKeywordSearchPort` 的字符重合度打分只有个位数，
      会低于绝对阈值导致该路被判定为「无有效召回」。
    """

    def __init__(self, hits: list[KeywordHit]) -> None:
        super().__init__([h.chunk for h in hits])
        self._fixed = hits

    def search(self, query, *, top_k=20, allowed_permissions=None):
        self.calls.append("search")
        pool = self._fixed
        if allowed_permissions is not None:
            allowed = {str(p) for p in allowed_permissions}
            pool = [h for h in pool
                    if str(h.chunk.permission_level) in allowed]
        return pool[:top_k]


def _keyword() -> _FixedKeyword:
    return _FixedKeyword([KeywordHit(chunk=c, score=20.0) for c in _chunks()])


def _build(**overrides) -> AnswerQuestionUseCase:
    """默认：三路都有结果，LLM 返回合规回答。"""
    kwargs = dict(
        graph=FakeGraphQueryPort({"P001": _fish_oil(),
                                  "P016": _ordinary_food()}),
        vector=_vector(),
        keyword=_keyword(),
        llm=FakeLanguageModelPort(
            reply=("康源牌鱼油软胶囊是保健食品，批准功能是"
                   "「有助于维持血脂（胆固醇/甘油三酯）健康水平」。"
                   "保健食品不是药物，不能代替药物治疗疾病。")),
    )
    kwargs.update(overrides)
    return AnswerQuestionUseCase(**kwargs)


# ===================================================================
# ★ 红线 1：召回为空 → 拒答，且 LLM 绝不被调用
# ===================================================================
class TestRefuseWithoutLLM:
    def test_refuses_when_no_hits_without_llm(self) -> None:
        """三路召回全空 → 拒答，且**不调 LLM**。

        这是本项目最重要的一条测试：
        宁可说「暂无资料」，也不能让 LLM 凭参数知识编。
        """
        uc = AnswerQuestionUseCase(
            graph=FakeGraphQueryPort({}),          # 无产品
            vector=_FixedVector([]),               # 无文档
            keyword=FakeKeywordSearchPort([]),     # 无关键词命中
            llm=FailOnCallLLM(),                   # ← 一调就炸
        )

        result = uc.execute("知识库里完全没有的问题",
                            vector_collection="training")

        assert result.refused is True
        assert "暂无" in result.answer or "没有" in result.answer
        assert result.refusal_reason

    def test_refuses_when_vector_scores_below_threshold(self) -> None:
        """向量有返回但分数低于阈值 → 视为无召回 → 拒答。

        向量检索永远返回 Top-N，低分就是噪声。
        若这里放行，P4 会拿无关内容生成答案。
        """
        uc = AnswerQuestionUseCase(
            graph=FakeGraphQueryPort({}),
            vector=_FixedVector([VectorHit(chunk=c, score=0.31)
                                 for c in _chunks()]),
            keyword=FakeKeywordSearchPort([]),
            llm=FailOnCallLLM(),
        )
        result = uc.execute("无关问题", vector_collection="training")

        assert result.refused is True

    def test_refuse_answer_offers_next_step(self) -> None:
        """拒答不能只说「无法回答」，要给下一步建议。

        incremental-delivery 明确要求：拒答 + 替代建议。
        """
        uc = AnswerQuestionUseCase(
            graph=None, vector=_FixedVector([]), keyword=None,
            llm=FailOnCallLLM(),
        )
        result = uc.execute("问题", vector_collection="training")

        assert result.refused is True
        # 至少给出一种可行建议
        hints = ["咨询", "主管", "换个说法", "资料", "培训"]
        assert any(h in result.answer for h in hints), (
            f"拒答应给替代建议，实际: {result.answer}"
        )


# ===================================================================
# ★ 红线 2：生成后必须过护栏
# ===================================================================
class TestGuardrailIsMandatory:
    def test_violating_llm_output_is_blocked(self) -> None:
        """LLM 输出了违规内容 → 必须被护栏拦下，不能原样返回。

        护栏是必经节点（约束 #1）。即使 LLM「看起来说得挺顺」也要过。
        """
        uc = _build(llm=FakeLanguageModelPort(
            reply="这个能降血糖，很多人吃了都说好。"))

        result = uc.execute("这个能降血糖吗", vector_collection="training")

        assert result.refused is True, (
            "违规输出被拦后应转为拒答，不能把违规内容返回给客户")
        assert "降血糖" not in result.answer

    def test_ordinary_food_claim_blocked(self) -> None:
        """普通食品的功能声称必须拦（约束 #2）。"""
        uc = _build(llm=FakeLanguageModelPort(
            reply="这款固体饮料有助于增强免疫力，效果很好。"))

        result = uc.execute("这个固体饮料能增强免疫力吗",
                            vector_collection="training",
                            product_ids=("P016",))

        assert result.refused is True
        assert "增强免疫力" not in result.answer

    def test_missing_warning_repaired_not_refused(self) -> None:
        """保健食品回答缺警示语 → **补回警示语**，不整段拒答。

        ★ 行为变更（2026-10-23，产品决策）：
          原策略是「缺警示语就整段拒答」。
          改为：内容本身合规、只缺一句法定警示语时，把它补上再输出。

        ★ 为什么这样更好：
          缺警示语是**可修复**的问题——补一句就合规了，
          整段拒答等于把正确内容一起丢掉，学员什么也学不到。

        ★ 但底线不能松：
          safe_text 必须真的包含法定警示语（下面有断言），
          否则就是漏放违规输出。
        """
        uc = _build(llm=FakeLanguageModelPort(
            reply="这款鱼油有助于维持血脂健康水平。"))

        result = uc.execute("鱼油有什么功能", vector_collection="training",
                            product_ids=("P001",))

        assert result.refused is False
        assert "有助于维持血脂" in result.answer
        # ★ 底线：警示语必须被补上
        assert "不能代替药物治疗疾病" in result.answer
        assert result.meta.get("repaired_warning") is True

    def test_violating_output_still_refused_when_nothing_safe(self) -> None:
        """★ 全违规（无合规内容可留）时仍必须整段拒答。

        这是护栏的底线，不能被「部分保留」策略削弱。
        """
        uc = _build(llm=FakeLanguageModelPort(
            reply="这个能降血糖。"))

        result = uc.execute("能降血糖吗", vector_collection="training")

        assert result.refused is True
        assert "降血糖" not in result.answer

    def test_compliant_output_passes(self) -> None:
        """合规回答必须正常返回——不能过度拦截。"""
        uc = _build()
        result = uc.execute("鱼油有什么功能", vector_collection="training",
                            product_ids=("P001",))

        assert result.refused is False
        assert result.answer
        assert "有助于维持血脂" in result.answer


# ===================================================================
# 3. 正常路径
# ===================================================================
class TestNormalPath:
    def test_returns_answer_with_citations(self) -> None:
        """正常回答要带溯源（评委问「凭什么这么说」要能答）。"""
        uc = _build()
        result = uc.execute("鱼油有什么功能",
                            vector_collection="training",
                            product_ids=("P001",))

        assert not result.refused
        assert result.cited_chunk_ids, "必须带引用的知识块 id"
        assert result.product_ids == ("P001",)

    def test_llm_called_once(self) -> None:
        llm = FakeLanguageModelPort(reply="合规回答。" + "保健食品不是药物，不能代替药物治疗疾病。")
        uc = _build(llm=llm)
        uc.execute("鱼油有什么功能", vector_collection="training",
                   product_ids=("P001",))

        assert llm.calls.count("generate") == 1, "正常路径只该调一次 LLM"

    def test_prompt_contains_context(self) -> None:
        """prompt 里必须包含召回内容——否则 LLM 是在凭参数知识答。"""
        llm = FakeLanguageModelPort(
            reply="合规回答。" + "保健食品不是药物，不能代替药物治疗疾病。")
        uc = _build(llm=llm)
        uc.execute("鱼油有什么功能", vector_collection="training",
                   product_ids=("P001",))

        prompt = llm.prompts[0]
        assert "康源牌鱼油" in prompt, "prompt 应含产品事实"
        assert "有助于维持血脂" in prompt, "prompt 应含批准功能原文"

    def test_meta_records_diagnostics(self) -> None:
        """meta 要记录各路命中数与耗时，便于排障。"""
        uc = _build()
        result = uc.execute("鱼油有什么功能",
                            vector_collection="training",
                            product_ids=("P001",))

        assert "elapsed_ms" in result.meta
        assert "vector_hits" in result.meta or "raw_hits" in result.meta


# ===================================================================
# 4. 降级路径
# ===================================================================
class TestDegradation:
    def test_graph_down_still_answers(self) -> None:
        """图谱挂了仍能回答（三路容错）。"""
        uc = _build(graph=FakeGraphQueryPort({"P001": _fish_oil()},
                                            fail_on_call=True))
        result = uc.execute("鱼油有什么功能", vector_collection="training")

        assert not result.refused
        assert result.answer
        assert "graph" in result.meta.get("degraded_paths", [])

    def test_llm_down_returns_error_not_empty(self) -> None:
        """LLM 挂了必须明确报错，不能返回空回答。

        空回答会让用户以为「系统说没有」，实际是故障。
        """
        from domain.ports import LanguageModelPort

        class _DeadLLM(LanguageModelPort):
            def generate(self, prompt, **kw):
                from infrastructure.exceptions import LLMCallError
                raise LLMCallError("所有端点均失败")

            def generate_stream(self, prompt, **kw):
                from infrastructure.exceptions import LLMCallError
                raise LLMCallError("所有端点均失败")

            def classify(self, text, labels, **kw):
                return "unknown"

            def is_available(self):
                return False

        uc = _build(llm=_DeadLLM())
        result = uc.execute("鱼油有什么功能", vector_collection="training",
                            product_ids=("P001",))

        # 应转入拒答并说明原因，而不是返回空字符串
        assert result.refused is True
        assert result.refusal_reason, "必须说明为什么没答出来"


# ===================================================================
# 5. 权限透传（FR-050）
# ===================================================================
class TestPermission:
    def test_trainee_cannot_see_restricted_in_prompt(self) -> None:
        """学员视角下，受限内容不能出现在 prompt 里。

        即使检索层已过滤，也要守住「不喂给 LLM」这一关——
        一旦进了 prompt 就可能被复述出来。
        """
        restricted = ChunkMeta(
            chunk_id="Q001", doc_id="Q001",
            text="参考答案：这道题的答案是 B",
            permission_level=PermissionLevel.RESTRICTED)

        vec = _FixedVector([VectorHit(chunk=restricted, score=0.9)])
        llm = FakeLanguageModelPort(
            reply="合规回答。" + "保健食品不是药物，不能代替药物治疗疾病。")

        uc = AnswerQuestionUseCase(
            graph=FakeGraphQueryPort({}), vector=vec,
            keyword=FakeKeywordSearchPort([]), llm=llm)

        uc.execute("这道题答案是什么", vector_collection="training",
                   allowed_permissions=("public", "internal"))

        if llm.prompts:
            assert "答案：这道题的答案是 B" not in llm.prompts[0], (
                "受限内容泄漏进了 prompt"
            )


# ===================================================================
# 6. 参数校验
# ===================================================================
class TestValidation:
    def test_empty_question(self) -> None:
        uc = _build()
        result = uc.execute("", vector_collection="training")
        assert result.refused is True

    def test_whitespace_question(self) -> None:
        uc = _build()
        result = uc.execute("   ", vector_collection="training")
        assert result.refused is True
