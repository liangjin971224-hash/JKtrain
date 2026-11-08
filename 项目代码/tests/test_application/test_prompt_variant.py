"""系统提示词版本（P1-c）测试。

守住的语义：
    1. 默认是 v3（八条 + 回答前自检四条）
    2. v2 仍可显式回滚，且不含自检块
    3. 未知版本回退 v2（配置写错不能悄悄改变行为）
    4. 流式与非流式用同一个版本，不能各用各的

背景：v3 针对 45 题 rubric 里生成侧的四类失分——
    批准原文（ev2-01）/ 替代说法（ev2-03）/ 先问清（ev2-14）/ 来源分级（ev2-35）。
"""
from __future__ import annotations

from application.use_cases.answer_question import AnswerQuestionUseCase
from domain.value_objects import ChunkMeta, PermissionLevel, VectorHit
from tests.fakes import FakeLanguageModelPort, FakeVectorSearchPort

QUERY = "客户嫌贵怎么办"
_REPLY = "保健食品不是药物，不能代替药物治疗疾病。"


class _FixedVector(FakeVectorSearchPort):
    def __init__(self, hits: list[VectorHit]) -> None:
        super().__init__([h.chunk for h in hits])
        self._fixed = hits

    def search(self, query, *, collection="", top_k=20,
               filter_expr=None, allowed_permissions=None):
        self.calls.append("search")
        return self._fixed[:top_k]


def _uc(variant: str | None = None) -> tuple[AnswerQuestionUseCase, FakeLanguageModelPort]:
    chunk = ChunkMeta(
        chunk_id="D001_c001", doc_id="D001",
        text="客户嫌贵时先讲价值再谈价格。",
        title="接待话术", permission_level=PermissionLevel.INTERNAL,
    )
    llm = FakeLanguageModelPort(reply=_REPLY)
    kwargs = {}
    if variant is not None:
        kwargs["prompt_variant"] = variant
    uc = AnswerQuestionUseCase(
        vector=_FixedVector([VectorHit(chunk=chunk, score=0.9)]),
        keyword=None,
        llm=llm,
        **kwargs,
    )
    return uc, llm


def _sys_prompt(llm: FakeLanguageModelPort) -> str:
    assert llm.system_prompts, "应该调用过 LLM"
    return llm.system_prompts[0]


def test_default_is_v3() -> None:
    uc, llm = _uc()
    uc.execute(QUERY, vector_collection="health_training_docs")

    prompt = _sys_prompt(llm)
    assert "必须遵守" in prompt
    assert "回答前自检" in prompt


def test_explicit_v2_has_no_selfcheck() -> None:
    uc, llm = _uc("v2")
    uc.execute(QUERY, vector_collection="health_training_docs")

    prompt = _sys_prompt(llm)
    assert "必须遵守" in prompt
    assert "回答前自检" not in prompt


def test_v3_adds_four_selfcheck_rules() -> None:
    uc, llm = _uc("v3")
    uc.execute(QUERY, vector_collection="health_training_docs")

    prompt = _sys_prompt(llm)
    assert "回答前自检" in prompt
    assert "逐字使用" in prompt            # A 批准原文
    assert "规范替代说法" in prompt         # B 替代说法
    assert "先问清" in prompt              # C 缺信息
    assert "员工经验" in prompt            # D 来源分级


def test_unknown_variant_falls_back_to_v2() -> None:
    uc, llm = _uc("v9")
    uc.execute(QUERY, vector_collection="health_training_docs")

    prompt = _sys_prompt(llm)
    assert "回答前自检" not in prompt
    assert "必须遵守" in prompt


def test_stream_uses_same_variant() -> None:
    uc, llm = _uc("v3")
    # 流式路径同样应带 v3 自检块
    list(uc.execute_stream(QUERY, vector_collection="health_training_docs"))

    prompt = _sys_prompt(llm)
    assert "回答前自检" in prompt
