"""条件式答案补全（A4）的信号与用例接线。

不测「补全一定让答案更好」——那是 A/B 的事；这里只锁三件事：
1. 什么情况下触发补全；
2. 补全结果必须重新过护栏，被拦/为空时保留初稿；
3. 开关关闭时调用次数完全不增加。
"""

from __future__ import annotations

from collections.abc import Iterator

from application.services.answer_repair import (
    RepairSignal,
    build_repair_prompt,
    detect_gaps,
)
from application.use_cases.answer_question import AnswerQuestionUseCase
from domain.ports import GenerationResult, LanguageModelPort
from domain.value_objects import ChunkMeta, KeywordHit, PermissionLevel
from tests.fakes import FakeKeywordSearchPort, FakeVectorSearchPort


def _chunk(text: str) -> ChunkMeta:
    return ChunkMeta(
        chunk_id="D005_c999",
        doc_id="D005",
        text=text,
        title="功能声称规范",
        permission_level=PermissionLevel.PUBLIC,
    )


class _FixedKeyword(FakeKeywordSearchPort):
    """固定高分的假关键词路，避免替身打分低于 14.0 阈值。"""

    def __init__(self, chunks: list[ChunkMeta]) -> None:
        super().__init__(chunks)
        self._fixed = [KeywordHit(chunk=c, score=20.0) for c in chunks]

    def search(self, query, *, top_k=20, allowed_permissions=None):
        self.calls.append("search")
        return self._fixed[:top_k]


class _SequencedLLM(LanguageModelPort):
    """按顺序返回多条回复，用于「初稿 → 补全」两次调用。"""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        self.prompts.append(prompt)
        text = self._replies.pop(0) if self._replies else ""
        return GenerationResult(
            text=text,
            model="fake-sequenced",
            prompt_tokens=len(prompt),
            completion_tokens=len(text),
            finish_reason="stop",
        )

    def classify(self, text: str, labels: list[str], *,
                 allow_unknown: bool = True) -> str:
        return labels[0] if labels else "unknown"

    def generate_stream(self, prompt: str, *, system_prompt: str = "",
                        temperature: float = 0.0,
                        max_tokens: int = 1024) -> Iterator[str]:
        result = self.generate(prompt, system_prompt=system_prompt,
                               temperature=temperature, max_tokens=max_tokens)
        if result.text:
            yield result.text

    def is_available(self) -> bool:
        return True


# ===================================================================
# 纯信号
# ===================================================================
class TestDetectGaps:
    def test_spurious_no_info_with_strong_keyword_hit(self) -> None:
        signals = detect_gaps(
            "客户说「防癌」这种话我们能不能讲？",
            "资料里没有这个信息。",
            ["防癌属于疾病治疗类表述，禁止使用。"],
            keyword_score_max=20.0,
        )
        assert [s.code for s in signals] == ["spurious_no_info"]

    def test_no_info_is_kept_when_evidence_is_not_strong(self) -> None:
        signals = detect_gaps(
            "量子能量水有什么功效？",
            "资料里没有这个信息。",
            ["一般产品说明。"],
            keyword_score_max=11.7,
        )
        assert signals == ()

    def test_medical_question_requires_escalation(self) -> None:
        signals = detect_gaps(
            "客户说吃了产品之后头晕，要求赔偿，怎么处理？",
            "记录并上报主管。",
            ["记录并上报主管。"],
        )
        assert [s.code for s in signals] == ["medical_escalation"]

    def test_medical_answer_with_doctor_is_not_flagged(self) -> None:
        signals = detect_gaps(
            "客户说吃了产品之后头晕，要求赔偿，怎么处理？",
            "建议客户先停用，并建议就医；记录并上报主管。",
            ["建议就医。上报主管。"],
        )
        assert signals == ()

    def test_claim_question_about_side_effects_does_not_trigger(self) -> None:
        signals = detect_gaps(
            "「纯天然，没有副作用」这种话能不能说？",
            "不能。这属于绝对化承诺。",
            ["不能使用「纯天然，没有副作用」。"],
        )
        assert signals == ()

    def test_case_question_mentioning_stop_medication_does_not_trigger(self) -> None:
        signals = detect_gaps(
            "检测表里涉及承诺停药时间的那一类包含哪些说法？",
            "涉及「给时间承诺」这一类。",
            ["给时间承诺：吃几个月就能停、一年左右、三个疗程。"],
        )
        assert signals == ()

    def test_numeric_question_without_number_is_flagged(self) -> None:
        signals = detect_gaps(
            "乳清蛋白粉一罐多少克、大约多少份？",
            "每罐分量很足，可以吃很久。",
            ["每罐 400g，约 20 份。"],
        )
        assert [s.code for s in signals] == ["numeric_missing"]

    def test_numeric_answer_with_number_is_not_flagged(self) -> None:
        signals = detect_gaps(
            "乳清蛋白粉一罐多少克、大约多少份？",
            "每罐 400g，约 20 份。",
            ["每罐 400g，约 20 份。"],
        )
        assert signals == ()

    def test_repair_prompt_contains_draft_and_checklist(self) -> None:
        prompt = build_repair_prompt(
            "【员工的问题】\n客户问防癌",
            "资料里没有这个信息。",
            (RepairSignal("spurious_no_info", "请基于参考资料回答。"),),
        )
        assert "【初稿】" in prompt
        assert "资料里没有这个信息。" in prompt
        assert "【必须补全】" in prompt
        assert "请基于参考资料回答。" in prompt


# ===================================================================
# 用例接线
# ===================================================================
def _build_uc(llm: LanguageModelPort, *, enabled: bool) -> AnswerQuestionUseCase:
    chunks = [_chunk("防癌属于疾病治疗类表述，禁止使用。")]
    return AnswerQuestionUseCase(
        graph=None,
        vector=FakeVectorSearchPort(chunks),
        keyword=_FixedKeyword(chunks),
        llm=llm,
        answer_repair_enabled=enabled,
    )


class TestRepairIntegration:
    def test_repair_used_when_signal_present(self) -> None:
        repaired = "不能讲。该类表述禁止使用，请改用目录原文。"
        llm = _SequencedLLM(["资料里没有这个信息。", repaired])
        uc = _build_uc(llm, enabled=True)

        result = uc.execute("防癌这种话能讲吗", vector_collection="training")

        assert result.answer == repaired
        assert result.meta["repair_used"] is True
        assert result.meta["repair_signals"] == ["spurious_no_info"]
        assert len(llm.prompts) == 2
        assert "【必须补全】" in llm.prompts[1]

    def test_repair_disabled_keeps_single_call(self) -> None:
        llm = _SequencedLLM(["资料里没有这个信息。", "不会用到"])
        uc = _build_uc(llm, enabled=False)

        result = uc.execute("防癌这种话能讲吗", vector_collection="training")

        assert result.answer == "资料里没有这个信息。"
        assert result.meta["repair_used"] is False
        assert len(llm.prompts) == 1

    def test_blocked_repair_keeps_draft(self) -> None:
        llm = _SequencedLLM([
            "资料里没有这个信息。",
            "这个产品可以治疗糖尿病。",
        ])
        uc = _build_uc(llm, enabled=True)

        result = uc.execute("防癌这种话能讲吗", vector_collection="training")

        assert result.answer == "资料里没有这个信息。"
        assert result.meta["repair_used"] is False
        assert len(llm.prompts) == 2

    def test_empty_repair_keeps_draft(self) -> None:
        llm = _SequencedLLM(["资料里没有这个信息。", ""])
        uc = _build_uc(llm, enabled=True)

        result = uc.execute("防癌这种话能讲吗", vector_collection="training")

        assert result.answer == "资料里没有这个信息。"
        assert result.meta["repair_used"] is False
