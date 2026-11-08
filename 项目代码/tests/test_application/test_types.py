"""应用层基础类型测试。"""
from __future__ import annotations

from application.types import QuestionIntent, UseCaseResult


class TestUseCaseResult:
    def test_defaults(self) -> None:
        r = UseCaseResult(answer="正常回答")
        assert r.refused is False
        assert r.refusal_reason == ""
        assert r.cited_chunk_ids == ()

    def test_refused_semantics(self) -> None:
        """拒答与「正常回答里说不可以」是两件事，必须能区分。

        - refused=True ：召回到不了，系统主动拒答，此时 LLM 不应被调用
        - refused=False：正常回答，即使内容含「不建议」也不算拒答
        """
        refused = UseCaseResult(answer="暂无相关资料", refused=True,
                                refusal_reason="三路召回全空")
        normal = UseCaseResult(answer="不可以，建议遵医嘱")

        assert refused.refused is True
        assert normal.refused is False
        # 两者都可能含"不"，但语义完全不同
        assert refused.refusal_reason != ""

    def test_frozen(self) -> None:
        r = UseCaseResult(answer="a")
        try:
            r.answer = "b"        # type: ignore[misc]
        except (AttributeError, TypeError):
            pass
        else:
            raise AssertionError("UseCaseResult 应不可变")

    def test_with_meta_returns_new_instance(self) -> None:
        """追加元信息不能原地改（保持不可变语义）。"""
        r = UseCaseResult(answer="a")
        r2 = r.with_meta(intent="产品功效", elapsed_ms=120)

        assert r.meta == {}                       # 原对象未变
        assert r2.meta["intent"] == "产品功效"
        assert r2.answer == "a"                   # 其余字段保留

    def test_with_meta_merges(self) -> None:
        r = UseCaseResult(answer="a", meta={"x": 1})
        r2 = r.with_meta(y=2)
        assert r2.meta == {"x": 1, "y": 2}


class TestQuestionIntent:
    def test_safety_labels_flagged(self) -> None:
        """安全类意图必须被标记——这些查询要上调图谱权重。"""
        for label in ("禁忌咨询", "特殊人群", "要停药", "要诊断", "要处方", "超量服用"):
            assert QuestionIntent.of(label).is_safety_sensitive is True, label

    def test_normal_labels_not_flagged(self) -> None:
        for label in ("产品功效", "业务流程", "开放问答"):
            assert QuestionIntent.of(label).is_safety_sensitive is False, label

    def test_confidence_preserved(self) -> None:
        i = QuestionIntent.of("产品功效", confidence=0.92)
        assert i.confidence == 0.92

    def test_default_confidence(self) -> None:
        assert QuestionIntent.of("产品功效").confidence == 0.0
