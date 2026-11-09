"""口语词检索扩展（h2-03 的「三高」）。"""

from __future__ import annotations

from application.services.query_expansion import expand_query
from application.use_cases.answer_question import AnswerQuestionUseCase
from domain.value_objects import ChunkMeta, PermissionLevel
from tests.fakes import (
    FakeKeywordSearchPort,
    FakeLanguageModelPort,
    FakeVectorSearchPort,
)


def _chunk() -> ChunkMeta:
    return ChunkMeta(
        chunk_id="D006_c999",
        doc_id="D006",
        text="高血压 高血糖 高血脂 对应有助于维持血压/血糖/血脂健康水平。",
        title="三高对应说法",
        permission_level=PermissionLevel.PUBLIC,
    )


class _RecordingVector(FakeVectorSearchPort):
    def __init__(self) -> None:
        super().__init__([_chunk()])
        self.last_query = ""

    def search(self, query, **kwargs):
        self.last_query = query
        return super().search(query, **kwargs)


class TestExpandQuery:
    def test_three_highs_is_expanded(self) -> None:
        expanded = expand_query("有没有适合三高的？")
        assert expanded.startswith("有没有适合三高的？")
        for word in ("高血压", "高血糖", "高血脂", "血压", "血糖", "血脂"):
            assert word in expanded

    def test_plain_query_is_unchanged(self) -> None:
        assert expand_query("鱼油有什么功能") == "鱼油有什么功能"

    def test_already_formal_query_is_unchanged(self) -> None:
        text = "高血压 高血糖 高血脂 血压 血糖 血脂 有哪些产品"
        assert expand_query(text) == text


class TestAnswerUsesExpandedQueryForRetrievalOnly:
    def test_retrieval_query_contains_formal_terms_but_prompt_keeps_original(self) -> None:
        vector = _RecordingVector()
        llm = FakeLanguageModelPort(
            reply="没有直接标「三高」的产品；可按有助于维持血压/血糖/血脂健康水平介绍。"
        )
        uc = AnswerQuestionUseCase(
            graph=None,
            vector=vector,
            keyword=FakeKeywordSearchPort([]),
            llm=llm,
            min_vector_score=0.1,
        )

        result = uc.execute("有没有适合三高的？", vector_collection="training")

        assert result.refused is False
        assert "高血压" in vector.last_query
        assert "【员工的问题】\n有没有适合三高的？" in llm.prompts[0]
