"""检索结果值对象。

Meta 类型覆盖 4 个 collection 的元数据字段
（与 `08_数据入库/milvus/build_collections.py` 的 schema 对齐）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class PermissionLevel(StrEnum):
    """知识权限级别（FR-050）。

    学员只能看 public / internal；
    restricted 仅主管可见（案例、题库、参考答案）。
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"

    @classmethod
    def trainee_allowed(cls) -> tuple[PermissionLevel, ...]:
        """学员可见的级别。"""
        return (cls.PUBLIC, cls.INTERNAL)

    @classmethod
    def manager_allowed(cls) -> tuple[PermissionLevel, ...]:
        """主管可见的级别。"""
        return (cls.PUBLIC, cls.INTERNAL, cls.RESTRICTED)


class ContentCategory(StrEnum):
    """知识块类型，与 Milvus 的 `category` 字段取值一致。"""

    TRAINING = "training"
    CASE = "case"
    QUESTION = "question"
    SELF_TEST = "self_test"


@dataclass(frozen=True, slots=True)
class ChunkMeta:
    """一个知识块的元数据。

    不含 `vector`——向量只在基础设施层内部流转，不进入领域。
    """

    chunk_id: str
    doc_id: str
    text: str
    title: str = ""
    section: str = ""
    category: ContentCategory | str = ContentCategory.TRAINING
    permission_level: PermissionLevel | str = PermissionLevel.INTERNAL
    review_status: str = "approved"
    data_origin: str = "real"
    source_file: str = ""
    chunk_index: int = 0
    # 题库专用（其余 collection 为空）
    knowledge_point: str = ""
    question_type: str = ""
    difficulty: int = 0


@dataclass(frozen=True, slots=True)
class VectorHit:
    """一次向量的命中结果。

    `score` 是余弦相似度（0–1，越大越相关），
    注意：它是**检索分数**，不是评阅分数——摘要卡里不允许出现任何分数。
    """

    chunk: ChunkMeta
    score: float


@dataclass(frozen=True, slots=True)
class KeywordHit:
    """关键词检索的命中结果。"""

    chunk: ChunkMeta
    score: float


@dataclass(frozen=True, slots=True)
class FusedHit:
    """RRF 融合后的结果。"""

    chunk_id: str
    fused_score: float
    chunk: ChunkMeta | None = None
    # 该结果来自哪几路召回，便于排查"为什么这条排前面"
    sources: tuple[str, ...] = field(default_factory=tuple)
