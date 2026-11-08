"""向量检索端口。

数据源：Milvus 4 个 collection（training 293 / case 117 / question 140 / self_test 30）
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from domain.value_objects import ChunkMeta, VectorHit


class VectorSearchPort(ABC):
    """语义检索端口。"""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        collection: str,
        top_k: int = 20,
        filter_expr: str | None = None,
        allowed_permissions: tuple[str, ...] | None = None,
    ) -> list[VectorHit]:
        """语义检索知识块。

        ★ 权限过滤必须在**检索时**生效，不能检索后过滤。
          原因：top-20 里若有 18 条受限内容，过滤后只剩 2 条，召回质量崩溃。

        Args:
            query: 用户问题原文。
            collection: collection 名（走配置，不硬编码）。
            top_k: 返回条数。
            filter_expr: Milvus 标量过滤表达式（原始表达式，供特殊场景使用）。
            allowed_permissions: 允许的权限级别。传 None 表示不过滤（仅限内部调用）。
                学员传 ("public", "internal")，主管传含 "restricted"。

        Returns:
            按相似度降序的命中列表。无结果返回空列表。
        """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。

        Returns:
            与输入等长的向量列表，每维必须等于 collection 的维度（1024）。

        Raises:
            EmbeddingDimensionError: 维度与配置不符。
        """

    @abstractmethod
    def get_chunks_by_ids(
        self, chunk_ids: list[str], *, collection: str
    ) -> list[ChunkMeta]:
        """按 chunk_id 批量取元数据（融合后回填用）。

        融合阶段只需要 id，最终展示才需要正文，这里按需回填。
        """
