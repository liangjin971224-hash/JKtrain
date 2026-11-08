"""关键词检索端口。

容易被漏掉但必不可少的一路：
图谱和向量都做不了「XX 牌 2000IU」这种编号/型号类精确匹配。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from domain.value_objects import KeywordHit


class KeywordSearchPort(ABC):
    """精确匹配检索端口（BM25 / 倒排索引）。"""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        allowed_permissions: tuple[str, ...] | None = None,
    ) -> list[KeywordHit]:
        """关键词检索。

        擅长：产品名、批准文号、规格型号等**字面精确**的查询。
        不擅长：口语化表达（那是向量的事）。

        Args:
            query: 用户问题原文。
            top_k: 返回条数。
            allowed_permissions: 允许的权限级别，语义同 VectorSearchPort。

        Returns:
            按 BM25 分数降序的命中列表。
        """

    @abstractmethod
    def build_index(self, chunks: list[object]) -> int:
        """构建/重建索引。

        Args:
            chunks: 待索引的知识块（由调用方从向量库或数据源取出）。

        Returns:
            实际索引的条数。
        """
