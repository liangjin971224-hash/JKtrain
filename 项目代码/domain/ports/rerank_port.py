"""重排序端口。

职责：对「已经过权限过滤与阈值判空」的候选块，按 query 相关性重新排序。
不参与判空、不参与权限过滤——那两道防线在检索层（见 TDD_检索质量优化 §4.3）。

为什么是独立端口而不是 VectorSearchPort 的方法：
    rerank 是可选能力（模型缺失/关闭时问答必须照常工作），
    独立端口让「有没有 rerank」成为装配决策，用例代码零改动。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from domain.value_objects import RankedHit


class RerankPort(ABC):
    """CrossEncoder 重排序端口。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[RankedHit]:
        """按相关性对候选重排。

        Args:
            query: 用户问题原文。
            candidates: (chunk_id, chunk_text) 列表。调用方保证已过权限过滤；
                chunk_id 顺序即重排前的名次（original_rank 由此推导）。

        Returns:
            按相关性**降序**的 RankedHit 列表，chunk_id 与输入一一对应
            （不增不减不重复）。空输入返回空列表。

        Raises:
            infrastructure.exceptions.RerankError: 模型推理失败。
                调用方（RetrieveUseCase）捕获后降级为原排序——
                rerank 挂了问答不能挂。
        """
