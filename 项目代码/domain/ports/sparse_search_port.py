"""稀疏检索端口（bge-m3 lexical_weights → Milvus SPARSE_FLOAT_VECTOR）。

────────────────────────────────────────────────────────────────────
为什么与 `VectorSearchPort` 并列，而不是合并成一个端口
────────────────────────────────────────────────────────────────────
两者分数**不同量纲**：
    dense  = COSINE，有界 0~1，实测相关集中在 0.55~0.78
    sparse = IP，无上界，随查询词权重变化（实测 0.04~1.27）

`VectorSearchPort` 的契约里带 `min_vector_score` 语义（0.50 是余弦尺度）。
把稀疏塞进同一个端口，等于让一个阈值管两种分数——线上必然出现
「案例库全被卡掉」或「噪声全放过」。阈值必须按路分开，端口也就必须分开。

为什么不用现有的 BM25 关键词路代替：
    BM25 是 2-gram 字面匹配，bge-m3 稀疏是**学习出来的词权重**。
    实测（205 题、案例库 34 题）：三路 RRF（含 BM25）案例库 21/34，
    与纯 dense 完全相同；而 bge-m3 稀疏单独就有 25/34，
    并集可达 28/34。BM25 补不上表格型内容（要素 ID / 权重 / 正则）那一块。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from domain.value_objects import VectorHit


class SparseSearchPort(ABC):
    """词法匹配端口（学习式稀疏，非 BM25）。

    ★ 方法名必须是 `search_sparse` 而不是 `search`：
      同一个适配器类可以同时实现 `VectorSearchPort` 与 `SparseSearchPort`，
      若两者都叫 `search`，后实现的那一个会覆盖前者——
      实测踩过：稀疏路实际跑成了第二遍 dense 搜索（n_sparse=20 但结果不变）。
    """

    @abstractmethod
    def search_sparse(
        self,
        query: str,
        *,
        collection: str,
        top_k: int = 20,
        filter_expr: str | None = None,
        allowed_permissions: tuple[str, ...] | None = None,
    ) -> list[VectorHit]:
        """按词权重匹配知识块。

        ★ 权限过滤同样必须在检索时生效（与 `VectorSearchPort` 同一红线）。

        Args:
            query: 用户问题原文。
            collection: collection 名。
            top_k: 返回条数。
            filter_expr: Milvus 标量过滤表达式（原始表达式）。
            allowed_permissions: 允许的权限级别，None 表示不过滤。

        Returns:
            按稀疏内积（IP）降序的命中列表。`score` 是 **IP 分数**，
            与 dense 的余弦分数不可直接比较，调用方必须用各自的阈值。
        """
