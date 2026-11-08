"""稀疏检索端口的测试替身。

★ 与 `FakeVectorSearchPort` 的两点关键差异（必须保持一致，否则测出假绿）：
    1. **分数是 IP 尺度**，不是余弦——所以它故意给 0~3 这类
       "大于 1、量纲不同"的值，用来证明调用方没有拿余弦阈值去卡它；
    2. 权限过滤同样在检索阶段生效。
"""
from __future__ import annotations

from domain.ports import SparseSearchPort
from domain.value_objects import ChunkMeta, VectorHit


class FakeSparseSearchPort(SparseSearchPort):
    """内存稀疏库。按指定顺序返回命中的块。"""

    def __init__(
        self,
        chunks: list[ChunkMeta] | None = None,
        *,
        score: float = 1.0,
        fail_on_call: bool = False,
    ) -> None:
        self._chunks = chunks or []
        self._score = score
        self._fail_on_call = fail_on_call
        self.calls: list[str] = []
        self.last_allowed_permissions = None
        self.last_collection: str | None = None

    def search_sparse(
        self,
        query: str,
        *,
        collection: str = "",
        top_k: int = 20,
        filter_expr: str | None = None,
        allowed_permissions: tuple[str, ...] | None = None,
    ) -> list[VectorHit]:
        self.calls.append("search_sparse")
        self.last_allowed_permissions = allowed_permissions
        self.last_collection = collection
        if self._fail_on_call:
            raise AssertionError("FakeSparseSearchPort 不应被调用")

        pool = self._chunks
        if allowed_permissions is not None:
            allowed = {str(p) for p in allowed_permissions}
            pool = [c for c in pool if str(c.permission_level) in allowed]

        # 分数递减，保持"按 IP 降序"的契约
        return [
            VectorHit(chunk=c, score=max(self._score - i * 0.01, 0.0))
            for i, c in enumerate(pool[:top_k])
        ]
