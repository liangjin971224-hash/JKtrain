"""向量检索端口的测试替身。

★ 关键：权限过滤在替身里也必须真的生效。
   如果替身忽略 allowed_permissions，那用它测出来的
   「学员搜不到受限内容」就是假绿——这正是 P8 最容易自欺的地方。
"""
from __future__ import annotations

from domain.ports import VectorSearchPort
from domain.value_objects import ChunkMeta, VectorHit


class FakeVectorSearchPort(VectorSearchPort):
    """内存向量库。用字符重合度模拟相似度，不做真实向量计算。"""

    def __init__(
        self,
        chunks: list[ChunkMeta] | None = None,
        *,
        dim: int = 1024,
        fail_on_call: bool = False,
    ) -> None:
        self._chunks = chunks or []
        self._dim = dim
        self._fail_on_call = fail_on_call
        self.calls: list[str] = []
        #: 最近一次 search 收到的权限参数。
        #: ★ 用来验证「权限在检索阶段生效」而不是「检索后过滤」——
        #:   两种做法返回值一样，只有传参能区分实现方式。
        self.last_allowed_permissions = None
        #: 最近一次 search 收到的 filter_expr
        self.last_filter_expr: str | None = None

    def _guard(self, name: str) -> None:
        self.calls.append(name)
        if self._fail_on_call:
            raise AssertionError(f"FakeVectorSearchPort 不应被调用: {name}")

    def search(
        self,
        query: str,
        *,
        collection: str = "",
        top_k: int = 20,
        filter_expr: str | None = None,
        allowed_permissions: tuple[str, ...] | None = None,
    ) -> list[VectorHit]:
        self._guard("search")
        self.last_allowed_permissions = allowed_permissions
        self.last_filter_expr = filter_expr

        # ★ 权限过滤：与真实现一致，检索时就过滤
        pool = self._chunks
        if allowed_permissions is not None:
            allowed = {str(p) for p in allowed_permissions}
            pool = [c for c in pool if str(c.permission_level) in allowed]

        q = set(query)
        scored: list[VectorHit] = []
        for c in pool:
            hits = len(q & set(c.text))
            if hits == 0:
                continue
            scored.append(VectorHit(chunk=c, score=hits / max(len(q), 1)))

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """返回固定维度的假向量（不需要模型）。"""
        self._guard("embed")
        return [[0.0] * self._dim for _ in texts]

    def get_chunks_by_ids(
        self, chunk_ids: list[str], *, collection: str = ""
    ) -> list[ChunkMeta]:
        self._guard("get_chunks_by_ids")
        wanted = set(chunk_ids)
        return [c for c in self._chunks if c.chunk_id in wanted]
