"""关键词检索端口的测试替身。"""
from __future__ import annotations

from domain.ports import KeywordSearchPort
from domain.value_objects import ChunkMeta, KeywordHit


class FakeKeywordSearchPort(KeywordSearchPort):
    """内存倒排。按子串命中计数打分，够验证融合逻辑。"""

    def __init__(
        self,
        chunks: list[ChunkMeta] | None = None,
        *,
        fail_on_call: bool = False,
    ) -> None:
        self._chunks = chunks or []
        self._fail_on_call = fail_on_call
        self.calls: list[str] = []

    def _guard(self, name: str) -> None:
        self.calls.append(name)
        if self._fail_on_call:
            raise AssertionError(f"FakeKeywordSearchPort 不应被调用: {name}")

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        allowed_permissions: tuple[str, ...] | None = None,
    ) -> list[KeywordHit]:
        self._guard("search")

        pool = self._chunks
        if allowed_permissions is not None:
            allowed = {str(p) for p in allowed_permissions}
            pool = [c for c in pool if str(c.permission_level) in allowed]

        # 粗略分词：连续中文按 2 字滑窗，够模拟"精确匹配"的排序特征
        tokens = {query[i:i + 2] for i in range(max(len(query) - 1, 1))}
        out: list[KeywordHit] = []
        for c in pool:
            hit = sum(1 for t in tokens if t and t in c.text)
            if hit:
                out.append(KeywordHit(chunk=c, score=float(hit)))

        out.sort(key=lambda h: h.score, reverse=True)
        return out[:top_k]

    def build_index(self, chunks: list[object]) -> int:
        self._guard("build_index")
        self._chunks = [c for c in chunks if isinstance(c, ChunkMeta)]
        return len(self._chunks)
