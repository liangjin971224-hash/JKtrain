"""重排端口的测试替身。

两种用途：
  · `FakeRerankPort`  —— 按「关键词命中数」打分，模拟 cross-encoder 的
    「问题-文档相关性」判断；同时记录调用入参，供断言「是否真的调用了」。
  · `ExplodingRerankPort` —— 一调就抛异常，专门证明**降级路径**真的生效
    （即：重排挂了，问答依然可用，而不是靠读代码断言）。
"""
from __future__ import annotations

from domain.ports import RerankPort
from domain.value_objects import FusedHit


class FakeRerankPort(RerankPort):
    """按「问题与文档的字符重叠」粗糙打分。

    ★ 故意写得比真模型简单得多：这里只验证**编排是否正确**
      （候选截断、top_k、降级、空集处理），
      不验证重排质量——那是真模型的事，用真实语料测。
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def rerank(
        self,
        query: str,
        candidates: list[FusedHit],
        *,
        top_k: int = 5,
    ) -> list[FusedHit]:
        self.calls.append({
            "query": query,
            "candidate_ids": [h.chunk_id for h in candidates],
            "top_k": top_k,
        })
        if not candidates or top_k <= 0:
            return []

        def overlap(hit: FusedHit) -> int:
            if hit.chunk is None:
                return -1                      # 无正文的排最后
            text = hit.chunk.text or ""
            return sum(1 for ch in set(query) if ch in text)

        ordered = sorted(candidates, key=lambda h: (-overlap(h), ))
        return ordered[:top_k]


class ExplodingRerankPort(RerankPort):
    """一调就抛异常。用于验证「重排失败不影响出答案」。"""

    def __init__(self, message: str = "rerank exploded") -> None:
        self.message = message
        self.called = 0

    def rerank(
        self,
        query: str,
        candidates: list[FusedHit],
        *,
        top_k: int = 5,
    ) -> list[FusedHit]:
        self.called += 1
        raise RuntimeError(self.message)
