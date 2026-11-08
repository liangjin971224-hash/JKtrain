"""重排序端口的测试替身。

★ 关键纪律：Fake 必须遵守真实现对外承诺的契约——
   返回列表与输入候选**不增不减不重复**（RetrieveUseCase 依赖这一点）。
   如果 Fake 随手丢掉候选，用它测出来的「rerank 异常降级」就是假绿。
"""
from __future__ import annotations

from domain.ports import RerankPort
from domain.value_objects import RankedHit


class FakeRerankPort(RerankPort):
    """可控排序的替身。

    Args:
        order: "passthrough" 保持输入序；"reverse" 完全倒序。
        preferred_chunk_id: 给定则把该候选排最前（其余按输入序），
            用于验证「重排真的改变了 fused 顺序」。
        raise_on_call: True 时模拟模型故障，抛 RuntimeError
            （用例层应捕获并降级为原序）。
    """

    def __init__(
        self,
        *,
        order: str = "passthrough",
        preferred_chunk_id: str | None = None,
        raise_on_call: bool = False,
    ) -> None:
        self._order = order
        self._preferred = preferred_chunk_id
        self._raise = raise_on_call
        #: 每次 rerank 收到的 (query, 候选id列表)，供测试断言「送了什么进模型」
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[RankedHit]:
        ids = [cid for cid, _ in candidates]
        self.calls.append((query, list(ids)))

        if self._raise:
            raise RuntimeError("fake rerank 故障")

        if self._preferred is not None and self._preferred in ids:
            ids = ([self._preferred]
                   + [i for i in ids if i != self._preferred])
        elif self._order == "reverse":
            ids = list(reversed(ids))

        return [
            RankedHit(chunk_id=cid, rerank_score=float(len(ids) - rank),
                      original_rank=ids.index(cid))
            for rank, cid in enumerate(ids)
        ]


class ExplodingRerankPort(FakeRerankPort):
    """必然抛错的替身（旧接口名，保留以兼容既有测试）。"""

    def __init__(self) -> None:
        super().__init__(raise_on_call=True)
