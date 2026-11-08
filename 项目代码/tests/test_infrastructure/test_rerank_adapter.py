"""BgeRerankerAdapter 的契约测试（不加载真模型）。

覆盖 D5 决定的四件事：
1. 返回 `RankedHit`、**不增不减不重复**（RetrieveUseCase 依赖这一点）
2. 按 `rerank_score` 降序，`original_rank` 记录重排前名次
3. 空候选 / 空查询 → 不加载模型、原序返回
4. 推理异常 → 抛 `RerankError`（由用例降级，不是静默吞掉）
"""
from __future__ import annotations

import pytest

from domain.value_objects import RankedHit
from infrastructure.exceptions import RerankError
from infrastructure.rerank.bge_reranker_adapter import BgeRerankerAdapter


class _FakeModel:
    """按文本里的数字给分的假 CrossEncoder。"""

    def __init__(self, *, boom: bool = False) -> None:
        self.boom = boom
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, *, batch_size, show_progress_bar):
        self.calls.append(list(pairs))
        if self.boom:
            raise RuntimeError("CUDA out of memory")
        return [float(t.split("#")[-1]) for _, t in pairs]


def _adapter(model: _FakeModel) -> BgeRerankerAdapter:
    a = BgeRerankerAdapter(model_dir="unused", slow_warn_sec=99)
    a._model = model                      # 绕过惰性加载
    return a


CANDIDATES = [("c1", "doc#0.1"), ("c2", "doc#0.9"), ("c3", "doc#0.5")]


class TestContract:
    def test_returns_ranked_hits_one_to_one(self) -> None:
        out = _adapter(_FakeModel()).rerank("q", CANDIDATES)
        assert all(isinstance(h, RankedHit) for h in out)
        assert sorted(h.chunk_id for h in out) == ["c1", "c2", "c3"]

    def test_sorted_by_score_desc_with_original_rank(self) -> None:
        out = _adapter(_FakeModel()).rerank("q", CANDIDATES)
        assert [h.chunk_id for h in out] == ["c2", "c3", "c1"]
        assert [h.original_rank for h in out] == [1, 2, 0]
        assert out[0].rerank_score == pytest.approx(0.9)

    def test_empty_candidates_returns_empty(self) -> None:
        assert _adapter(_FakeModel()).rerank("q", []) == []

    def test_empty_query_keeps_order_without_model(self) -> None:
        model = _FakeModel()
        out = _adapter(model).rerank("   ", CANDIDATES)
        assert [h.chunk_id for h in out] == ["c1", "c2", "c3"]
        assert model.calls == []          # 没有查询就不该跑模型


class TestFailure:
    def test_predict_error_becomes_rerank_error(self) -> None:
        with pytest.raises(RerankError):
            _adapter(_FakeModel(boom=True)).rerank("q", CANDIDATES)

    def test_model_load_error_becomes_rerank_error(self) -> None:
        """模型缺失/依赖缺失也必须抛 `RerankError`（用例据此降级，不 500）。"""
        a = BgeRerankerAdapter(model_dir="D:/does/not/exist", slow_warn_sec=99)
        with pytest.raises(RerankError):
            a.rerank("q", CANDIDATES)
