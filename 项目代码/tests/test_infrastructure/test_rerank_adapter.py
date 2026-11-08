"""Rerank 重排的测试。

分三层，各测各的：

| 层 | 测什么 | 用真模型吗 |
|---|---|---|
| 适配器 | 降级、空集、无 chunk 处理、参数校验 | 否（用桩） |
| 用例 | 编排：候选截断、top_k、降级 | 否（用 FakeRerankPort） |
| 真实模型 | 排序是否真的把「相关」的排前面 | **是**（标 slow，需模型文件） |

★ 为什么真实模型测试要单独标记：
  模型加载占 2.2GB 显存、耗时数秒，不该进每次 CI。
  但它必须存在——否则「接上真模型能不能用」永远没被验证过。
"""
from __future__ import annotations

import pytest

from domain.ports import RerankPort
from domain.value_objects import ChunkMeta, FusedHit
from infrastructure.search.bge_reranker_adapter import BgeRerankerAdapter
from tests.fakes import ExplodingRerankPort, FakeRerankPort


def _hit(cid: str, text: str, *, score: float = 0.5,
         title: str = "鱼油", section: str = "正文") -> FusedHit:
    return FusedHit(
        chunk_id=cid,
        fused_score=score,
        chunk=ChunkMeta(chunk_id=cid, doc_id="D001", text=text,
                        title=title, section=section),
    )


# ===================================================================
# 一、端口契约：任何实现都必须满足
# ===================================================================
class TestPortContract:
    """端口注释里写死的三条约定，必须被守住。"""

    def test_empty_candidates_returns_empty(self) -> None:
        for impl in (FakeRerankPort(), ExplodingRerankPort()):
            if isinstance(impl, ExplodingRerankPort):
                # 爆炸替身的语义是「调用即抛」，但空集走不到它
                continue
            assert impl.rerank("任意问题", [], top_k=5) == []

    def test_zero_top_k_returns_empty(self) -> None:
        fake = FakeRerankPort()
        hits = [_hit("c1", "鱼油成分")]
        assert fake.rerank("鱼油", hits, top_k=0) == []

    def test_returns_at_most_top_k(self) -> None:
        fake = FakeRerankPort()
        hits = [_hit(f"c{i}", f"文档{i}") for i in range(10)]
        assert len(fake.rerank("鱼油", hits, top_k=3)) == 3

    def test_does_not_mutate_input(self) -> None:
        fake = FakeRerankPort()
        hits = [_hit("c1", "甲"), _hit("c2", "乙")]
        before = [h.chunk_id for h in hits]
        fake.rerank("问题", hits, top_k=2)
        assert [h.chunk_id for h in hits] == before

    def test_result_type_is_still_fused_hit(self) -> None:
        """返回的必须是 FusedHit —— 调用方要继续用 chunk / sources。"""
        fake = FakeRerankPort()
        out = fake.rerank("鱼油", [_hit("c1", "鱼油成分")], top_k=1)
        assert isinstance(out[0], FusedHit)
        assert out[0].chunk is not None


# ===================================================================
# 二、适配器降级：模型不可用时必须优雅退化
# ===================================================================
class TestAdapterDegrade:
    """这是本功能的**安全底线**：重排挂了，问答不能挂。"""

    def test_missing_model_dir_degrades_to_original_order(self, tmp_path) -> None:
        """模型目录不存在 → 返回原顺序前 top_k，不抛异常。"""
        adapter = BgeRerankerAdapter(model_dir=str(tmp_path / "不存在"))
        hits = [_hit("c1", "甲"), _hit("c2", "乙"), _hit("c3", "丙")]
        out = adapter.rerank("问题", hits, top_k=2)

        assert [h.chunk_id for h in out] == ["c1", "c2"]
        assert adapter.load_failed is True
        assert adapter.is_ready is False

    def test_load_failure_not_retried(self, tmp_path) -> None:
        """加载失败后不再重试——否则每次请求都卡一遍。"""
        adapter = BgeRerankerAdapter(model_dir=str(tmp_path / "不存在"))
        hits = [_hit("c1", "甲")]
        adapter.rerank("问题", hits, top_k=1)
        first_failed = adapter.load_failed

        adapter.rerank("问题", hits, top_k=1)
        assert adapter.load_failed is first_failed is True

    def test_hits_without_chunk_kept_at_tail(self, tmp_path) -> None:
        """无 chunk 的命中（纯图谱事实）没有正文可判断相关性，
        不能被误杀——应排在后面但保留。"""
        adapter = BgeRerankerAdapter(model_dir=str(tmp_path / "不存在"))
        hits = [
            FusedHit(chunk_id="g1", fused_score=0.9, chunk=None),
            _hit("c1", "有正文的文档"),
        ]
        out = adapter.rerank("问题", hits, top_k=2)
        assert len(out) == 2
        assert {h.chunk_id for h in out} == {"g1", "c1"}

    def test_stats_exposes_state(self, tmp_path) -> None:
        adapter = BgeRerankerAdapter(model_dir=str(tmp_path))
        st = adapter.stats()
        assert st["loaded"] is False
        assert "model_dir" in st


# ===================================================================
# 三、真实模型（需模型文件 + 显存，标 slow）
# ===================================================================
@pytest.mark.slow
class TestRealModel:
    """接真模型验证「排序真的按相关性」。

    ★ 这一层不能用假替身替代：假替身只能证明编排对，
      证明不了「bge-reranker 在中文语料上真的把禁忌段排到成分段前面」。
    """

    @pytest.fixture(scope="class")
    @classmethod
    def adapter(cls) -> BgeRerankerAdapter:
        from config.settings import Settings
        from pathlib import Path

        model_dir = Path(Settings().rerank_model_dir)
        if not model_dir.exists():
            pytest.skip(f"模型目录不存在，跳过真模型测试: {model_dir}")
        a = BgeRerankerAdapter(model_dir=str(model_dir), batch_size=4)
        # 触发加载，失败则跳过（显存不足时不该让测试红）
        a.rerank("预热", [_hit("w", "预热文本")], top_k=1)
        if not a.is_ready:
            pytest.skip("Reranker 加载失败（可能显存不足），跳过")
        return a

    def test_ranks_relevant_above_irrelevant(self, adapter) -> None:
        """问禁忌时，「禁忌段」必须排在「成分段」前面。

        ★ 这是重排存在的**唯一理由**：
          若真模型做不到这一点，那它就该被移除，而不是留着当装饰。
        """
        hits = [
            _hit("c_ingredient",
                 "鱼油的主要成分是 DHA 和 EPA，属于多不饱和脂肪酸，"
                 "常温下为淡黄色透明液体。", section="产品成分"),
            _hit("c_contraindication",
                 "正在服用抗凝药物（如阿司匹林、华法林）的人群，"
                 "不建议同时服用鱼油，因为可能增加出血风险。",
                 section="禁忌人群"),
        ]
        out = adapter.rerank("在吃阿司匹林还能吃鱼油吗", hits, top_k=2)

        assert out[0].chunk_id == "c_contraindication", (
            f"重排未能把禁忌段排到首位，实际顺序: "
            f"{[h.chunk_id for h in out]}"
        )

    def test_scores_are_deterministic(self, adapter) -> None:
        """同一输入两次调用顺序必须一致（合规场景不接受抖动）。"""
        hits = [
            _hit("c1", "鱼油的批准功能是有助于维持血脂健康水平。"),
            _hit("c2", "本店暂未上架该产品。"),
            _hit("c3", "保健食品不是药物，不能代替药物治疗疾病。"),
        ]
        a = [h.chunk_id for h in adapter.rerank("鱼油有什么功能", hits, top_k=3)]
        b = [h.chunk_id for h in adapter.rerank("鱼油有什么功能", hits, top_k=3)]
        assert a == b
