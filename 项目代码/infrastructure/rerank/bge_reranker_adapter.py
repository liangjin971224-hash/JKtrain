"""bge-reranker CrossEncoder 适配器。

实现 `RerankPort`。模型：BAAI/bge-reranker-v2-m3（~2.2GB，与 bge-m3 同家族）。

────────────────────────────────────────────────────────────────────
设计约束（与 MilvusSearchAdapter 同规矩）
────────────────────────────────────────────────────────────────────
1. **惰性加载**：`import sentence_transformers` + 模型加载要数秒，
   不能在模块导入或 __init__ 里做。首次 `rerank()` 才初始化。
2. **GPU 失败回退 CPU**：本机无 N 卡时走 CPU；20 条候选约几百毫秒，
   问答总延迟可接受（实测见 train/eval_runs/ 的 rerank 延迟记录）。
3. **失败抛 RerankError**：由 RetrieveUseCase 捕获并降级为原排序——
   rerank 挂了问答不能挂。
4. 离线模式：TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE，与向量路一致，
   避免启动时联网检查。
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from domain.ports import RerankPort
from domain.value_objects import RankedHit
from infrastructure.exceptions import RerankError

logger = logging.getLogger(__name__)

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


class BgeRerankerAdapter(RerankPort):
    """bge-reranker-v2-m3 重排序实现。"""

    def __init__(
        self,
        *,
        model_dir: str,
        max_length: int = 512,
        batch_size: int = 16,
        slow_warn_sec: float = 3.0,
    ) -> None:
        """
        Args:
            model_dir: 模型目录（settings.rerank_model_dir）。
            max_length: query+chunk 拼接后的最大 token 数。
                512 覆盖绝大多数切块；CPU 下再长会明显变慢。
            batch_size: CrossEncoder 推理批大小。
            slow_warn_sec: 单次调用超过该值记 warning（软性观测，
                不做硬超时——跨平台硬超时复杂度高，v1 不做）。
        """
        self._model_dir = model_dir
        self._max_length = max_length
        self._batch_size = batch_size
        self._slow_warn_sec = slow_warn_sec
        self._model: Any = None          # 惰性加载，见 _ensure_model

    def _ensure_model(self) -> Any:
        """惰性加载 CrossEncoder。失败回退 CPU 的逻辑与向量路一致。"""
        if self._model is not None:
            return self._model

        from sentence_transformers import CrossEncoder

        logger.info("首次加载 reranker: %s", self._model_dir)
        try:
            model = CrossEncoder(self._model_dir, max_length=self._max_length,
                                 device="cuda")
        except Exception as exc:              # noqa: BLE001 无卡/显存不足 → CPU
            logger.warning(
                "reranker GPU 加载失败（%s），回退 CPU。", exc)
            model = CrossEncoder(self._model_dir, max_length=self._max_length,
                                 device="cpu")
        self._model = model
        return model

    # ---------------- 端口实现 ----------------
    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[RankedHit]:
        if not candidates:
            return []
        q = (query or "").strip()
        if not q:
            # 没有查询就没有排序依据：保持原序返回（语义上的「无操作」）
            return [RankedHit(chunk_id=cid, rerank_score=0.0,
                              original_rank=i)
                    for i, (cid, _) in enumerate(candidates)]

        model = self._ensure_model()
        t0 = time.perf_counter()
        try:
            scores = model.predict(
                [(q, text) for _, text in candidates],
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:              # noqa: BLE001 统一转 RerankError
            raise RerankError(f"rerank 推理失败: {type(exc).__name__}: {exc}") from exc

        elapsed = time.perf_counter() - t0
        if elapsed > self._slow_warn_sec:
            logger.warning("rerank 耗时 %.2fs（>%.1fs，候选 %d 条）",
                           elapsed, self._slow_warn_sec, len(candidates))

        order = sorted(range(len(candidates)),
                       key=lambda i: -float(scores[i]))
        out = [
            RankedHit(
                chunk_id=candidates[i][0],
                rerank_score=float(scores[i]),
                original_rank=i,
            )
            for i in order
        ]
        logger.debug("rerank 完成: %d 候选, %.0fms, top1=%s(%.3f, 原 rank=%d)",
                     len(candidates), elapsed * 1000,
                     out[0].chunk_id, out[0].rerank_score, out[0].original_rank)
        return out
