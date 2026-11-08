"""BGE Reranker 适配器（cross-encoder 重排）。

实现 `RerankPort`，加载 `bge-reranker-v2-m3`。

────────────────────────────────────────────────────────────────────
设计约束（违反即不可用）
────────────────────────────────────────────────────────────────────
1. **模型惰性加载**。`CrossEncoder(...)` 会占约 2.2GB 显存，
   而 bge-m3 嵌入模型已占约 2.5GB（8GB 卡）。若在模块导入时就加载，
   会导致：启动即吃满显存 → 后续任何 CUDA 分配都 OOM。
   所以**首次 `rerank()` 才初始化**。

2. **失败必须降级，不能抛到调用方**。重排是「锦上添花」：
   模型未就绪（文件缺失 / 显存不足）时，应原样返回前 top_k 条，
   让系统按原 RRF 顺序工作，**而不是整个问答失败**。
   这一点与 LLM 通道相反——LLM 挂了必须报错（不能给假答案），
   但重排挂了只是排序不最优，结果依然可用。

3. **batch 控制**。CrossEncoder 一次把 20 条候选全部送入，
   序列长度 = query + doc，可能很长。用 batch_size 限制峰值显存。

4. **不重新排序空集**。空候选直接返回空，不加载模型
   （避免「用户查了个生僻问题」也把 2.2GB 显存吃掉）。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from domain.ports import RerankPort
from domain.value_objects import FusedHit

logger = logging.getLogger(__name__)

#: 单条文档送入 cross-encoder 时截断的字符数。
#: 培训文档的 chunk 多在 300–800 字，取 512 覆盖绝大多数，
#: 同时把「query + doc」的总长度压在模型可承受范围内。
_MAX_DOC_CHARS = 512


class BgeRerankerAdapter(RerankPort):
    r"""用 bge-reranker-v2-m3 做候选重排。

    用法::

        r = BgeRerankerAdapter(model_dir=r"D:\...\bge-reranker-v2-m3")
        top5 = r.rerank(question, fused_hits[:20], top_k=5)
    """

    def __init__(
        self,
        *,
        model_dir: str,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        """
        Args:
            model_dir: 模型目录（本地路径，不走联网）。
            device: "cuda" / "cpu"；None 表示自动（有卡用卡）。
            batch_size: 前向批大小，控制峰值显存。
            max_length: 模型侧最大序列长度。
        """
        self._model_dir = model_dir
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length

        self._model: Any = None
        self._load_failed = False
        #: 惰性加载 + 线程安全（并发请求时避免重复加载两份模型）
        self._lock = threading.Lock()

    # ---------------- 内部 ----------------
    def _ensure_model(self) -> Any:
        """惰性加载模型。失败一次后不再重试（避免每次请求都卡一遍）。"""
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None

        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_failed:
                return None

            path = Path(self._model_dir)
            if not path.exists():
                logger.warning(
                    "Reranker 模型目录不存在，重排降级为原始顺序: %s", path)
                self._load_failed = True
                return None

            try:
                from sentence_transformers import CrossEncoder

                device = self._device
                if device is None:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                self._model = CrossEncoder(
                    str(path),
                    device=device,
                    max_length=self._max_length,
                )
                logger.info("Reranker 已加载: %s (device=%s)", path.name, device)
            except Exception as exc:                       # noqa: BLE001
                logger.warning("Reranker 加载失败，重排降级: %s", exc)
                self._load_failed = True
                return None

        return self._model

    # ---------------- 端口实现 ----------------
    def rerank(
        self,
        query: str,
        candidates: list[FusedHit],
        *,
        top_k: int = 5,
    ) -> list[FusedHit]:
        """重排并截断。失败时降级为原顺序。"""
        if not candidates or top_k <= 0:
            return []

        model = self._ensure_model()
        if model is None:
            # 降级：保持 RRF 原顺序
            return candidates[:top_k]

        # 只对「有 chunk」的候选打分；无 chunk 的（纯图谱命中）保持原位。
        # 这类命中没有正文可判断相关性，硬塞给 cross-encoder 只会得 0 分被误杀。
        scored: list[tuple[float, int, FusedHit]] = []
        pairs: list[tuple[str, str]] = []
        pair_index: list[int] = []

        for idx, hit in enumerate(candidates):
            if hit.chunk is None:
                continue
            pairs.append((query, (hit.chunk.text or "")[:_MAX_DOC_CHARS]))
            pair_index.append(idx)

        if not pairs:
            return candidates[:top_k]

        try:
            scores = model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:                           # noqa: BLE001
            # ★ 运行期失败也降级：用户不该因为重排出错而拿不到答案
            logger.warning("Reranker 推理失败，降级为原始顺序: %s", exc)
            return candidates[:top_k]

        for pos, idx in enumerate(pair_index):
            scored.append((float(scores[pos]), idx, candidates[idx]))

        # 有正文的按分数降序；同分时用原 RRF 位置做稳定排序（保证结果确定）
        scored.sort(key=lambda x: (-x[0], x[1]))
        merged = [hit for _, _, hit in scored]

        # 把无正文的候选接在后面（它们没被重排，但也不该丢）
        no_chunk = [h for h in candidates if h.chunk is None]
        merged.extend(no_chunk)

        return merged[:top_k]

    # ---------------- 可观测 ----------------
    @property
    def is_ready(self) -> bool:
        """模型是否已加载（供健康检查 / 自检用，不触发加载）。"""
        return self._model is not None

    @property
    def load_failed(self) -> bool:
        return self._load_failed

    def stats(self) -> dict[str, Any]:
        return {
            "model_dir": self._model_dir,
            "loaded": self._model is not None,
            "load_failed": self._load_failed,
            "batch_size": self._batch_size,
            "max_length": self._max_length,
        }
