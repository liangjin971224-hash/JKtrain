"""运行指标（D3）：进程内计数 + 最近延迟窗口。

★ 为什么放内存：
  演示 / 单实例 PoC，指标只需「最近一段」；接 Prometheus 时再换适配器。
  重启清零是可接受的，且不会因为指标写失败影响问答主链路。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from statistics import mean

#: 延迟窗口大小（只保留最近 N 次，算 p50/p95）。
DEFAULT_WINDOW = 200


class OpsMetrics:
    """线程安全的运行指标。

    计数器只有四个，刻意保持最小：
      answer_total / refused / guardrail_blocked / empty_answer
    足以回答「有没有在拒答、护栏是不是误拦、有没有空答案」。
    """

    def __init__(self, *, window: int = DEFAULT_WINDOW,
                 started_at: float | None = None) -> None:
        self._lock = threading.Lock()
        self._counters = {
            "answer_total": 0,
            "refused": 0,
            "guardrail_blocked": 0,
            "empty_answer": 0,
        }
        self._latencies: deque[float] = deque(maxlen=max(1, int(window)))
        self._last_retrieval_ms: float | None = None
        self._started_at = started_at if started_at is not None else time.time()

    # ---------------- 记录 ----------------
    def record_answer(
        self, *, latency_ms: float, refused: bool,
        blocked: bool, empty: bool,
    ) -> None:
        with self._lock:
            self._counters["answer_total"] += 1
            if refused:
                self._counters["refused"] += 1
            if blocked:
                self._counters["guardrail_blocked"] += 1
            if empty:
                self._counters["empty_answer"] += 1
            self._latencies.append(max(float(latency_ms), 0.0))

    def record_retrieval(self, elapsed_ms: float) -> None:
        with self._lock:
            self._last_retrieval_ms = max(float(elapsed_ms), 0.0)

    # ---------------- 快照 ----------------
    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self._latencies)
            counters = dict(self._counters)
            last_retrieval = self._last_retrieval_ms
            uptime = time.time() - self._started_at

        def pct(p: float) -> float | None:
            if not lat:
                return None
            idx = min(len(lat) - 1, max(0, int(round(p * (len(lat) - 1)))))
            return round(lat[idx], 1)

        return {
            "counters": counters,
            "latency_ms": {
                "n": len(lat),
                "p50": pct(0.50),
                "p95": pct(0.95),
                "avg": round(mean(lat), 1) if lat else None,
            },
            "last_retrieval_ms": (round(last_retrieval, 1)
                                  if last_retrieval is not None else None),
            "uptime_sec": round(uptime, 1),
        }

    def reset(self) -> None:
        with self._lock:
            for k in self._counters:
                self._counters[k] = 0
            self._latencies.clear()
            self._last_retrieval_ms = None


__all__ = ["OpsMetrics", "DEFAULT_WINDOW"]
