"""内存任务队列（开发/测试用）。

────────────────────────────────────────────────────────────────────
它的定位：**不是「简化版 RabbitMQ」，而是「同一个端口的另一种实现」**
────────────────────────────────────────────────────────────────────
所以它必须满足与 RabbitMQ 版**相同的语义**，否则用它的测试就是假绿：

| 语义 | 本实现 |
|---|---|
| 至少一次投递 | ✅ 失败会重入队 |
| 失败重试到上限 | ✅ `max_attempts` |
| 超过上限放弃（不死循环） | ✅ 记 dead，不重入队 |
| 处理失败不杀死消费者 | ✅ 异常按失败处理 |
| prefetch = 同时处理上限 | ✅ 用线程池（真的并发） |
| 处理函数返回 False = 失败 | ✅ |

★ 它**做不到**的（这也是为什么生产要用 RabbitMQ）：
    · 进程重启任务就丢（存在内存 list 里）
    · 多进程不共享
    · 没有管理界面

★ 为什么还要它：
    1. 单元测试不需要 Docker
    2. 本地开发能在没有 RabbitMQ 时跑通全链路
    3. 装配根按 `QUEUE_BACKEND` 切换，**业务代码零改动**
"""
from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import uuid
from typing import Any, Callable

from domain.ports import TaskQueuePort

logger = logging.getLogger(__name__)

ATTEMPT_KEY = "_attempt"


class InMemoryTaskQueue(TaskQueuePort):
    """进程内任务队列（线程 + 队列）。"""

    def __init__(self, *, max_attempts: int = 3,
                 retry_delay_sec: float = 0.0) -> None:
        """
        Args:
            max_attempts: 默认最大尝试次数。
            retry_delay_sec: 重试前等待秒数。
                ★ 测试里设 0 让重试立刻发生；生产建议 >0（避免忙循环）。
        """
        self._q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._max_attempts = max(1, max_attempts)
        self._retry_delay = max(0.0, retry_delay_sec)

        self._processed = 0
        self._failed = 0
        self._dead = 0
        #: 消费线程池（并发度 = prefetch）。None 表示还没启动。
        #: ★ 为什么需要它：单线程 `while get()` 是**串行**的，
        #:   prefetch 只控制「预取多少」，不产生并发。
        #:   实测 20 人串行 6s → 队列 8.8s（更慢了），就是因为没并发。
        self._pool: ThreadPoolExecutor | None = None
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        #: 供测试断言「哪些任务被判死了」
        self.dead_tasks: list[dict[str, Any]] = []

    # ---------------- 端口实现 ----------------
    def publish(self, payload: dict[str, Any], *, task_id: str = "") -> str:
        tid = task_id or uuid.uuid4().hex[:16]
        body = dict(payload)
        body.setdefault(ATTEMPT_KEY, 0)
        body.setdefault("max_attempts", self._max_attempts)
        body["task_id"] = tid
        self._q.put(body)
        return tid

    def start_consumer(self, handler: Callable[[dict[str, Any]], bool],
                       *, prefetch: int = 2) -> None:
        """启动消费者线程。

        ★ 与 RabbitMQ 版一致：已在跑就不再起第二个。
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            # ★ 并发度 = prefetch：同时最多处理这么多个任务
            self._pool = ThreadPoolExecutor(
                max_workers=max(1, prefetch),
                thread_name_prefix="memory-queue-worker")
            self._thread = threading.Thread(
                target=self._loop, args=(handler, prefetch),
                name="memory-queue-dispatcher", daemon=True)
            self._thread.start()

    def stop_consumer(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=5)
        self._thread = None
        # ★ 等在途任务跑完再关池，否则会中断正在写的库操作
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def stats(self) -> dict[str, Any]:
        return {
            "queue": "memory",
            "pending": self._q.qsize(),
            "processed": self._processed,
            "failed": self._failed,
            "dead": self._dead,
            "consuming": bool(
                self._thread and self._thread.is_alive()),
        }

    def is_ready(self) -> bool:
        return True

    # ---------------- 内部 ----------------
    def _loop(self, handler: Callable[[dict], bool], concurrency: int) -> None:
        """派发循环：从队列取任务，交给线程池执行。

        ★ 与 RabbitMQ 版的关键差别：
          RabbitMQ 的 `basic_consume` 由客户端库负责并发（预取 N 条同时回调）。
          这里没有库帮我做，必须自己用线程池 ——
          否则「prefetch」形同虚设，20 个任务还是一个一个跑。
        """
        limit = max(1, concurrency)
        while not self._stop.is_set():
            # 在途任务已满 → 等一下再取，避免把任务全捞出来堆在内存
            with self._inflight_lock:
                if self._inflight >= limit:
                    full = True
                else:
                    full = False
            if full:
                time.sleep(0.01)
                continue

            try:
                payload = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            with self._inflight_lock:
                self._inflight += 1

            pool = self._pool
            if pool is None:
                with self._inflight_lock:
                    self._inflight -= 1
                self._q.put(payload)
                return
            pool.submit(self._run_one, payload, handler)

    def _run_one(self, payload: dict, handler: Callable[[dict], bool]) -> None:
        """在工作线程里处理一个任务。"""
        try:
            self._process(payload, handler)
        finally:
            with self._inflight_lock:
                self._inflight -= 1

    def _process(self, payload: dict, handler: Callable[[dict], bool]) -> None:
        attempt = int(payload.get(ATTEMPT_KEY, 0))
        max_attempts = int(payload.get("max_attempts", self._max_attempts))

        ok = False
        try:
            ok = bool(handler(payload))
        except Exception as exc:                   # noqa: BLE001
            logger.warning("任务处理抛异常（按失败处理）: %s: %s",
                           type(exc).__name__, exc)
            ok = False

        if ok:
            self._processed += 1
            return

        self._failed += 1
        attempt += 1
        if attempt >= max_attempts:
            logger.error("任务超过最大重试(%d)，放弃: %s",
                         max_attempts, payload.get("task_id"))
            self._dead += 1
            self.dead_tasks.append(payload)
            return

        if self._retry_delay:
            if self._stop.wait(self._retry_delay):
                return
        payload[ATTEMPT_KEY] = attempt
        self._q.put(payload)

    # ---------------- 测试辅助 ----------------
    def drain(self, timeout: float = 10.0) -> bool:
        """等队列跑空（测试用）。

        Returns:
            True = 在超时前跑空；False = 还有积压。
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._q.qsize() == 0:
                return True
            time.sleep(0.02)
        return self._q.qsize() == 0


__all__ = ["ATTEMPT_KEY", "InMemoryTaskQueue"]
