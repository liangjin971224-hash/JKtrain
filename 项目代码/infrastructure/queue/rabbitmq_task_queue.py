"""RabbitMQ 任务队列适配器。

────────────────────────────────────────────────────────────────────
设计要点（都是踩过或想清楚的）
────────────────────────────────────────────────────────────────────
1. **持久化**：队列 durable + 消息 delivery_mode=2。
   否则 broker 重启任务就没了，等于没上队列。

2. **手动 ack**：`auto_ack=False`。
   处理成功才 ack；失败 nack 重入队。
   自动 ack 会在「消息刚拿到、还没处理完」就删掉——
   进程崩了任务就丢了。

3. **prefetch 限流**：默认 2。
   本项目每个任务是 LLM 调用，并发太高会打满配额、吃紧 8GB 显存。

4. **重试靠「重新投递 + 计数」**，不用 RabbitMQ 的 x-death。
   原因：x-death 的语义绕（要开 dead-letter exchange、计数在 header 里、
   不同版本行为有差异）。而这个项目只需要「最多 3 次」，
   在消息体里带 `_attempt` 更直观、更好排查。

5. **毒消息**：超过 max_attempts 直接 ack 丢弃并写日志。
   ★ 不 nack 无限重入队 —— 那会让一条坏消息堵住整个队列。
   本项目把「放弃的任务」记在 `report_task.status='dead'` 里，可人工介入。

────────────────────────────────────────────────────────────────────
★ 消费是阻塞的，必须跑在独立线程
────────────────────────────────────────────────────────────────────
`start_consumer` 内部用 `start_consuming()`，会阻塞当前线程。
调用方（worker 脚本或后端启动钩子）要自己放线程里，或直接用它做独立进程。
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import Any, Callable

from domain.ports import TaskQueuePort

logger = logging.getLogger(__name__)

#: 队列名。单一队列 + 单一消费组，符合本项目规模。
QUEUE_NAME = "health_training.report"

#: 消息里的重试计数键。
ATTEMPT_KEY = "_attempt"


class RabbitMQTaskQueue(TaskQueuePort):
    """基于 RabbitMQ 的任务队列。"""

    def __init__(
        self,
        *,
        url: str,
        queue: str = QUEUE_NAME,
        heartbeat: int = 60,
        blocked_connection_timeout: int = 30,
    ) -> None:
        """
        Args:
            url: `amqp://user:pass@host:port/vhost`。
                ★ 必须由 settings 注入（硬编码连接串会被架构门禁拦）。
            queue: 队列名。
            heartbeat: 心跳秒数。默认 60。
                ★ 处理 LLM 任务可能耗时数十秒，心跳太短会误断连接。
            blocked_connection_timeout: broker 阻塞超时。
        """
        self._url = url
        self._queue = queue
        self._heartbeat = heartbeat
        self._blocked_timeout = blocked_connection_timeout

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._consumer_thread: threading.Thread | None = None
        self._consumer_tag: str | None = None
        #: 保存 channel，stop 时调 stop_consuming 才算真的停
        #: ★ 实测踩坑：只 join 线程不够 —— broker 侧消费者还挂着，
        #:   下次 start 又起一个，consumers 变成 2，消息会被两边抢。
        self._channel: Any = None
        #: 保存 connection，用于跨线程安全地停止消费
        self._conn: Any = None
        self._processed = 0
        self._failed = 0
        self._dead = 0

    # ---------------- 连接 ----------------
    def _params(self) -> Any:
        import pika

        return pika.URLParameters(self._url)

    def _connect(self) -> Any:
        import pika

        return pika.BlockingConnection(self._params())

    def _declare(self, channel: Any) -> None:
        """声明持久化队列。"""
        channel.queue_declare(queue=self._queue, durable=True)

    # ---------------- 端口实现 ----------------
    def publish(self, payload: dict[str, Any], *, task_id: str = "") -> str:
        """投递任务。"""
        tid = task_id or uuid.uuid4().hex[:16]
        body = dict(payload)
        body.setdefault(ATTEMPT_KEY, 0)

        conn = self._connect()
        try:
            ch = conn.channel()
            self._declare(ch)
            ch.basic_publish(
                exchange="",
                routing_key=self._queue,
                body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                # ★ delivery_mode=2 才是持久化消息
                properties=self._props(delivery_mode=2),
            )
            logger.info("投递任务: %s → %s", tid, self._queue)
        finally:
            try:
                conn.close()
            except Exception:                      # noqa: BLE001
                pass
        return tid

    @staticmethod
    def _props(**kw: Any) -> Any:
        import pika

        return pika.BasicProperties(content_type="application/json", **kw)

    def start_consumer(
        self,
        handler: Callable[[dict[str, Any]], bool],
        *,
        prefetch: int = 2,
    ) -> None:
        """启动消费者（**阻塞**）。

        ★ 幂等：已在消费中再次调用会直接返回，不会起第二个消费者。
        """
        with self._lock:
            if self._consumer_thread is not None and self._consumer_thread.is_alive():
                logger.info("消费者已在运行，忽略重复启动")
                return
            self._stop_event.clear()
            self._consumer_thread = threading.Thread(
                target=self._consume_loop,
                args=(handler, prefetch),
                name="rabbit-consumer",
                daemon=True,
            )
            self._consumer_thread.start()
            logger.info("消费者已启动: queue=%s prefetch=%d",
                        self._queue, prefetch)

    def stop_consumer(self) -> None:
        """停止消费（幂等）。

        ★ 必须调 `stop_consuming()`，只 join 线程不够：
          `start_consuming()` 阻塞在 IO 循环里，join 超时后线程还在，
          broker 侧的消费者没摘掉。下次 start 就会有两个消费者抢消息
          （实测：consumers 从 1 变 2）。

        ★★ 而且**不能直接在主线程调 `channel.stop_consuming()`**：
          pika 的 `BlockingConnection` 不是线程安全的，跨线程调用会静默失败
          （实测：调了没用，线程照样卡着，10s 后告警）。
          必须走 `add_callback_threadsafe`，把调用投递到 IO 线程自己执行。
        """
        self._stop_event.set()

        conn = self._conn
        if conn is not None:
            try:
                # ★ 关键：跨线程安全
                conn.add_callback_threadsafe(self._safe_stop_consuming)
            except Exception as exc:               # noqa: BLE001
                logger.debug("投递 stop 回调失败（连接可能已断）: %s", exc)

        thread = self._consumer_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
            if thread.is_alive():
                logger.warning("消费者线程未在 10s 内退出")
        self._consumer_thread = None
        self._channel = None
        logger.info("消费者已停止")

    def _safe_stop_consuming(self) -> None:
        """在 IO 线程里执行 stop_consuming（线程安全）。"""
        ch = self._channel
        if ch is None:
            return
        try:
            ch.stop_consuming()
        except Exception as exc:                   # noqa: BLE001
            logger.debug("stop_consuming 异常（通常无害）: %s", exc)

    def stats(self) -> dict[str, Any]:
        """队列状态。"""
        out: dict[str, Any] = {
            "queue": self._queue,
            "processed": self._processed,
            "failed": self._failed,
            "dead": self._dead,
            "consuming": bool(
                self._consumer_thread and self._consumer_thread.is_alive()),
        }
        try:
            conn = self._connect()
            try:
                ch = conn.channel()
                self._declare(ch)
                q = ch.queue_declare(queue=self._queue, durable=True, passive=True)
                out["messages"] = q.method.message_count
                out["consumers"] = q.method.consumer_count
            finally:
                conn.close()
        except Exception as exc:                   # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    def is_ready(self) -> bool:
        try:
            conn = self._connect()
            try:
                conn.channel()
                return True
            finally:
                conn.close()
        except Exception:                          # noqa: BLE001
            return False

    # ---------------- 内部 ----------------
    def _consume_loop(self, handler: Callable[[dict], bool],
                      prefetch: int) -> None:
        """消费循环。

        ★ 断线重连：LLM 任务耗时长，长时间空闲可能被 broker 断掉；
          外层 while 保证断了能自愈，而不是永久停摆。
        """
        while not self._stop_event.is_set():
            try:
                conn = self._connect()
            except Exception as exc:               # noqa: BLE001
                logger.warning("连接 broker 失败，5s 后重试: %s", exc)
                if self._stop_event.wait(5):
                    return
                continue

            try:
                self._conn = conn
                ch = conn.channel()
                self._channel = ch
                self._declare(ch)
                ch.basic_qos(prefetch_count=max(1, prefetch))

                def _on_message(chn, method, props, raw) -> None:
                    self._handle(chn, method, raw, handler)

                ch.basic_consume(
                    queue=self._queue, on_message_callback=_on_message)
                logger.info("开始消费: %s", self._queue)
                ch.start_consuming()
            except Exception as exc:               # noqa: BLE001
                if self._stop_event.is_set():
                    return
                logger.warning("消费循环异常，5s 后重连: %s", exc)
                if self._stop_event.wait(5):
                    return
            finally:
                self._channel = None
                self._conn = None
                try:
                    conn.close()
                except Exception:                  # noqa: BLE001
                    pass

    def _handle(self, channel: Any, method: Any, raw: bytes,
                handler: Callable[[dict], bool]) -> None:
        """处理一条消息。任何异常都不能让 worker 死掉。"""
        attempt = 0
        payload: dict[str, Any] = {}
        try:
            payload = json.loads(raw.decode("utf-8"))
            attempt = int(payload.get(ATTEMPT_KEY, 0))
        except Exception as exc:                   # noqa: BLE001
            # 消息体都解不出来 → 毒消息，直接丢弃
            logger.error("消息体非法，丢弃: %s", exc)
            channel.basic_ack(delivery_tag=method.delivery_tag)
            self._dead += 1
            return

        max_attempts = int(payload.get("max_attempts", 3))
        ok = False
        try:
            ok = bool(handler(payload))
        except Exception as exc:                   # noqa: BLE001
            logger.warning("任务处理抛异常（按失败处理）: %s: %s",
                           type(exc).__name__, exc, exc_info=True)
            ok = False

        if ok:
            channel.basic_ack(delivery_tag=method.delivery_tag)
            self._processed += 1
            return

        self._failed += 1
        attempt += 1
        if attempt >= max_attempts:
            # ★ 超过上限直接 ack 丢弃并记 dead，不无限重入队
            logger.error("任务超过最大重试(%d)，放弃: %s",
                         max_attempts, self._brief(payload))
            channel.basic_ack(delivery_tag=method.delivery_tag)
            self._dead += 1
            return

        # 重新投递（带 +1 的计数）
        payload[ATTEMPT_KEY] = attempt
        logger.info("任务失败第 %d/%d 次，重新入队: %s",
                    attempt, max_attempts, self._brief(payload))
        try:
            channel.basic_publish(
                exchange="",
                routing_key=self._queue,
                body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                properties=self._props(delivery_mode=2),
            )
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as exc:                   # noqa: BLE001
            logger.error("重新入队失败，nack 让 broker 处理: %s", exc)
            try:
                channel.basic_nack(
                    delivery_tag=method.delivery_tag, requeue=True)
            except Exception:                      # noqa: BLE001
                pass

    @staticmethod
    def _brief(payload: dict) -> str:
        return (f"task={payload.get('task_id', '?')} "
                f"trainee={payload.get('trainee_id', '?')} "
                f"period={payload.get('period', '?')}")


__all__ = ["ATTEMPT_KEY", "QUEUE_NAME", "RabbitMQTaskQueue"]
