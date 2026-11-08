"""任务队列端口。

────────────────────────────────────────────────────────────────────
为什么需要它（而不用 `ThreadPoolExecutor`）
────────────────────────────────────────────────────────────────────
| 能力 | 线程池 | 队列 |
|---|---|---|
| 请求返回后继续跑 | ✅ | ✅ |
| 进程重启任务不丢 | ❌ | ✅（消息在 broker 里） |
| 失败重试 + 退避 | 自己写 | ✅ 原生 |
| 多个 worker 分担 | ❌（同一进程） | ✅ |
| 限流（并发数） | 自己写 | ✅ 预取数控制 |
| 任务可观测（积压/死信） | 自己写 | ✅ 管理界面 |

本项目当前只有 2 个员工，线程池够用。
但**这个端口的存在就是为了「不必改业务代码就能换实现」**：
装配根按 `QUEUE_BACKEND` 选 memory 或 rabbitmq，用例层只依赖端口。

────────────────────────────────────────────────────────────────────
★ 幂等由调用方保证，不由队列保证
────────────────────────────────────────────────────────────────────
队列投递语义是「至少一次」——网络抖动、worker 崩溃都可能导致重复投递。
所以消费端必须自己保证幂等：

    · `report_task` 表上 `(trainee_id, period)` 唯一键
    · 消费前先看任务是否已完成，已完成就跳过

★ 把幂等放在调用方而不是队列里，是因为队列改不了这一点——
  RabbitMQ 的 ack 机制只能保证「不丢」，保证不了「不重复」。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class TaskQueuePort(ABC):
    """任务队列端口。"""

    @abstractmethod
    def publish(self, payload: dict[str, Any], *, task_id: str = "") -> str:
        """投递一个任务。

        Args:
            payload: 任务内容（必须是 JSON 可序列化的 dict）。
            task_id: 任务标识。为空时由实现生成。
                同一 task_id 重复投递时，实现**不保证**去重
                （去重责任在消费端，见模块注释）。

        Returns:
            实际使用的 task_id。
        """

    @abstractmethod
    def start_consumer(
        self,
        handler: Callable[[dict[str, Any]], bool],
        *,
        prefetch: int = 2,
    ) -> None:
        """启动消费者（阻塞式，通常跑在独立线程/进程）。

        Args:
            handler: 处理函数。返回 True 表示处理成功（ack），
                False 表示失败（nack）。

                ★ 实现必须把 handler 抛出的异常当作「失败」处理，
                  不能让 worker 因一条毒消息而死掉。
            prefetch: 预取数 = 同时处理的任务上限。
                ★ 这个值是**限流关键**：本项目每个任务是 LLM 调用，
                  并发太高会打满千问配额，也会让 8GB 卡吃紧。
                  建议 2。
        """

    @abstractmethod
    def stop_consumer(self) -> None:
        """停止消费（幂等：未启动时调用不报错）。"""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """队列状态（积压量、消费者数等），供健康检查与管理页用。"""

    @abstractmethod
    def is_ready(self) -> bool:
        """队列是否可用。不可用时调用方应降级为同步执行。"""


__all__ = ["TaskQueuePort"]
