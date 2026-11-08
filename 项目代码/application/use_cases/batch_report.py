"""批量报告生成任务（P3）：把 N 个人的报告生成拆成 N 个队列任务。

────────────────────────────────────────────────────────────────────
它解决什么
────────────────────────────────────────────────────────────────────
单人生成（P2）是 `graph.invoke` 同步跑，约 5–15 秒。
20 人批量就是 100–300 秒 —— **HTTP 请求必然超时**（前端默认 30–60 秒）。

本模块把「一次批量」拆成 N 个独立任务：
    · 每个任务只负责一个人
    · 失败只重跑那一个人
    · 任务状态落 `report_task` 表，前端可轮询进度

────────────────────────────────────────────────────────────────────
★★★ 幂等在哪里保证
────────────────────────────────────────────────────────────────────
队列只保证「至少一次」，不保证「恰好一次」。
所以两层防重复：

    ① 投递前：`report_task` 上有唯一键 `(trainee_id, period)`。
       重复点击「生成」不会产生两条任务（用 upsert）。

    ② 消费前：先看任务状态。
       已经 `pending_review` / `done` 的**直接跳过**，
       不再跑一次图（否则会重复调 LLM，且可能覆盖主管已复核的定稿）。

★ 第 ② 条尤其重要：没有它，网络抖动导致的重投
  会让同一个人被生成两遍，第二次可能覆盖已定稿的报告。

────────────────────────────────────────────────────────────────────
任务状态机
────────────────────────────────────────────────────────────────────
    queued → running → pending_review → done
                    ↘ failed（可重试）→ dead（超上限，人工介入）

★ `pending_review` 与 `done` 分开：
  「报告已生成，等主管确认」不等于「已完成」。
  只有主管点了同意（定稿落库）才算 done。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------- 任务状态 ----------------
ST_QUEUED = "queued"
ST_RUNNING = "running"
ST_PENDING_REVIEW = "pending_review"
ST_DONE = "done"
ST_FAILED = "failed"
ST_DEAD = "dead"

#: 这些状态的任一人都不需要再投（幂等的依据）。
#:
#: ★ 含 `queued`：已排队就不该再排一次。
#:   实测踩坑：一开始漏了它，导致「重复点生成」把同一批人投两遍，
#:   LLM 调用翻倍且可能覆盖已生成的结果。
#:
#: ★ 不含 `failed` / `dead`：
#:   失败的任务需要人工重试，必须允许重新投递。
TERMINAL_OR_WAITING = (ST_QUEUED, ST_RUNNING, ST_PENDING_REVIEW, ST_DONE)


@dataclass(frozen=True, slots=True)
class BatchResult:
    """一次批量投递的结果。"""

    period: str
    total: int = 0
    enqueued: int = 0
    skipped: int = 0
    task_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "total": self.total,
            "enqueued": self.enqueued,
            "skipped": self.skipped,
            "task_ids": list(self.task_ids),
        }


class BatchReportUseCase:
    """批量投递报告生成任务。"""

    def __init__(
        self,
        *,
        repository: Any,
        queue: Any,
        max_attempts: int = 3,
    ) -> None:
        """
        Args:
            repository: 仓储（读写 `report_task`）。
            queue: `TaskQueuePort` 实现（RabbitMQ 或内存版）。
            max_attempts: 单任务最大尝试次数，写进消息体供消费者读。
        """
        self._repo = repository
        self._queue = queue
        self._max_attempts = max(1, max_attempts)

    # ---------------- 投递 ----------------
    def enqueue_batch(
        self,
        *,
        period: str,
        trainee_ids: list[str],
        since: str = "",
        until: str = "",
    ) -> BatchResult:
        """把一批学员的报告生成任务投进队列。

        ★ 已在处理中/已完成的**跳过**（幂等）。
          主管重复点「生成本月报告」不会造成重复生成。
        """
        task_ids: list[str] = []
        enqueued = skipped = 0

        for tid in trainee_ids:
            existing = self._get_task(tid, period)
            if existing is not None and existing["status"] in TERMINAL_OR_WAITING:
                logger.info("跳过（状态=%s）: %s %s",
                            existing["status"], tid, period)
                skipped += 1
                continue

            task_id = self._task_id(tid, period)
            self._upsert_task(task_id, tid, period, ST_QUEUED)
            self._queue.publish(
                {
                    "task_id": task_id,
                    "trainee_id": tid,
                    "period": period,
                    "since": since,
                    "until": until,
                    "max_attempts": self._max_attempts,
                },
                task_id=task_id,
            )
            task_ids.append(task_id)
            enqueued += 1

        logger.info("批量投递: period=%s 总计=%d 投递=%d 跳过=%d",
                    period, len(trainee_ids), enqueued, skipped)
        return BatchResult(
            period=period,
            total=len(trainee_ids),
            enqueued=enqueued,
            skipped=skipped,
            task_ids=tuple(task_ids),
        )

    # ---------------- 消费 ----------------
    def make_handler(self, graph_factory) -> Any:
        """构造消费者处理函数。

        Args:
            graph_factory: 无参可调用，返回编译好的图。
                ★ 每任务新建图实例（与接口层一致）：
                  图无状态，中断态在 Redis。

        Returns:
            `handler(payload) -> bool`。True = 成功（ack）。
        """
        def _handle(payload: dict[str, Any]) -> bool:
            task_id = str(payload.get("task_id") or "")
            trainee_id = str(payload.get("trainee_id") or "")
            period = str(payload.get("period") or "")
            if not trainee_id or not period:
                logger.error("任务缺少 trainee_id/period，丢弃: %s", payload)
                return True                    # 不可重试的错误 → ack 丢弃

            # ① 幂等：已完成/待复核的直接跳过
            existing = self._get_task(trainee_id, period)
            if existing is not None and existing["status"] in (ST_PENDING_REVIEW, ST_DONE):
                logger.info("跳过已完成任务: %s %s (%s)",
                            trainee_id, period, existing["status"])
                return True

            self._mark(task_id, trainee_id, period, ST_RUNNING)
            started = time.time()
            # ★ thread_id 必须在 try 之前算好：
            #   否则 graph_factory() 抛异常时，except 分支引用它 → UnboundLocalError，
            #   失败处理本身崩掉，任务永远停在 running（实测踩过）。
            from application.graph.monthly_report_graph import new_thread_id

            thread_id = (existing or {}).get("thread_id") or \
                new_thread_id(trainee_id, period)
            cfg = {"configurable": {"thread_id": thread_id}}
            try:
                graph = graph_factory()

                out = graph.invoke(
                    {
                        "trainee_id": trainee_id,
                        "period": period,
                        "since": payload.get("since", ""),
                        "until": payload.get("until", ""),
                        "thread_id": thread_id,
                    },
                    cfg,
                )
                # ★★ 必须检查返回里的 error（2026-10-31 修）：
                #   图的 `load_data` 捕获异常后返回 {"error": ...} 而不是抛出，
                #   原来只要 invoke 不抛就判成功 →
                #   **"拉数据失败"被当成成功**，任务标 pending_review，
                #   主管看到一份 0 次演练的"完美"空报告。
                #
                #   现在：图内 error → 判失败 → 走重试 → 最终判死（可人工介入）。
                if isinstance(out, dict) and out.get("error"):
                    raise RuntimeError(out["error"])
            except Exception as exc:           # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"[:500]
                logger.warning("任务执行失败: %s %s: %s",
                               trainee_id, period, err)

                # ★ 判断这次失败之后还会不会再重试（2026-10-31 修）
                #   队列侧「超过上限就 ack 丢弃」时**不会通知业务层**，
                #   所以任务表原来永远停在 `failed`，从不进 `dead` ——
                #   我在状态机文档里写了 `failed → dead` 这条边，但没实现。
                #
                #   这里自己按「本次是第几次失败」判断：
                #   队列的 ATTEMPT_KEY 从 0 起，本次失败后会被置为 n+1；
                #   当 n+1 >= max_attempts 时队列不会再重投 → 业务层标 dead。
                attempt_now = int(payload.get("_attempt", 0)) + 1
                final = attempt_now >= self._max_attempts
                self._mark(
                    task_id, trainee_id, period,
                    ST_DEAD if final else ST_FAILED,
                    error=err, thread_id=thread_id)
                if final:
                    logger.error("任务已放弃（累计失败 %d 次）: %s %s",
                                 attempt_now, trainee_id, period)
                return False                   # → 重试（队列自己判上限）

            elapsed = time.time() - started
            self._mark(task_id, trainee_id, period, ST_PENDING_REVIEW,
                       thread_id=thread_id)
            logger.info("任务完成（待复核）: %s %s 用时 %.1fs",
                        trainee_id, period, elapsed)
            return True

        return _handle

    # ---------------- 仓储辅助 ----------------
    @staticmethod
    def _task_id(trainee_id: str, period: str) -> str:
        return f"{trainee_id}:{period}"

    def _get_task(self, trainee_id: str, period: str) -> dict[str, Any] | None:
        try:
            return self._repo.get_report_task(trainee_id, period)
        except Exception as exc:                   # noqa: BLE001
            logger.warning("读任务失败: %s", exc)
            return None

    def _upsert_task(self, task_id: str, trainee_id: str,
                     period: str, status: str) -> None:
        try:
            self._repo.upsert_report_task(
                task_id=task_id, trainee_id=trainee_id,
                period=period, status=status)
        except Exception as exc:                   # noqa: BLE001
            logger.warning("写任务失败（不影响投递）: %s", exc)

    def _mark(self, task_id: str, trainee_id: str, period: str,
              status: str, *, error: str = "",
              thread_id: str = "") -> None:
        try:
            self._repo.upsert_report_task(
                task_id=task_id, trainee_id=trainee_id, period=period,
                status=status, error=error, thread_id=thread_id)
        except Exception as exc:                   # noqa: BLE001
            logger.warning("更新任务状态失败: %s", exc)

    # ---------------- 进度 ----------------
    def progress(self, period: str) -> dict[str, Any]:
        """查某周期的批量进度（前端轮询用）。"""
        try:
            rows = self._repo.list_report_tasks(period=period)
        except Exception as exc:                   # noqa: BLE001
            logger.warning("读任务列表失败: %s", exc)
            rows = []

        counts: dict[str, int] = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1

        total = len(rows)
        done = counts.get(ST_DONE, 0) + counts.get(ST_PENDING_REVIEW, 0)
        return {
            "period": period,
            "total": total,
            "generated": done,
            "queued": counts.get(ST_QUEUED, 0),
            "running": counts.get(ST_RUNNING, 0),
            "pending_review": counts.get(ST_PENDING_REVIEW, 0),
            "done": counts.get(ST_DONE, 0),
            "failed": counts.get(ST_FAILED, 0),
            "dead": counts.get(ST_DEAD, 0),
            "percent": round(done / total * 100, 1) if total else 0.0,
        }


__all__ = [
    "BatchReportUseCase",
    "BatchResult",
    "ST_DEAD",
    "ST_DONE",
    "ST_FAILED",
    "ST_PENDING_REVIEW",
    "ST_QUEUED",
    "ST_RUNNING",
    "TERMINAL_OR_WAITING",
]
