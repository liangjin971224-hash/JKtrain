"""批量报告任务（P3）的测试。

重点三条：
  1. **幂等**：重复投递不会重复生成（任务表 + 消费前检查）
  2. **重试上限**：失败到上限就放弃，不会无限循环堵住队列
  3. **队列语义一致**：内存版与 RabbitMQ 版行为相同（否则测出的是假绿）
"""
from __future__ import annotations

import time

import pytest

from application.use_cases.batch_report import (
    ST_DEAD,
    ST_DONE,
    ST_FAILED,
    ST_PENDING_REVIEW,
    ST_QUEUED,
    ST_RUNNING,
    BatchReportUseCase,
)
from infrastructure.queue.inmemory_task_queue import InMemoryTaskQueue
from tests.fakes import FakeRepositoryPort


def _repo() -> FakeRepositoryPort:
    return FakeRepositoryPort()


class TestEnqueue:
    def test_enqueue_all(self) -> None:
        repo = _repo()
        q = InMemoryTaskQueue()
        uc = BatchReportUseCase(repository=repo, queue=q)

        res = uc.enqueue_batch(
            period="2026-10", trainee_ids=["E001", "E002", "E003"])

        assert res.total == 3
        assert res.enqueued == 3
        assert res.skipped == 0
        assert q.stats()["pending"] == 3
        # 任务表里有 3 条 queued
        rows = repo.list_report_tasks(period="2026-10")
        assert len(rows) == 3
        assert all(r["status"] == ST_QUEUED for r in rows)

    def test_idempotent_repeat_enqueue(self) -> None:
        """★ 重复投递被跳过（主管重复点按钮不会重复生成）。"""
        repo = _repo()
        q = InMemoryTaskQueue()
        uc = BatchReportUseCase(repository=repo, queue=q)

        uc.enqueue_batch(period="2026-10", trainee_ids=["E001"])
        # 手工把状态推到 pending_review（模拟已生成）
        repo.upsert_report_task(
            task_id="E001:2026-10", trainee_id="E001",
            period="2026-10", status=ST_PENDING_REVIEW)

        res2 = uc.enqueue_batch(period="2026-10", trainee_ids=["E001"])
        assert res2.enqueued == 0
        assert res2.skipped == 1

    def test_running_is_also_skipped(self) -> None:
        """正在跑的也不该再投一次。"""
        repo = _repo()
        q = InMemoryTaskQueue()
        uc = BatchReportUseCase(repository=repo, queue=q)
        repo.upsert_report_task(task_id="E001:2026-10", trainee_id="E001",
                                period="2026-10", status=ST_RUNNING)
        res = uc.enqueue_batch(period="2026-10", trainee_ids=["E001"])
        assert res.enqueued == 0

    def test_failed_can_requeue(self) -> None:
        """失败状态允许重新投递（人工重试场景）。"""
        repo = _repo()
        q = InMemoryTaskQueue()
        uc = BatchReportUseCase(repository=repo, queue=q)
        repo.upsert_report_task(task_id="E001:2026-10", trainee_id="E001",
                                period="2026-10", status="failed")
        res = uc.enqueue_batch(period="2026-10", trainee_ids=["E001"])
        assert res.enqueued == 1


class TestConsumer:
    def test_handler_success_marks_pending_review(self) -> None:
        repo = _repo()
        q = InMemoryTaskQueue()
        uc = BatchReportUseCase(repository=repo, queue=q)

        calls = []

        def _factory():
            class _G:
                def invoke(self, state, cfg):
                    calls.append(state["trainee_id"])
                    return {}

            return _G()

        q.start_consumer(uc.make_handler(_factory), prefetch=2)
        try:
            uc.enqueue_batch(period="2026-10", trainee_ids=["E001", "E002"])
            assert q.drain(timeout=5) is True
            # 等消费完成
            for _ in range(100):
                rows = repo.list_report_tasks(period="2026-10")
                if all(r["status"] == ST_PENDING_REVIEW for r in rows):
                    break
                time.sleep(0.02)
        finally:
            q.stop_consumer()

        rows = repo.list_report_tasks(period="2026-10")
        assert len(rows) == 2
        assert all(r["status"] == ST_PENDING_REVIEW for r in rows)
        assert sorted(calls) == ["E001", "E002"]

    def test_handler_failure_retries_then_dead(self) -> None:
        """★ 一直失败 → 重试到上限 → 判死，不无限循环。"""
        repo = _repo()
        q = InMemoryTaskQueue(max_attempts=3, retry_delay_sec=0)
        uc = BatchReportUseCase(repository=repo, queue=q, max_attempts=3)

        attempts = []

        def _factory():
            class _G:
                def invoke(self, state, cfg):
                    attempts.append(1)
                    raise RuntimeError("boom")

            return _G()

        q.start_consumer(uc.make_handler(_factory), prefetch=1)
        try:
            uc.enqueue_batch(period="2026-10", trainee_ids=["E001"])
            # 等它跑到 dead
            for _ in range(200):
                st = q.stats()
                if st["dead"] >= 1:
                    break
                time.sleep(0.02)
        finally:
            q.stop_consumer()

        assert len(attempts) == 3, f"应重试 3 次，实际 {len(attempts)}"
        assert q.stats()["dead"] == 1
        assert len(q.dead_tasks) == 1

    def test_skip_already_done(self) -> None:
        """★ 消费前检查：已待复核的任务直接跳过（幂等的第二层）。"""
        repo = _repo()
        q = InMemoryTaskQueue()
        uc = BatchReportUseCase(repository=repo, queue=q)
        repo.upsert_report_task(task_id="E001:2026-10", trainee_id="E001",
                                period="2026-10", status=ST_PENDING_REVIEW)

        called = []

        def _factory():
            class _G:
                def invoke(self, state, cfg):
                    called.append(1)
                    return {}

            return _G()

        # 直接手工投一条（绕过 enqueue 的跳过逻辑）
        q.publish({"task_id": "E001:2026-10", "trainee_id": "E001",
                   "period": "2026-10"})
        q.start_consumer(uc.make_handler(_factory), prefetch=1)
        try:
            q.drain(timeout=5)
            time.sleep(0.2)
        finally:
            q.stop_consumer()

        assert called == [], "已待复核的任务不该再跑图"

    def test_missing_fields_acked(self) -> None:
        """缺 trainee_id/period 的消息：ack 丢弃，不进重试。"""
        repo = _repo()
        q = InMemoryTaskQueue()
        uc = BatchReportUseCase(repository=repo, queue=q)
        ok = uc.make_handler(lambda: None)({"task_id": "x"})
        assert ok is True
        assert q.stats()["failed"] == 0

    def test_handler_exception_returns_false(self) -> None:
        """处理函数抛异常 → 返回 False（触发重试），不崩消费者。"""
        repo = _repo()
        q = InMemoryTaskQueue()
        uc = BatchReportUseCase(repository=repo, queue=q)

        def _factory():
            raise RuntimeError("build failed")

        ok = uc.make_handler(_factory)(
            {"task_id": "t1", "trainee_id": "E001", "period": "2026-10"})
        assert ok is False


class TestProgress:
    def test_progress_counts(self) -> None:
        repo = _repo()
        q = InMemoryTaskQueue()
        uc = BatchReportUseCase(repository=repo, queue=q)

        for i, st in enumerate([ST_QUEUED, ST_RUNNING,
                                ST_PENDING_REVIEW, ST_DONE]):
            repo.upsert_report_task(
                task_id=f"E00{i}:2026-10", trainee_id=f"E00{i}",
                period="2026-10", status=st)

        prog = uc.progress("2026-10")
        assert prog["total"] == 4
        assert prog["queued"] == 1
        assert prog["running"] == 1
        assert prog["pending_review"] == 1
        assert prog["done"] == 1
        # generated = pending_review + done
        assert prog["generated"] == 2
        assert prog["percent"] == 50.0

    def test_progress_empty_period(self) -> None:
        uc = BatchReportUseCase(repository=_repo(), queue=InMemoryTaskQueue())
        prog = uc.progress("2026-01")
        assert prog["total"] == 0
        assert prog["percent"] == 0.0


class TestQueueSemantics:
    """内存队列与 RabbitMQ 版的**共同契约**。

    ★ 这些测试是「换实现不改业务代码」的信心来源：
      如果内存版行为与 RabbitMQ 版不同，用它跑的测试就是假绿。
    """

    def test_is_ready(self) -> None:
        assert InMemoryTaskQueue().is_ready() is True

    def test_publish_returns_task_id(self) -> None:
        q = InMemoryTaskQueue()
        tid = q.publish({"a": 1}, task_id="fixed")
        assert tid == "fixed"
        tid2 = q.publish({"a": 1})
        assert tid2 and tid2 != "fixed"

    def test_stats_shape(self) -> None:
        st = InMemoryTaskQueue().stats()
        for k in ("processed", "failed", "dead", "consuming"):
            assert k in st

    def test_stop_idempotent(self) -> None:
        q = InMemoryTaskQueue()
        q.stop_consumer()
        q.stop_consumer()          # 不该抛

    def test_double_start_ignored(self) -> None:
        """重复启动不会起第二个消费者（否则会重复消费）。"""
        q = InMemoryTaskQueue()
        q.start_consumer(lambda p: True, prefetch=1)
        try:
            q.start_consumer(lambda p: True, prefetch=1)
            assert q.stats()["consuming"] is True
        finally:
            q.stop_consumer()


class TestQueuedIsIdempotent:
    """★ 真实 bug（2026-10-31）：queued 状态没被算作「已排队」。

    最初 `TERMINAL_OR_WAITING` 只含 pending_review/done/running，
    漏了 queued。结果主管刚点完「生成」，再点一次 →
    同一批人被投两遍，LLM 调用翻倍。

    ★ 为什么 failed/dead 不能含进去：
      失败的任务需要人工重试，必须允许重新投递。
    """

    def test_immediate_repeat_skipped(self) -> None:
        """刚投递完立刻再投 → 全部跳过。"""
        repo = _repo()
        q = InMemoryTaskQueue()
        uc = BatchReportUseCase(repository=repo, queue=q)

        first = uc.enqueue_batch(
            period="2026-10", trainee_ids=["E001", "E002"])
        assert first.enqueued == 2

        # 状态还是 queued（消费者没跑）
        second = uc.enqueue_batch(
            period="2026-10", trainee_ids=["E001", "E002"])
        assert second.enqueued == 0
        assert second.skipped == 2
        # 队列里只有 2 条，不是 4 条
        assert q.stats()["pending"] == 2

    def test_failed_still_requeued(self) -> None:
        """失败态仍允许重投（人工重试）。"""
        repo = _repo()
        q = InMemoryTaskQueue()
        uc = BatchReportUseCase(repository=repo, queue=q)
        repo.upsert_report_task(task_id="E001:2026-10", trainee_id="E001",
                                period="2026-10", status="failed")
        res = uc.enqueue_batch(period="2026-10", trainee_ids=["E001"])
        assert res.enqueued == 1

    def test_dead_still_requeued(self) -> None:
        repo = _repo()
        q = InMemoryTaskQueue()
        uc = BatchReportUseCase(repository=repo, queue=q)
        repo.upsert_report_task(task_id="E001:2026-10", trainee_id="E001",
                                period="2026-10", status=ST_DEAD)
        res = uc.enqueue_batch(period="2026-10", trainee_ids=["E001"])
        assert res.enqueued == 1


class TestFailureIsNotSilentlySwallowed:
    """★ 两个真实 bug（2026-10-31 造「必然失败」数据时发现）。

    造失败的方式：给一个非法日期区间（`2026-13-45`），
    MySQL 会抛 `Incorrect DATETIME value`。

    ── bug ①：图内 error 被吞掉 ──
    `load_data` 捕获异常后返回 `{"error": ...}` 而不是抛出，
    但原来**没有任何条件边检查 error** → 图继续跑，
    用空数据算出「0 次演练」的报告，最后还写进定稿。
    表现：任务标 `pending_review`（成功），主管看到一份"完美"空报告。

    ── bug ②：failed 从不转 dead ──
    队列超过重试上限会 ack 丢弃，但**不通知业务层**，
    任务表永远停在 `failed`，从不进 `dead`——
    状态机文档里写了这条边，但没实现。
    """

    def test_graph_stops_on_load_error(self) -> None:
        """★ 拉数据失败时图必须终止，不能产出空报告。"""
        from langgraph.checkpoint.memory import InMemorySaver

        from application.graph.monthly_report_graph import build_report_graph

        class _BoomRepo(FakeRepositoryPort):
            def list_drill_reviews(self, **kw):
                raise RuntimeError("Incorrect DATETIME value")

        g = build_report_graph(
            repository=_BoomRepo(), llm=None,
            checkpointer=InMemorySaver())
        cfg = {"configurable": {"thread_id": "boom"}}
        out = g.invoke({"trainee_id": "E001", "period": "2026-10",
                        "thread_id": "boom"}, cfg)

        assert out.get("error"), "应记录 error"
        # ★ 关键：不能挂起等复核（那意味着它当成成功了）
        assert "__interrupt__" not in out, (
            "拉数据失败却进了复核节点 —— 说明失败被当成成功")
        st = g.get_state(cfg)
        assert not st.next, f"图应终止，实际还有待执行节点 {st.next}"

    def test_handler_treats_graph_error_as_failure(self) -> None:
        """★ handler 要把图返回的 error 判为失败，触发重试。"""
        repo = _repo()
        q = InMemoryTaskQueue(max_attempts=3, retry_delay_sec=0)
        uc = BatchReportUseCase(repository=repo, queue=q, max_attempts=3)

        def _bad_factory():
            class _G:
                def invoke(self, state, cfg):
                    # 图内部失败，但**不抛异常**，只返回 error
                    return {"error": "拉取数据失败: OperationalError"}

            return _G()

        ok = uc.make_handler(_bad_factory)(
            {"task_id": "t1", "trainee_id": "E001", "period": "2026-10"})
        assert ok is False, (
            "图返回 error 却判成功 —— 失败会被静默吞掉")

    def test_final_attempt_marks_dead(self) -> None:
        """★ 最后一次失败要标 `dead`，不能永远停在 `failed`。"""
        repo = _repo()
        q = InMemoryTaskQueue(max_attempts=2, retry_delay_sec=0)
        uc = BatchReportUseCase(repository=repo, queue=q, max_attempts=2)

        def _bad_factory():
            class _G:
                def invoke(self, state, cfg):
                    raise RuntimeError("boom")

            return _G()

        handler = uc.make_handler(_bad_factory)
        # 第 1 次失败（_attempt=0 → 本次是第 1 次，还没到最后）
        handler({"task_id": "E001:2026-10", "trainee_id": "E001",
                 "period": "2026-10", "_attempt": 0, "max_attempts": 2})
        assert repo.get_report_task("E001", "2026-10")["status"] == ST_FAILED

        # 第 2 次失败（_attempt=1 → 本次是第 2 次 = 上限 → 判死）
        handler({"task_id": "E001:2026-10", "trainee_id": "E001",
                 "period": "2026-10", "_attempt": 1, "max_attempts": 2})
        assert repo.get_report_task("E001", "2026-10")["status"] == ST_DEAD, (
            "重试耗尽仍是 failed —— 永远进不了 dead，主管看不到"
            "「需人工介入」")
