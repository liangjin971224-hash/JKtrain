"""队列消费者启动的测试。

★ 为什么单独测这个
  批量投递只是把任务塞进队列，**没有消费者就永远是 queued**。
  P3 首次交付时漏了在 lifespan 里启动消费者 ——
  压测脚本自己 `start_consumer` 过，所以压测全绿，
  但真实服务里进度条永远 0%。

  这正是「单测绿 ≠ 可用」的又一例。这组测试守住：
    · lifespan 启动时确实会调 start_consumer
    · 关闭时会 stop
    · 开关关掉时不启动（多进程部署场景）
"""
from __future__ import annotations

import pytest

from application.use_cases.report_context import case_titles, doc_catalog
from tests.fakes import FakeRepositoryPort


class _FakeQueue:
    """记录调用情况的假队列。"""

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.prefetch = None
        self.handler = None

    def start_consumer(self, handler, *, prefetch=2):
        self.started += 1
        self.handler = handler
        self.prefetch = prefetch

    def stop_consumer(self):
        self.stopped += 1

    def publish(self, payload, *, task_id=""):
        return task_id or "t"

    def stats(self):
        return {}

    def is_ready(self):
        return True


class _FakeContainer:
    def __init__(self, queue=None):
        self.task_queue = queue
        self.repository = FakeRepositoryPort()
        self.llm = None
        self.checkpoint_saver = None
        self.case_catalog = None
        self.milvus_client = None
        self.settings = None


class TestStartConsumer:
    def test_starts_consumer(self) -> None:
        """★ 核心断言：lifespan 启动时必须启动消费者。

        没有这条，批量任务投进去没人处理，进度条永远 0%。
        """
        from interfaces.http.deps import _start_queue_consumer

        q = _FakeQueue()
        c = _FakeContainer(q)
        _start_queue_consumer(c)

        assert q.started == 1, "lifespan 没有启动消费者"

    def test_passes_handler(self) -> None:
        """必须传 handler，否则消费了也不知道干什么。"""
        from interfaces.http.deps import _start_queue_consumer

        q = _FakeQueue()
        _start_queue_consumer(_FakeContainer(q))

        assert q.handler is not None
        assert callable(q.handler)

    def test_no_queue_does_not_crash(self) -> None:
        """队列不可用时不崩（只影响批量，不该影响整个应用启动）。"""
        from interfaces.http.deps import _start_queue_consumer

        _start_queue_consumer(_FakeContainer(None))    # 不抛即通过

    def test_disabled_by_setting(self, monkeypatch) -> None:
        """多进程部署时用开关关掉进程内消费者。"""
        from interfaces.http import deps

        q = _FakeQueue()
        real = deps.get_settings

        class _S:
            queue_consumer_in_web = False
            queue_max_attempts = 3
            queue_prefetch = 2

        monkeypatch.setattr(deps, "get_settings", lambda: _S())
        try:
            deps._start_queue_consumer(_FakeContainer(q))
        finally:
            monkeypatch.setattr(deps, "get_settings", real)

        assert q.started == 0

    def test_stop_consumer(self) -> None:
        from interfaces.http.deps import _stop_queue_consumer

        q = _FakeQueue()
        _stop_queue_consumer(_FakeContainer(q))
        assert q.stopped == 1

    def test_stop_idempotent(self) -> None:
        from interfaces.http.deps import _stop_queue_consumer

        _stop_queue_consumer(_FakeContainer(None))
        _stop_queue_consumer(_FakeContainer(None))


class TestReportContext:
    """`report_context` 的两个函数被路由和 worker 共用。

    ★ 抽出来的原因：`deps.py` 若 import `routers.reports` 就是反向依赖，
      是循环依赖的前兆。
    """

    def test_case_titles_no_catalog(self) -> None:
        assert case_titles(_FakeContainer()) == {}

    def test_case_titles_from_catalog(self) -> None:
        class _Spec:
            def __init__(self, cid, t):
                self.case_id = cid
                self.title = t

        class _Cat:
            def list_all(self):
                return [_Spec("C01", "鱼油"), _Spec("C04", "停药")]

        c = _FakeContainer()
        c.case_catalog = _Cat()
        assert case_titles(c) == {"C01": "鱼油", "C04": "停药"}

    def test_case_titles_survives_error(self) -> None:
        """目录报错时返回空 dict，不能让报告整体失败。"""
        class _BadCat:
            def list_all(self):
                raise RuntimeError("boom")

        c = _FakeContainer()
        c.case_catalog = _BadCat()
        assert case_titles(c) == {}

    def test_doc_catalog_no_milvus(self) -> None:
        assert doc_catalog(_FakeContainer()) == {}

    def test_doc_catalog_from_milvus(self) -> None:
        class _Milvus:
            def query(self, **kw):
                return [{"doc_id": "D005", "title": "功能声称规范"},
                        {"doc_id": "D005", "title": "重复"},
                        {"doc_id": "D009", "title": "异议处理"}]

        class _S:
            coll_training = "health_training_docs"

        c = _FakeContainer()
        c.milvus_client = _Milvus()
        c.settings = _S()
        out = doc_catalog(c)
        # 同 doc_id 只留第一条
        assert out == {"D005": "功能声称规范", "D009": "异议处理"}

    def test_doc_catalog_survives_error(self) -> None:
        class _Bad:
            def query(self, **kw):
                raise RuntimeError("milvus down")

        class _S:
            coll_training = "x"

        c = _FakeContainer()
        c.milvus_client = _Bad()
        c.settings = _S()
        assert doc_catalog(c) == {}
