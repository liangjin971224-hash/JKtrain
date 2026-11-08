"""HTTP 层的依赖注入。

────────────────────────────────────────────────────────────────────
为什么必须是进程级单例
────────────────────────────────────────────────────────────────────
`build_container` 会做两件昂贵的事：
    · 加载 bge-m3（约 14 秒）
    · 建 BM25 索引（约 1 秒）

如果每个请求都装配一次容器，第 1 个请求要等 15 秒，
而且每次都重复占显存 —— 三个请求就把 8GB 显存吃光。

所以：**启动时装配一次，之后复用**。

────────────────────────────────────────────────────────────────────
为什么用 lifespan 而不是模块级装配
────────────────────────────────────────────────────────────────────
模块级装配会导致「import 即加载模型」：
    · 跑任何 import 这个模块的测试都会等 15 秒
    · 单元测试无法绕过

用 FastAPI 的 lifespan，模型加载时机由应用启动决定，
测试可以注入假容器（见 `tests/test_interfaces/`）。
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config.assembly import Container, build_container
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

#: 进程级容器。应用启动时填充，关闭时释放。
_container: Container | None = None


def get_container() -> Container:
    """取当前容器。

    Raises:
        RuntimeError: 容器尚未初始化（应用没通过 lifespan 启动）。
    """
    if _container is None:
        raise RuntimeError(
            "容器未初始化。请通过 FastAPI 的 lifespan 启动应用，"
            "或在测试中调用 set_container() 注入假容器。")
    return _container


def set_container(container: Container | None) -> None:
    """注入容器（供测试使用）。"""
    global _container
    _container = container


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动装配 → 关闭释放。"""
    global _container

    settings = get_settings()
    logger.info("正在装配依赖容器（首次加载 bge-m3 约 15 秒）…")

    _container = build_container(settings)
    ready = _container.ready_ports()
    logger.info("容器就绪: %d/7 端口 %s", len(ready), ready)
    if _container.pending:
        logger.warning("待实现端口: %s", _container.pending)

    # ★ 启动队列消费者（P3 批量生成）
    #   没有它，批量任务投进去就一直是 queued，永远没人处理 ——
    #   这是 P3 交付时的真实缺口（压测脚本自己 start 了，生产代码没有）。
    _start_queue_consumer(_container)

    try:
        yield
    finally:
        _stop_queue_consumer(_container)
        logger.info("正在释放容器…")
        if _container is not None:
            _container.close()
        _container = None


def current_settings() -> Settings:
    """取配置（每次读环境变量，便于测试覆盖）。"""
    return get_settings()


__all__ = [
    "current_settings",
    "get_container",
    "lifespan",
    "set_container",
]


# ===================================================================
# 队列消费者（P3）
# ===================================================================
def _start_queue_consumer(container) -> None:
    """在 Web 进程内启动批量任务消费者。

    ★ 为什么要它：批量投递只是把任务塞进队列，
      没有消费者就永远是 `queued`。P3 交付时漏了这一步
      （压测脚本里手工 start 过，生产代码没有），
      表现为「前端进度条一直是 0%」。

    ★ 为什么用开关：多进程部署（`--workers N`）会起 N 个消费者，
      与独立 worker 进程叠加会重复抢任务。虽然幂等能兜住不重复生成，
      但会白跑 LLM。所以留了 `QUEUE_CONSUMER_IN_WEB` 开关。

    ★ 失败不影响启动：消费者起不来只影响批量生成，
      单人生成和问答不受影响，不该让整个应用起不来。
    """
    settings = get_settings()
    if not getattr(settings, "queue_consumer_in_web", True):
        logger.info("QUEUE_CONSUMER_IN_WEB=false，跳过进程内消费者")
        return
    if container is None or getattr(container, "task_queue", None) is None:
        logger.warning("任务队列不可用，跳过消费者启动")
        return

    try:
        from application.use_cases.batch_report import BatchReportUseCase

        uc = BatchReportUseCase(
            repository=container.repository,
            queue=container.task_queue,
            max_attempts=settings.queue_max_attempts,
        )
        container.task_queue.start_consumer(
            uc.make_handler(lambda: _build_report_graph(container)),
            prefetch=settings.queue_prefetch,
        )
        logger.info("队列消费者已启动: prefetch=%s", settings.queue_prefetch)
    except Exception as exc:                       # noqa: BLE001
        logger.error("队列消费者启动失败（批量生成将不可用）: %s: %s",
                     type(exc).__name__, exc, exc_info=True)


def _stop_queue_consumer(container) -> None:
    """关闭消费者（幂等）。"""
    if container is None or getattr(container, "task_queue", None) is None:
        return
    try:
        container.task_queue.stop_consumer()
        logger.info("队列消费者已停止")
    except Exception as exc:                       # noqa: BLE001
        logger.warning("停止消费者失败: %s", exc)


def _build_report_graph(container):
    """构造月度报告图（供 Worker 用）。

    ★ 必须**每任务新建**：图无状态，中断态在 Redis checkpoint 里。
    """
    from application.graph.monthly_report_graph import build_report_graph
    from application.use_cases.report_context import case_titles, doc_catalog

    return build_report_graph(
        repository=container.repository,
        llm=container.llm,
        checkpointer=container.checkpoint_saver,
        case_titles=case_titles(container),
        doc_catalog=doc_catalog(container),
    )
