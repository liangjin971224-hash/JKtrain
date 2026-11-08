"""基于项目 `CachePort` 的 LangGraph checkpoint 存储器。

────────────────────────────────────────────────────────────────────
为什么不用官方 `langgraph-checkpoint-redis`
────────────────────────────────────────────────────────────────────
| 对比项 | 官方包 | 本实现 |
|---|---|---|
| 额外依赖 | 要装 `langgraph-checkpoint-redis` | **不装**（`langgraph-checkpoint` 已有基类） |
| 连接 | 自己建一套连接池 | **复用 `container.cache`** |
| 键前缀 | 自己一套 | 统一 `health_training:`，`clear_prefix` 能一次清干净 |
| 切换后端 | 只能 Redis | 同一个类，`CACHE_BACKEND=memory` 也能跑 |

★ 第三点是关键：**P2 的开发与测试都跑在内存版上**，
  上线切 Redis 只改 `.env`，代码零改动。官方包做不到这点。

────────────────────────────────────────────────────────────────────
存储布局
────────────────────────────────────────────────────────────────────
    report:ckpt:{thread_id}:{checkpoint_id}          → checkpoint 元组
    report:ckpt:{thread_id}:{checkpoint_id}:writes   → 该步的 writes

★ 为什么 writes 单独一个键：
  `put_writes` 与 `put` 是两个不同的调用时机（writes 是节点产出，
  checkpoint 是状态快照）。合成一个键就得「读-改-写」，
  而 Redis 的 GET/SET 不是原子的 —— 并发恢复时会丢写入。

────────────────────────────────────────────────────────────────────
★ 序列化必须用 `JsonPlusSerializer`
────────────────────────────────────────────────────────────────────
LangGraph 的 state 里可能有 tuple / set / 自定义 dataclass。
直接 `json.dumps` 会失败。基类自带的 `JsonPlusSerializer` 处理了这些类型，
`dumps_typed` / `loads_typed` 会额外带上类型标记（能还原 tuple 与 bytes）。

★ 而 `CachePort` 的 Redis 实现只接受 JSON 可序列化的值 ——
  所以这里存的是 **serializer 产出的 (type, base64) 二元组**，
  这是一个纯 list，JSON 能存。读回来再 `loads_typed` 还原。
"""
from __future__ import annotations

import logging
from typing import Any, Iterator, Sequence

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)

#: 键前缀。与项目其他缓存键共用 `health_training:` 命名空间，
#: 本类内部再加 `report:ckpt:` 便于 `clear_prefix("report:ckpt:")` 单清。
_KEY_PREFIX = "report:ckpt:"


class CacheCheckpointSaver(BaseCheckpointSaver):
    """把 LangGraph 的 checkpoint 存进 `CachePort`。

    用法::

        saver = CacheCheckpointSaver(container.cache, ttl_sec=7 * 86400)
        graph = builder.compile(checkpointer=saver)
    """

    def __init__(
        self,
        cache: Any,
        *,
        ttl_sec: int | None = 7 * 86400,
        serde: JsonPlusSerializer | None = None,
    ) -> None:
        """
        Args:
            cache: `CachePort` 实现（内存版或 Redis 版都行）。
            ttl_sec: 过期秒数。默认 7 天 —— 够主管隔周复核。
                ★ 过期后图无法恢复，只能重跑。这是可接受的：
                  重跑代价是一次 LLM 调用，比永久占内存划算。
            serde: 序列化器。默认 `JsonPlusSerializer`。
        """
        super().__init__(serde=serde or JsonPlusSerializer())
        self._cache = cache
        self._ttl = ttl_sec

    # ---------------- 内部 ----------------
    @staticmethod
    def _ckpt_key(thread_id: str, checkpoint_id: str) -> str:
        return f"{_KEY_PREFIX}{thread_id}:{checkpoint_id}"

    @staticmethod
    def _writes_key(thread_id: str, checkpoint_id: str) -> str:
        return f"{_KEY_PREFIX}{thread_id}:{checkpoint_id}:writes"

    def _dump(self, value: Any) -> list:
        """序列化成 JSON 可存的 list：`[type, base64]`。"""
        typ, payload = self.serde.dumps_typed(value)
        # payload 是 bytes，转成可 JSON 存的字符串
        import base64

        return [typ, base64.b64encode(payload).decode("ascii")]

    def _load(self, raw: Any) -> Any:
        """从 list 还原对象。结构不对返回 None（当未命中处理）。"""
        import base64

        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return None
        typ, b64 = raw
        try:
            payload = base64.b64decode(b64.encode("ascii"))
            return self.serde.loads_typed((typ, payload))
        except Exception as exc:                   # noqa: BLE001
            logger.warning("checkpoint 反序列化失败，按未命中处理: %s", exc)
            return None

    # ---------------- BaseCheckpointSaver 实现 ----------------
    def get_tuple(self, config: Any) -> CheckpointTuple | None:
        """按 config 取最近的 checkpoint。

        `config` 形如 `{"configurable": {"thread_id": ..., "checkpoint_id": ...}}`。
        没给 `checkpoint_id` 时取该 thread 最新一条。
        """
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        if not thread_id:
            return None

        checkpoint_id = conf.get("checkpoint_id")
        if not checkpoint_id:
            # 没指定 → 从索引里取最新
            checkpoint_id = self._latest_id(thread_id)
            if not checkpoint_id:
                return None

        raw = self._cache.get(self._ckpt_key(thread_id, checkpoint_id))
        restored = self._load(raw)
        if restored is None:
            return None

        checkpoint, metadata, parent_id = restored

        writes_raw = self._cache.get(
            self._writes_key(thread_id, checkpoint_id)) or []
        pending_writes: list[tuple[str, str, Any]] = []
        for item in writes_raw:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            task_id, channel, payload = item
            value = self._load(payload)
            pending_writes.append((task_id, channel, value))

        cfg: dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": conf.get("checkpoint_ns", ""),
                "checkpoint_id": checkpoint_id,
            }
        }
        if parent_id:
            cfg["configurable"]["parent_checkpoint_id"] = parent_id

        return CheckpointTuple(
            config=cfg,
            checkpoint=checkpoint,
            metadata=metadata,
            pending_writes=pending_writes,
            parent_config=(
                {"configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": conf.get("checkpoint_ns", ""),
                    "checkpoint_id": parent_id,
                }} if parent_id else None
            ),
        )

    def _latest_id(self, thread_id: str) -> str:
        """取该 thread 最新的 checkpoint_id。"""
        raw = self._cache.get(f"{_KEY_PREFIX}{thread_id}:latest")
        return raw if isinstance(raw, str) else ""

    def put(
        self,
        config: Any,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> Any:
        """写入一个 checkpoint，返回带 checkpoint_id 的新 config。"""
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id") or ""
        checkpoint_id = checkpoint.get("id") or ""
        parent_id = conf.get("checkpoint_id", "")

        if not thread_id or not checkpoint_id:
            logger.warning("checkpoint 缺少 thread_id/id，跳过写入")
            return config

        payload = self._dump((checkpoint, metadata, parent_id))
        self._cache.set(
            self._ckpt_key(thread_id, checkpoint_id), payload, ttl_sec=self._ttl)
        # 最新指针也要存，否则 get_tuple 不知道从哪找
        self._cache.set(
            f"{_KEY_PREFIX}{thread_id}:latest", checkpoint_id, ttl_sec=self._ttl)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": conf.get("checkpoint_ns", ""),
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: Any,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """写入某一步的节点产出。

        ★ 与 `put` 分开存（见模块注释）：合并会导致读-改-写丢数据。
        """
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id") or ""
        checkpoint_id = conf.get("checkpoint_id") or ""
        if not thread_id or not checkpoint_id:
            return

        key = self._writes_key(thread_id, checkpoint_id)
        existing = self._cache.get(key) or []
        if not isinstance(existing, list):
            existing = []

        for channel, value in writes:
            existing.append([task_id, channel, self._dump(value)])

        self._cache.set(key, existing, ttl_sec=self._ttl)

    def list(self, config: Any, **kwargs: Any) -> Iterator[CheckpointTuple]:
        """列出 checkpoint。

        ★ 简化实现：只返回**最新一条**。
          完整历史遍历需要额外维护一个 checkpoint 列表索引，
          而本项目没有「回放历史步骤」的需求（只有「恢复最新」）。
          如果将来要单步回放，再加索引。
        """
        conf = (config or {}).get("configurable", {})
        thread_id = conf.get("thread_id")
        if not thread_id:
            return iter(())
        latest = self.get_tuple({"configurable": {"thread_id": thread_id}})
        return iter([latest] if latest else [])

    def delete_thread(self, thread_id: str) -> None:
        """删掉一个 thread 的所有 checkpoint。

        ★ 报告被驳回重跑时调用：避免旧 checkpoint 一直堆着到 TTL。
        """
        self._cache.clear_prefix(f"{_KEY_PREFIX}{thread_id}:")
        logger.info("已清理 thread 的 checkpoint: %s", thread_id)

    def is_ready(self) -> bool:
        """存储器是否可用（Redis 挂了会返回 False）。"""
        try:
            self._cache.set("report:ckpt:__probe__", "1", ttl_sec=5)
            return self._cache.get("report:ckpt:__probe__") == "1"
        except Exception:                              # noqa: BLE001
            return False


__all__ = ["CacheCheckpointSaver"]
