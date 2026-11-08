"""内存缓存适配器。

P0 阶段 Redis 未安装，先用进程内实现把 CachePort 跑通。
换来的是：Assembly 里 cache 端口真的就绪，而不是永远 pending。

★ 限制：多进程/多实例不共享。生产部署需换 RedisAdapter。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from domain.ports import CachePort

logger = logging.getLogger(__name__)


class InMemoryCacheAdapter(CachePort):
    """进程内缓存，线程安全，TTL 惰性过期。"""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._deadline: dict[str, float] = {}
        self._lock = threading.RLock()

    def _gc(self, key: str) -> bool:
        """检查并清理已过期键。返回 True 表示已过期。

        惰性过期而非后台线程：P0 的数据量小，扫描成本可忽略，
        避免引入后台线程带来的退出时序问题。
        """
        deadline = self._deadline.get(key)
        if deadline is None:
            return False
        if time.monotonic() < deadline:
            return False
        self._data.pop(key, None)
        self._deadline.pop(key, None)
        return True

    def get(self, key: str) -> Any | None:
        with self._lock:
            if self._gc(key):
                return None
            return self._data.get(key)

    def set(self, key: str, value: Any, *, ttl_sec: int | None = None) -> None:
        with self._lock:
            self._data[key] = value
            if ttl_sec is None:
                self._deadline.pop(key, None)
            else:
                self._deadline[key] = time.monotonic() + ttl_sec

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._deadline.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def clear_prefix(self, prefix: str) -> int:
        with self._lock:
            keys = [k for k in self._data if k.startswith(prefix)]
            for k in keys:
                self.delete(k)
            if keys:
                logger.info("按前缀清理缓存: prefix=%s, count=%d", prefix, len(keys))
            return len(keys)
