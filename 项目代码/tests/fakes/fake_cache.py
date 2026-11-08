"""缓存端口的测试替身。内存字典 + TTL 惰性过期。"""
from __future__ import annotations

import time
from typing import Any

from domain.ports import CachePort


class FakeCachePort(CachePort):
    """内存缓存。TTL 用单调时钟，不依赖系统时间跳变。"""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._expire_at: dict[str, float] = {}

    def _expired(self, key: str) -> bool:
        deadline = self._expire_at.get(key)
        if deadline is None:
            return False
        if time.monotonic() >= deadline:
            self._data.pop(key, None)
            self._expire_at.pop(key, None)
            return True
        return False

    def get(self, key: str) -> Any | None:
        if self._expired(key):
            return None
        return self._data.get(key)

    def set(self, key: str, value: Any, *, ttl_sec: int | None = None) -> None:
        self._data[key] = value
        if ttl_sec is None:
            self._expire_at.pop(key, None)
        else:
            self._expire_at[key] = time.monotonic() + ttl_sec

    def delete(self, key: str) -> None:
        self._data.pop(key, None)
        self._expire_at.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def clear_prefix(self, prefix: str) -> int:
        keys = [k for k in self._data if k.startswith(prefix)]
        for k in keys:
            self.delete(k)
        return len(keys)
