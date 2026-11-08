"""缓存端口。

用途仅限两类：会话态（演练多轮对话）、临时结果（检索缓存）。
★ 不做业务数据持久化 —— 那是 RepositoryPort 的事。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CachePort(ABC):
    """缓存端口。"""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """读缓存。不存在返回 None（不抛异常）。"""

    @abstractmethod
    def set(self, key: str, value: Any, *, ttl_sec: int | None = None) -> None:
        """写缓存。

        Args:
            key: 键。
            value: 值（实现层需可序列化）。
            ttl_sec: 过期秒数。None 表示不过期。
        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """删除缓存。不存在时静默返回（幂等）。"""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """判断键是否存在。"""

    @abstractmethod
    def clear_prefix(self, prefix: str) -> int:
        """按前缀批量清除（如演练结束后清 `drill:{id}:*`）。

        Returns:
            实际删除的键数量。
        """
