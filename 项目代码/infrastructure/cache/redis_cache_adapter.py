"""Redis 缓存适配器。

把 `CachePort` 从进程内存换到 Redis，解决两个内存版做不到的事：

| 能力 | InMemory | Redis |
|---|---|---|
| 多实例共享（演练游标跨进程） | ❌ 各存各的 | ✅ |
| 重启不丢（会话可恢复） | ❌ | ✅（本容器开了 AOF） |

────────────────────────────────────────────────────────────────────
★ 与内存版最大的差别：必须序列化
────────────────────────────────────────────────────────────────────
内存版可以把任意 Python 对象直接塞进 dict，Redis 只能存 bytes/str。
所以写入前要 `json.dumps`，读出后要 `json.loads`。

这带来一条**必须遵守的约定**：

    只能存 JSON 可序列化的值 —— dict / list / str / int / float / bool / None

    不能存 dataclass 实例、set、datetime、Decimal。

★ 为什么不用 pickle：
  pickle 反序列化会执行任意代码，缓存内容若被污染就是远程代码执行。
  缓存是多实例共享的东西，攻击面比特意序列化大得多。
  JSON 虽有限制，但安全边界清楚 —— 这个项目存的是会话摘要与规划结果，
  用 JSON 完全够。

★ 读不出来怎么办：
  版本升级后旧缓存可能是别的结构（或非 JSON）。一律当作「未命中」返回 None，
  不要让反序列化异常冒到业务层 —— 缓存失败绝不该影响功能可用性。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from domain.ports import CachePort

logger = logging.getLogger(__name__)


class RedisCacheAdapter(CachePort):
    """Redis 版缓存。值以 JSON 存储。

    ★ url 无默认值：本项目禁止在代码里写死连接串，
      必须由装配根从 `settings.redis_url` 传入。
    """

    def __init__(
        self,
        *,
        url: str,
        key_prefix: str = "",
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
    ) -> None:
        """
        Args:
            url: 连接串，**必须由 settings 注入**（`settings.redis_url`）。
                不设默认值是有意的：硬编码连接串会被架构门禁拦下。
            key_prefix: 全局键前缀。多项目共用一个 Redis 实例时用它隔离，
                避免误删（`clear_prefix` 只在命名空间内生效）。
            socket_timeout / socket_connect_timeout: 都要设。
                ★ 不设超时的话，Redis 卡住会拖死整个请求线程 ——
                  缓存是加速手段，不该成为阻塞点。
        """
        import redis

        self._key_prefix = key_prefix
        self._pool = redis.ConnectionPool.from_url(
            url,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=True,      # 直接拿 str，省一层 decode
            health_check_interval=30,   # 长时间空闲后自动探活，避免用死连接
        )
        self._client = redis.Redis(connection_pool=self._pool)

    # ---------------- 内部 ----------------
    def _k(self, key: str) -> str:
        return f"{self._key_prefix}{key}" if self._key_prefix else key

    @staticmethod
    def _dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _loads(raw: str | None) -> Any | None:
        """反序列化。任何异常都当作未命中。"""
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            # 旧格式/被污染/非 JSON —— 当未命中，不抛
            logger.warning("缓存值无法反序列化，按未命中处理: %.60s", raw)
            return None

    # ---------------- 端口实现 ----------------
    def get(self, key: str) -> Any | None:
        try:
            return self._loads(self._client.get(self._k(key)))
        except Exception as exc:                       # noqa: BLE001
            # ★ 缓存读失败绝不能影响业务：降级为「未命中」
            logger.warning("缓存读失败（按未命中处理）key=%s: %s", key, exc)
            return None

    def set(self, key: str, value: Any, *, ttl_sec: int | None = None) -> None:
        try:
            raw = self._dumps(value)
        except (TypeError, ValueError) as exc:
            # 不可序列化的值：不写，但也不炸调用方
            logger.warning(
                "缓存值不可 JSON 序列化，跳过写入 key=%s type=%s: %s",
                key, type(value).__name__, exc)
            return

        try:
            if ttl_sec is None:
                self._client.set(self._k(key), raw)
            else:
                # ttl 必须为正，否则 Redis 报错；<=0 直接不写（等价于立即过期）
                if ttl_sec <= 0:
                    self.delete(key)
                    return
                self._client.set(self._k(key), raw, ex=ttl_sec)
        except Exception as exc:                       # noqa: BLE001
            logger.warning("缓存在写失败 key=%s: %s", key, exc)

    def delete(self, key: str) -> None:
        try:
            self._client.delete(self._k(key))
        except Exception as exc:                       # noqa: BLE001
            logger.warning("缓存删除失败 key=%s: %s", key, exc)

    def exists(self, key: str) -> bool:
        try:
            return bool(self._client.exists(self._k(key)))
        except Exception as exc:                       # noqa: BLE001
            logger.warning("缓存 exists 失败 key=%s: %s", key, exc)
            return False

    def clear_prefix(self, prefix: str) -> int:
        """按前缀批量删除。

        ★ 用 SCAN 而不是 KEYS：
          KEYS 会阻塞 Redis（单线程模型），大库上可能卡几秒，
          影响所有使用这个实例的应用。SCAN 是游标式，分批不阻塞。

        ★ 只在自身 key_prefix 命名空间内扫描：
          避免多项目共用实例时误删别人的键。
        """
        pattern = f"{self._k(prefix)}*"
        removed = 0
        try:
            batch: list[str] = []
            for key in self._client.scan_iter(match=pattern, count=500):
                batch.append(key)
                # 分批删，避免单条命令塞太多键
                if len(batch) >= 500:
                    removed += self._client.delete(*batch)
                    batch = []
            if batch:
                removed += self._client.delete(*batch)
        except Exception as exc:                       # noqa: BLE001
            logger.warning("按前缀清理失败 prefix=%s: %s", prefix, exc)
            return removed

        if removed:
            logger.info("按前缀清理缓存(redis): prefix=%s, count=%d",
                        prefix, removed)
        return removed

    # ---------------- 运维辅助 ----------------
    def ping(self) -> bool:
        """连通性检查。供健康检查与装配时探活用。"""
        try:
            return bool(self._client.ping())
        except Exception:                              # noqa: BLE001
            return False

    def close(self) -> None:
        try:
            self._pool.disconnect()
        except Exception:                              # noqa: BLE001
            pass
