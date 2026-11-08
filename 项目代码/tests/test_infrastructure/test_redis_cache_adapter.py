"""Redis 缓存适配器测试。

★ 核心思路：**用同一套契约测试跑两个实现**。
  如果 Redis 版和内存版行为不一致，那就是 bug——业务代码不该因为
  换后端而改变行为（除了「跨进程共享」这个预期内的差异）。

分两层：
| 层 | 需要真 Redis 吗 | 验证什么 |
|---|---|---|
| 契约测试 | **是**（标 integration） | 5 个方法的行为与内存版一致 |
| 降级测试 | 否 | Redis 不可用时不影响调用方 |
"""
from __future__ import annotations

import pytest

from domain.ports import CachePort
from infrastructure.cache.inmemory_cache_adapter import InMemoryCacheAdapter

#: 真 Redis 是否可用。用于决定 integration 测试是跑还是跳过。
def _redis_available(url: str = "redis://127.0.0.1:6379/0") -> bool:
    try:
        import redis

        r = redis.Redis.from_url(url, socket_timeout=1.5,
                                 socket_connect_timeout=1.5)
        return bool(r.ping())
    except Exception:                                  # noqa: BLE001
        return False


REDIS_URL = "redis://127.0.0.1:6379/0"
HAS_REDIS = _redis_available()

skip_no_redis = pytest.mark.skipif(
    not HAS_REDIS, reason="本机 Redis 不可用（docker start health-training-redis）")


# ===================================================================
# 契约测试：两个实现必须通过同一套断言
# ===================================================================
@pytest.fixture(params=["memory", "redis"])
def cache(request) -> CachePort:
    """参数化夹具：同一组测试分别跑内存版与 Redis 版。"""
    if request.param == "memory":
        return InMemoryCacheAdapter()

    if not HAS_REDIS:
        pytest.skip("Redis 不可用")

    from infrastructure.cache.redis_cache_adapter import RedisCacheAdapter

    adapter = RedisCacheAdapter(
        url=REDIS_URL, key_prefix="test:cache:contract:")
    # 每个用例前清干净，避免相互污染
    adapter.clear_prefix("")
    return adapter


class TestContract:
    """CachePort 的通用契约。两个实现都必须满足。"""

    def test_get_missing_returns_none(self, cache: CachePort) -> None:
        assert cache.get("不存在的键") is None

    def test_set_then_get(self, cache: CachePort) -> None:
        cache.set("k1", {"a": 1, "b": [1, 2]})
        assert cache.get("k1") == {"a": 1, "b": [1, 2]}

    def test_exists(self, cache: CachePort) -> None:
        assert cache.exists("k2") is False
        cache.set("k2", "v")
        assert cache.exists("k2") is True

    def test_delete_is_idempotent(self, cache: CachePort) -> None:
        cache.set("k3", "v")
        cache.delete("k3")
        assert cache.get("k3") is None
        # 再删一次不该抛
        cache.delete("k3")
        assert cache.get("k3") is None

    def test_clear_prefix(self, cache: CachePort) -> None:
        cache.set("drill:a:1", 1)
        cache.set("drill:a:2", 2)
        cache.set("other:b", 3)

        removed = cache.clear_prefix("drill:a:")
        assert removed == 2
        assert cache.get("drill:a:1") is None
        assert cache.get("other:b") == 3

    def test_clear_prefix_no_match(self, cache: CachePort) -> None:
        assert cache.clear_prefix("完全不存在的前缀:") == 0

    def test_ttl_expires(self, cache: CachePort) -> None:
        """TTL 必须真的生效。用 1 秒 + sleep 验证。"""
        import time

        cache.set("k_ttl", "v", ttl_sec=1)
        assert cache.get("k_ttl") == "v"
        time.sleep(1.2)
        assert cache.get("k_ttl") is None

    def test_none_value_is_not_cached(self, cache: CachePort) -> None:
        """存 None 等价于「未命中」。

        ★ 这不是缺陷，是接口本身的语义：
          内存版 `exists()` 实现为 `get(...) is not None`，
          所以存 None 与没存在行为上不可区分。
          两个实现保持一致即可 —— 业务层不该依赖「存了 None」这件事。
        """
        cache.set("k_none", None)
        assert cache.get("k_none") is None


# ===================================================================
# Redis 特有行为
# ===================================================================
@skip_no_redis
class TestRedisSpecific:
    def _adapter(self, prefix: str = "test:cache:specific:"):
        from infrastructure.cache.redis_cache_adapter import RedisCacheAdapter

        a = RedisCacheAdapter(url=REDIS_URL, key_prefix=prefix)
        a.clear_prefix("")
        return a

    def test_ping(self) -> None:
        assert self._adapter().ping() is True

    def test_key_prefix_isolation(self) -> None:
        """不同前缀互不影响 —— 多项目共用实例时不误删。"""
        a = self._adapter("test:iso:a:")
        b = self._adapter("test:iso:b:")

        a.set("shared", "from-a")
        b.set("shared", "from-b")

        assert a.get("shared") == "from-a"
        assert b.get("shared") == "from-b"

        # a 清空自己的命名空间，不影响 b
        a.clear_prefix("")
        assert a.get("shared") is None
        assert b.get("shared") == "from-b"
        b.clear_prefix("")

    def test_unserializable_value_skipped(self) -> None:
        """不可 JSON 序列化的值：跳过写入，但不抛异常。"""
        a = self._adapter()

        class Custom:
            pass

        a.set("k_custom", Custom())     # 不该抛
        assert a.get("k_custom") is None

    def test_corrupted_value_treated_as_miss(self) -> None:
        """缓存里是非 JSON 内容 → 当未命中，不抛异常。

        ★ 这个场景真实存在：版本升级后旧缓存结构变了。
          缓存读失败绝不该让业务崩掉。
        """
        a = self._adapter()
        # 绕过 set 直接塞一个非 JSON 字符串
        a._client.set(a._k("k_bad"), "not-a-json{{{")
        assert a.get("k_bad") is None

    def test_bad_url_does_not_raise(self) -> None:
        """连不上的 Redis：方法调用都不该抛，只是返回「未命中」。"""
        from infrastructure.cache.redis_cache_adapter import RedisCacheAdapter

        a = RedisCacheAdapter(
            url="redis://127.0.0.1:6399/0",     # 故意错的端口
            socket_timeout=0.5, socket_connect_timeout=0.5,
        )
        assert a.ping() is False
        assert a.get("k") is None            # 不抛
        a.set("k", "v")                      # 不抛
        assert a.exists("k") is False
        a.delete("k")                        # 不抛
        assert a.clear_prefix("x") == 0
