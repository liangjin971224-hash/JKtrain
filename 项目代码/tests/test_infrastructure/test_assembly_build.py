"""装配根的真实构建测试。

P0 阶段大部分适配器还没实现，所以重点是验证
「缺适配器时优雅降级」这一行为——这是 P0「空服务能启动」的关键。
"""
from __future__ import annotations

import sys

import pytest

from config.assembly import build_container
from config.settings import Settings


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,                                   # type: ignore[call-arg]
        neo4j_uri="bolt://localhost:7687",
        neo4j_database="jiankangshujuku",
        milvus_host="localhost",
        milvus_port=19530,
        embed_model_dir=str(tmp_path / "model"),
        file_storage_root=str(tmp_path / "data"),
    )


class TestBuildContainerDegrades:
    """缺适配器时必须降级，不能抛异常。"""

    def test_pending_ports_recorded_with_reason(self, tmp_path,
                                                monkeypatch) -> None:
        """未实现的适配器要登记到 pending，并说明是哪个阶段补。"""
        # 屏蔽真连接，避免依赖本机服务是否在线
        monkeypatch.setitem(sys.modules, "neo4j", None)
        monkeypatch.setitem(sys.modules, "pymilvus", None)

        c = build_container(_settings(tmp_path))
        try:
            # P0 只实现了 cache 与 file_storage
            ready = c.ready_ports()
            assert "cache" in ready
            assert "file_storage" in ready

            # 其他的必须在 pending 里有原因，不能静默为 None
            for port in ("graph", "vector", "keyword", "llm", "repository"):
                if getattr(c, port) is None:
                    assert port in c.pending, f"{port} 为 None 却没登记原因"
                    assert c.pending[port], f"{port} 的 pending 原因为空"
        finally:
            c.close()

    def test_does_not_raise_when_neo4j_down(self, tmp_path, monkeypatch) -> None:
        """Neo4j 连不上时容器仍要能构建出来（P0 要求空服务可启动）。"""
        class _Driver:
            def __init__(self, *a, **kw) -> None:
                raise ConnectionError("服务未启动")

            def close(self) -> None:
                pass

        import types

        fake = types.ModuleType("neo4j")
        fake.GraphDatabase = types.SimpleNamespace(driver=_Driver)
        monkeypatch.setitem(sys.modules, "neo4j", fake)

        c = build_container(_settings(tmp_path))       # 不应抛
        try:
            assert c.neo4j_driver is None
            assert "neo4j_driver" in c.pending
        finally:
            c.close()

    def test_close_safe_when_nothing_built(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "neo4j", None)
        monkeypatch.setitem(sys.modules, "pymilvus", None)
        c = build_container(_settings(tmp_path))
        c.close()
        c.close()                                       # 幂等


class TestBuildContainerWithRealServices:
    """真实服务可用时，两个已实现的适配器应就绪。"""

    def test_cache_and_storage_ready(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "neo4j", None)
        monkeypatch.setitem(sys.modules, "pymilvus", None)

        c = build_container(_settings(tmp_path))
        try:
            from infrastructure.cache.inmemory_cache_adapter import (
                InMemoryCacheAdapter,
            )
            from infrastructure.storage.local_file_storage_adapter import (
                LocalFileStorageAdapter,
            )

            assert isinstance(c.cache, InMemoryCacheAdapter)
            assert isinstance(c.file_storage, LocalFileStorageAdapter)
        finally:
            c.close()
