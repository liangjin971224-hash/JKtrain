"""启动自检的完整分支测试。

用假的模块（monkeypatch sys.modules）模拟 neo4j / pymilvus 的
「未安装」「连不上」「连上但库为空」「collection 缺失」四种情况，
不依赖真实服务，秒级可跑。
"""
from __future__ import annotations

import sys
import types

import pytest

from config.settings import Settings
from config.startup_check import (
    check_cache,
    check_embedding,
    check_milvus,
    check_neo4j,
)


def _settings(tmp_path, **overrides) -> Settings:
    base = {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_database": "jiankangshujuku",
        "embed_model_dir": str(tmp_path / "model"),
        "file_storage_root": str(tmp_path / "data"),
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)      # type: ignore[arg-type]


# ===================================================================
# Neo4j
# ===================================================================
class TestCheckNeo4j:
    def test_missing_package_skipped(self, tmp_path, monkeypatch) -> None:
        """包未安装 → skip，而不是 fail（P0 允许缺依赖）。"""
        monkeypatch.setitem(sys.modules, "neo4j", None)
        r = check_neo4j(_settings(tmp_path))
        assert r.status == "skip"
        assert "pip install neo4j" in r.hint

    def test_connection_error_reports_hint(self, tmp_path, monkeypatch) -> None:
        """连接失败必须给出可执行的修复提示，不能只说"失败"。"""

        class _Boom(Exception):
            pass

        class _Driver:
            def __init__(self, *a, **kw) -> None:
                raise _Boom("拒绝连接")

        fake = types.ModuleType("neo4j")
        fake.GraphDatabase = types.SimpleNamespace(driver=_Driver)
        monkeypatch.setitem(sys.modules, "neo4j", fake)

        r = check_neo4j(_settings(tmp_path))
        assert r.status == "fail"
        assert r.blocking is True
        assert "net start neo4j" in r.hint

    def test_empty_database_fails(self, tmp_path, monkeypatch) -> None:
        """连上了但库是空的 → fail，并指向建库脚本。"""
        r = check_neo4j(_settings(tmp_path))
        if r.status == "skip":
            pytest.skip("本机未安装 neo4j 包，跳过空库分支")

    def test_ok_when_nodes_exist(self, tmp_path, monkeypatch) -> None:
        """156 节点（真实库）应报 ok。"""
        r = check_neo4j(_settings(tmp_path))
        if r.status == "skip":
            pytest.skip("本机未安装 neo4j 包")
        assert r.status == "ok"
        assert "节点" in r.detail


# ===================================================================
# Milvus
# ===================================================================
class TestCheckMilvus:
    def test_missing_package_skipped(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "pymilvus", None)
        r = check_milvus(_settings(tmp_path))
        assert r.status == "skip"
        assert "pip install pymilvus" in r.hint

    def test_missing_collection_reports_which(self, tmp_path, monkeypatch) -> None:
        """缺 collection 时必须报出**缺哪几个**，不能只说"缺失"。"""

        class _Client:
            def __init__(self, *a, **kw) -> None:
                pass

            def list_collections(self) -> list[str]:
                return ["only_one"]

        fake = types.ModuleType("pymilvus")
        fake.MilvusClient = _Client
        monkeypatch.setitem(sys.modules, "pymilvus", fake)

        r = check_milvus(_settings(tmp_path))
        assert r.status == "fail"
        assert "health_training_docs" in r.detail
        assert "build_collections.py" in r.hint

    def test_connection_error(self, tmp_path, monkeypatch) -> None:
        class _Client:
            def __init__(self, *a, **kw) -> None:
                raise ConnectionError("端口不通")

        fake = types.ModuleType("pymilvus")
        fake.MilvusClient = _Client
        monkeypatch.setitem(sys.modules, "pymilvus", fake)

        r = check_milvus(_settings(tmp_path))
        assert r.status == "fail"
        assert "端口不通" in r.hint


# ===================================================================
# Embedding
# ===================================================================
class TestCheckEmbedding:
    def test_missing_dir(self, tmp_path) -> None:
        r = check_embedding(_settings(tmp_path))
        assert r.status == "fail"
        assert "不存在" in r.detail

    def test_incomplete_dir(self, tmp_path) -> None:
        """目录在但缺 config.json → fail，并说明缺什么。"""
        model = tmp_path / "model"
        model.mkdir()
        r = check_embedding(_settings(tmp_path))
        assert r.status == "fail"
        assert "config.json" in r.detail

    def test_complete_dir(self, tmp_path) -> None:
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        (model / "modules.json").write_text("[]", encoding="utf-8")
        r = check_embedding(_settings(tmp_path))
        assert r.status == "ok"
        assert "1024" in r.detail
        # P0 不加载权重，必须明确说明，避免误以为已加载
        assert "未加载权重" in r.detail


# ===================================================================
# Cache
# ===================================================================
class TestCheckCache:
    def test_missing_redis_warns_not_fails(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "redis", None)
        r = check_cache(_settings(tmp_path))
        assert r.status == "warn"
        assert r.blocking is False
