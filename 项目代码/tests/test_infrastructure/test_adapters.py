"""已实现的适配器集成测试（真实磁盘，但不连数据库）。

P0 阶段真正写完的适配器只有两个：
    infrastructure/cache/inmemory_cache_adapter.py
    infrastructure/storage/local_file_storage_adapter.py

为什么这两个要真测：
    文件存储的三条安全要求（路径穿越、扩展名、大小）一旦漏，
    就是安全漏洞而不是功能缺陷。Fake 测过不等于 Adapter 也过。
"""
from __future__ import annotations

import pytest

from domain.exceptions import (
    FileTooLargeError,
    PathTraversalError,
    UnsupportedFileTypeError,
)
from infrastructure.cache.inmemory_cache_adapter import InMemoryCacheAdapter
from infrastructure.storage.local_file_storage_adapter import (
    LocalFileStorageAdapter,
)


# ===================================================================
# 本地文件存储适配器
# ===================================================================
class TestLocalFileStorageAdapter:
    """三条安全要求必须在真实现上也成立。"""

    @pytest.fixture()
    def storage(self, tmp_path):
        return LocalFileStorageAdapter(root=tmp_path, max_bytes=1024)

    def test_save_read_roundtrip(self, storage, tmp_path) -> None:
        stored = storage.save(
            content=b"%PDF-1.4 hello",
            file_name="E001.pdf",
            category="submissions",
            sub_path="2026/11/S1-20261105-001",
        )
        assert stored.file_size == len(b"%PDF-1.4 hello")
        assert stored.mime_type == "application/pdf"
        # 数据库只存相对路径，且必须是正斜杠
        assert stored.relative_path.startswith("submissions/2026/11/")
        assert "\\" not in stored.relative_path
        # 真实落到磁盘
        assert (tmp_path / stored.relative_path).is_file()
        assert storage.read(stored.relative_path) == b"%PDF-1.4 hello"

    def test_reject_exe(self, storage) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            storage.save(content=b"MZ", file_name="x.exe",
                         category="submissions", sub_path="a")

    def test_reject_php(self, storage) -> None:
        """伪造成 pdf 后缀也要能过，但 .php 必须挡。"""
        with pytest.raises(UnsupportedFileTypeError):
            storage.save(content=b"<?php", file_name="shell.php",
                         category="submissions", sub_path="a")

    def test_reject_oversize(self, storage) -> None:
        with pytest.raises(FileTooLargeError):
            storage.save(content=b"0" * 1025, file_name="big.pdf",
                         category="submissions", sub_path="a")

    def test_reject_path_traversal_in_sub_path(self, storage) -> None:
        """`sub_path` 来自用户输入，必须挡住 `../`。"""
        with pytest.raises(PathTraversalError):
            storage.save(content=b"%PDF", file_name="a.pdf",
                         category="submissions", sub_path="../../../evil")

    def test_reject_path_traversal_in_file_name(self, storage) -> None:
        """文件名也可能被构造，同样要挡。"""
        with pytest.raises(PathTraversalError):
            storage.save(content=b"%PDF", file_name="../../evil.pdf",
                         category="submissions", sub_path="a")

    def test_read_missing_raises(self, storage) -> None:
        with pytest.raises(FileNotFoundError):
            storage.read("submissions/nope.pdf")

    def test_delete_is_idempotent(self, storage) -> None:
        stored = storage.save(content=b"%PDF", file_name="a.pdf",
                              category="exports", sub_path="tmp")
        storage.delete(stored.relative_path)
        storage.delete(stored.relative_path)     # 第二次不抛
        assert storage.exists(stored.relative_path) is False

    def test_exists_false_for_traversal(self, storage) -> None:
        """越权路径的 exists 应返回 False 而不是抛异常。"""
        assert storage.exists("../../etc/passwd") is False

    def test_jpg_png_allowed(self, storage) -> None:
        for name, mime in (("a.jpg", "image/jpeg"),
                           ("b.png", "image/png")):
            stored = storage.save(content=b"img", file_name=name,
                                  category="submissions", sub_path="x")
            assert stored.mime_type == mime


# ===================================================================
# 内存缓存适配器
# ===================================================================
class TestInMemoryCacheAdapter:
    @pytest.fixture()
    def cache(self):
        return InMemoryCacheAdapter()

    def test_set_get(self, cache) -> None:
        cache.set("k", {"a": 1})
        assert cache.get("k") == {"a": 1}

    def test_miss_returns_none(self, cache) -> None:
        assert cache.get("none") is None

    def test_delete(self, cache) -> None:
        cache.set("k", 1)
        cache.delete("k")
        assert cache.exists("k") is False

    def test_delete_missing_ok(self, cache) -> None:
        cache.delete("never")            # 幂等，不抛

    def test_clear_prefix(self, cache) -> None:
        cache.set("drill:1:turn1", "a")
        cache.set("drill:1:turn2", "b")
        cache.set("qa:cache:1", "c")
        assert cache.clear_prefix("drill:1:") == 2
        assert cache.exists("qa:cache:1") is True

    def test_ttl_expired(self, cache) -> None:
        cache.set("k", "v", ttl_sec=0)
        assert cache.get("k") is None

    def test_ttl_not_expired(self, cache) -> None:
        cache.set("k", "v", ttl_sec=300)
        assert cache.get("k") == "v"

    def test_overwrite_clears_ttl(self, cache) -> None:
        cache.set("k", "v1", ttl_sec=0)
        cache.set("k", "v2")             # 不带 TTL → 覆盖后应长期有效
        assert cache.get("k") == "v2"
