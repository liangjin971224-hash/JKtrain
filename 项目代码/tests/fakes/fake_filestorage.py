"""文件存储端口的测试替身。内存实现，不碰磁盘。

★ 三条安全要求也要在替身里成立（扩展名白名单 / 大小限制 / 路径穿越），
   否则用它测 P6 的「上传 .exe 应被拒」就是假绿。
"""
from __future__ import annotations

from pathlib import PurePosixPath

from domain.exceptions import (
    FileTooLargeError,
    PathTraversalError,
    UnsupportedFileTypeError,
)
from domain.ports import FileStoragePort, StoredFile

_ALLOWED_SUFFIX = {".pdf", ".jpg", ".jpeg", ".png"}
_MIME = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


class InMemoryFileStorage(FileStoragePort):
    """内存文件存储。"""

    MAX_SIZE = 20 * 1024 * 1024

    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}

    def save(
        self,
        *,
        content: bytes,
        file_name: str,
        category: str,
        sub_path: str,
    ) -> StoredFile:
        if len(content) > self.MAX_SIZE:
            raise FileTooLargeError(len(content), self.MAX_SIZE)

        suffix = PurePosixPath(file_name).suffix.lower()
        if suffix not in _ALLOWED_SUFFIX:
            raise UnsupportedFileTypeError(suffix)

        # ★ 与 LocalFileStorageAdapter 同语义：逐个校验输入，而不是只看拼接结果。
        #   `submissions/a` + `../../evil.pdf` 拼接后仍在根内，
        #   但用户已能控制落盘目录，必须挡。
        for field, value in (("category", category), ("file_name", file_name)):
            normalized = value.replace("\\", "/")
            if not normalized.strip() or "/" in normalized or ":" in normalized:
                raise PathTraversalError(f"{field}={value}")

        normalized_sub = sub_path.replace("\\", "/").strip("/")
        if not normalized_sub:
            raise PathTraversalError("sub_path 为空")
        for part in normalized_sub.split("/"):
            if part in ("", ".", "..") or ":" in part:
                raise PathTraversalError(f"sub_path={sub_path}")

        rel = PurePosixPath(category) / normalized_sub / file_name
        key = str(rel)
        self._files[key] = content
        return StoredFile(
            relative_path=key,
            file_name=file_name,
            file_size=len(content),
            mime_type=_MIME.get(suffix, ""),
        )

    def read(self, relative_path: str) -> bytes:
        if relative_path not in self._files:
            raise FileNotFoundError(relative_path)
        return self._files[relative_path]

    def delete(self, relative_path: str) -> None:
        self._files.pop(relative_path, None)

    def exists(self, relative_path: str) -> bool:
        return relative_path in self._files
