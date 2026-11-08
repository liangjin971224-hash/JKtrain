"""本地文件存储适配器（答卷归档）。

三条安全要求（缺一不可）：
    1. 路径穿越防护   —— sub_path 来自用户输入，必须校验解析后仍在 root 内
    2. 扩展名白名单   —— 防止上传 .exe / .php
    3. 只存相对路径   —— 换机器/换目录不用改数据

★ 校验顺序很关键：先挡路径穿越，再判类型。
  否则一个 `../../x.pdf` 会先通过扩展名检查再写到根目录外。
"""
from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath

from domain.exceptions import (
    FileTooLargeError,
    PathTraversalError,
    UnsupportedFileTypeError,
)
from domain.ports import FileStoragePort, StoredFile

logger = logging.getLogger(__name__)

ALLOWED_SUFFIX_MIME = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

DEFAULT_MAX_BYTES = 20 * 1024 * 1024


class LocalFileStorageAdapter(FileStoragePort):
    """本地磁盘存储。"""

    def __init__(self, root: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._root = Path(root).resolve()
        self._max_bytes = max_bytes

    # ---------------- 内部 ----------------
    def _validate_name(self, value: str, field: str) -> None:
        """校验文件名/子路径不含路径分隔符与上跳。

        ★ 必须在拼接**之前**逐个校验。
          原因：`PurePosixPath("a") / "../../evil.pdf"` 不会规范化 `..`，
          拼接结果 `submissions/a/../../evil.pdf` 解析后落到根目录下——
          虽然仍在 root 内，但用户已经能控制落盘目录，属于越权写入。
          这类问题只用「解析后是否在 root 内」这一道检查是拦不住的。

        Raises:
            PathTraversalError: 含分隔符、`..` 或为空。
        """
        if not value or not value.strip():
            raise PathTraversalError(f"{field} 为空")
        # 统一按正反斜杠检查，避免 Windows 上用 `/` 绕过
        normalized = value.replace("\\", "/")
        if "/" in normalized:
            raise PathTraversalError(f"{field} 含路径分隔符: {value}")
        if normalized in (".", ".."):
            raise PathTraversalError(f"{field} 为相对路径标记: {value}")
        # Windows 盘符与保留设备名
        if ":" in normalized:
            raise PathTraversalError(f"{field} 含盘符: {value}")

    def _validate_sub_path(self, value: str) -> None:
        """校验多级子路径：允许 `2026/11/E001`，但不允许 `..` 与绝对路径。

        Raises:
            PathTraversalError: 含上跳或为绝对路径。
        """
        if not value or not value.strip():
            raise PathTraversalError("sub_path 为空")
        normalized = value.replace("\\", "/").strip("/")
        parts = normalized.split("/")
        for part in parts:
            if part in ("", ".", ".."):
                raise PathTraversalError(f"sub_path 含非法段: {value}")
            if ":" in part:
                raise PathTraversalError(f"sub_path 含盘符: {value}")

    def _resolve(self, relative_path: str) -> Path:
        """把相对路径解析为绝对路径，并校验未逃出根目录。

        这是最后一道防线：即使前面漏了，这里也不允许写出去。

        Raises:
            PathTraversalError: 解析后不在 root 内。
        """
        candidate = (self._root / relative_path).resolve()
        if not candidate.is_relative_to(self._root):
            raise PathTraversalError(relative_path)
        return candidate

    # ---------------- 端口实现 ----------------
    def save(
        self,
        *,
        content: bytes,
        file_name: str,
        category: str,
        sub_path: str,
    ) -> StoredFile:
        # 1. 大小
        if len(content) > self._max_bytes:
            raise FileTooLargeError(len(content), self._max_bytes)

        # 2. 扩展名白名单
        suffix = PurePosixPath(file_name).suffix.lower()
        if suffix not in ALLOWED_SUFFIX_MIME:
            raise UnsupportedFileTypeError(suffix)

        # 3. 逐个校验输入（顺序重要：先挡住 `..`，再拼接）
        self._validate_name(category, "category")
        self._validate_name(file_name, "file_name")
        self._validate_sub_path(sub_path)

        # 4. 拼接后统一再做一次「是否逃出 root」的解析校验
        rel = PurePosixPath(category) / sub_path / file_name
        abs_path = self._resolve(str(rel))

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(content)

        rel_str = str(rel).replace("\\", "/")
        logger.info(
            "文件已保存: path=%s, size=%d, mime=%s",
            rel_str, len(content), ALLOWED_SUFFIX_MIME.get(suffix, ""),
        )
        return StoredFile(
            relative_path=rel_str,
            file_name=file_name,
            file_size=len(content),
            mime_type=ALLOWED_SUFFIX_MIME.get(suffix, ""),
        )

    def read(self, relative_path: str) -> bytes:
        abs_path = self._resolve(relative_path)
        if not abs_path.is_file():
            raise FileNotFoundError(relative_path)
        return abs_path.read_bytes()

    def delete(self, relative_path: str) -> None:
        abs_path = self._resolve(relative_path)
        # 幂等：不存在时静默返回
        if abs_path.is_file():
            abs_path.unlink()
            logger.info("文件已删除: %s", relative_path)

    def exists(self, relative_path: str) -> bool:
        try:
            return self._resolve(relative_path).is_file()
        except PathTraversalError:
            return False
