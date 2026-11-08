"""文件存储端口（答卷归档专用）。

为什么单独一个 Port 而不是塞进 RepositoryPort：
答卷归档是「文件」不是「数据」。文件可独立迁移到对象存储，
测试时可用内存实现，换存储只换 Adapter。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StoredFile:
    """已存储文件的描述。

    `relative_path` 是相对路径，**不暴露绝对路径**——
    换机器/换目录时不用改数据，也不泄漏服务器目录结构。
    """

    relative_path: str
    file_name: str
    file_size: int
    mime_type: str = ""


class FileStoragePort(ABC):
    """文件存储端口。"""

    @abstractmethod
    def save(
        self,
        *,
        content: bytes,
        file_name: str,
        category: str,
        sub_path: str,
    ) -> StoredFile:
        """保存文件。

        Args:
            content: 文件二进制内容。
            file_name: 原始文件名（用于展示）。
            category: 分类目录，如 'submissions' / 'exports'。
            sub_path: 子路径，如 '2026/11/S1-20261105-001'。

        Returns:
            存储结果，含相对路径。

        Raises:
            FileTooLargeError: 超过大小限制（默认 20MB）。
            UnsupportedFileTypeError: 不在允许列表（pdf/jpg/png）。
            PathTraversalError: sub_path 解析后逃出存储根目录。
        """

    @abstractmethod
    def read(self, relative_path: str) -> bytes:
        """读取文件内容。不存在时抛 FileNotFoundError。"""

    @abstractmethod
    def delete(self, relative_path: str) -> None:
        """删除文件。不存在时静默返回（幂等）。"""

    @abstractmethod
    def exists(self, relative_path: str) -> bool:
        """判断文件是否存在。"""
