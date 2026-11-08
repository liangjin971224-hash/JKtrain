"""基础设施层异常。

三层异常体系的第二层：连接、超时、协议错误。
每个异常必须带上下文（product_id / database / 原始异常），
否则线上排障只能靠猜。
"""
from __future__ import annotations


class InfrastructureError(Exception):
    """所有基础设施异常的基类。"""


class GraphQueryError(InfrastructureError):
    """图谱查询失败（连接、Cypher 语法、超时）。"""

    def __init__(self, message: str, *, product_id: str = "",
                 database: str = "") -> None:
        parts = [message]
        if product_id:
            parts.append(f"product_id={product_id}")
        if database:
            parts.append(f"database={database}")
        super().__init__(" | ".join(parts))
        self.product_id = product_id
        self.database = database


class VectorSearchError(InfrastructureError):
    """向量检索失败。"""

    def __init__(self, message: str, *, collection: str = "") -> None:
        super().__init__(f"{message} | collection={collection}" if collection
                         else message)
        self.collection = collection


class RepositoryError(InfrastructureError):
    """业务库读写失败。"""


class CacheError(InfrastructureError):
    """缓存读写失败。"""


class FileStorageError(InfrastructureError):
    """文件存储失败。"""


class LLMCallError(InfrastructureError):
    """模型调用失败（网络/鉴权/限流）。"""

    def __init__(self, message: str, *, model: str = "") -> None:
        super().__init__(f"{message} | model={model}" if model else message)
        self.model = model


class LLMResponseFormatError(InfrastructureError):
    """模型返回无法解析（如要求 JSON 却给了散文）。"""


class EmbeddingError(InfrastructureError):
    """向量化失败。"""
