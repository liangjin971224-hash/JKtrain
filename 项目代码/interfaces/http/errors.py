"""协议层错误映射（三层异常体系的第三层）。

把领域异常、基础设施异常映射为 HTTP 状态码与统一错误体。
★ 注意：业务结果（禁忌命中、合规违规）不走这里——它们是正常返回值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.exceptions import (
    CaseNotFoundError,
    DomainError,
    DuplicateDailySummaryError,
    DraftAlreadySubmittedError,
    ExamPaperLockedError,
    FileTooLargeError,
    PathTraversalError,
    ProductNotFoundError,
    QuestionNotFoundError,
    UnsupportedFileTypeError,
)


@dataclass(frozen=True, slots=True)
class ErrorBody:
    """统一错误响应体。"""

    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


#: 领域异常 → (HTTP 状态码, 业务错误码)
DOMAIN_ERROR_MAP: tuple[tuple[type[DomainError], int, str], ...] = (
    (ProductNotFoundError, 404, "PRODUCT_NOT_FOUND"),
    (CaseNotFoundError, 404, "CASE_NOT_FOUND"),
    (QuestionNotFoundError, 404, "QUESTION_NOT_FOUND"),
    (ExamPaperLockedError, 409, "EXAM_PAPER_LOCKED"),
    (DuplicateDailySummaryError, 409, "DUPLICATE_DAILY_SUMMARY"),
    (DraftAlreadySubmittedError, 403, "DRAFT_ALREADY_SUBMITTED"),
    (UnsupportedFileTypeError, 415, "UNSUPPORTED_FILE_TYPE"),
    (FileTooLargeError, 413, "FILE_TOO_LARGE"),
    (PathTraversalError, 400, "INVALID_PATH"),
)


#: 认证/授权异常的 HTTP 状态码。
#:
#: ★ 401 与 403 必须分开：
#:     401 = 你是谁我不知道 → 前端去登录页
#:     403 = 我知道你是谁，但你不能做这个 → 前端提示权限不足
#:   混用会让前端无法正确处理。
_AUTH_STATUS: dict[str, int] = {
    "AuthenticationError": 401,
    "PermissionDeniedError": 403,
}


def map_domain_error(exc: DomainError) -> tuple[int, ErrorBody]:
    """把领域异常映射为 (状态码, 错误体)。

    未登记的 DomainError 一律 400，避免被当成 500 掩盖。
    """
    # ★ 认证/授权类优先（按名字判断，避免循环 import）
    for cls in type(exc).__mro__:
        status = _AUTH_STATUS.get(cls.__name__)
        if status is not None:
            return status, ErrorBody(
                code="AUTHENTICATION_ERROR" if status == 401
                     else "PERMISSION_DENIED",
                message=str(exc),
                detail={"type": type(exc).__name__},
            )

    for exc_type, status, code in DOMAIN_ERROR_MAP:
        if isinstance(exc, exc_type):
            return status, ErrorBody(
                code=code,
                message=str(exc),
                detail={"type": type(exc).__name__},
            )
    return 400, ErrorBody(
        code="DOMAIN_ERROR",
        message=str(exc),
        detail={"type": type(exc).__name__},
    )


# ===================================================================
# 基础设施异常 → 503
# ===================================================================
# ★ 为什么用「按名字判断」而不是 import InfrastructureError：
#
#   门禁 `test_interfaces_does_not_import_infrastructure` 禁止接口层
#   直接 import infrastructure。理由很实在：
#     · 换实现要改路由
#     · 无法用假实现测接口层
#
#   但「基础设施故障要返回 503 而不是 500」是协议层的正当职责
#   （503 表示下游不可用，500 表示代码 bug，两者给运维的信号完全不同）。
#
#   解法：按**类名 + MRO 名字**判断，不 import 具体类。
#   这样既满足门禁，又能正确区分 503 / 500。
#
#   已知取舍：改名会失效。所以有测试守住
#   （`test_infrastructure_error_maps_to_503`）。

#: 基础设施异常类名后缀（InfrastructureError 及其子类都以此结尾或其子类）
_INFRA_BASE_NAMES = frozenset({
    "InfrastructureError",
    "VectorSearchError",
    "LLMCallError",
    "LLMResponseFormatError",
    "GraphQueryError",
    "StorageError",
})


def is_infrastructure_error(exc: BaseException) -> bool:
    """判断异常是否属于基础设施故障。

    不看类型看名字，避免接口层 import infrastructure。

    Args:
        exc: 待判断的异常。

    Returns:
        True 表示基础设施故障（应返回 503）。
    """
    for cls in type(exc).__mro__:
        if cls.__name__ in _INFRA_BASE_NAMES:
            return True
    return False


def map_infrastructure_error(exc: BaseException) -> ErrorBody:
    """把基础设施异常转为统一错误体。"""
    return ErrorBody(
        code="INFRASTRUCTURE_ERROR",
        message=str(exc) or type(exc).__name__,
        detail={"type": type(exc).__name__},
    )
