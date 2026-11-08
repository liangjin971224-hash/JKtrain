"""错误映射与异常层测试。

覆盖两块 P0 就写好但还没被测的代码：
    infrastructure/exceptions.py   —— 基础设施异常必须带上下文
    interfaces/http/errors.py      —— 领域异常到 HTTP 状态码的映射
"""
from __future__ import annotations

import pytest

from domain.exceptions import (
    CaseNotFoundError,
    DomainError,
    DuplicateDailySummaryError,
    DraftAlreadySubmittedError,
    ExamPaperLockedError,
    FileTooLargeError,
    InvalidPermissionLevelError,
    PathTraversalError,
    ProductNotFoundError,
    QuestionNotFoundError,
    UnsupportedFileTypeError,
)
from infrastructure.exceptions import (
    CacheError,
    EmbeddingError,
    FileStorageError,
    GraphQueryError,
    InfrastructureError,
    LLMCallError,
    LLMResponseFormatError,
    RepositoryError,
    VectorSearchError,
)
from interfaces.http.errors import DOMAIN_ERROR_MAP, ErrorBody, map_domain_error


# ===================================================================
# 领域异常：必须带可定位的上下文
# ===================================================================
class TestDomainExceptions:
    """异常信息里必须能看出是哪个对象出的问题。"""

    def test_product_not_found_carries_id(self) -> None:
        exc = ProductNotFoundError("P999")
        assert exc.product_id == "P999"
        assert "P999" in str(exc)

    def test_case_not_found_carries_id(self) -> None:
        assert CaseNotFoundError("C99").case_id == "C99"

    def test_question_not_found_carries_id(self) -> None:
        assert QuestionNotFoundError("Q999").question_id == "Q999"

    def test_file_too_large_reports_both_sizes(self) -> None:
        exc = FileTooLargeError(30 * 1024 * 1024, 20 * 1024 * 1024)
        text = str(exc)
        assert "30.0MB" in text and "20MB" in text

    def test_unsupported_type_reports_suffix(self) -> None:
        exc = UnsupportedFileTypeError(".exe")
        assert ".exe" in str(exc)
        # 必须提示允许的类型，否则用户不知道该怎么办
        assert ".pdf" in str(exc)

    def test_all_inherit_domain_error(self) -> None:
        for cls in (ProductNotFoundError, CaseNotFoundError,
                    QuestionNotFoundError, InvalidPermissionLevelError,
                    UnsupportedFileTypeError, PathTraversalError,
                    ExamPaperLockedError):
            assert issubclass(cls, DomainError)

    def test_exam_paper_locked_reports_status(self) -> None:
        exc = ExamPaperLockedError("E001", "active")
        assert "active" in str(exc)
        assert exc.status == "active"

    def test_duplicate_summary_reports_key(self) -> None:
        exc = DuplicateDailySummaryError("T001", "2026-10-22")
        assert "T001" in str(exc) and "2026-10-22" in str(exc)


# ===================================================================
# 基础设施异常：带上下文
# ===================================================================
class TestInfrastructureExceptions:
    def test_graph_error_appends_context(self) -> None:
        """排障要能看出是查哪个产品、哪个库出的错。"""
        exc = GraphQueryError("查询超时", product_id="P001", database="jiankangshujuku")
        text = str(exc)
        assert "查询超时" in text
        assert "product_id=P001" in text
        assert "database=jiankangshujuku" in text

    def test_graph_error_without_optional_context(self) -> None:
        exc = GraphQueryError("连接失败")
        assert str(exc) == "连接失败"

    def test_vector_error_appends_collection(self) -> None:
        exc = VectorSearchError("检索失败", collection="health_training_docs")
        assert "health_training_docs" in str(exc)

    def test_llm_error_appends_model(self) -> None:
        exc = LLMCallError("限流", model="deepseek-chat")
        assert "deepseek-chat" in str(exc)

    def test_all_inherit_infrastructure_error(self) -> None:
        for cls in (GraphQueryError, VectorSearchError, RepositoryError,
                    CacheError, FileStorageError, LLMCallError,
                    LLMResponseFormatError, EmbeddingError):
            assert issubclass(cls, InfrastructureError)


# ===================================================================
# HTTP 错误映射
# ===================================================================
class TestErrorMapping:
    """领域异常 → (状态码, 错误码)。"""

    @pytest.mark.parametrize(("exc", "status", "code"), [
        (ProductNotFoundError("P1"), 404, "PRODUCT_NOT_FOUND"),
        (CaseNotFoundError("C1"), 404, "CASE_NOT_FOUND"),
        (QuestionNotFoundError("Q1"), 404, "QUESTION_NOT_FOUND"),
        (ExamPaperLockedError("E1", "active"), 409, "EXAM_PAPER_LOCKED"),
        (DuplicateDailySummaryError("T1", "d"), 409, "DUPLICATE_DAILY_SUMMARY"),
        (DraftAlreadySubmittedError("T1", "d"), 403, "DRAFT_ALREADY_SUBMITTED"),
        (UnsupportedFileTypeError(".exe"), 415, "UNSUPPORTED_FILE_TYPE"),
        (FileTooLargeError(1, 1), 413, "FILE_TOO_LARGE"),
        (PathTraversalError("../x"), 400, "INVALID_PATH"),
    ])
    def test_mapping(self, exc: DomainError, status: int, code: str) -> None:
        got_status, body = map_domain_error(exc)
        assert got_status == status
        assert body.code == code
        assert body.message            # 错误信息不能为空

    def test_unregistered_domain_error_falls_back_to_400(self) -> None:
        """未登记的领域异常必须落到 400，不能被当成 500 掩盖。"""
        status, body = map_domain_error(InvalidPermissionLevelError("secret"))
        assert status == 400
        assert body.code == "DOMAIN_ERROR"

    def test_body_contains_type(self) -> None:
        _, body = map_domain_error(ProductNotFoundError("P1"))
        assert body.detail["type"] == "ProductNotFoundError"

    def test_body_to_dict(self) -> None:
        body = ErrorBody(code="X", message="m", detail={"a": 1})
        assert body.to_dict() == {"code": "X", "message": "m", "detail": {"a": 1}}

    def test_map_covers_all_registered_types(self) -> None:
        """映射表里的每个类型都必须真的能命中（防止表与实现脱节）。"""
        for exc_type, status, code in DOMAIN_ERROR_MAP:
            # 用最小参数构造
            exc = _construct(exc_type)
            got_status, body = map_domain_error(exc)
            assert got_status == status, f"{exc_type.__name__} 状态码不符"
            assert body.code == code, f"{exc_type.__name__} 错误码不符"


def _construct(exc_type: type[DomainError]) -> DomainError:
    """按参数名构造异常实例（测试辅助）。

    注意不能用 `param.annotation` 判断类型：
    本模块用了 `from __future__ import annotations`，
    注解全是字符串，`ann is int` 永远为 False。
    改按参数名判断——异常构造参数命名是稳定的。
    """
    import inspect

    sig = inspect.signature(exc_type.__init__)
    args = []
    for name in sig.parameters:
        if name == "self":
            continue
        if any(k in name for k in ("size", "limit", "bytes", "count")):
            args.append(1)
        else:
            args.append("X")
    return exc_type(*args)
