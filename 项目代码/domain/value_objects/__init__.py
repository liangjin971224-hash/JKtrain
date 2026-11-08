"""领域值对象统一出口。

外部只需 `from domain.value_objects import ProductFacts, ...`。
"""
from __future__ import annotations

from domain.value_objects.approval_claim import (
    ApprovalClaim,
    Contraindication,
    ContraindicationHit,
    EfficacyRef,
    ProductRef,
    SimilarPair,
    Usage,
)
from domain.value_objects.role import Role
from domain.value_objects.employee import (
    DailySummary,
    EmployeeProfile,
    MissLogEntry,
    SummaryStatus,
)
from domain.value_objects.exam import (
    ExamPaper,
    ExamSpec,
    ExamStatus,
    QuestionSnapshot,
    make_paper_no,
)
from domain.value_objects.account import (
    AuditAction,
    AuditLogEntry,
    AuditResult,
    EmployeeAccount,
)
from domain.value_objects.drill_review import (
    CaseSpec,
    ElementCheck,
    ElementSpec,
    ForbiddenRule,
    RiskItem,
    RiskLevel,
    ScriptedFollowUp,
    SummaryCard,
    TraineeTurn,
)
from domain.value_objects.product_category import ProductCategory
from domain.value_objects.product_facts import ProductFacts
from domain.value_objects.search_result import (
    ChunkMeta,
    ContentCategory,
    FusedHit,
    KeywordHit,
    PermissionLevel,
    VectorHit,
)
from domain.value_objects.severity import Severity

__all__ = [
    "ApprovalClaim",
    "CaseSpec",
    "ChunkMeta",
    "ContentCategory",
    "Contraindication",
    "ContraindicationHit",
    "EfficacyRef",
    "AuditAction",
    "AuditLogEntry",
    "AuditResult",
    "DailySummary",
    "ElementCheck",
    "EmployeeAccount",
    "EmployeeProfile",
    "ExamPaper",
    "ExamSpec",
    "ExamStatus",
    "ElementSpec",
    "ForbiddenRule",
    "FusedHit",
    "MissLogEntry",
    "KeywordHit",
    "PermissionLevel",
    "ProductCategory",
    "ProductFacts",
    "ProductRef",
    "QuestionSnapshot",
    "RiskItem",
    "RiskLevel",
    "Role",
    "ScriptedFollowUp",
    "Severity",
    "SummaryStatus",
    "SimilarPair",
    "SummaryCard",
    "TraineeTurn",
    "Usage",
    "VectorHit",
    "make_paper_no",
]
