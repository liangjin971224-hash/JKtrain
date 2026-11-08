"""员工与日报路由（P7）。

    GET  /api/employees                员工列表
    POST /api/employees/import         批量导入
    POST /api/summaries                保存/提交日报
    GET  /api/summaries                日报列表
    POST /api/summaries/review         主管评语
    GET  /api/miss-logs                未命中问题清单

★ 角色通过请求参数传（`role`），P8 接真鉴权后改成从 token 取。
  现在这样便于测试，也不会把角色硬编码在服务端。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from application.use_cases.employee import (
    ImportEmployeesUseCase,
    ListEmployeesUseCase,
    ListMissLogsUseCase,
    ListSummariesUseCase,
    ReviewSummaryUseCase,
    SaveSummaryUseCase,
)
from config.assembly import Container
from domain.exceptions import (
    DailySummaryFrozenError,
    DataPermissionError,
)
from interfaces.http.auth_deps import (
    Identity,
    current_identity,
    require_feature,
)
from interfaces.http.deps import get_container
from interfaces.http.schemas.models import (
    EmployeeView,
    ImportEmployeesRequest,
    ImportResultResponse,
    MissLogsResponse,
    ReviewRequest,
    SaveSummaryRequest,
    SummaryView,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["员工与日报"])


def _get_repo(container: Container):
    if container.repository is None:
        raise HTTPException(
            503, detail="MySQL 仓储未就绪，员工与日报不可用")
    return container.repository


def _to_view(s) -> SummaryView:
    """转响应模型。

    ★ 兼容两种输入（实测踩过）：
      · `DailySummary` 值对象（用例层返回）
      · `DailySummaryRow` 数据行（仓储直接返回）
      后者没有 `can_edit` 属性，所以要按状态现算。
    """
    status = str(getattr(s, "status", "draft"))
    can_edit = getattr(s, "can_edit", None)
    if can_edit is None:
        can_edit = status == "draft"

    return SummaryView(
        trainee_no=s.trainee_no,
        summary_date=s.summary_date,
        content=s.content,
        status=status,
        submitted_at=getattr(s, "submitted_at", "") or "",
        review_note=getattr(s, "review_note", "") or "",
        can_edit=can_edit,
    )


# ===================================================================
# 员工
# ===================================================================
@router.get("/employees", response_model=list[EmployeeView],
            summary="员工列表")
def list_employees(
    team: str = "",
    limit: int = 500,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> list[EmployeeView]:
    """列员工。

    ★ 身份从 token 取（2026-10-23 修）：原实现让前端传 requester/role，
      学员传 role=manager 就能看到全部员工档案。
    ★ 学员只能看到自己 —— 用例层按角色强制。
    """
    repo = _get_repo(container)
    uc = ListEmployeesUseCase(repository=repo)
    try:
        rows = uc.execute(role=identity.role.value,
                          requester=identity.employee_no,
                          team=team, limit=limit)
    except DataPermissionError as exc:
        raise HTTPException(403, detail=str(exc)) from exc

    # ★ 不能用 `vars(e)`：EmployeeProfile 是 slots 值对象，没有 __dict__。
    #   显式列字段虽然啰嗦，但字段名变更时编译期就能发现（属性不存在会报错）。
    return [
        EmployeeView(
            employee_no=e.employee_no,
            name=e.name,
            hire_date=e.hire_date,
            position=e.position,
            team=e.team,
            mentor=e.mentor,
            active=e.active,
        )
        for e in rows
    ]


@router.post("/employees/import", response_model=ImportResultResponse,
             summary="批量导入员工",
             dependencies=[Depends(require_feature("employee_import"))])
def import_employees(
    req: ImportEmployeesRequest,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> ImportResultResponse:
    """批量导入。

    ★ 坏行跳过并报告，不让整批失败 —— 导 30 人时错一行不该全废。
    """
    repo = _get_repo(container)
    uc = ImportEmployeesUseCase(repository=repo)
    try:
        # ★ 角色从 token 取，不信请求体里的 operator_role
        result = uc.execute(req.rows, role=identity.role.value)
    except DataPermissionError as exc:
        raise HTTPException(403, detail=str(exc)) from exc

    return ImportResultResponse(**result)


# ===================================================================
# 每日总结
# ===================================================================
# ★★ 身份一律从 `Identity`（令牌）取，**绝不从请求参数取**（2026-10-23 修）。
#
#    原实现让前端传 `requester` 和 `role`，后果（已实测验证）：
#      · 学员传 requester=M001 就能写主管的日报
#      · 学员传 role=manager 就能看到全部人的日报
#
#    这类漏洞的根源是「把身份当成普通参数」。
#    身份是**由令牌证明的**，不是由调用方声明的。
@router.post("/summaries", response_model=SummaryView,
             summary="保存/提交日报")
def save_summary(
    req: SaveSummaryRequest,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> SummaryView:
    """保存**自己的**日报。

    - 草稿可反复保存
    - `submit=true` 提交后**冻结**，再改返回 409
    - 每人每天只有一条

    ★ 工号不由前端传 —— 从令牌取。想写别人的日报是不可能的。
    """
    repo = _get_repo(container)
    uc = SaveSummaryUseCase(repository=repo)

    try:
        saved = uc.execute(
            requester=identity.employee_no,
            role=identity.role.value,
            summary_date=req.summary_date,
            content=req.content,
            submit=req.submit,
        )
    except DailySummaryFrozenError as exc:
        # 已提交还想改 → 409 Conflict（不是 400：请求本身合法，只是状态不允许）
        raise HTTPException(409, detail=str(exc)) from exc
    except DataPermissionError as exc:
        raise HTTPException(403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc

    return _to_view(saved)


@router.get("/summaries", response_model=list[SummaryView],
            summary="日报列表")
def list_summaries(
    trainee_no: str = "",
    start_date: str = "",
    end_date: str = "",
    status: str = "",
    limit: int = 100,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> list[SummaryView]:
    """列日报。

    - 学员：**永远只能看到自己的**（`trainee_no` 参数对学员无效）
    - 主管：可按 `trainee_no` 筛选，不传则看全部
    """
    repo = _get_repo(container)
    uc = ListSummariesUseCase(repository=repo)

    rows = uc.execute(
        requester=identity.employee_no,
        role=identity.role.value,
        trainee_no=trainee_no,
        start_date=start_date, end_date=end_date,
        status=status, limit=limit)
    return [_to_view(s) for s in rows]


@router.post("/summaries/review", response_model=SummaryView,
             summary="主管填写评语",
             dependencies=[Depends(require_feature("daily_review"))])
def review_summary(
    req: ReviewRequest,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> SummaryView:
    """填写/修改评语（仅主管）。

    ★ 主管只能改评语，**不能改学员正文** —— 用例层只调 `save_review`。
    """
    repo = _get_repo(container)
    uc = ReviewSummaryUseCase(repository=repo)

    try:
        saved = uc.execute(
            requester=identity.employee_no,
            role=identity.role.value,
            trainee_no=req.trainee_no,
            summary_date=req.summary_date, note=req.note)
    except DataPermissionError as exc:
        raise HTTPException(403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(404, detail=str(exc)) from exc

    return _to_view(saved)


# ===================================================================
# 未命中日志
# ===================================================================
@router.get("/miss-logs", response_model=MissLogsResponse,
            summary="未命中问题清单",
            dependencies=[Depends(require_feature("miss_log_view"))])
def list_miss_logs(
    limit: int = 100,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> MissLogsResponse:
    """只看「问题」，不看「谁问的」。

    ★★ FR-047 原文：「该日志不得用于对学员的绩效评价」。
      所以返回结构里**没有学员维度**，也没有按学员的排序。
    """
    repo = _get_repo(container)
    uc = ListMissLogsUseCase(repository=repo)

    try:
        data = uc.execute(role=identity.role.value, limit=limit)
    except DataPermissionError as exc:
        raise HTTPException(403, detail=str(exc)) from exc

    return MissLogsResponse(**data)
