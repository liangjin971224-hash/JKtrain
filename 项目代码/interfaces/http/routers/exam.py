"""试卷与组卷路由（P6）。

    POST /api/exam/build            组卷
    GET  /api/exam/papers           试卷列表
    GET  /api/exam/papers/{no}      试卷详情（含题目）
    POST /api/exam/papers/{no}/status  状态流转
    POST /api/exam/papers/{no}/export  导出 Markdown

★ 题库来源：Milvus（`application/services/question_bank.py`）
★ 试卷落库：MySQL（`RepositoryPort`）
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from application.services.paper_export import (
    render_answer_key,
    render_exam,
    render_summary,
)
from application.services.question_bank import build_bank
from application.use_cases.build_paper import (
    BuildPaperUseCase,
    ChangePaperStatusUseCase,
)
from config.assembly import Container
from domain.exceptions import ExamBuildError, InvalidExamTransition
from domain.value_objects.exam import ExamSpec, ExamStatus
from interfaces.http.auth_deps import (
    Identity,
    current_identity,
    require_feature,
)
from interfaces.http.deps import current_settings, get_container
from interfaces.http.schemas.models import (
    BuildPaperRequest,
    ExportResponse,
    PaperListItem,
    PaperStatusRequest,
    PaperView,
    QuestionView,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/exam", tags=["试卷"])

#: 题库缓存（进程内）。
#: 题目是只读内容资产，改题库要重启服务——与案例目录同样的约定。
_bank_cache: dict[str, list] = {}


def _load_bank(container: Container, stage: str) -> list:
    """从 Milvus 取题库（带缓存）。"""
    if stage in _bank_cache:
        return _bank_cache[stage]

    if container.milvus_client is None:
        raise HTTPException(503, detail="Milvus 未就绪，无法读取题库")

    settings = current_settings()
    rows = container.milvus_client.query(
        collection_name=settings.coll_question,
        filter="id >= 0",
        output_fields=[
            "chunk_id", "text", "knowledge_point", "kp_name",
            "question_type", "difficulty", "review_status", "extra",
        ],
        limit=1000,
    )
    bank = build_bank(rows, stage=stage)
    _bank_cache[stage] = bank
    logger.info("题库已加载并缓存: stage=%s, %d 题", stage, len(bank))
    return bank


def _get_repo(container: Container):
    if container.repository is None:
        raise HTTPException(
            503, detail="MySQL 仓储未就绪，试卷无法保存。请检查 DATABASE_URL")
    return container.repository


# ===================================================================
# 组卷
# ===================================================================
@router.post("/build", response_model=PaperView, summary="组卷",
             dependencies=[Depends(require_feature("exam_build"))])
def build_exam(
    req: BuildPaperRequest,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> PaperView:
    """按规则组一份试卷。

    - 知识库覆盖是**硬约束**，凑不齐直接报错并说明缺哪个知识点
    - 难度配比是**软约束**，凑不齐会降级并在 `warnings` 里说明
    - 30 天内该学员抽过的题不会重复（按 `trainee_id` 隔离）
    """
    # 难度配比键是字符串（JSON 限制），转成 int
    #
    # ★ 必须在这里校验范围：`ExamSpec.__post_init__` 也会校验，
    #   但那个 ValueError 不是 DomainError，会一路冒成 500。
    #   接口层提前挡成 400，前端才能显示「难度配比填错了」。
    quota: dict[int, int] = {}
    for k, v in (req.difficulty_quota or {}).items():
        try:
            d = int(k)
        except (TypeError, ValueError):
            raise HTTPException(400, detail=f"难度配比的键必须是数字: {k!r}")
        if d not in (1, 2, 3):
            raise HTTPException(
                400, detail=f"难度只能是 1/2/3，收到 {d}")
        if int(v) < 0:
            raise HTTPException(400, detail=f"难度 {d} 的题数不能为负")
        quota[d] = int(v)

    spec = ExamSpec(
        total=req.total,
        knowledge_points=tuple(req.knowledge_points),
        difficulty_quota=quota,
        product_ids=tuple(req.product_ids),
        shuffle_options=req.shuffle_options,
        stage=req.stage,
    )

    repo = _get_repo(container) if req.persist else None
    uc = BuildPaperUseCase(
        repository=repo,
        bank_loader=lambda st: _load_bank(container, st),
    )

    try:
        paper = uc.execute(
            spec,
            title=req.title,
            # ★ 创建人 = 登录身份，不接受前端声明（2026-10-23 修）
            created_by=identity.employee_no,
            # trainee_id 是「替哪个学员组卷」——主管的业务输入，保留。
            # 但它只影响防重复窗口，不构成越权（组卷本身要 exam_build 权限）。
            trainee_id=req.trainee_id,
            seed=req.seed,
            persist=req.persist,
        )
    except ExamBuildError as exc:
        # 组卷规则无法满足 → 400，并把原因原文返回（管理员要知道补什么）
        raise HTTPException(400, detail=str(exc)) from exc

    return _to_view(paper)


# ===================================================================
# 列表 / 详情
# ===================================================================
@router.get("/papers", response_model=list[PaperListItem], summary="试卷列表",
            dependencies=[Depends(require_feature("exam_manage"))])
def list_papers(
    stage: str = "",
    status: str = "",
    limit: int = 50,
    container: Container = Depends(get_container),
) -> list[PaperListItem]:
    repo = _get_repo(container)
    return [PaperListItem(**p) for p in repo.list_exam_papers(
        stage=stage, status=status, limit=limit)]


@router.get("/papers/{paper_no}", response_model=PaperView, summary="试卷详情",
            dependencies=[Depends(require_feature("exam_manage"))])
def get_paper(
    paper_no: str,
    container: Container = Depends(get_container),
) -> PaperView:
    repo = _get_repo(container)
    paper = repo.get_exam_paper(paper_no)
    if paper is None:
        raise HTTPException(404, detail=f"试卷不存在: {paper_no}")

    return PaperView(
        paper_no=paper["paper_no"],
        stage=paper["stage"],
        title=paper["title"],
        status=paper["status"],
        total_count=paper["total_count"],
        created_by=paper["created_by"],
        created_at=paper["created_at"],
        questions=[QuestionView(**q) for q in paper["questions"]],
    )


# ===================================================================
# 状态流转
# ===================================================================
@router.post("/papers/{paper_no}/status",
             response_model=PaperListItem, summary="试卷状态流转",
             dependencies=[Depends(require_feature("exam_manage"))])
def change_status(
    paper_no: str,
    req: PaperStatusRequest,
    container: Container = Depends(get_container),
) -> PaperListItem:
    """草稿 → 生效 → 归档（单向，不可回退）。"""
    repo = _get_repo(container)

    try:
        target = ExamStatus(req.target)
    except ValueError:
        raise HTTPException(
            400,
            detail=f"状态非法: {req.target}。可用："
                   f"{[s.value for s in ExamStatus]}")

    uc = ChangePaperStatusUseCase(repository=repo)
    try:
        new_status = uc.execute(paper_no, target)
    except InvalidExamTransition as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(404, detail=str(exc)) from exc

    paper = repo.get_exam_paper(paper_no)
    return PaperListItem(
        paper_no=paper["paper_no"],
        stage=paper["stage"],
        title=paper["title"],
        status=str(new_status),
        total_count=paper["total_count"],
        created_by=paper["created_by"],
        created_at=paper["created_at"],
    )


# ===================================================================
# 导出
# ===================================================================
@router.post("/papers/{paper_no}/export",
             response_model=ExportResponse, summary="导出试卷（Markdown）",
             dependencies=[Depends(require_feature("exam_manage"))])
def export_paper(
    paper_no: str,
    container: Container = Depends(get_container),
) -> ExportResponse:
    """导出三份 Markdown：试卷页 / 答案页 / 组卷报告。

    ★ 只有「生效」或「归档」的试卷能导出。
      草稿还在变，导出没意义（需求 FR-040）。
    """
    repo = _get_repo(container)
    data = repo.get_exam_paper(paper_no)
    if data is None:
        raise HTTPException(404, detail=f"试卷不存在: {paper_no}")

    status = ExamStatus(data["status"])
    if not status.can_export:
        raise HTTPException(
            409,
            detail=f"试卷当前为「{status.label}」，不能导出。"
                   f"请先定稿为「生效」。")

    paper = _to_paper(data)

    return ExportResponse(
        paper_no=paper_no,
        exam_markdown=render_exam(paper, with_answer=False),
        answer_markdown=render_answer_key(paper),
        summary_markdown=render_summary(paper),
    )


# ===================================================================
# 转换
# ===================================================================
def _to_view(paper) -> PaperView:
    """domain `ExamPaper` → 响应模型。"""
    return PaperView(
        paper_no=paper.paper_no,
        stage=paper.stage,
        title=paper.title,
        status=str(paper.status),
        total_count=paper.count,
        created_by=paper.created_by,
        created_at=paper.created_at,
        warnings=list(paper.warnings),
        covered_knowledge_points=list(paper.covered_knowledge_points),
        difficulty_distribution={
            str(k): v for k, v in paper.difficulty_distribution.items()},
        questions=[
            QuestionView(
                seq=i,
                question_id=q.question_id,
                knowledge_point=q.knowledge_point,
                kp_name=q.kp_name,
                question_type=q.question_type,
                difficulty=q.difficulty,
                stem=q.stem,
                options=list(q.options),
                answer=q.answer,
                explanation=q.explanation,
                product_id=q.product_id,
            )
            for i, q in enumerate(paper.questions, start=1)
        ],
    )


def _to_paper(data: dict):
    """仓储字典 → domain `ExamPaper`（导出用）。"""
    from domain.value_objects.exam import (
        ExamPaper,
        ExamSpec,
        ExamStatus,
        QuestionSnapshot,
    )

    spec_data = data.get("spec") or {}

    # ★ 难度配比的键从 JSON 读回来是字符串（"1"/"2"/"3"），
    #   而 ExamSpec 要求 int 键。这里必须转换，否则重建值对象会抛
    #   「难度只能是 1/2/3」——实测踩过（导出 500）。
    quota_raw = spec_data.get("difficulty_quota") or {}
    quota: dict[int, int] = {}
    for k, v in quota_raw.items():
        try:
            quota[int(k)] = int(v)
        except (TypeError, ValueError):
            continue                        # 脏数据跳过，不让导出挂掉

    spec = ExamSpec(
        total=int(spec_data.get("total") or data.get("total_count") or 0),
        knowledge_points=tuple(spec_data.get("knowledge_points") or ()),
        difficulty_quota=quota,
        product_ids=tuple(spec_data.get("product_ids") or ()),
        shuffle_options=bool(spec_data.get("shuffle_options", True)),
        stage=data.get("stage") or "",
    )

    questions = tuple(
        QuestionSnapshot(
            question_id=q["question_id"],
            knowledge_point=q.get("knowledge_point", ""),
            kp_name=q.get("kp_name", ""),
            question_type=q.get("question_type", "单选"),
            difficulty=int(q.get("difficulty") or 2),
            stem=q.get("stem", ""),
            options=tuple(q.get("options") or ()),
            answer=q.get("answer", ""),
            explanation=q.get("explanation", ""),
            product_id=q.get("product_id", ""),
        )
        for q in data.get("questions", [])
    )

    warnings = spec_data.get("warnings") or ()

    return ExamPaper(
        paper_no=data["paper_no"],
        stage=data.get("stage", ""),
        spec=spec,
        questions=questions,
        status=ExamStatus(data.get("status", "draft")),
        created_at=data.get("created_at", ""),
        created_by=data.get("created_by", ""),
        title=data.get("title", ""),
        warnings=tuple(warnings),
    )
