"""演练与评阅路由。

流程：
    GET  /api/cases              案例列表
    GET  /api/cases/{id}         案例详情
    POST /api/drill/start        开始演练 → 返回开场白 + session_id
    POST /api/drill/answer       提交一轮回答 → 返回下一句客户台词
    POST /api/drill/review       结束并评阅 → 返回摘要卡

★ 会话状态在 `session_store`（进程内），前端只需保存 `session_id`。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from application.services.case_catalog import CaseCatalog
from application.use_cases.review_drill import ReviewDrillUseCase
from config.assembly import Container
from domain.ports.repository_port import (
    DrillReviewRow,
    DrillRiskItemRow,
)
from domain.value_objects import CaseSpec
from interfaces.http.auth_deps import (
    Identity,
    current_identity,
    require_feature,
)
from interfaces.http.deps import get_container
from interfaces.http.schemas.models import (
    AnswerDrillRequest,
    CaseDetail,
    CaseSummary,
    DrillStateResponse,
    ElementCheckResponse,
    ReviewDrillRequest,
    RiskItemResponse,
    StartDrillRequest,
    SummaryCardResponse,
)
from interfaces.http.session_store import SessionRecord, get_session_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["演练"])


# ===================================================================
# 案例
# ===================================================================
def _get_catalog(container: Container) -> CaseCatalog:
    """从容器取案例目录。

    ★ 接口层不直接读文件——那会触发「interfaces 依赖 infrastructure」门禁。
      案例目录由装配根构造好（loader 在那里注入），这里只取用。
    """
    catalog = getattr(container, "case_catalog", None)
    if catalog is None:
        raise HTTPException(
            503,
            detail="案例目录未加载，请检查 CASE_DIR 配置",
        )
    return catalog


def _summary_of(spec: CaseSpec) -> CaseSummary:
    return CaseSummary(
        case_id=spec.case_id,
        title=spec.title,
        question_type=spec.question_type,
        difficulty=spec.difficulty,
        stage=spec.stage,
        teaching_point=spec.teaching_point,
        element_count=len(spec.elements),
        rule_count=len(spec.forbidden_rules),
        is_red_line=spec.is_red_line,
    )


@router.get("/cases", response_model=list[CaseSummary], summary="案例列表",
            dependencies=[Depends(require_feature("drill"))])
def list_cases(
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> list[CaseSummary]:
    catalog = _get_catalog(container)
    return [_summary_of(s) for s in catalog.list_all()]


@router.get("/cases/{case_id}", response_model=CaseDetail, summary="案例详情",
            dependencies=[Depends(require_feature("drill"))])
def get_case(
    case_id: str,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> CaseDetail:
    """案例详情（内容按角色裁剪）。

    ★★ FR-050：「参考答案、评阅标准应设为受限级别，仅主管可见」。

      案例详情里三样东西属于评阅侧信息，学员不该看到：

        `judgement_note`（判定说明）—— 就是评分标准原文
            「出现「有」+ 产品名，或「没有」。不能是"好像有"…」
        `weight`（分值）—— 给了会引导学员只为高分答题
        `focus`（考察点）—— 等于提前告诉学员这道题考什么
        `trigger_note`（触发条件）—— 流程控制信息，学员不需要

      学员该看到的是：要素**名称**（不然不知道该说什么）
      和追问**问题**（客户会问什么）。

      ★ 过滤放在**接口层按角色构造**，而不是「先全给再让前端隐藏」——
        后者只要有人抓包就能看到，等于没做。
    """
    spec = _get_catalog(container).get(case_id)
    if spec is None:
        raise HTTPException(404, detail=f"案例不存在: {case_id}")

    is_manager = identity.is_manager

    base = _summary_of(spec)
    return CaseDetail(
        **base.model_dump(),
        opening_line=spec.opening_line,
        elements=[
            _element_view(e, is_manager=is_manager)
            for e in spec.elements
        ],
        follow_ups=[
            _follow_up_view(f, is_manager=is_manager)
            for f in spec.follow_ups
        ],
    )


def _element_view(e, *, is_manager: bool) -> dict:
    """要素视图。

    ★ 学员只给 `element_id` / `name` / `required`：
      知道「要说哪些点」和「哪些是必答」就够了。
      不给 `weight`（会只为高分答）、不给 `judgement_note`（评分标准）。
    """
    if is_manager:
        return {
            "element_id": e.element_id,
            "name": e.name,
            "weight": e.weight,
            "required": e.required,
            "judgement_note": e.judgement_note,
        }
    return {
        "element_id": e.element_id,
        "name": e.name,
        "required": e.required,
    }


def _follow_up_view(f, *, is_manager: bool) -> dict:
    """追问视图。

    ★ 学员只给问题本身：客户会问什么，这是演练必须知道的。
      `focus`（考察点）等于提前给答案，`trigger_note`（触发条件）
      是流程控制信息 —— 都不给学员。
    """
    if is_manager:
        return {
            "follow_up_id": f.follow_up_id,
            "question": f.question,
            "trigger_note": f.trigger_note,
            "focus": f.focus,
        }
    return {
        "follow_up_id": f.follow_up_id,
        "question": f.question,
    }


# ===================================================================
# 演练
# ===================================================================
def _persist_review(container: Container, record: SessionRecord, card) -> None:
    """把评阅结果落库（月度报告的数据源）。

    ★ 为什么放在接口层而不是用例里：
      用例 `ReviewDrillUseCase` 是**纯计算**（无 IO），
      加仓储会让它依赖持久化。落库是「记录」而非「判定」，
      与 `_log_miss` 一样属于旁路动作。

    ★ 为什么失败只告警不抛：
      评阅结果是学员**必须看到**的反馈。落库失败只是月度报告少一条数据，
      不能让学员连摘要卡都拿不到。这与 `_log_miss` 的处理原则一致。

    统计口径：
      · `element_hit/total` 只算**必答**要素（算要素分用）
      · `element_hit_all/total_all` 含选答（备用，若改用含选答的算法）
      · `case_type` 取 `CaseSpec.question_type` —— 能力项归并的唯一依据
    """
    repo = getattr(container, "repository", None)
    if repo is None:
        return
    try:
        spec = record.spec
        required = spec.required_elements
        required_ids = {e.element_id for e in required}
        required_hit = sum(
            1 for c in card.element_coverage
            if c.covered and c.element_id in required_ids)
        all_hit = sum(1 for c in card.element_coverage if c.covered)

        levels = [r.level.value for r in card.risk_items]
        # ★ 用 session_id 作为 drill_id：它唯一标识一次演练，
        #   SessionRecord 没有独立 drill_id 字段。
        review = DrillReviewRow(
            drill_id=record.session_id,
            # ★ 不能用 case_id 兜底 —— 那会让「谁练的」变成「练的什么」
            trainee_id=card.trainee_id or "",
            case_id=spec.case_id,
            case_type=(spec.question_type or "")[:64],
            duration_sec=card.duration_sec,
            element_hit=required_hit,
            element_total=len(required),
            element_hit_all=all_hit,
            element_total_all=len(card.element_coverage),
            risk_count=len(card.risk_items),
            high_count=levels.count("high"),
            medium_count=levels.count("medium"),
            critical_count=levels.count("critical"),
        )
        items = tuple(
            DrillRiskItemRow(
                drill_id=record.session_id,
                turn_index=r.turn_index,
                sentence_index=r.sentence_index,
                rule_type=r.rule_type,
                matched=r.matched,
                level=r.level.value,
                reason=r.reason,
            )
            for r in card.risk_items
        )
        repo.save_drill_review(review, items)
        logger.info(
            "评阅结果已落库: drill=%s case=%s type=%s 违规=%d",
            record.session_id, spec.case_id, review.case_type, len(items))
    except Exception as exc:                       # noqa: BLE001 落库失败不阻塞
        logger.warning("评阅结果落库失败（不影响返回）: %s: %s",
                       type(exc).__name__, exc, exc_info=True)


def _get_own_session(session_id: str, identity) -> SessionRecord:
    """取会话，并校验归属。

    ★★ 为什么必须校验（2026-10-23 修）：
      原实现只按 session_id 取，不检查是谁的会话 ——
      只要知道 session_id，任何人都能替别人作答、看别人的摘要卡。
      这是典型的水平越权（IDOR）。

      session_id 是 16 位随机十六进制，不好猜，
      但它会出现在前端、浏览器历史、日志里，泄露面不小。

    Returns:
        会话记录。

    Raises:
        HTTPException: 404 会话不存在；403 不是自己的会话。
    """
    try:
        record = get_session_store().get(session_id)
    except KeyError as exc:
        raise HTTPException(404, detail=str(exc)) from exc

    owner = record.session.trainee_id or ""
    if owner and owner != identity.employee_no:
        # ★ 只在不相等时报错。owner 为空（历史遗留会话）时放行，
        #   避免把之前没记身份的会话全部变成不可用。
        logger.warning(
            "越权访问演练会话: %s 试图操作 %s 的会话 %s",
            identity.employee_no, owner, session_id)
        raise HTTPException(403, detail="这不是你的演练会话")

    return record


def _state_of(record: SessionRecord) -> DrillStateResponse:
    """把会话内部状态转成响应。

    ★ 对话记录要**含客户台词**——否则前端只能显示学员单方面发言，
      而评阅时人需要看「客户问了什么，学员怎么答的」。
    """
    s = record.session

    return DrillStateResponse(
        session_id=record.session_id,
        case_id=record.spec.case_id,
        case_title=record.spec.title,
        trainee_id=s.trainee_id,
        current_question=s.current_question,
        current_follow_up_id=s.current_follow_up_id,
        progress_current=s.progress[0],
        progress_total=s.progress[1],
        remaining=s.remaining,
        turn_count=s.turn_count,
        is_finished=s.is_finished,
        duration_sec=s.duration_sec,
        messages=s.messages(),
    )


@router.post("/drill/start", response_model=DrillStateResponse,
             summary="开始演练",
             dependencies=[Depends(require_feature("drill"))])
def start_drill(
    req: StartDrillRequest,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> DrillStateResponse:
    spec = _get_catalog(container).get(req.case_id)
    if spec is None:
        raise HTTPException(404, detail=f"案例不存在: {req.case_id}")
    if not spec.opening_line:
        raise HTTPException(400, detail=f"案例 {req.case_id} 缺少开场白")

    # ★ 演练人 = 登录身份，不接受前端声明（2026-10-23 修）
    record = get_session_store().create(
        spec, trainee_id=identity.employee_no)
    return _state_of(record)


@router.post("/drill/answer", response_model=DrillStateResponse,
             summary="提交一轮回答",
             dependencies=[Depends(require_feature("drill"))])
def answer_drill(
    req: AnswerDrillRequest,
    identity: Identity = Depends(current_identity),
) -> DrillStateResponse:
    record = _get_own_session(req.session_id, identity)

    try:
        record.session.answer(req.answer)
    except ValueError as exc:
        # 会话已结束 / 回答为空 → 400 而不是 500
        raise HTTPException(400, detail=str(exc)) from exc

    return _state_of(record)


@router.post("/drill/review", response_model=SummaryCardResponse,
             summary="评阅并生成摘要卡",
             dependencies=[Depends(require_feature("drill"))])
def review_drill(
    req: ReviewDrillRequest,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> SummaryCardResponse:
    record = _get_own_session(req.session_id, identity)

    s = record.session
    card = ReviewDrillUseCase().execute(
        record.spec,
        s.to_trainee_turns(),
        trainee_id=s.trainee_id,
        duration_sec=s.duration_sec,
    )

    # ★ 落库（月度报告的数据源）。
    #   写失败**不能**影响返回：评阅结果学生必须能看到，
    #   落库只是运营素材，与 `_log_miss` 同一处理原则。
    _persist_review(container, record, card)

    return SummaryCardResponse(
        trainee_id=card.trainee_id,
        case_id=card.case_id,
        case_title=record.spec.title,
        duration_sec=card.duration_sec,
        risk_items=[
            RiskItemResponse(
                level=r.level.value,
                rule_type=r.rule_type,
                matched=r.matched,
                location=r.location,
                reason=r.reason,
                is_critical=r.is_critical,
            )
            for r in card.risk_items
        ],
        element_coverage=[
            ElementCheckResponse(
                element_id=c.element_id,
                element_name=c.element_name,
                required=c.required,
                covered=c.covered,
                evidence=c.evidence,
                status_mark=c.status_mark,
            )
            for c in card.element_coverage
        ],
        pending_human=list(card.pending_human),
        has_critical_violation=card.has_critical_violation,
        critical_count=len(card.critical_items),
        missing_required_count=len(card.missing_required_elements),
    )
