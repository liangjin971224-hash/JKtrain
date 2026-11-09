"""月度培训报告路由。

端点：
    GET  /api/reports/monthly                      同步版（P1，无图、无复核）
    POST /api/reports/monthly/generate             跑图到挂起，返回 thread_id（P2）
    POST /api/reports/monthly/{thread_id}/review   复核决定，恢复图（P2）
    GET  /api/reports/monthly/{thread_id}/state    查任务状态（P2）

★ P1 与 P2 同时保留：前者「随便看看」，后者是「要主管确认的正式流程」。
  两者共用同一套领域逻辑，P2 只是外面包了一层图。

★ 图实例**每请求新建**（用户 2026-10-31 决策）：图无状态，
  真正的中断态在 Redis 的 checkpoint 里。唯一影响是进程内缓存为空，
  节点会走 `_rebuild_report()` 从 state 重建（P2 跨进程验证已覆盖该路径）。

★ 权限：主管看全部，学员只能看自己。
  校验放在用例外（本层），因为「谁能看谁的报告」是接口层的策略，
  不是领域规则。
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query

from application.use_cases.monthly_report import MonthlyReportUseCase
from application.use_cases.report_context import (
    case_titles as _case_titles,
    doc_catalog as _doc_catalog,
)
from config.assembly import Container
from interfaces.http.auth_deps import (
    Identity,
    current_identity,
    require_feature,
)
from interfaces.http.deps import get_container
from interfaces.http.schemas.models import (
    BatchGenerateRequest,
    BatchProgressResponse,
    BatchTaskItem,
    BatchTaskListResponse,
    GenerateReportRequest,
    ReportStateResponse,
    ReviewReportRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reports", tags=["月度报告"])

#: 每个报告接口都带这句：分数是参考分，不是判定。
#: ★ 这是项目的约束 #3（不给 verdict、判定由人下）在对外文案上的体现。
_DISCLAIMER = (
    "本报告的分数为参考分，仅用于辅助主管了解培训情况；"
    "正式判定权在企业，系统不提供自动判定结论。"
)


def _build_use_case(container: Container) -> MonthlyReportUseCase:
    return MonthlyReportUseCase(
        repository=container.repository,
        llm=container.llm,
        case_titles=_case_titles(container),
        doc_catalog=_doc_catalog(container),
    )


@router.get("/monthly", summary="月度培训报告（同步生成）",
            dependencies=[Depends(require_feature("drill"))])
def monthly_report(
    trainee_id: str = Query("", description="学员编号，空则用当前用户"),
    since: str = Query("", description="起始日期 YYYY-MM-DD（含）"),
    until: str = Query("", description="结束日期 YYYY-MM-DD（不含）"),
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> dict:
    """生成某学员某月的培训报告。

    ★ 三层结构：能力项（第1层）→ 案例（第2层）→ 违规明细（第3层）。

    ★ 分数是**参考分**，不是判定：
      项目的约束 #3 规定单次演练不给分，本接口的分只在月度汇总里出现，
      不写回 `SummaryCard`，也不作为任何自动判定的依据。
    """
    if container.repository is None:
        raise HTTPException(503, detail="仓储未就绪，报告不可用")

    # ---- 归属校验：学员只能看自己 ----
    target = (trainee_id or "").strip() or identity.employee_no
    if not _can_view(identity, target):
        # ★ 不暴露「该员工是否存在」——直接说无权
        raise HTTPException(403, detail="无权查看他人的报告")

    uc = _build_use_case(container)
    try:
        result = uc.execute(target, since=since, until=until)
    except Exception as exc:                       # noqa: BLE001
        logger.error("生成月度报告失败: %s: %s",
                     type(exc).__name__, exc, exc_info=True)
        raise HTTPException(500, detail="报告生成失败") from exc

    payload = result.summary
    payload["disclaimer"] = (
        "本报告的分数为参考分，仅用于辅助主管了解培训情况；"
        "正式判定权在企业，系统不提供自动判定结论。"
    )
    return payload


def _current_period() -> str:
    return time.strftime("%Y-%m")


def _can_view(identity: Identity, target_no: str) -> bool:
    """能否查看该员工的报告。

    主管/管理员可以看全部，学员只能看自己。
    ★ 这里不做「该员工是否存在」的判断，避免被用来探测工号。
    """
    own = getattr(identity, "employee_no", "") or ""
    if target_no == own:
        return True
    role = getattr(identity, "role", "") or ""
    role_value = getattr(role, "value", role)
    return str(role_value) in ("manager", "admin")


# ===================================================================
# P2：图编排 + 复核
# ===================================================================
def _checkpointer(container: Container):
    """从容器取 checkpoint 存储器。

    ★ 由装配根注入（`container.checkpoint_saver`），
      接口层不得自己 new —— 那样就要 import infrastructure，
      架构门禁会拦（`test_interfaces_does_not_import_infrastructure`）。

    ★ 取不到就报 503，不悄悄降级：
      没有 checkpointer 时 `interrupt` 会失效，
      图会一路跑到定稿，等于绕过主管复核。
    """
    return getattr(container, "checkpoint_saver", None)


def _build_graph(container: Container):
    """每请求新建图实例。"""
    from application.graph.monthly_report_graph import build_report_graph

    saver = _checkpointer(container)
    if saver is None:
        raise HTTPException(
            503, detail="checkpoint 存储不可用，无法发起需要复核的报告生成")

    return build_report_graph(
        repository=container.repository,
        llm=container.llm,
        checkpointer=saver,
        case_titles=_case_titles(container),
        doc_catalog=_doc_catalog(container),
    )


def _parse_thread_id(thread_id: str) -> tuple[str, str]:
    """从 `{trainee}:{period}:{uuid}` 取前两段。

    ★ 解析失败返回空元组，调用方必须拒绝 ——
      不能把无法解析的 id 当成「不属于任何人」而放行。
    """
    parts = (thread_id or "").split(":")
    if len(parts) < 3 or not parts[0] or not parts[1]:
        return "", ""
    return parts[0], parts[1]


def _weak_named(keys) -> list[dict]:
    """薄弱能力项：key → 带中文名的结构。

    ★ 为什么在这里转换而不是前端查表：
      中文名定义在领域层 `CAPABILITY_META`（唯一来源）。
      前端再维护一份就是两处实现，改名字会漂移。
    """
    from domain.services.monthly_report import CAPABILITY_META

    return [
        {"key": k, "name": CAPABILITY_META.get(k, {}).get("name", k)}
        for k in (keys or [])
    ]


def _state_of(container: Container, graph, thread_id: str,
              trainee_id: str, period: str):
    """组装状态响应。

    ★ 优先看 MySQL：有定稿就是 approved（唯一算数的状态）。
      没有定稿再问图的挂起状态。
    """
    from interfaces.http.schemas.models import ReportStateResponse

    row = None
    try:
        row = container.repository.get_monthly_report(trainee_id, period)
    except Exception as exc:                       # noqa: BLE001
        logger.warning("读定稿失败: %s", exc)

    if row is not None:
        return ReportStateResponse(
            thread_id=thread_id, status="approved",
            trainee_id=trainee_id, period=period,
            report=row.report_json or {}, disclaimer=_DISCLAIMER,
        )

    try:
        st = graph.get_state({"configurable": {"thread_id": thread_id}})
        values = dict(st.values or {})
        waiting = bool(st.next)
    except Exception as exc:                       # noqa: BLE001
        logger.warning("读图状态失败: %s", exc)
        values, waiting = {}, False

    if not waiting and not values:
        return ReportStateResponse(
            thread_id=thread_id, status="unknown",
            trainee_id=trainee_id, period=period,
            disclaimer=_DISCLAIMER,
        )

    overview = (values.get("summary") or {}).get("overview", {})
    weak_keys = values.get("weak") or []
    pending = {
        "total_drills": overview.get("total_drills", 0),
        "weak": weak_keys,
        "weak_caps": _weak_named(weak_keys),
        "suggestion": values.get("suggestion_text", ""),
        "suggestion_passed": values.get("suggestion_passed", True),
        "suggestion_note": values.get("suggestion_note", ""),
        "rewrites": values.get("rewrites", 0),
    }
    return ReportStateResponse(
        thread_id=thread_id,
        status="pending_review" if waiting else "rejected",
        trainee_id=trainee_id, period=period,
        pending=pending, disclaimer=_DISCLAIMER,
    )


# ===================================================================
# P3：批量生成（队列）
# ===================================================================
def _batch_use_case(container: Container):
    from application.use_cases.batch_report import BatchReportUseCase

    if container.task_queue is None:
        raise HTTPException(503, detail="任务队列未就绪，批量生成不可用")
    return BatchReportUseCase(
        repository=container.repository,
        queue=container.task_queue,
        max_attempts=container.settings.queue_max_attempts,
    )


@router.post("/monthly/batch", response_model=BatchProgressResponse,
             summary="批量投递报告生成任务（P3 队列）",
             dependencies=[Depends(require_feature("drill"))])
def batch_generate(
    req: BatchGenerateRequest,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> BatchProgressResponse:
    """把一批学员的报告生成投进队列，**立即返回**。

    ★ 这是批量场景的关键：20 人 × 5–15 秒 = 100–300 秒，
      同步接口必然超时。投递后由 worker 异步跑。

    ★ 幂等：已在处理/待复核/已完成的人会被跳过，
      主管重复点「生成本月报告」不会重复生成。

    ★ 权限：只有主管/管理员能批量（学员不该给别人生成）。
    """
    if container.repository is None:
        raise HTTPException(503, detail="仓储未就绪")
    if not _can_view(identity, "__batch__"):
        raise HTTPException(403, detail="只有主管可以批量生成报告")

    period = (req.period or "").strip() or _current_period()
    trainee_ids = [t.strip() for t in (req.trainee_ids or []) if t.strip()]
    if not trainee_ids:
        trainee_ids = _all_active_employees(container)
    if not trainee_ids:
        raise HTTPException(400, detail="没有可生成报告的学员")

    uc = _batch_use_case(container)
    try:
        result = uc.enqueue_batch(
            period=period, trainee_ids=trainee_ids,
            since=req.since, until=req.until)
    except Exception as exc:                       # noqa: BLE001
        logger.error("批量投递失败: %s: %s",
                     type(exc).__name__, exc, exc_info=True)
        raise HTTPException(500, detail="批量投递失败") from exc

    prog = uc.progress(period)
    return BatchProgressResponse(
        period=period,
        total=prog["total"], generated=prog["generated"],
        queued=prog["queued"], running=prog["running"],
        pending_review=prog["pending_review"], done=prog["done"],
        failed=prog["failed"], dead=prog["dead"], percent=prog["percent"],
        enqueued_now=result.enqueued, skipped=result.skipped,
    )


@router.get("/monthly/batch/progress", response_model=BatchProgressResponse,
            summary="查询批量进度",
            dependencies=[Depends(require_feature("drill"))])
def batch_progress(
    period: str = Query("", description="周期，空则当前月"),
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> BatchProgressResponse:
    """查某周期的批量进度（前端轮询用）。"""
    if container.repository is None:
        raise HTTPException(503, detail="仓储未就绪")
    if not _can_view(identity, "__batch__"):
        raise HTTPException(403, detail="只有主管可以查看批量进度")

    p = (period or "").strip() or _current_period()
    prog = _batch_use_case(container).progress(p)
    return BatchProgressResponse(
        period=p, total=prog["total"], generated=prog["generated"],
        queued=prog["queued"], running=prog["running"],
        pending_review=prog["pending_review"], done=prog["done"],
        failed=prog["failed"], dead=prog["dead"], percent=prog["percent"],
    )


def _all_active_employees(container: Container) -> list[str]:
    """取所有在职员工编号（批量生成的默认范围）。"""
    try:
        rows = container.repository.list_employees(limit=1000)
    except Exception as exc:                       # noqa: BLE001
        logger.warning("读取员工列表失败: %s", exc)
        return []
    out: list[str] = []
    for r in rows:
        no = getattr(r, "employee_no", "") or ""
        active = getattr(r, "active", True)
        if no and active:
            out.append(no)
    return out


@router.get("/monthly/batch/tasks", response_model=BatchTaskListResponse,
            summary="批量任务明细（逐人状态）",
            dependencies=[Depends(require_feature("drill"))])
def batch_tasks(
    period: str = Query("", description="周期，空则当前月"),
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> BatchTaskListResponse:
    """列出某周期每个学员的任务状态。

    ★ 为什么需要它：进度条只给「20/20 完成」，
      但主管要的是「**谁**卡住了、谁失败了」。
      没有明细，`failed=1` 这种数字无法定位到人。
    """
    if container.repository is None:
        raise HTTPException(503, detail="仓储未就绪")
    if not _can_view(identity, "__batch__"):
        raise HTTPException(403, detail="只有主管可以查看批量任务")

    p = (period or "").strip() or _current_period()
    rows = container.repository.list_report_tasks(period=p, limit=500)

    # 工号 → 姓名（任务表不存姓名，要现查）
    names = _employee_names(container)
    labels = {
        "queued": "排队中", "running": "生成中",
        "pending_review": "待复核", "done": "已定稿",
        "failed": "失败", "dead": "已放弃",
    }

    items = [
        BatchTaskItem(
            trainee_no=r.get("trainee_id", "") or "",
            name=names.get(r.get("trainee_id", ""), ""),
            status=r.get("status", "") or "",
            status_label=labels.get(r.get("status", ""), r.get("status", "")),
            attempts=int(r.get("attempts") or 0),
            last_error=(r.get("last_error") or "")[:200],
            thread_id=r.get("thread_id", "") or "",
        )
        for r in rows
    ]
    return BatchTaskListResponse(period=p, items=items)


def _employee_names(container: Container) -> dict[str, str]:
    """`工号 → 姓名`。取不到就返回空 dict（前端只显示工号）。"""
    try:
        rows = container.repository.list_employees(limit=1000)
        return {getattr(r, "employee_no", ""): getattr(r, "name", "")
                for r in rows if getattr(r, "employee_no", "")}
    except Exception as exc:                       # noqa: BLE001
        logger.warning("读取姓名失败（只显示工号）: %s", exc)
        return {}


@router.post("/monthly/generate", response_model=ReportStateResponse,
             summary="生成报告并挂起等待复核（P2 图编排）",
             dependencies=[Depends(require_feature("drill"))])
def generate_report(
    req: GenerateReportRequest,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> ReportStateResponse:
    """跑图到「主管复核」节点，返回 thread_id。

    ★ 此时**不写 MySQL**：库里没有定稿，状态只在 Redis 的 checkpoint 里。
    """
    if container.repository is None:
        raise HTTPException(503, detail="仓储未就绪")

    target = (req.trainee_id or "").strip() or identity.employee_no
    if not _can_view(identity, target):
        raise HTTPException(403, detail="无权为他人生成报告")

    period = (req.period or "").strip() or _current_period()
    graph = _build_graph(container)

    from application.graph.monthly_report_graph import new_thread_id

    thread_id = new_thread_id(target, period)
    cfg = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke({
            "trainee_id": target, "period": period,
            "since": req.since, "until": req.until,
            "thread_id": thread_id,
        }, cfg)
    except Exception as exc:                       # noqa: BLE001
        logger.error("报告图执行失败: %s: %s",
                     type(exc).__name__, exc, exc_info=True)
        raise HTTPException(500, detail="报告生成失败") from exc

    itf = result.get("__interrupt__") or []
    if not itf:
        logger.warning("图未在复核节点挂起，thread=%s", thread_id)

    return ReportStateResponse(
        thread_id=thread_id, status="pending_review",
        trainee_id=target, period=period,
        pending=dict(itf[0].value) if itf else {},
        disclaimer=_DISCLAIMER,
    )


@router.post("/monthly/{thread_id}/review", response_model=ReportStateResponse,
             summary="复核报告（同意 / 驳回重生成）",
             dependencies=[Depends(require_feature("drill"))])
def review_report(
    thread_id: str,
    req: ReviewReportRequest,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> ReportStateResponse:
    """提交复核决定。

    - `approve`：走完最后一步，**写 MySQL 定稿**
    - `reject` ：回「生成建议」重新写（**不重算数据**），仍停在待复核

    ★ 复核人从 token 里取，**不接受请求参数传** —— 否则可伪造。
    """
    if container.repository is None:
        raise HTTPException(503, detail="仓储未就绪")

    decision = (req.decision or "").strip().lower()
    if decision not in ("approve", "reject"):
        raise HTTPException(400, detail="decision 只能是 approve 或 reject")

    trailer, period = _parse_thread_id(thread_id)
    if not trailer:
        raise HTTPException(400, detail="thread_id 格式不正确")
    if not _can_view(identity, trailer):
        raise HTTPException(403, detail="无权复核他人的报告")

    graph = _build_graph(container)
    cfg = {"configurable": {"thread_id": thread_id}}

    from langgraph.types import Command

    try:
        graph.invoke(Command(resume={
            "decision": decision,
            "note": req.note or "",
            "reviewed_by": getattr(identity, "employee_no", "") or "",
        }), cfg)
        # 人工复核走的是图，不是批量 worker；同步一下批量任务表，
        # 否则批量进度会一直停在 pending_review。
        if decision == "approve":
            repo = container.repository
            if (hasattr(repo, "get_report_task")
                    and hasattr(repo, "upsert_report_task")):
                try:
                    task = repo.get_report_task(trailer, period)
                    if task:
                        repo.upsert_report_task(
                            task_id=str(task.get("task_id")
                                        or f"{trailer}:{period}"),
                            trainee_id=trailer,
                            period=period,
                            status="done",
                            thread_id=thread_id,
                        )
                except Exception as task_exc:       # noqa: BLE001 辅助状态不阻断复核
                    logger.warning("同步报告任务状态失败: %s", task_exc)
    except Exception as exc:                       # noqa: BLE001
        logger.error("复核失败 thread=%s: %s: %s",
                     thread_id, type(exc).__name__, exc, exc_info=True)
        raise HTTPException(500, detail="复核失败") from exc

    return _state_of(container, graph, thread_id, trailer, period)


@router.get("/monthly/{thread_id}/state", response_model=ReportStateResponse,
            summary="查询报告任务状态",
            dependencies=[Depends(require_feature("drill"))])
def report_state(
    thread_id: str,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> ReportStateResponse:
    """查任务进度：待复核 / 已定稿。"""
    trailer, period = _parse_thread_id(thread_id)
    if not trailer:
        raise HTTPException(400, detail="thread_id 格式不正确")
    if not _can_view(identity, trailer):
        raise HTTPException(403, detail="无权查看他人的报告")

    graph = _build_graph(container)
    return _state_of(container, graph, thread_id, trailer, period)