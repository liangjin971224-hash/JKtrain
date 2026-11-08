"""月度报告生成图（P2）：8 节点 LangGraph 编排。

────────────────────────────────────────────────────────────────────
为什么要用图（而不是继续用 P1 的直线调用）
────────────────────────────────────────────────────────────────────
P1 是一条直线：拉数据 → 聚合 → 生成建议 → 校验 → 返回。
它有三个图才能表达好的东西：

    ① 条件分支：只有薄弱项才下钻取明细（否则跳过，省一次查询）
    ② 退回环：  校验不通过 → 回「生成建议」重写（本质是循环）
    ③ **中断等人**：主管复核必须挂起，隔天/隔周回来还能继续

第 ③ 条是决定性的 —— 直线代码里没法「挂起等外部输入」，
只能拆成两个接口 + 自己存状态，那就是手写一个图。

────────────────────────────────────────────────────────────────────
★ 领域逻辑零改动
────────────────────────────────────────────────────────────────────
聚合与评分（`domain/services/monthly_report.py`）**一行不改**，
图只是按顺序调用它。这是 P1 把评分做成纯函数的价值所在。

────────────────────────────────────────────────────────────────────
★ 驳回只有一条路：回「生成建议」（用户 2026-10-31 决策）
────────────────────────────────────────────────────────────────────
不做「回上游重算数据」，原因（实测得出）：

    1. `operator.add` reducer 的字段在回上游后会**翻倍**
       （实测 ['风险1','风险2'] → ['风险1','风险2','风险1','风险2']）
    2. 「重算数据」的语义是模糊的：只重算聚合？还是连数据重拉？
       重拉的话 checkpoint 价值就没了，等于整图重跑
    3. 如果数据真算错了，那应该修代码 + 重跑，而不是给主管一个
       「重算」按钮 —— **按钮会把 bug 掩盖成操作问题**

所以 `review` 的驳回分支固定指向 `suggest`。
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from domain.services.monthly_report import (
    CAPABILITY_META,
    MonthlyReport,
    build_report,
)

logger = logging.getLogger(__name__)

# ---------------- 节点名（字符串常量，避免拼错） ----------------
N_LOAD = "load_data"
N_AGG_CASE = "aggregate_by_case"
N_MERGE_CAP = "merge_capabilities"
N_DRILL = "drill_down"
N_SUGGEST = "make_suggestion"
N_CHECK = "guardrail_check"
N_REVIEW = "await_review"
N_PERSIST = "persist"

#: 复核决定的取值。
DECISION_APPROVE = "approve"
DECISION_REJECT = "reject"

#: 建议被校验拦截的最大重写次数（与 P1 保持一致）。
MAX_REWRITES = 1


class ReportState(TypedDict, total=False):
    """图的共享状态。

    ★ 所有字段都用**覆盖语义**，不用 `Annotated[..., operator.add]`。
      原因见模块注释：追加型 reducer 在回上游时会翻倍。
      本图确实只有一条退回边（回 suggest），不回累积型节点，
      但保持覆盖语义是更安全的默认。

    字段分三组：
        · 输入：trainee_id / period / since / until
        · 过程：report / suggestion_text / ...
        · 输出：persisted / thread_id
    """

    # ---- 输入 ----
    trainee_id: str
    period: str
    since: str
    until: str

    # ---- 过程 ----
    #: ★ 中间传参：load_data → aggregate_by_case。
    #:   必须在 State 里声明，否则 LangGraph 会丢弃未声明字段
    #:   （实测踩坑：不声明时 aggregate 拿到空列表，报告全是 0）。
    _reviews: list
    #: 聚合后的报告（dataclass，不参与序列化对比，只在本进程内存里流转）
    report: Any
    #: 报告的三层结构摘要（dict，可序列化）
    summary: dict
    #: 薄弱能力项键列表
    weak: list
    #: 下钻得到的明细（只有薄弱项才有）
    drilldown: list
    #: 建议正文
    suggestion_text: str
    suggestion_passed: bool
    suggestion_note: str
    #: 已重写次数
    rewrites: int
    #: 复核决定
    decision: str
    review_note: str
    #: ★ 主管驳回意见（与 suggestion_note 分开！）
    #:
    #: 踩过的坑：一开始复用 suggestion_note 传驳回意见，
    #: 但它在校验通过时会被置空 → 驳回后读不到意见，提示词里没有它。
    #: 两个字段语义不同：suggestion_note 是「机器判定的失败原因」，
    #: review_feedback 是「人给的修改要求」。
    review_feedback: str
    #: 复核人（从 interrupt 的 resume 值里带进来，落库要用）
    reviewed_by: str

    # ---- 输出 ----
    thread_id: str
    persisted: bool
    error: str


def _to_matrix(state: dict) -> dict:
    """把 dataclass 报告转成可序列化的 dict（checkpoint 要存）。"""
    report = state.get("report")
    if report is None:
        return {}
    try:
        from application.use_cases.monthly_report import _to_summary
        from application.use_cases.monthly_report import GeneratedSuggestion

        return _to_summary(
            report,
            GeneratedSuggestion(
                text=state.get("suggestion_text", ""),
                passed=state.get("suggestion_passed", True),
                reason=state.get("suggestion_note", ""),
            ),
        )
    except Exception as exc:                       # noqa: BLE001
        logger.warning("报告转摘要失败: %s", exc)
        return {}


def _rebuild_report(
    trainee_id: str, period: str, summary: dict,
) -> MonthlyReport:
    """从 `_to_summary()` 产出的 dict 还原 `MonthlyReport`。

    ★ 为什么需要它：State 里只存 dict（dataclass 会让 checkpointer 报警告），
      但 `drill_down` / `make_suggestion` 需要对象。
      跨进程恢复时内存缓存是空的，必须能从 dict 重建。

    ★ 只还原**后续节点真正用到的字段**：
      capabilities（判 has_weak）、outcomes（下钻明细）、weak_capabilities。
      其余字段（total_duration_sec 等）填 0 —— 它们只在 summary 里被读，
      不参与后续计算，重建时丢了不影响。
    """
    from domain.services.monthly_report import (
        CapabilityScore,
        CaseOutcome,
        MonthlyReport,
    )

    caps = tuple(
        CapabilityScore(
            capability=c.get("key", ""),
            name=c.get("name", ""),
            samples=int(c.get("samples") or 0),
            score=c.get("score"),
            level=c.get("level", ""),
            level_label=c.get("level_label", ""),
            critical_count=int(c.get("critical_count") or 0),
            note=c.get("note", ""),
        )
        for c in (summary.get("capabilities") or [])
    )
    outcomes = tuple(
        CaseOutcome(
            drill_id=o.get("drill_id", ""),
            case_id=o.get("case_id", ""),
            case_title=o.get("case_title", ""),
            case_type=o.get("case_type", ""),
            capability=o.get("capability", ""),
            duration_sec=0,
            element_hit=int(o.get("element_hit") or 0),
            element_total=int(o.get("element_total") or 0),
            risk_count=int(o.get("risk_count") or 0),
            high_count=0,
            medium_count=0,
            critical_count=int(o.get("critical_count") or 0),
            score=float(o.get("score") or 0.0),
            score_note=o.get("score_note", ""),
            created_at=o.get("created_at", ""),
            risks=tuple(o.get("risks") or ()),
        )
        for o in (summary.get("cases") or [])
    )
    overview = summary.get("overview") or {}
    return MonthlyReport(
        trainee_id=trainee_id,
        period=period,
        total_drills=int(overview.get("total_drills") or 0),
        total_duration_sec=int(overview.get("total_duration_sec") or 0),
        distinct_cases=int(overview.get("distinct_cases") or 0),
        capabilities=caps,
        outcomes=outcomes,
        weak_capabilities=tuple(summary.get("weak_capabilities") or ()),
    )


def build_report_graph(
    *,
    repository: Any,
    llm: Any = None,
    guardrail: Any = None,
    case_titles: dict[str, str] | None = None,
    doc_catalog: dict[str, str] | None = None,
    checkpointer: Any = None,
) -> Any:
    """构建并编译月度报告图。

    Args:
        repository: 仓储（读评阅结果 + 写定稿）。
        llm: 语言模型。None 时只出数据不出建议（便于无 LLM 测试）。
        guardrail: 报告专用校验器（`ReportGuardrail`）。
        case_titles: `case_id → 标题`。
        doc_catalog: `doc_id → 文档标题`（真实清单，防编造资料名）。
        checkpointer: checkpoint 存储器。
            ★ 必须传才支持 interrupt 恢复：`interrupt()` 依赖
              checkpointer 保存挂起状态，不传会直接报错。
            生产用 `CacheCheckpointSaver`，测试可用 `InMemorySaver`。

    Returns:
        编译后的图。
    """
    from application.use_cases.monthly_report import (
        build_suggestion_prompt,
    )
    from domain.services.report_guardrail import ReportGuardrail

    titles = case_titles or {}
    docs = doc_catalog or {}
    gr = guardrail or ReportGuardrail()

    # ================================================================
    # 节点实现
    # ================================================================
    def load_data(state: ReportState) -> dict:
        """① 拉本月评阅记录。"""
        tid = state.get("trainee_id", "")
        try:
            reviews = repository.list_drill_reviews(
                trainee_id=tid,
                since=state.get("since", ""),
                until=state.get("until", ""),
                with_items=True,
            )
        except Exception as exc:                   # noqa: BLE001
            logger.error("拉取评阅记录失败: %s", exc)
            return {"error": f"拉取数据失败: {type(exc).__name__}"}
        logger.info("月度报告 ① 拉数据: %d 条", len(reviews))
        # ★ 数据存进程内缓存，State 里只放一个标记
        #   （dataclass 进 checkpoint 会报 unregistered type）
        _reviews_cache[state.get("thread_id", "")] = reviews
        return {"_reviews_marker": True}

    #: 图执行期间持有 MonthlyReport 对象的地方（**仅内存缓存，不可依赖**）。
    #:
    #: ★ 为什么不放进 State：`MonthlyReport` 是 dataclass，
    #:   LangGraph 的 checkpointer 序列化时会报
    #:   「Deserializing unregistered type ... This will be blocked in a future version」。
    #:
    #: ★★ 为什么每个节点都要能从 state 重建，不能只读这个字典：
    #:   进程重启后字典是空的。虽然本图的中断点在 `await_review`、
    #:   其后的 `persist` 只用 `summary`（dict）所以碰巧能恢复，
    #:   但这是**偶然**——一旦将来把中断点前移，就会静默失败。
    #:   所以这里只是「同一次 invoke 内的优化」，不是状态来源。
    _reports: dict[str, Any] = {}
    #: 同理：`DrillReviewRow` 是 dataclass，也不能进 State
    _reviews_cache: dict[str, list] = {}

    def _report_from_state(state: ReportState) -> MonthlyReport | None:
        """取报告：先查内存缓存，查不到从 state 重建。

        ★ 这段「重建」逻辑是跨进程恢复的关键：
          状态里只存了 summary（dict），但 drill_down / make_suggestion
          需要 MonthlyReport 对象。从 summary 重建即可，
          不必把 dataclass 塞进 checkpoint。
        """
        tid = state.get("thread_id", "")
        cached = _reports.get(tid)
        if cached is not None:
            return cached
        summary = state.get("summary") or {}
        if not summary:
            return None
        try:
            report = _rebuild_report(state.get("trainee_id", ""),
                                     state.get("period", ""), summary)
            _reports[tid] = report
            return report
        except Exception as exc:                   # noqa: BLE001
            logger.warning("从 state 重建报告失败: %s", exc)
            return None

    def aggregate_by_case(state: ReportState) -> dict:
        """②③ 聚合 + 能力项归并（一次调收益，避免拆两个空跑节点）。"""
        # ★ 优先读缓存；跨进程恢复时缓存空 → 重新拉一次
        tid = state.get("thread_id", "")
        reviews = _reviews_cache.get(tid)
        if reviews is None:
            try:
                reviews = repository.list_drill_reviews(
                    trainee_id=state.get("trainee_id", ""),
                    since=state.get("since", ""), until=state.get("until", ""),
                    with_items=True)
                _reviews_cache[tid] = reviews
            except Exception as exc:               # noqa: BLE001
                logger.warning("重拉评阅记录失败: %s", exc)
                reviews = []
        report = build_report(
            trainee_id=state.get("trainee_id", ""),
            period=state.get("period", ""),
            reviews=reviews,
            case_titles=titles,
        )
        logger.info("月度报告 ②③ 聚合: %d 次演练 / %d 个能力项",
                    report.total_drills, len(report.capabilities))
        # ★ 只把可序列化的 summary 放进 state；dataclass 存在进程内字典里
        tid = state.get("thread_id", "")
        _reports[tid] = report
        return {
            "summary": _to_matrix({"report": report}),
            "weak": list(report.weak_capabilities),
        }

    def drill_down(state: ReportState) -> dict:
        """④ 下钻：只取薄弱能力项下的案例明细。

        ★ 只有在「有薄弱项」时才执行（条件边控制）。
          这是图的第一处真实价值：没薄弱项就不做这一步。
        """
        report = _report_from_state(state)
        weak = set(state.get("weak") or [])
        if report is None or not weak:
            # ★ 显式返回空列表（不能返回 {}，否则状态里根本没有这个键，
            #   下游 .get("drilldown") 拿到 None 而不是 []）
            return {"drilldown": []}

        rows = []
        for o in report.outcomes:
            if o.capability not in weak:
                continue
            rows.append({
                "case_id": o.case_id,
                "case_title": o.case_title,
                "capability": o.capability,
                "score": o.score,
                "risks": list(o.risks),
            })
        logger.info("月度报告 ④ 下钻: %d 个案例", len(rows))
        return {"drilldown": rows}

    def make_suggestion(state: ReportState) -> dict:
        """⑥ 生成建议（LLM 调用点，全图唯一）。"""
        report = _report_from_state(state)
        if report is None or not report.has_weak:
            # ★ 无薄弱项不生成建议 —— 硬凑就是让模型编
            return {
                "suggestion_text": "本月各能力项均无明显问题，建议增加案例覆盖广度。",
                "suggestion_passed": True,
                "suggestion_note": "",
                "skipped": True,
            }
        if llm is None:
            return {
                "suggestion_text": "",
                "suggestion_passed": True,
                "suggestion_note": "未配置 LLM，仅输出数据部分",
                "skipped": True,
            }

        prompt = build_suggestion_prompt(report, docs)

        # ★ 两种「要改」的来源不同，分别处理：
        #   ① 校验没通过（机器判定）→ suggestion_note
        #   ② 主管驳回（人给的）→ review_feedback
        machine_note = state.get("suggestion_note", "")
        human_note = state.get("review_feedback", "")
        if machine_note:
            prompt += (f"\n\n【上次的问题】\n你的上一次输出违反规则：{machine_note}\n"
                       "请重写，务必不出现任何功效宣称或绝对化表述。")
        if human_note:
            prompt += (f"\n\n【主管的修改要求】\n{human_note}\n"
                       "请按这个要求重写。")

        try:
            res = llm.generate(
                prompt,
                system_prompt=("你是保健品门店的培训主管，负责写学员的月度改进建议。"
                               "只输出建议正文，不解释。严禁出现产品功效表述。"),
                temperature=0.0,
                max_tokens=600,
            )
            text = (getattr(res, "text", "") or "").strip()
        except Exception as exc:                   # noqa: BLE001
            logger.warning("生成建议失败: %s", exc)
            text = ""

        logger.info("月度报告 ⑥ 生成建议: %d 字（第 %d 次）",
                    len(text), state.get("rewrites", 0) + 1)
        return {
            "suggestion_text": text,
            "suggestion_passed": bool(text),
            "suggestion_note": "" if text else "LLM 未返回内容",
        }

    def guardrail_check(state: ReportState) -> dict:
        """⑦ 校验建议。"""
        text = state.get("suggestion_text", "")
        if not text:
            return {"suggestion_passed": False,
                    "suggestion_note": state.get("suggestion_note") or "无内容"}
        try:
            v = gr.check(text)
        except Exception as exc:                   # noqa: BLE001
            logger.warning("校验异常，按未通过处理: %s", exc)
            return {"suggestion_passed": False,
                    "suggestion_note": f"校验异常: {type(exc).__name__}"}

        logger.info("月度报告 ⑦ 校验: %s", "通过" if not v.blocked else "不通过")
        return {
            "suggestion_passed": not v.blocked,
            "suggestion_note": v.reason if v.blocked else "",
        }

    def await_review(state: ReportState) -> dict:
        """⑧ 断点：等主管确认。

        ★ 这是整张图存在的**核心理由**：
          直线代码无法「挂起等外部输入」，只能拆两个接口自己存状态。

        `interrupt` 的返回值就是 `Command(resume=...)` 传进来的内容。
        """
        # ★ 薄弱项同时给 key 与中文名：
        #   key（C_contraindication）机器用，name（禁忌与配伍）给人看。
        #   原实现只给了 key，前端直接把 key 显示成「薄弱能力项」，
        #   主管看到的是英文标识符 —— 等于没告诉他哪里弱。
        weak_keys = state.get("weak") or []
        weak_named = [
            {"key": k, "name": CAPABILITY_META.get(k, {}).get("name", k)}
            for k in weak_keys
        ]
        payload = {
            "trainee_id": state.get("trainee_id", ""),
            "period": state.get("period", ""),
            "total_drills": (state.get("summary") or {}).get("overview", {})
                            .get("total_drills", 0),
            # 保留 weak（key 列表，兼容既有前端/脚本），另给 weak_caps（带名字）
            "weak": weak_keys,
            "weak_caps": weak_named,
            "suggestion": state.get("suggestion_text", ""),
            "suggestion_passed": state.get("suggestion_passed", True),
            "suggestion_note": state.get("suggestion_note", ""),
        }
        answer = interrupt(payload)

        if isinstance(answer, dict):
            decision = str(answer.get("decision") or DECISION_APPROVE)
            note = str(answer.get("note") or "")
            reviewer = str(answer.get("reviewed_by") or "")
        else:
            decision = DECISION_APPROVE if answer else DECISION_REJECT
            note = ""
            reviewer = ""

        logger.info("月度报告 ⑧ 复核决定: %s", decision)
        # ★ 驳回意见同时写进 review_feedback（供重生成时用）
        #   和 review_note（落库给审计看）
        return {"decision": decision, "review_note": note,
                "review_feedback": note if decision == DECISION_REJECT else "",
                "reviewed_by": reviewer}

    def persist(state: ReportState) -> dict:
        """⑨ 写 MySQL 定稿（唯一一次业务库写入）。"""
        if state.get("decision") != DECISION_APPROVE:
            return {"persisted": False}

        from domain.ports.repository_port import MonthlyReportRow

        summary = dict(state.get("summary") or {})
        summary["suggestion"] = {
            "text": state.get("suggestion_text", ""),
            "passed": state.get("suggestion_passed", True),
            "reason": state.get("suggestion_note", ""),
        }
        summary["drilldown"] = state.get("drilldown") or []

        row = MonthlyReportRow(
            trainee_id=state.get("trainee_id", ""),
            period=state.get("period", ""),
            thread_id=state.get("thread_id", ""),
            status="approved",
            report_json=summary,
            suggestion=state.get("suggestion_text", ""),
            suggestion_passed=bool(state.get("suggestion_passed", True)),
            suggestion_note=state.get("suggestion_note", ""),
            total_drills=(summary.get("overview") or {}).get("total_drills", 0),
            total_critical=(summary.get("overview") or {}).get("total_critical", 0),
            weak_capabilities=tuple(state.get("weak") or ()),
            graph_steps=9,
            reviewed_by=state.get("reviewed_by", ""),
            reviewed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            review_note=state.get("review_note", ""),
        )
        try:
            repository.save_monthly_report(row)
        except Exception as exc:                   # noqa: BLE001
            logger.error("写定稿失败: %s", exc, exc_info=True)
            return {"persisted": False,
                    "error": f"写定稿失败: {type(exc).__name__}"}

        tid = state.get("thread_id", "")
        _reports.pop(tid, None)
        _reviews_cache.pop(tid, None)
        logger.info("月度报告 ⑨ 已定稿: %s %s",
                    row.trainee_id, row.period)
        return {"persisted": True}

    # ================================================================
    # 路由
    # ================================================================
    def after_load(state: ReportState) -> str:
        """拉数据出错 → 直接结束，不要把空报告当成功往下跑。

        ★★ 这是真实 bug（2026-10-31 实测发现）：
          `load_data` 捕获异常后返回 `{"error": ...}`，
          但原来**没有任何条件边检查 error** → 图继续跑，
          用空数据算出「0 次演练」的报告，最后还写进了定稿。

          后果：任务表显示 `pending_review`（成功），
          而实际上数据一条都没拉到 —— **失败被静默吞掉了**，
          主管看到的是一份"完美"的空报告。

          现在：有 error 直接 END，让消费端 handler 判失败 → 重试 / 判死。
        """
        if state.get("error"):
            logger.error("图终止（拉数据失败）: %s", state.get("error"))
            return END
        return N_AGG_CASE

    def after_aggregate(state: ReportState) -> str:
        """有薄弱项才下钻。"""
        return N_DRILL if state.get("weak") else N_SUGGEST

    def after_check(state: ReportState) -> str:
        """校验不通过 → 回 ⑥ 重写；通过 → 复核。

        ★ 退回边是图不可替代的一处：它是**循环**，
          直线代码要写成 while，且「重写了什么」不可见。
        """
        if state.get("suggestion_passed"):
            return N_REVIEW
        if state.get("rewrites", 0) < MAX_REWRITES:
            return N_SUGGEST
        # 重试用尽仍不通过 → 仍然进复核，但带着「未通过」标记
        #   ★ 不因为建议不合规就整份报告作废 —— 数据部分依然有价值
        return N_REVIEW

    def bump_rewrites(state: ReportState) -> dict:
        """退回 ⑥ 前把重写次数 +1（单独节点，便于观测）"""
        return {"rewrites": state.get("rewrites", 0) + 1}

    def after_review(state: ReportState) -> str:
        """复核决定路由。

        ★ 驳回**固定回 ⑥**（重新生成建议），不回上游重算数据。
          实测回上游会让 `operator.add` 字段翻倍，且「重算什么」语义模糊。

        ★ 驳回也要经过 `bump_rewrites`：
          最初只在校验失败分支累加，导致「主管驳回 3 次，rewrites 仍是 0」。
          rewrites 的语义是「建议被重写了多少次」，**不区分谁要求重写**
          （机器判定不合格 / 主管不满意都算）。
        """
        if state.get("decision") == DECISION_REJECT:
            return "bump_rewrites"
        return N_PERSIST

    # ================================================================
    # 组装
    # ================================================================
    g = StateGraph(ReportState)
    g.add_node(N_LOAD, load_data)
    g.add_node(N_AGG_CASE, aggregate_by_case)
    g.add_node(N_DRILL, drill_down)
    g.add_node(N_SUGGEST, make_suggestion)
    g.add_node(N_CHECK, guardrail_check)
    g.add_node(N_REVIEW, await_review)
    g.add_node(N_PERSIST, persist)
    g.add_node("bump_rewrites", bump_rewrites)

    g.add_edge(START, N_LOAD)
    g.add_conditional_edges(
        N_LOAD, after_load,
        {N_AGG_CASE: N_AGG_CASE, END: END})
    g.add_conditional_edges(
        N_AGG_CASE, after_aggregate,
        {N_DRILL: N_DRILL, N_SUGGEST: N_SUGGEST})
    g.add_edge(N_DRILL, N_SUGGEST)
    g.add_edge(N_SUGGEST, N_CHECK)
    g.add_conditional_edges(
        N_CHECK, after_check,
        {N_REVIEW: N_REVIEW, N_SUGGEST: "bump_rewrites"})
    g.add_edge("bump_rewrites", N_SUGGEST)
    g.add_conditional_edges(
        N_REVIEW, after_review,
        {"bump_rewrites": "bump_rewrites", N_PERSIST: N_PERSIST})
    g.add_edge(N_PERSIST, END)

    return g.compile(checkpointer=checkpointer)


def new_thread_id(trainee_id: str, period: str) -> str:
    """生成 thread_id。

    格式 `{trainee}:{period}:{uuid8}` —— 前两段便于人工排查「这是谁的哪个月」。
    """
    return f"{trainee_id}:{period}:{uuid.uuid4().hex[:8]}"


__all__ = [
    "DECISION_APPROVE",
    "DECISION_REJECT",
    "MAX_REWRITES",
    "N_AGG_CASE",
    "N_CHECK",
    "N_DRILL",
    "N_LOAD",
    "N_PERSIST",
    "N_REVIEW",
    "N_SUGGEST",
    "ReportState",
    "build_report_graph",
    "new_thread_id",
]
