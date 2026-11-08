"""知识问答路由。

两个入口：
    POST /api/ask          普通返回（一次给完整答案）
    POST /api/ask/stream   SSE 流式（首句 1–2 秒到，体验更好）

★ 两者都走同一个用例，区别只在「整段返回」还是「逐句返回」。
  护栏在用例层已经强制，接口层不需要也不应该重复判断。

★ 多轮问答（B2~B4）：只加「会话 + 改写 + 接口」三层。
  `session_id` 缺失 / 开关关闭 / 非法时，下面的用例调用参数、
  响应字段与原实现**完全一致**（响应模型用 exclude_unset 挡住新字段）。
  改写后的独立问题走原链路，检索 / 融合 / 窗口 / 生成 / 护栏 / prompt
  一行未改。
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from application.services.conversation import ConversationSession, ConversationTurn
from application.services.query_rewriter import QueryRewriter, RewriteResult
from application.use_cases.answer_question import AnswerQuestionUseCase
from application.use_cases.query_plan import QueryPlanUseCase
from config.assembly import Container
from interfaces.http.auth_deps import (
    Identity,
    current_identity,
    require_feature,
)
from interfaces.http.deps import current_settings, get_container
from interfaces.http.schemas.models import (
    AnnotationResponse,
    AskRequest,
    AskResponse,
    AskSegment,
    HopInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["问答"])

#: session_id 只允许 8~64 位字母/数字/下划线/连字符。
#: ★ 非法一律按「无会话」处理，不返回 4xx —— 前端老版本不带这个字段，
#:   带了脏数据也不该让整个问答挂掉。
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{8,64}")


def _build_use_case(container: Container) -> AnswerQuestionUseCase:
    """按当前容器装配问答用例。

    ★ 每请求新建用例对象是可以的——它是**轻量**的（只持有端口引用），
      昂贵的模型加载在端口里，已经由容器复用。
    """
    if container.llm is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 未配置，问答功能不可用。请在 .env 填 QWEN_API_KEY 或 DEEPSEEK_API_KEY",
        )
    return AnswerQuestionUseCase(
        graph=container.graph,
        vector=container.vector,
        keyword=container.keyword,
        # ★ 稀疏并集（P0-3）：只对配置的 collection 生效（默认案例库）。
        sparse=container.sparse,
        sparse_collections=frozenset(
            c.strip() for c in container.settings.sparse_recall_collections.split(",")
            if c.strip()),
        min_sparse_score=container.settings.min_sparse_score,
        # ★ 每路头部窗口（P1-a）：默认只对文档库开；配置置空即回滚纯 RRF。
        path_head_collections=frozenset(
            c.strip() for c in container.settings.path_head_collections.split(",")
            if c.strip()),
        # ★ 参考资料条数（P1-b）：默认 5，经答案级 A/B 后再调大。
        context_docs=container.settings.context_docs,
        # ★ 系统提示词版本（P1-c）：默认 v3（已通过主集 + 独立 held-out 验证）。
        prompt_variant=container.settings.prompt_variant,
        llm=container.llm,
        # ★ A3：给推理模型留足 token，避免 reasoning 吃满导致空正文。
        llm_max_tokens=container.settings.llm_max_tokens,
        # ★ 传入仓储：召回为空时会写未命中日志（FR-047）。
        #   仓储为 None 时功能照常，只是不记日志。
        repository=container.repository,
        # ★ rerank（可选）：容器未启用时为 None，链路与历史行为一致。
        #   多跳复用同一个 uc，自动获得同样的重排行为（P5 单一生成路径纪律）。
        rerank=container.rerank,
        rerank_candidate_k=container.settings.rerank_candidate_k,
    )


def _build_multi_hop(
    container: Container, uc: AnswerQuestionUseCase, settings,
    default_product_ids: tuple[str, ...] = (),
) -> QueryPlanUseCase:
    """装配多跳用例。

    Args:
        default_product_ids: 多轮「产品切换」解析出的新产品 ID。
            `QueryPlanUseCase.execute` 不接受 product_ids，而纠正类问句
            （「不是这款，是 P002」）按设计**不改写句子**，只能靠它把
            新主语带进每一跳。单轮路径恒为空元组，行为完全不变。

    ★ 把**已经装配好的** `uc` 作为单跳实现注入，而不是在这里重新 new：

      · 护栏配置只有一份（`AnswerQuestionUseCase` 内部）
      · 意图分类器只有一份
      · 召回阈值只有一份

      多跳最容易犯的错就是「另起一条生成路径」，
      然后护栏/阈值/意图在其中一条路径上失效（P5 踩过规则漂移）。
    """
    def answer_one(question: str, *, allowed_permissions=None,
                   fact_hints=None, product_ids=()):
        return uc.execute(
            question,
            vector_collection=settings.coll_training,
            allowed_permissions=(
                allowed_permissions
                if allowed_permissions is not None
                else ("public", "internal")),
            # ★★ 这两个必须透传（2026-10-23 HTTP 链路实测暴露）：
            #   漏掉 `fact_hints` 会让多跳查到的事实**传不到生成层**，
            #   结果是接口里又编出「目录中没有改善体脂等表述」——
            #   而直连用例的测试是好的。**单测绿 ≠ 接口可用**。
            #   漏掉 `product_ids` 会让护栏拿不到产品类别，
            #   普通食品的功能声称就判不出来。
            fact_hints=fact_hints,
            product_ids=tuple(product_ids or ()) or default_product_ids,
        )

    return QueryPlanUseCase(
        answer_one=answer_one,
        graph=container.graph,
    )


def _hops_of(result) -> list[HopInfo]:
    """把多跳明细转成响应模型。

    ★ 一步时不返回（见 `AskResponse.hops` 的说明）。
    """
    hops = getattr(result, "hops", ()) or ()
    if len(hops) <= 1:
        return []
    return [
        HopInfo(
            order=h.order,
            question=h.question,
            kind=h.kind,
            depends_on_previous=h.candidates_source == "previous_hop",
            product_ids=list(h.product_ids),
            candidate_ids=list(h.candidate_ids),
            candidate_count=len(h.candidate_ids),
            refused=h.refused,
        )
        for h in hops
    ]


def _annotation_of(result) -> AnnotationResponse | None:
    """把用例层的标注结果转成响应模型。

    ★ `result.answer` 已经是剔除违规句后的合规版本，
      这里只搬标注明细给前端渲染标红。
    """
    ann = getattr(result, "annotation", None)
    if ann is None or not ann.segments:
        return None

    return AnnotationResponse(
        has_safe_content=ann.has_safe_content,
        segment_count=len(ann.segments),
        violation_count=ann.violation_count,
        repaired_warning=ann.repaired_warning,
        segments=[
            AskSegment(
                text=s.text,
                ok=s.ok,
                rule_types=list(s.rule_types),
                reason=s.reason,
            )
            for s in ann.segments
        ],
    )


# ===================================================================
# 多轮问答：会话解析、改写、落轮次
# ===================================================================
def _resolve_session(
    req: AskRequest, identity: Identity, container: Container,
) -> ConversationSession | None:
    """解析会话；返回 None 表示本轮按单轮处理。

    ★ 开关从**容器里的配置**取（装配时定下的，测试注入什么就用什么），
      不用 `current_settings()` —— 那个读进程环境变量，测试覆盖不到。

    ★ user_key 绑定工号与角色：同一个 session_id 换了身份就当新会话，
      不泄露上一身份的上下文（TDD §5）。
    """
    settings = getattr(container, "settings", None)
    if settings is None or not getattr(settings, "multi_turn_enabled", False):
        return None

    sid = (req.session_id or "").strip()
    if not sid or _SESSION_ID_RE.fullmatch(sid) is None:
        logger.info("忽略非法 session_id，按单轮处理")
        return None

    store = getattr(container, "conversation_store", None)
    if store is None:
        logger.warning("多轮会话存储未装配，降级为单轮")
        return None

    user_key = f"{identity.employee_no}:{identity.role}"
    try:
        session = store.get_or_create(sid, user_key)
        # ★ 前端点「新对话」：只清历史、不换 id（主干既有语义，合并后保留）。
        if getattr(req, "new_conversation", False):
            store.reset_session(sid, user_key)
            session = store.get_or_create(sid, user_key)
            logger.info("会话 %s 历史已按请求清空（新对话）", sid)
        return session
    except Exception as exc:                  # noqa: BLE001 store 异常降级单轮
        logger.warning("会话读取失败，降级为单轮: %s: %s",
                       type(exc).__name__, exc)
        return None


def _rewrite_question(
    question: str, session: ConversationSession | None, container: Container,
) -> tuple[str, RewriteResult | None]:
    """有会话才改写；任何异常都退回原问题。"""
    if session is None:
        return question, None
    try:
        rewriter = QueryRewriter(
            graph=container.graph, llm=container.llm, timeout_sec=2.0)
        result = rewriter.rewrite(question, session)
        return result.question, result
    except Exception as exc:                  # noqa: BLE001 改写失败降级单轮
        logger.warning("查询改写失败，回退原问题: %s: %s",
                       type(exc).__name__, exc)
        return question, None


def _product_ids_for(req: AskRequest, rewrite: RewriteResult | None) -> tuple[str, ...]:
    """改写器给出新产品 ID 时优先用它（产品切换只换 ID、不改句子）。"""
    if rewrite is not None and rewrite.product_ids:
        return rewrite.product_ids
    return tuple(req.product_ids)


def _intent_from_meta(meta: dict | None) -> str:
    """从用例 meta 里取意图标签（多跳是 intents 数组）。"""
    meta = meta or {}
    label = str(meta.get("intent", "") or "")
    if label:
        return label
    intents = meta.get("intents") or ()
    return str(intents[0]) if intents else ""


def _make_turn(
    req: AskRequest, rewritten: str, *, answer: str,
    cited=(), products=(), intent: str = "", refused: bool = False,
) -> ConversationTurn:
    """组装一轮记录。

    ★ 只存 `answer`（合规正文 / 拒答话术），绝不写
      `annotation.segments` —— 那里含违规原文，按约定不得入库。
    """
    return ConversationTurn(
        question=req.question,
        rewritten=rewritten,
        answer=str(answer or ""),
        cited_chunk_ids=tuple(cited or ()),
        product_ids=tuple(products or ()),
        intent=str(intent or ""),
        refused=bool(refused),
    )


def _append_turn(store, session_id: str, turn: ConversationTurn) -> int:
    """落一轮，返回轮次号；存储异常返回 0（本轮回答照常返回）。"""
    try:
        return int(store.append(session_id, turn) or 0)
    except Exception as exc:                  # noqa: BLE001
        logger.warning("多轮会话写入失败，本轮不保存: %s: %s",
                       type(exc).__name__, exc)
        return 0


def _multi_turn_fields(
    req: AskRequest, rewritten: str, turn_no: int,
    rewrite: RewriteResult | None,
) -> dict:
    """多轮响应新增的五个字段（单轮路径不会调用）。"""
    original = (req.question or "").strip()
    return {
        "session_id": req.session_id,
        "rewritten_question": "" if rewritten == original else rewritten,
        "turn": int(turn_no),
        # 主干前端在用 history_turns 显示「已记 N 轮」，值等于 turn。
        "history_turns": int(turn_no),
        "multi_turn": True,
        "rewrite_source": rewrite.source if rewrite is not None else "none",
    }


def _record_metrics(container: Container, t0: float, *,
                    result=None, answer: str = "", refused: bool = False,
                    meta: dict | None = None) -> None:
    """D3：记录一次问答（失败不影响回答）。"""
    metrics = getattr(container, "metrics", None)
    if metrics is None:
        return
    try:
        m = meta if meta is not None else (getattr(result, "meta", {}) or {})
        text = (getattr(result, "answer", "") if result is not None else answer)
        metrics.record_answer(
            latency_ms=(time.perf_counter() - t0) * 1000,
            refused=bool(getattr(result, "refused", False) if result is not None
                         else refused),
            blocked=bool(m.get("blocked_rules")),
            empty=not (text or "").strip(),
        )
    except Exception as exc:                  # noqa: BLE001
        logger.warning("运行指标记录失败（不影响回答）: %s: %s",
                       type(exc).__name__, exc)


@router.post("/ask", response_model=AskResponse, summary="知识问答",
             response_model_exclude_unset=True,
             dependencies=[Depends(require_feature("ask"))])
def ask(
    req: AskRequest,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> AskResponse:
    """回答员工的问题。

    **两种降级行为**：
    - 召回为空 → 拒答且**不调 LLM**，`refused=true`
    - 输出护栏命中 → 部分保留：`refused=false`，`answer` 是合规部分，
      `annotation.segments` 标出违规句与原因

    ★★ 字段使用约定：
    - **`answer`** = 合规版正文，可直接对外（展示 / 复制 / 日志 / 转发）
    - **`annotation`** = 逐句明细，**含违规原文**，仅限前端渲染标红
      如果只要正文，忽略这个字段即可

    ★ `response_model_exclude_unset=True`：没显式赋值的多轮字段不会
      出现在单轮响应里，老前端拿到的 JSON 与改造前逐字节一致。
    """
    settings = current_settings()
    uc = _build_use_case(container)
    t0 = time.perf_counter()
    session = _resolve_session(req, identity, container)
    rewritten, rewrite = _rewrite_question(req.question, session, container)
    product_ids = _product_ids_for(req, rewrite)
    store = getattr(container, "conversation_store", None)
    sid = req.session_id or ""

    # ★ 产品切换按设计不改写句子，多跳路径要把新 ID 透传给每一跳，
    #   否则「不是这款，是 P002」在多跳下会被当成无主语的泛问。
    #   单轮（无会话）时恒为空，原链路一字不变。
    switch_ids: tuple[str, ...] = ()
    if (session is not None and rewrite is not None
            and rewrite.source == "rule" and rewrite.product_ids
            and rewrite.question == (req.question or "").strip()):
        switch_ids = rewrite.product_ids

    # ---- 多跳（拆子查询 / 前一步结果作后一步前提）----
    if req.multi_hop:
        hop_uc = _build_multi_hop(container, uc, settings, switch_ids)
        hop = hop_uc.execute(
            rewritten,
            allowed_permissions=identity.role.retrieval_levels,
        )
        _record_metrics(container, t0, result=hop)
        extra: dict = {}
        if session is not None and store is not None:
            turn_no = _append_turn(store, sid, _make_turn(
                req, rewritten,
                answer=hop.answer,
                cited=hop.cited_chunk_ids,
                products=hop.product_ids,
                intent=_intent_from_meta(hop.meta),
                refused=hop.refused,
            )) or len(session.turns) + 1
            extra = _multi_turn_fields(req, rewritten, turn_no, rewrite)
        return AskResponse(
            answer=hop.answer,
            refused=hop.refused,
            refusal_reason="",
            cited_chunk_ids=list(hop.cited_chunk_ids),
            product_ids=list(hop.product_ids),
            meta={
                **hop.meta,
                "multi_hop": True,
                "cheapest": (
                    {
                        "product_id": hop.cheapest.product_id,
                        "name": hop.cheapest.name,
                        "price": hop.cheapest.price,
                        "category": hop.cheapest.category,
                    } if hop.cheapest else None
                ),
                "notes": list(hop.notes),
            },
            hops=_hops_of(hop),
            annotation=_annotation_of(hop),
            **extra,
        )

    result = uc.execute(
        rewritten,
        vector_collection=settings.coll_training,
        product_ids=product_ids,
        # ★★ 权限级别由**角色**决定，不接受前端声明（2026-10-23 修）。
        #    原实现用 req.allowed_permissions，学员传
        #    ["public","internal","restricted","confidential"]
        #    就能越权检索受限内容（FR-050 明确要求检索阶段过滤）。
        allowed_permissions=identity.role.retrieval_levels,
        top_k=req.top_k,
    )
    _record_metrics(container, t0, result=result)

    extra = {}
    if session is not None and store is not None:
        turn_no = _append_turn(store, sid, _make_turn(
            req, rewritten,
            answer=result.answer,
            cited=result.cited_chunk_ids,
            products=result.product_ids,
            intent=_intent_from_meta(result.meta),
            refused=result.refused,
        )) or len(session.turns) + 1
        extra = _multi_turn_fields(req, rewritten, turn_no, rewrite)

    return AskResponse(
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        cited_chunk_ids=list(result.cited_chunk_ids),
        product_ids=list(result.product_ids),
        meta=result.meta,
        hops=[],
        annotation=_annotation_of(result),
        **extra,
    )


@router.post("/ask/stream", summary="知识问答（SSE 流式）",
             dependencies=[Depends(require_feature("ask"))])
def ask_stream(
    req: AskRequest,
    identity: Identity = Depends(current_identity),
    container: Container = Depends(get_container),
) -> StreamingResponse:
    """流式回答（Server-Sent Events）。

    事件格式：
        data: {"kind":"text","text":"..."}
        data: {"kind":"done","refused":false,"meta":{...}}
        data: [DONE]

    ★ 前端必须按「逐行 data:」解析，不能用一次性 fetch().json()。
    ★ 带 session 时，多轮字段挂在事件 payload 上；`done` 事件后落一轮，
      下一轮追问就能接着上文。无 session 时 payload 一个键都不多。
    """
    settings = current_settings()
    t0 = time.perf_counter()
    session = _resolve_session(req, identity, container)
    rewritten, rewrite = _rewrite_question(req.question, session, container)
    product_ids = _product_ids_for(req, rewrite)
    store = getattr(container, "conversation_store", None)
    sid = req.session_id or ""

    extra: dict = {}
    if session is not None and store is not None:
        try:
            turn_no = store.next_turn(sid)
        except Exception:                     # noqa: BLE001 存储异常不阻断流式
            turn_no = len(session.turns) + 1
        extra = _multi_turn_fields(req, rewritten, turn_no, rewrite)

    uc = _build_use_case(container)

    def event_stream() -> Iterator[str]:
        answer_parts: list[str] = []
        try:
            for ev in uc.execute_stream(
                rewritten,
                vector_collection=settings.coll_training,
                product_ids=product_ids,
                # ★ 同上：权限由角色决定，不信前端
                allowed_permissions=identity.role.retrieval_levels,
                top_k=req.top_k,
            ):
                payload = {
                    "kind": ev.kind,
                    "text": ev.text,
                    "refused": ev.refused,
                    "refusal_reason": ev.refusal_reason,
                    "meta": ev.meta,
                }
                if ev.kind == "text" and (ev.text or "").strip():
                    answer_parts.append(ev.text)
                if ev.kind == "done":
                    # ★ 指标不分是否多轮：所有流式请求都记。
                    _record_metrics(
                        container, t0, answer="".join(answer_parts),
                        refused=bool(ev.refused), meta=ev.meta)
                    if extra:
                        # 只累积非空白正文；拒答也落轮次，让下一轮仍能承接场景。
                        meta = ev.meta or {}
                        _append_turn(store, sid, _make_turn(
                            req, rewritten,
                            answer="".join(answer_parts).strip(),
                            cited=meta.get("cited_chunk_ids") or (),
                            products=meta.get("product_ids") or (),
                            intent=_intent_from_meta(meta),
                            refused=bool(ev.refused),
                        ))
                if extra:
                    payload.update(extra)
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as exc:                # noqa: BLE001
            # 流已经开始，无法再改 HTTP 状态码，只能发一个错误事件
            logger.error("流式问答失败: %s", exc, exc_info=True)
            err = {"kind": "done", "refused": True,
                   "refusal_reason": f"服务异常: {type(exc).__name__}"}
            _record_metrics(container, t0, answer="".join(answer_parts),
                            refused=True)
            if extra:
                err.update(extra)
                _append_turn(store, sid, _make_turn(
                    req, rewritten,
                    answer="".join(answer_parts).strip(),
                    refused=True,
                ))
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # 防 nginx 缓冲导致「不流」
        },
    )
