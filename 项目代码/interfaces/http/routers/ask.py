"""知识问答路由。

两个入口：
    POST /api/ask          普通返回（一次给完整答案）
    POST /api/ask/stream   SSE 流式（首句 1–2 秒到，体验更好）

★ 两者都走同一个用例，区别只在「整段返回」还是「逐句返回」。
  护栏在用例层已经强制，接口层不需要也不应该重复判断。
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

import logging
from dataclasses import dataclass

from fastapi import APIRouter, Depends

from application.services.llm_query_planner import (
    CachedQueryPlanner,
    LLMQueryPlanner,
)
from application.services.semantic_efficacy import SemanticEfficacyMatcher
from application.use_cases.answer_question import AnswerQuestionUseCase
from application.use_cases.query_plan import QueryPlanUseCase
from config.assembly import Container
from interfaces.http.ask_session_store import (
    get_ask_session_store,
    make_turn,
)
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["问答"])


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
    settings = container.settings
    return AnswerQuestionUseCase(
        graph=container.graph,
        vector=container.vector,
        keyword=container.keyword,
        llm=container.llm,
        # ★ 传入仓储：召回为空时会写未命中日志（FR-047）。
        #   仓储为 None 时功能照常，只是不记日志。
        repository=container.repository,
        # ★ 重排（可选）：容器里为 None 时用例自动走 RRF 原顺序。
        #   这里显式传参，避免以后新增入口又漏了它
        #   （历史上路由漏透传可选参数踩过 4 次）。
        reranker=container.rerank,
        rerank_candidates=settings.rerank_candidates,
        rerank_top_k=settings.rerank_top_k,
    )


def _build_multi_hop(
    container: Container, uc: AnswerQuestionUseCase, settings,
) -> QueryPlanUseCase:
    """装配多跳用例。

    ★ 把**已经装配好的** `uc` 作为单跳实现注入，而不是在这里重新 new：

      · 护栏配置只有一份（`AnswerQuestionUseCase` 内部）
      · 意图分类器只有一份
      · 召回阈值只有一份

      多跳最容易犯的错就是「另起一条生成路径」，
      然后护栏/阈值/意图在其中一条路径上失效（P5 踩过规则漂移）。
    """
    def answer_one(question: str, *, allowed_permissions=None,
                   fact_hints=None, product_ids=(), history=""):
        return uc.execute(
            question,
            vector_collection=settings.coll_training,
            allowed_permissions=(
                allowed_permissions
                if allowed_permissions is not None
                else ("public", "internal")),
            # ★★ 这三个必须透传（2026-10-23 与 10-24 两次实测踩到）：
            #   漏 `fact_hints` → 多跳查到的事实传不到生成层，
            #     接口里又编出「目录中没有改善体脂等表述」（直连用例却是对的）
            #   漏 `product_ids` → 护栏拿不到产品类别，普通食品声称判不出来
            #   漏 `history` → **追问接不上**，模型反问「您问的是哪一款」
            #     （第 3 次踩同一个坑：参数透传不能靠记性）
            fact_hints=fact_hints,
            product_ids=tuple(product_ids or ()),
            history=history or "",
        )

    settings = container.settings

    # ★ LLM 多跳规划复核（可选）。
    #   规则先跑；只有结果可疑（如多步却一个依赖都没判出来）时才让模型复核。
    #   每次请求新建实例 = 每次请求一份预算，不会跨请求累计。
    llm_planner = None
    if settings.planner_use_llm and container.llm is not None:
        planner = LLMQueryPlanner(
            container.llm,
            max_calls=settings.planner_max_calls,
            max_steps=settings.planner_max_steps,
        )
        # 包一层缓存：同一问题 N 秒内复用拆解结果。
        # ★ 只缓存「LLM 真的参与过」的结果 —— 规则直出本来就不贵，
        #   缓存它反而会在问题微变时读到旧拆解。
        if container.cache is not None:
            planner = CachedQueryPlanner(
                planner, container.cache,
                ttl_sec=settings.planner_cache_ttl_sec)
        llm_planner = planner

    return QueryPlanUseCase(
        answer_one=answer_one,
        graph=container.graph,
        # ★ LLM 语义映射兜底：只在规则全失败且预算内才真正调用。
        #   每次请求新建一个实例 = 每次请求一份预算（max_calls=1），
        #   不会跨请求累计，也不会因为前面几次问答把额度用光而失效。
        semantic_matcher=SemanticEfficacyMatcher(container.llm, max_calls=1),
        llm_planner=llm_planner,
        max_steps=settings.planner_max_steps,
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


@dataclass
class _SessionCtx:
    """本次请求的会话上下文（只在本函数内传递）。"""

    session_id: str
    #: 渲染好的历史文本（空 = 没有历史 / 已清空）
    history: str = ""
    #: 本轮开始前已有几轮（用于 meta 观测）
    turns_before: int = 0
    #: 本轮是否已计入（用于构造响应时显示"当前共几轮"）
    turns_now: int = 0
    #: 上一轮涉及的产品 ID。
    #:
    #: ★★ 这一项**必须进召回**，是历史里唯一的例外（2026-10-24 实测）：
    #:    第 1 轮问「鱼油有什么功能」，第 2 轮问「那它有什么禁忌」——
    #:    「它」指鱼油，但这句里**没有任何产品名**。
    #:    若不给召回层这个 id，向量/关键词都搜不到鱼油，
    #:    图谱也拿不到事实，模型只能反问"您问的是哪一款"。
    #:    实测就是这么失败的。
    #:
    #:    为什么只带产品、不带其它历史：产品是**实体锚点**，
    #:    带了它召回才准；带整段历史会把召回带偏（问禁忌却召回一堆功能描述）。
    last_product_ids: tuple[str, ...] = ()

    @property
    def has_history(self) -> bool:
        return bool(self.history.strip())


def _resolve_session(req: AskRequest, identity: Identity) -> _SessionCtx:
    """取出/新建会话，并按需清空历史。

    ★ 三条设计取舍（都为了"不要让会话机制本身成为新的故障源"）：

      ① 会话读写**失败不影响问答**。存储是内存的，任何异常都只记日志，
         然后当作"没有历史"继续。学员要的是答案，不是会话。
      ② `new_conversation=true` 只清历史、不换 id。
         前端不用改本地存的值，也避免"点了新对话但旧请求还在飞"造成串话。
      ③ owner 用**工号**。知道 session_id 就能读别人对话（可能含过敏史、
         用药情况），这是隐私，必须校验（演练那边踩过同类）。
    """
    store = get_ask_session_store()
    owner = identity.employee_no
    try:
        sess = store.get_or_create(req.session_id, owner)
        if req.new_conversation:
            store.reset(sess.session_id, owner)
            logger.info("会话 %s 历史已按请求清空（新对话）", sess.session_id)
            return _SessionCtx(session_id=sess.session_id)
        return _SessionCtx(
            session_id=sess.session_id,
            history=sess.render_history(),
            turns_before=len(sess.turns),
            turns_now=len(sess.turns),
            last_product_ids=sess.last_product_ids(),
        )
    except Exception as exc:                        # noqa: BLE001 会话不该拖垮问答
        logger.warning("会话处理失败，按无历史继续: %s", exc)
        return _SessionCtx(session_id="")


def _effective_product_ids(
    req: AskRequest, sess: "_SessionCtx"
) -> tuple[str, ...]:
    """本轮召回该用哪些产品 ID。

    规则（按优先级）：
        ① 本轮显式选了产品（前端下拉框）→ 用它，**忽略历史**
           （用户明确切换了话题对象，历史锚点必须让位）
        ② 本轮没选、但有历史锚点 → 用历史锚点
        ③ 都没有 → 空（正常全库检索）

    ★ ①必须优先于②：用户手动选 P003 时，如果还沿用上一轮的 P001，
      会答成一个他没问的产品 —— 这比"接不上"更糟。
    """
    picked = tuple(req.product_ids or ())
    if picked:
        return picked
    return tuple(sess.last_product_ids or ())


def _record_turn(
    req: AskRequest,
    identity: Identity,
    ctx: _SessionCtx,
    answer: str,
    *,
    product_ids: tuple[str, ...] = (),
    refused: bool = False,
    intent: str = "",
) -> None:
    """把本轮问答记进会话。

    ★ `answer` 必须是**合规版正文**。绝不能传 `annotation.segments` ——
      那里含违规原文，回灌进 prompt 等于教模型说错话。
    """
    if not ctx.session_id:
        return
    store = get_ask_session_store()
    try:
        turn = make_turn(
            req.question, answer,
            product_ids=product_ids, intent=intent, refused=refused)
        ok = store.append(ctx.session_id, identity.employee_no, turn)
        if ok:
            ctx.turns_now = min(
                ctx.turns_before + 1, store.max_turns)
    except Exception as exc:                        # noqa: BLE001
        logger.warning("记录会话轮次失败（不影响回答）: %s", exc)


@router.post("/ask", response_model=AskResponse, summary="知识问答",
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
    """
    settings = current_settings()
    uc = _build_use_case(container)

    # ---- 会话（多轮追问）----
    #
    # ★ 会话读写在**接口层**，不下沉到应用层。
    #   理由：会话是"存哪里、留几轮、怎么过期"的存储问题，
    #   应用层只需要一个渲染好的 history 字符串。
    #   把存储下沉会让应用层依赖 HTTP 层的 session 概念。
    sess = _resolve_session(req, identity)

    # ---- 多跳（拆子查询 / 前一步结果作后一步前提）----
    if req.multi_hop:
        hop_uc = _build_multi_hop(container, uc, settings)
        hop = hop_uc.execute(
            req.question,
            allowed_permissions=identity.role.retrieval_levels,
            history=sess.history,
            seed_product_ids=_effective_product_ids(req, sess),
        )
        _record_turn(req, identity, sess, hop.answer,
                     product_ids=hop.product_ids,
                     refused=hop.refused,
                     intent=str(hop.meta.get("intents", [""])[0]
                                if hop.meta.get("intents") else ""))
        return AskResponse(
            answer=hop.answer,
            refused=hop.refused,
            session_id=sess.session_id,
            history_turns=sess.turns_now,
            cited_chunk_ids=list(hop.cited_chunk_ids),
            product_ids=list(hop.product_ids),
            meta={
                **hop.meta,
                "multi_hop": True,
                "history_turns_used": sess.turns_before,
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
        )

    result = uc.execute(
        req.question,
        vector_collection=settings.coll_training,
        # ★ 本轮显式指定的产品优先；没指定就沿用上一轮的产品锚点。
        #   这就是「那它有什么禁忌」里的"它"能接上的关键。
        product_ids=_effective_product_ids(req, sess),
        # ★★ 权限级别由**角色**决定，不接受前端声明（2026-10-23 修）。
        #    原实现用 req.allowed_permissions，学员传
        #    ["public","internal","restricted","confidential"]
        #    就能越权检索受限内容（FR-050 明确要求检索阶段过滤）。
        allowed_permissions=identity.role.retrieval_levels,
        top_k=req.top_k,
        history=sess.history,
    )

    _record_turn(req, identity, sess, result.answer,
                 product_ids=tuple(result.product_ids),
                 refused=result.refused,
                 intent=str(result.meta.get("intent", "")))

    return AskResponse(
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        session_id=sess.session_id,
        history_turns=sess.turns_now,
        cited_chunk_ids=list(result.cited_chunk_ids),
        product_ids=list(result.product_ids),
        meta={**result.meta, "history_turns_used": sess.turns_before},
        annotation=_annotation_of(result),
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
    """
    settings = current_settings()
    uc = _build_use_case(container)

    # ★ 流式也要支持多轮。会话在生成器**外面**先解析好 ——
    #   生成器是延迟执行的，在里面解析会让「session_id 什么时候生成的」
    #   变得不可预测（万一客户端断开，可能根本没执行到）。
    sess = _resolve_session(req, identity)

    def event_stream() -> Iterator[str]:
        #: 累积正文。流式过程中我们只有增量，最后要拼起来才能记历史。
        collected: list[str] = []
        final: dict[str, object] = {}
        try:
            for ev in uc.execute_stream(
                req.question,
                vector_collection=settings.coll_training,
                product_ids=tuple(req.product_ids),
                # ★ 同上：权限由角色决定，不信前端
                allowed_permissions=identity.role.retrieval_levels,
                top_k=req.top_k,
                history=sess.history,
            ):
                if ev.kind == "text" and ev.text:
                    collected.append(ev.text)
                if ev.kind == "done":
                    final = {
                        "refused": ev.refused,
                        "meta": ev.meta,
                    }
                payload = {
                    "kind": ev.kind,
                    "text": ev.text,
                    "refused": ev.refused,
                    "refusal_reason": ev.refusal_reason,
                    "meta": ev.meta,
                    # 前端拿这个值存起来，下次请求带回才能追问
                    "session_id": sess.session_id,
                    "history_turns": sess.turns_now,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # ★ 流结束后才记历史：中途断开就不记，避免"答了一半进上下文"
            #
            # ★ 产品 ID 从 done 事件的 meta 里取（`StreamEvent` 没有独立字段）。
            #   不取的话这轮历史就没有产品锚点，下一句「那它有什么禁忌」
            #   里的"它"接不上 —— 流式追问会失效，而非流式却是好的。
            meta = final.get("meta") if isinstance(final.get("meta"), dict) else {}
            _record_turn(
                req, identity, sess, "".join(collected),
                product_ids=tuple(meta.get("product_ids") or ()),
                refused=bool(final.get("refused", False)),
                intent=str(meta.get("intent", "")),
            )
        except Exception as exc:                # noqa: BLE001
            # 流已经开始，无法再改 HTTP 状态码，只能发一个错误事件
            logger.error("流式问答失败: %s", exc, exc_info=True)
            err = {"kind": "done", "refused": True,
                   "refusal_reason": f"服务异常: {type(exc).__name__}",
                   "session_id": sess.session_id}
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
