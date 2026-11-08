"""多跳问答用例：拆子查询 → 逐跳执行 → 合并回答。

────────────────────────────────────────────────────────────────────
两类跳法，语义不同，不能混
────────────────────────────────────────────────────────────────────
并列（`depends_on_previous=False`）
    「补钙的有什么推荐，有什么禁忌」
    两跳各自独立召回，互不影响。合并成一段回答。

依赖（`depends_on_previous=True`）
    「改善体脂都可以吃什么保健品，其中那种最便宜」
    第 2 跳的候选集**只来自**第 1 跳，不允许自己重新全库检索。

    这是用户明确要求的「前一句话查询的结果是后一个查询的前提」。

────────────────────────────────────────────────────────────────────
为什么「候选集」必须显式传递，不能靠上下文顺延
────────────────────────────────────────────────────────────────────
两种做法在**结果上**能分得出来：

    集合传参：第 2 跳的候选数 ≤ 第 1 跳的结果数
    上下文顺延：第 2 跳重新全库检索，可能比出第 1 跳里没有的产品

真实库里有现成的坑：全店最便宜的保健食品是「维生素C咀嚼片 98 元」，
它和体脂毫无关系，但「最便宜」这个条件它满足。
第 2 跳一旦自由检索，答案就是它 —— **错得非常自然，看起来有理有据**。

所以本模块的铁律：
    `depends_on_previous=True` 的步骤，候选集为前一跳的 product_ids；
    候选集为空 → 明确说明「前面没找到，无法比价」，**不许自己去全库找**。

────────────────────────────────────────────────────────────────────
比价为什么不调 LLM
────────────────────────────────────────────────────────────────────
价格是**客观数字**。让模型参与等于给它编造的机会：
    「最便宜的是代餐奶昔，218 元，每天一袋很划算」←「很划算」是编的
所以比价走纯模板：数字从图谱取，句式固定，最后再过一次护栏。
这也让「单步问题只调一次 LLM」这条性能约束自动成立。

────────────────────────────────────────────────────────────────────
三个必须守住的约束
────────────────────────────────────────────────────────────────────
1. **每跳的生成都要过护栏**（约束 #1）。这里复用 `AnswerQuestionUseCase`，
   护栏在它内部，本模块不重复判断，也不能绕过。
2. **普通食品不得被当作有功效的产品推荐**（约束 #2）。若候选里混入普通食品，
   必须在 notes 里标注，不能只靠正文措辞。
3. **单步问题不许退化成多跳**。多一步就多一次召回 + 一次生成，
   简单问题走多跳会让 P95 直接恶化。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from domain.ports import GraphQueryPort
from domain.services.demand_matcher import (
    absent_reason,
    demand_terms,
    guess_efficacy_ids,
)
from domain.services.output_guardrail import OutputGuardrail
from domain.services.query_planner import (
    dropped_fragments,
    HOP_PRICE_RANK,
    SubQuery,
    plan_queries,
)
from domain.value_objects import ProductCategory, ProductRef

logger = logging.getLogger(__name__)

#: 候选集来源标签。
SOURCE_OWN_RECALL = "own_recall"
SOURCE_PREVIOUS_HOP = "previous_hop"

#: 需求词 → 图谱 Need 节点。
#:
#: ★ 这张表是「第一跳」的入口：客户不会说「NEED_LIPID」，
#:   只会说「血脂高」「想控体脂」。
#:   映射不到的就不映射 —— 宁可空手回来，也不要猜一个需求去反查，
#:   猜错会让整条多跳链路从一个错误的前提开始。
#:
#: 注：真实库中「体重/体脂」暂无对应 Need 节点，
#: 所以这里映射到 NEED_WEIGHT 会返回空 —— 这正是正确答案
#: （目录里有「有助于控制体内脂肪」，但本店没有产品声称它）。

#: 比价话术模板。
#:
#: ★ 只陈述价格与类别，不写任何功效。
#:   「最便宜」是事实，写在这里不会构成宣传；一旦加上「效果好」就是违规。
_PRICE_PICK = "在上一问找到的 {n} 款里，价格最低的是：{name}（{pid}），{price} 元。"
_PRICE_PICK_WITH_OTHERS = (
    "\n价格从低到高依次为：{listing}。"
)
_PRICE_NO_CANDIDATES = (
    "上一问没有找到符合条件的产品，所以无法比价。"
    "\n原因见上一段说明。"
)
_PRICE_NO_PRICE = (
    "上一问找到的产品里没有可用的价格信息，无法比价。"
    "建议查看价签或咨询门店主管。"
)
_ORDINARY_FOOD_NOTE = (
    "{name}（{pid}）是**普通食品**，不是保健食品，不得宣称任何保健功能，"
    "价格较低不代表有相应功效。"
)


def _spoken_not_equal(question: str, catalog_name: str) -> bool:
    """客户的说话与目录原文是不是明显不同的说法。

    ★ 为什么要判这个：批准功能只能用目录原文表述。
      客户说「改善精力」而目录写「缓解体力疲劳」，
      模型很容易顺着客户的话写成「批准功能是改善精力」——那是违规声称。
      这里只要判断「两者字面差异大」，就加一句强约束提示。

    判据：目录名的主要实词没有出现在问句里 → 说明客户用的是另一套说法。
    """
    core = catalog_name
    for prefix in ("有助于", "辅助", "补充"):
        core = core.replace(prefix, "")
    # 取目录名里最长的两个字组合看是否出现在问句里
    q = (question or "")
    for i in range(len(core) - 1):
        if core[i:i + 2] in q:
            return False
    return True


def _is_signature_mismatch(exc: TypeError) -> bool:
    """这个 TypeError 是不是「参数不被接受」。

    ★ 必须区分两种情况，否则会吞掉真 bug：
        · `got an unexpected keyword argument 'fact_hints'` → 签名不匹配，可降级
        · 实现内部 `int('abc')` 抛的 TypeError → 真错误，必须抛出去
    """
    msg = str(exc)
    return ("unexpected keyword argument" in msg
            or "takes no keyword arguments" in msg
            or "positional argument" in msg and "was given" in msg)


@dataclass(frozen=True, slots=True)
class PricePick:
    """比价结果。纯客观信息，不含评价。"""

    product_id: str
    name: str
    price: int
    category: str = ""


@dataclass(frozen=True, slots=True)
class HopResult:
    """一跳的执行结果。

    Attributes:
        order: 序号（从 0 开始）。
        question: 这一跳实际用的问句（可能已补全主题）。
        kind: `simple` / `price_rank`。
        answer: 这一跳的正文（已过护栏）。
        refused: 这一跳是否拒答。
        product_ids: 这一跳确认的产品。
        candidate_ids: 这一跳的**候选集**。

            ★ 与 `product_ids` 分开：候选集是「可选项」，
              产品是「确认命中」。依赖跳的候选集来自前一跳的 product_ids。
        candidates_source: `own_recall` / `previous_hop`。
        notes: 这一跳的提示（如普通食品声明）。
    """

    order: int
    question: str
    kind: str = "simple"
    answer: str = ""
    refused: bool = False
    product_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    candidates_source: str = SOURCE_OWN_RECALL
    notes: tuple[str, ...] = ()
    error: str = ""
    #: 溯源：这一跳引用的知识块 id。
    #:
    #: ★ 多跳不合并溯源会丢掉「凭什么这么说」这条能力。
    #:   单跳时它等于 `AnswerQuestionUseCase.cited_chunk_ids`；
    #:   比价跳没有引用文档（价格来自图谱），所以为空。
    cited_chunk_ids: tuple[str, ...] = ()
    #: 这一跳的意图（决定权重）。
    intent: str = ""
    #: 这一跳是否调用了 LLM。
    #:
    #: ★ 这个字段存在的原因是「召回为空不调 LLM」那条红线：
    #:   接口层测试靠 `meta.llm_called` 证明该路径没调模型。
    #:   多跳若不把它带出来，那条证明就断了。
    llm_called: bool = False


@dataclass(frozen=True, slots=True)
class MultiHopAnswer:
    """多跳问答的完整结果。"""

    answer: str
    hops: tuple[HopResult, ...] = ()
    cheapest: PricePick | None = None
    product_ids: tuple[str, ...] = ()
    refused: bool = False
    notes: tuple[str, ...] = ()
    #: 合并后的溯源（所有跳的引用去重保序）。
    cited_chunk_ids: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


class QueryPlanUseCase:
    """多跳问答用例。

    ★ 构造函数接受 Port，不碰具体适配器。
    """

    def __init__(
        self,
        *,
        answer_one: Callable[..., Any],
        graph: GraphQueryPort | None = None,
        guardrail: OutputGuardrail | None = None,
        planner: Callable[[str], tuple[SubQuery, ...]] | None = None,
        max_steps: int = 3,
        semantic_matcher: Any = None,
        llm_planner: Any = None,
        planner_source: str = "",
    ) -> None:
        """
        Args:
            answer_one: 单跳回答函数，签名 `(question, *, product_ids) -> UseCaseResult`。

                ★ 用注入而不是自己 new 一个 `AnswerQuestionUseCase`：
                  护栏、意图分类、召回阈值这些配置都在调用方手上，
                  在这里重建一份会导致两处配置漂移
                  （P5 踩过：评阅与护栏各写一份规则，漂移后互相矛盾）。

            graph: 图谱端口（比价要读价格）。
            guardrail: 护栏。比价话术是模板拼的，仍要过一遍（约束 #1）。
            planner: 规划函数。默认 `plan_queries`，测试可替换。
            max_steps: 最多几跳。
            semantic_matcher: LLM 语义映射兜底（`SemanticEfficacyMatcher`）。
                None = 不启用，只走规则。生产装配要传；
                测试默认不传，保证「不接 LLM 也能全跑通」。

            llm_planner: LLM 多跳规划复核（`LLMQueryPlanner`）。
                None = 不启用，只用规则拆解。

                ★ 它是**复核**不是替代：规则先跑，只有结果可疑时才调它。
                  所以 None 时行为与改造前完全一致 —— 这是默认值的安全保证。

            planner_source: 本次请求的规划来源标记，会写进 meta
                （`rule` / `llm` / `cache`），便于审计「这次为什么拆成这样」。
        """
        self._answer_one = answer_one
        self._graph = graph
        self._guardrail = guardrail or OutputGuardrail()
        self._planner = planner or plan_queries
        self._max_steps = max(1, max_steps)
        self._matcher = semantic_matcher
        #: LLM 多跳规划复核（可选）。None 时纯规则。
        self._llm_planner = llm_planner
        self._planner_source = planner_source
        #: 本次请求实际走了几次语义兜底（用于观测真实覆盖率）。
        self._semantic_calls = 0
        #: 便于测试断言「这次跑了几跳」。
        self.last_hops: tuple[HopResult, ...] = ()
        #: 目录快照缓存（十几项，一次问答内复用）。
        self._catalog_cache: list[tuple[str, str]] | None = None
        self._catalog_names: dict[str, str] = {}
        self._catalog_items: dict[str, Any] = {}

    # ---------------- 主流程 ----------------
    def execute(
        self,
        question: str,
        *,
        allowed_permissions: tuple[str, ...] | None = None,
        history: str = "",
        seed_product_ids: tuple[str, ...] = (),
    ) -> MultiHopAnswer:
        """执行多跳问答。

        Args:
            question: 用户原始问题。
            allowed_permissions: 权限级别，透传给每一跳的召回。
            history: 前几轮对话摘要。

                ★ 只透传给**最后一跳**（也就是真正生成最终答案的那一跳）。
                  每一跳都带历史的话，第 1 跳会被前文干扰
                  （它本来就是在独立查一个子问题），
                  而最后一跳需要它来把指代接上。
                  实测场景里的追问几乎都在最后一句（「其中那种最便宜」）。
            seed_product_ids: 上游已经确定的产品（例如上一轮聊的鱼油）。
                会作为**第 1 跳的已知产品**参与召回。

                ★ 这是「那它有什么禁忌」这种纯指代追问能接上的关键：
                  这句话里没有任何产品名，不给锚点就召回不到任何东西。

        Returns:
            合并后的回答。单步问题与普通问答行为一致。
        """
        t0 = time.perf_counter()
        text = (question or "").strip()
        if not text:
            self.last_hops = ()
            return MultiHopAnswer(
                answer="请输入您的问题。", refused=True,
                meta={"hops": 0},
            )

        steps = self._plan(text)
        hops: list[HopResult] = []
        notes: list[str] = []

        #: 前一跳确认的产品 —— 这是「后一跳的前提」。
        #: ★ 初值来自会话锚点：纯指代追问（「那它有什么禁忌」）里
        #:   没有产品名，只能靠上一轮的产品把召回拉回来。
        previous_products: tuple[str, ...] = tuple(seed_product_ids or ())

        for step in steps:
            try:
                hop = self._run_step(
                    step,
                    previous_products=previous_products,
                    allowed_permissions=allowed_permissions,
                    # ★ 只给最后一跳带历史：前面几跳在独立查子问题，
                    #   带上前文反而会被干扰。
                    history=(history if step is steps[-1] else ""),
                )
            except Exception as exc:                # noqa: BLE001 单跳失败不能全崩
                logger.error("第 %d 跳执行失败: %s: %s",
                             step.order, type(exc).__name__, exc, exc_info=True)
                hop = HopResult(
                    order=step.order, question=step.question, kind=step.kind,
                    refused=True, candidates_source=(
                        SOURCE_PREVIOUS_HOP if step.depends_on_previous
                        else SOURCE_OWN_RECALL),
                    error=f"{type(exc).__name__}: {exc}",
                )
                notes.append(
                    f"第 {step.order + 1} 部分没有答出来（{type(exc).__name__}），"
                    f"以上其余部分仍然有效。")

            hops.append(hop)
            notes.extend(hop.notes)

            # ★ 只有「确认到手的产品」才作为下一跳的前提。
            #   依赖跳自己不再产生新候选，否则会出现
            #   「A 依赖 B，B 又依赖 A 的结果」这种不可控的链。
            if not step.depends_on_previous and hop.product_ids:
                previous_products = hop.product_ids

        cheapest = self._cheapest_of(hops)
        answer = self._compose(hops, notes)
        # ---- ★ 合成后再过一遍护栏（按「最严格类别」）----
        #
        # 为什么单跳时也要：单跳走的是 `AnswerQuestionUseCase`，它内部已过护栏；
        # 但**多跳合成**会把这些内容拼到一起，拼接本身可能产生新问题
        # （比如第 2 跳的比价标题与第 1 跳的功效句连读成一句新的声称）。
        #
        # ★ 类别取「候选中出现的最严格类别」：
        #   只要有一款普通食品，整段就按普通食品校验。
        #   理由：普通食品那部分内容一旦夹带功效声称就是违法，
        #   而读者不会区分「这句说的是哪款产品」。
        #   实测踩坑：候选里只有普通食品（代餐奶昔）时，
        #   若按 None 校验，护栏不做类别判定 → 「能改善体脂」被放行。
        category = self._strictest_category(hops)
        final = self._guardrail.check(answer, category=category)
        if final.blocked:
            logger.warning("多跳合成结果被护栏拦截: %s", final.reason[:100])
            # ★ 不能「换个排版再放出去」。
            #   拼接层发现违规，说明**某个违规句确实在正文里**，
            #   换分隔符治不了它（红灯抓出：safe_mode 换完仍然是同一句）。
            #
            #   正确做法与单跳的「部分保留」一致：剔除违规句，保留合规内容，
            #   并把剔除说明附在 notes 里。这样学员仍能看到有用的部分，
            #   也不会看到违规原文。
            annotation = self._guardrail.annotate(answer, category=category)
            if annotation.has_safe_content:
                answer = annotation.safe_text
                notes.append(
                    f"部分内容因不符合功能声称规范已隐藏"
                    f"（{annotation.violation_count} 处）。")
                if annotation.repaired_warning:
                    logger.info("多跳合成：警示语已补回")
            else:
                answer = (
                    "这个问题的回答涉及不宜直接表述的内容，我不能按上述方式作答。\n"
                    "如果涉及产品的批准功能、适宜人群、禁忌或用法用量，"
                    "我可以如实告诉您；涉及疾病治疗、用药调整或效果承诺，"
                    "请咨询医生或门店主管。")
                notes.append("回答因合规原因未展示。")

        self.last_hops = tuple(hops)
        return MultiHopAnswer(
            answer=answer,
            hops=tuple(hops),
            cheapest=cheapest,
            product_ids=previous_products,
            refused=all(h.refused for h in hops) if hops else True,
            notes=tuple(dict.fromkeys(notes)),
            # 溯源：合并所有跳的引用（去重保序）。
            # 单跳时等于原答案的 cited_chunk_ids，接口契约不变。
            cited_chunk_ids=tuple(dict.fromkeys(
                cid for h in hops for cid in h.cited_chunk_ids)),
            meta={
                "hops": len(hops),
                "step_kinds": [h.kind for h in hops],
                "depends": [h.candidates_source == SOURCE_PREVIOUS_HOP
                            for h in hops],
                "intents": [h.intent for h in hops],
                # ★ 语义兜底观测：本次请求走了几次 LLM 映射。
                #   这是判断「规则漏多少」的唯一依据。
                #   长期看这个数字：一直为 0 说明规则够用（可以关掉兜底省成本）；
                #   持续 >0 说明规则有系统性缺口，应把高频案例补进 `_SYNONYMS`。
                "semantic_fallback_calls": self._semantic_calls,
                # ★ 沿用单跳的语义：任何一跳调了 LLM 就算调了。
                #   全都没调时才是 False —— 接口层测试靠它证明
                #   「召回为空不调 LLM」这条红线在多跳路径上仍然成立。
                "llm_called": any(h.llm_called for h in hops),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        )

    # ---------------- 单跳 ----------------
    def _run_step(
        self,
        step: SubQuery,
        *,
        previous_products: tuple[str, ...],
        allowed_permissions: tuple[str, ...] | None,
        history: str = "",
    ) -> HopResult:
        """执行一跳。

        Args:
            history: 前几轮对话摘要。只有最后一跳传（见 `execute` 的说明）。
        """
        # ---- 依赖前一步：候选集锁定，不重新检索 ----
        if step.depends_on_previous:
            return self._rank_by_price(
                step, candidates=previous_products)

        # ---- 独立召回 ----
        #
        # ★ 先把「已核实的事实」算出来，再交给生成层。
        #   顺序不能反：生成层要拿事实去写答案，不是先写再对账。
        #   实测（2026-10-23）：「补钙的有什么推荐」先走召回 → 段落里没讲
        #   产品清单 → 模型只能追问客户，而图谱明明已查到 P011。
        evidence = self._gather_evidence(step.question)
        hints = evidence["hints"]

        # ★ 召回入口 = 本轮图谱查到的产品 + 会话给的锚点（去重保序）。
        #
        #   为什么要把 previous_products 并进来：纯指代追问
        #   （「那它有什么禁忌」）里没有产品名，图谱查不到、向量也搜不到，
        #   唯一能定位的就是上一轮聊的产品。
        #   实测：不并进来时，模型只能反问「您问的是哪一款」。
        entry_ids = tuple(dict.fromkeys(
            list(evidence["product_ids"]) + list(previous_products)))

        result = self._answer(
            step.question,
            allowed_permissions=allowed_permissions,
            fact_hints=hints,
            # 图谱查到的产品也要作为入口传进去（护栏判类别要用）
            product_ids=entry_ids,
            history=history,
        )
        product_ids = tuple(getattr(result, "product_ids", ()) or ())
        hop_notes: list[str] = []

        # 图谱反查的产品 + 会话锚点都要进候选集（去重保序）。
        # 图谱是权威结构化数据，文本段落可能过期。
        merged = list(product_ids)
        for pid in list(evidence["product_ids"]) + list(previous_products):
            if pid not in merged:
                merged.append(pid)
        product_ids = tuple(merged)

        # 普通食品混入时必须标注（约束 #2）
        hop_notes.extend(self._ordinary_food_notes(product_ids))

        meta = getattr(result, "meta", {}) or {}
        return HopResult(
            order=step.order,
            question=step.question,
            kind=step.kind,
            answer=getattr(result, "answer", "") or "",
            refused=bool(getattr(result, "refused", False)),
            product_ids=product_ids,
            candidate_ids=product_ids,
            candidates_source=SOURCE_OWN_RECALL,
            notes=tuple(hop_notes),
            # 溯源与诊断信息要带出去，否则多跳会丢掉「凭什么这么说」
            cited_chunk_ids=tuple(
                getattr(result, "cited_chunk_ids", ()) or ()),
            intent=str(meta.get("intent", "")),
            llm_called=bool(meta.get("llm_called", False)),
        )

    def _gather_evidence(self, question: str) -> dict:
        """查「这个问题涉及哪些事实」，供生成层直接引用。

        Returns:
            `{"product_ids": [...], "hints": [...]}`。
            `hints` 是**给人读的权威事实句**，会进 prompt 的最高优先分区。

        ★ 这一步是整个多跳的价值所在（2026-10-23 端到端实测结论）：
          不查证据，多跳就只是「问两遍」——
          第一遍照样答不出「有哪些产品」，第二遍照样比不了价
          （候选集是空的）。

          只让比价跳用候选集，是**用了一半的前提**：
          回答正文仍然不知道前提是什么，学员看到的还是「先问客户几个问题」。

        ★ 三种证据，缺一不可：
          ① 目录里有没有这个功能 → 决定「法规上能不能这么说」
          ② 本店哪些产品声称了它 → 决定「有什么可推荐」
          ③ 两者都空 → 必须明确说「目录里没有」而不是含糊
        """
        empty = {"product_ids": [], "hints": []}
        if self._graph is None:
            return empty

        hints: list[str] = []
        product_ids: list[str] = []

        # ---- ⓪ 先查「目录里确定没有」的高频需求 ----
        #
        # ★ 为什么单独一步、且必须在映射之前（2026-10-24 实测）：
        #   问「提升关节舒适度」，规则映射为空，系统答「资料里没有这个信息」。
        #   这句话不算错，但让人以为「知识库没收录」——
        #   真相是**27 项批准功能里根本没有关节这一项**。
        #   对学员来说这两句话的意义完全不同：
        #     「资料里没有」    → 我再翻翻别的资料
        #     「法规没有这一项」 → 这句不能说，说了就违规
        absent = absent_reason(question)
        if absent:
            hints.append(
                f"【法规事实】{absent}。"
                f"因此**不能**用这个说法描述任何产品的功效，"
                f"也**不要**回答「知识库没收录」——知识库没问题，是法规里没有这一项。")

        ids = guess_efficacy_ids(question, self._catalog_snapshot())

        # ---- ★ 规则全失败时才调 LLM（本请求最多 1 次）----
        #
        # 红线：「召回为空不调 LLM」。这里是唯一破例点，且条件很窄：
        #   · 规则一个都没映射出来（ids 为空）
        #   · 没有命中「目录确定没有」表（那种情况已有确定答案，不需要模型）
        #   · 预算未用完
        # 绝大多数问题（含乱码）走不到这里。
        if not ids and not absent and self._matcher is not None:
            terms = demand_terms(question)
            hit = self._semantic_fallback(terms, question)
            if hit:
                ids = [hit]

        if not ids:
            return empty

        for eid in ids:
            # ① 目录项（法规层面）
            catalog_item = self._catalog_item(eid)
            # ② 声称该功能的产品（本店层面）
            try:
                refs = self._graph.find_products_by_efficacy(eid)
            except Exception as exc:                # noqa: BLE001 反查失败不崩
                logger.warning("按功效反查产品失败 %s: %s", eid, exc)
                refs = []

            if catalog_item is not None:
                name = catalog_item.name
                no = catalog_item.catalog_no
                label = f"第 {no} 项" if no else "营养素补充剂类"
                # ★ 客户的说话与目录原文不同时，必须点明「最接近 ≠ 等同」。
                #   实测：客户说「改善精力」，目录最接近「缓解体力疲劳」。
                #   若模型顺势说「本品批准功能是改善精力」——那是违规声称，
                #   批准功能只能用目录原文，改一个字都不行。
                wording = ""
                if _spoken_not_equal(question, name):
                    wording = (
                        f"注意：客户说的是口语说法，与目录原文不同。"
                        f"对外只能表述为「{name}」，"
                        f"**不能**把客户的说法当成批准功能。")
                if refs:
                    listing = "、".join(f"{r.name}（{r.product_id}）" for r in refs)
                    hints.append(
                        f"批准功能目录中「{name}」（{label}）是允许声称的功能；"
                        f"本店在售产品里声称该功能的有：{listing}。" + wording)
                    product_ids.extend(
                        r.product_id for r in refs
                        if r.product_id not in product_ids)
                else:
                    # ★ 这条最关键：目录里有，但本店没有。
                    #   实测若不给这句，模型会编成「目录中没有这个功能」。
                    hints.append(
                        f"批准功能目录中确实有「{name}」（{label}）这一项，"
                        f"但本店在售的保健食品**没有任何一款**声称该功能。"
                        f"所以不能说「本店有满足该需求的产品」，"
                        f"也**不能**说「法规不允许这个功能」——"
                        f"法规是允许的，只是本店没有对应产品。" + wording)
            else:
                # 目录里查不到 → 只能说明「没查到」，不能替法规下结论
                if refs:
                    listing = "、".join(f"{r.name}（{r.product_id}）" for r in refs)
                    hints.append(
                        f"本店声称与该需求相关功能的产品有：{listing}。")
                    product_ids.extend(
                        r.product_id for r in refs
                        if r.product_id not in product_ids)
                else:
                    hints.append(
                        "在批准功能目录与本店产品中都没有查到与该需求对应的一项。"
                        "请如实说明「没有查到」，不要推断法规允许或不允许。")

        return {"product_ids": product_ids, "hints": hints}

    def _semantic_fallback(self, terms: list[str], question: str) -> str:
        """规则映射失败时，用 LLM 兜一次。

        Args:
            terms: `demand_terms` 抽出的需求片段。
            question: 原问句（片段为空时退化用它）。

        Returns:
            命中的 efficacy_id；空字符串 = 没兜到或校验没过。

        ★ 三层校验在这里的落法：
            L1/L2 在 `SemanticEfficacyMatcher` 内部完成；
            **L3（该功能是否真有产品声称）在这里补**，因为要用图谱。
            没有产品的项**也返回**——它仍然是有用的事实
            （「法规允许但没有货」本身就是要告诉学员的话）。
        """
        if self._matcher is None:
            return ""
        catalog = self._catalog_snapshot()
        if not catalog:
            return ""

        demand = terms[0] if terms else (question or "").strip()[:40]
        match = self._matcher.match(demand, catalog)
        self._semantic_calls += 1

        logger.info(
            "语义映射: demand=%r → eid=%r verified=%s reason=%s",
            demand[:30], match.efficacy_id, match.verified, match.reason[:60])

        if not match.usable:
            return ""

        # L3：该功能有没有产品声称（只记录，不阻断）
        try:
            refs = self._graph.find_products_by_efficacy(match.efficacy_id)
        except Exception as exc:                    # noqa: BLE001
            logger.warning("L3 校验查产品失败 %s: %s", match.efficacy_id, exc)
            refs = []
        if not refs:
            logger.info(
                "语义映射命中 %s 但本店无产品声称（只答法规，不推荐）",
                match.efficacy_id)
        return match.efficacy_id

    def _catalog_snapshot(self) -> list[tuple[str, str]]:
        """目录快照 `[(id, name)]`，用于口语匹配。缓存一次，避免每跳重查。"""
        if self._catalog_cache is not None:
            return self._catalog_cache
        out: list[tuple[str, str]] = []
        if self._graph is not None:
            try:
                refs = self._graph.list_efficacy_catalog()
                out = [(r.efficacy_id, r.name) for r in refs]
                self._catalog_names = {r.efficacy_id: r.name for r in refs}
                self._catalog_items = {r.efficacy_id: r for r in refs}
            except Exception as exc:                # noqa: BLE001
                logger.warning("取目录快照失败: %s", exc)
        self._catalog_cache = out
        return out

    def _catalog_item(self, efficacy_id: str):
        """取目录项详情（名称与目录编号）。走快照，不再单独查库。"""
        self._catalog_snapshot()
        return self._catalog_items.get(efficacy_id)

    def _answer(
        self, question: str, *, allowed_permissions,
        fact_hints: list[str] | None = None,
        product_ids: tuple[str, ...] = (),
        history: str = "",
    ) -> Any:
        """调用单跳回答函数。

        ★ 用「逐个试参数」而不是「一个签名到底」：
          不同实现可能不收某个可选参数，但那不该让整个多跳链路失败。
          注意**只对 TypeError 兜底**，其他异常要照常抛 ——
          把逻辑错误当签名不匹配吞掉，会变成最难查的一类 bug。
        """
        kwargs: dict[str, Any] = {"allowed_permissions": allowed_permissions}
        if fact_hints:
            kwargs["fact_hints"] = fact_hints
        if product_ids:
            kwargs["product_ids"] = product_ids
        if history:
            kwargs["history"] = history

        try:
            return self._answer_one(question, **kwargs)
        except TypeError as exc:
            # ★ 只在「错误信息确实指向签名不匹配」时才降级。
            #
            #   实测踩坑（2026-10-23）：原来无条件捕获 TypeError 降级，
            #   结果把**单跳实现内部抛的 TypeError**（比如测试注入的
            #   故意抛错）也吞掉了 —— 表面看是「参数不兼容」，
            #   实际是真 bug，最难查。
            if not _is_signature_mismatch(exc):
                raise
            logger.debug("单跳实现不接受可选参数，降级调用: %s", exc)
            try:
                return self._answer_one(
                    question, allowed_permissions=allowed_permissions)
            except TypeError as exc2:
                if not _is_signature_mismatch(exc2):
                    raise
                return self._answer_one(question)

    # ---------------- 比价 ----------------
    def _rank_by_price(
        self, step: SubQuery, *, candidates: tuple[str, ...]
    ) -> HopResult:
        """在**给定的候选集**里比价。

        ★ 候选集为空 → 直接说明无法比价，绝不自己去全库找。
          这是本模块最关键的语义，见模块头部注释。
        """
        if not candidates:
            return HopResult(
                order=step.order, question=step.question, kind=HOP_PRICE_RANK,
                answer=_PRICE_NO_CANDIDATES,
                refused=False,                 # 有明确解释，不算拒答
                candidate_ids=(), candidates_source=SOURCE_PREVIOUS_HOP,
            )

        priced: list[PricePick] = []
        for pid in candidates:
            facts = self._facts(pid)
            if facts is None or facts.retail_price is None:
                continue
            priced.append(PricePick(
                product_id=facts.product_id, name=facts.name,
                price=facts.retail_price, category=facts.category,
            ))

        if not priced:
            return HopResult(
                order=step.order, question=step.question, kind=HOP_PRICE_RANK,
                answer=_PRICE_NO_PRICE,
                refused=False,
                candidate_ids=tuple(candidates),
                candidates_source=SOURCE_PREVIOUS_HOP,
            )

        priced.sort(key=lambda p: (p.price, p.product_id))
        best = priced[0]
        body = _PRICE_PICK.format(
            n=len(priced), name=best.name, pid=best.product_id,
            price=best.price)

        if len(priced) > 1:
            listing = "；".join(
                f"{p.name} {p.price} 元" for p in priced)
            body += _PRICE_PICK_WITH_OTHERS.format(listing=listing)

        hop_notes: list[str] = []
        if self._is_ordinary_food(best.product_id):
            note = _ORDINARY_FOOD_NOTE.format(
                name=best.name, pid=best.product_id)
            hop_notes.append(note)
            body += "\n\n⚠️ " + note.replace("**", "")

        # ★ 比价跳只准说价格（2026-10-23 设计约束）。
        #
        #   好奇点：本跳**根本不调用 LLM**，为什么还要在这里拦功效词？
        #   因为「比价」这个动作天然诱惑人去补一句「所以它最划算/最能瘦」。
        #   模板里今天没有，明天有人加一句就漏了 —— 而护栏**拦不住**：
        #   实测「能改善体脂」不在 `_ORDINARY_FOOD_FORBIDDEN` 里
        #   （表里是「控制体内脂肪」「减脂」，没有「体脂」），
        #   所以护栏会放行。护栏的完整性不能假定，本层必须自己守住。
        #
        #   判据：价格是客观事实，功效是法律许可的声称，两者不能混。
        efficacy = self._guardrail.check(
            body, category=ProductCategory.ORDINARY_FOOD)
        if efficacy.blocked:
            logger.warning("比价话术含功能声称，已拦截: %s", efficacy.reason[:80])
            body = ("价格信息无法展示（" + efficacy.reason.split("；")[0] + "）。")
            hop_notes.append("价格段落因含不合规表述已隐藏。")

        return HopResult(
            order=step.order, question=step.question, kind=HOP_PRICE_RANK,
            answer=body, refused=False,
            product_ids=(best.product_id,),
            candidate_ids=tuple(candidates),
            candidates_source=SOURCE_PREVIOUS_HOP,
            notes=tuple(hop_notes),
        )

    # ---------------- 事实 / 组合 ----------------
    def _facts(self, product_id: str):
        if self._graph is None:
            return None
        try:
            return self._graph.get_product_facts(product_id)
        except Exception as exc:                    # noqa: BLE001 取事实失败不崩
            logger.warning("取产品事实失败 %s: %s", product_id, exc)
            return None

    def _is_ordinary_food(self, product_id: str) -> bool:
        facts = self._facts(product_id)
        if facts is None:
            return False
        return facts.category_enum is ProductCategory.ORDINARY_FOOD

    def _ordinary_food_notes(self, product_ids: tuple[str, ...]) -> list[str]:
        """普通食品被推荐时必须标注（约束 #2 的落点）。"""
        out: list[str] = []
        for pid in product_ids:
            if self._is_ordinary_food(pid):
                facts = self._facts(pid)
                name = facts.name if facts else pid
                out.append(_ORDINARY_FOOD_NOTE.format(name=name, pid=pid))
        return out

    def _cheapest_of(self, hops: list[HopResult]) -> PricePick | None:
        """从比价跳里取结构化结果（`result.cheapest`）。

        ★ 这里要重新读一次事实而不是从文本里反解：
          文本是给人看的，结构化字段是给接口/前端用的，两者不能互相推导。
        """
        for hop in reversed(hops):
            if hop.kind != HOP_PRICE_RANK or not hop.product_ids:
                continue
            pid = hop.product_ids[0]
            facts = self._facts(pid)
            if facts is None or facts.retail_price is None:
                return None
            return PricePick(
                product_id=facts.product_id, name=facts.name,
                price=facts.retail_price, category=facts.category,
            )
        return None

    def _strictest_category(self, hops: list[HopResult]):
        """取所有跳里「最严格」的产品类别。

        ★ 为什么取最严格而不是「各自校验」：
          回答是一整段给人读的文本，读者不会区分「这句话说的是哪款产品」。
          只要里面有一款普通食品，整段的功效表述就都有违法风险。
          宁可多拦（内容仍然给出，只是换措辞），也不能漏。
        """
        has_ordinary = False
        has_health = False
        for hop in hops:
            for pid in hop.product_ids:
                facts = self._facts(pid)
                if facts is None:
                    continue
                cat = facts.category_enum
                if cat is ProductCategory.ORDINARY_FOOD:
                    has_ordinary = True
                elif cat is ProductCategory.HEALTH_FOOD:
                    has_health = True

        if has_ordinary:
            return ProductCategory.ORDINARY_FOOD
        if has_health:
            return ProductCategory.HEALTH_FOOD
        # 没有任何产品事实 → None（护栏只做禁用词检测，不猜类别）
        return None

    @staticmethod
    def _compose(hops: list[HopResult], notes: list[str],
                 *, safe_mode: bool = False) -> str:
        """把多跳结果合成一段回答。

        ★ 单跳时**原样返回**，不加任何标题前缀。
          原因：单步是最常见路径，加「第 1 部分」这种字样纯属噪声，
          而且会让「单步与普通问答一致」这条测试变得难以判断。

        Args:
            safe_mode: 拼接处检出问题时退化的安全模式 ——
                去掉 markdown 分隔与加粗，改成纯文本分段，
                减少「两段连读成一句新声称」的可能。
        """
        if not hops:
            return ""

        if len(hops) == 1:
            body = hops[0].answer
        else:
            blocks: list[str] = []
            for hop in hops:
                if not hop.answer:
                    continue
                if hop.kind == HOP_PRICE_RANK:
                    blocks.append(f"**关于价格**\n\n{hop.answer}")
                else:
                    blocks.append(hop.answer)
            if safe_mode:
                body = "\n\n".join(blocks)
            else:
                body = "\n\n---\n\n".join(blocks)

        if notes:
            body = body.rstrip() + "\n\n" + "\n".join(
                f"> {n}" for n in dict.fromkeys(notes))
        return body

    # ---------------- 规划 ----------------
    def _plan(self, text: str) -> tuple[SubQuery, ...]:
        """规划：规则先跑 → 结果可疑时 LLM 复核 → 校验 → 采纳或退回。

        ★ 分层顺序不能反：
          1. 规则永远先跑（零成本、完全可复现）
          2. 只有 `plan_looks_suspicious` 认为可疑才调 LLM
          3. LLM 输出必须过 `validate_plan` 六道校验
          4. 任何一步失败 → 退回规则结果，**不抛异常**

          所以「LLM 规划挂了」的最坏后果是「行为与改造前一致」，
          而不是「问答不可用」。
        """
        try:
            steps = self._planner(text)
        except Exception as exc:                    # noqa: BLE001
            logger.warning("查询规划失败，按单步处理: %s: %s",
                           type(exc).__name__, exc)
            steps = ()

        if not steps:
            steps = (SubQuery(question=text, original_fragment=text),)
        steps = tuple(steps[:self._max_steps])

        # ---- LLM 复核（可选） ----
        if self._llm_planner is None:
            return steps

        try:
            dropped = dropped_fragments(text)
            reviewed = self._llm_planner.plan(
                text, steps, dropped_fragments=dropped)
            reviewed = tuple(reviewed[:self._max_steps]) if reviewed else steps
        except Exception as exc:                    # noqa: BLE001
            # 复核出任何问题都不影响主流程：退回规则结果
            logger.warning("LLM 规划复核失败，沿用规则结果: %s: %s",
                           type(exc).__name__, exc)
            return steps

        # 记录来源（进 meta 便于审计）
        try:
            self._planner_source = str(
                (self._llm_planner.last_decision or {}).get("source", "rule"))
        except Exception:                           # noqa: BLE001
            self._planner_source = "rule"

        return reviewed if reviewed else steps


__all__ = [
    "HopResult",
    "MultiHopAnswer",
    "PricePick",
    "QueryPlanUseCase",
    "SOURCE_OWN_RECALL",
    "SOURCE_PREVIOUS_HOP",
]
