"""查询规划：把一句复合问题拆成有序的子查询。

纯领域服务：**零框架、零 IO、零 LLM**。可秒级全测。

────────────────────────────────────────────────────────────────────
解决两个不同的东西
────────────────────────────────────────────────────────────────────
① 并列多意图
    「补钙的有什么推荐，有什么禁忌」
    两个问题**互不依赖**，各自独立召回，最后合并成一段回答。

② 顺序多跳（后一步以前一步结果为前提）
    「改善体脂都可以吃什么保健品，其中那种最便宜」
    第 2 步的候选集**完全来自**第 1 步的结果。

    用户原话：「前一句话查询的结果是后一个查询的前提」。

────────────────────────────────────────────────────────────────────
为什么这两件事必须分开建模
────────────────────────────────────────────────────────────────────
并列的两问拆开后可以**同时**跑；有依赖的必须等前一步出结果。
混为一谈会犯两种错：

  · 把并列当依赖 → 「有什么禁忌」被塞进第 1 步的产品集合，
    变成「在推荐的这几个里找禁忌」，范围被无端缩小
  · 把依赖当并列 → 「那种最便宜」重新全库检索，
    比出一个第 1 步里根本没有的产品

第二种错**在真实数据上会错得很自然**：全店最便宜的保健食品是
「维生素C咀嚼片 98 元」，它和体脂毫无关系，但「最便宜」这个条件它满足。
答案看起来有理有据，实际答非所问。

────────────────────────────────────────────────────────────────────
判断「是不是真的在指代前面」
────────────────────────────────────────────────────────────────────
只认**回指词**，不认「顺序连接」：
    「其中」「里面」「这几个」「上述」「这些」「那种/哪个」指代前面
    「有什么禁忌」只是第二个问题，没有回指

★ 这一条必须双向测（`test_dependency_not_triggered_by_plain_second_question`）：
  只测「该标的时候标了」会做出一个「永远标依赖」的实现，
  那比不做还糟 —— 它会把并列问题错误地收窄范围。

────────────────────────────────────────────────────────────────────
为什么不用 LLM 拆
────────────────────────────────────────────────────────────────────
1. 成本：每次问答多一次调用（~0.3s），且多数问题只有一步
2. 红线：`AnswerQuestionUseCase` 的约束是「召回为空不调 LLM」，
   规划器在链路最前面，一调就绕过了
3. 可复现：同一句话拆法不同，答案会不一样，排障时说不清

拆错比不拆更贵（多一次召回 + prompt 变长 + 合并逻辑出错），
所以规则刻意写得**保守**：拿不准就只保留一步。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: 步骤类型。执行层按它决定「要不要前一步的结果」。
HOP_SIMPLE = "simple"
HOP_PRICE_RANK = "price_rank"

#: 最多拆几步。超过就只取前 N 步。
#:
#: ★ 为什么必须有上限：
#:   拆一步 = 一次三路召回（向量 + 关键词 + 图谱）。
#:   一句塞满逗号的话拆出 10 步，一次问答就变成 30 次检索，
#:   P95 直接崩掉，prompt 也会超出上下文。
MAX_STEPS = 3

#: 并列分隔符。**顿号不算**（「疾病预防、治疗功能」是一个短语）。
_SPLIT = re.compile(r"[，,；;]|(?<=[？?])\s*")

#: 问句结束符（用来识别「上一句问完了，这是新的一句」）。
_QUESTION_END = re.compile(r"[？?]\s*")

#: 回指词：出现即说明这一步在说「前面那批」。
#:
#: ★ 为什么「那种/哪个」也算：中文口语里「其中那种最便宜」的「那种」
#:   指的就是前面那批产品，不是泛指。
_ANAPHORA = re.compile(
    r"其中|这里面|里面|这几个|这些个|这些|上述|前面(?:那|这)?[些批个]|"
    r"那些|那几款|这几款|上述产品|以上(?:产品|几款|这些)|"
    r"这种|那种|哪种|哪个|哪一个|哪款|哪一款|"
    r"最便宜|最贵|价格最低|价格最高|单价最低|哪个划算|性价比"
)

#: 比价意图。
_PRICE = re.compile(
    r"最便宜|最贵|价格最低|价格最高|哪个划算|性价比|多少钱|价格|价格排序|"
    r"贵|便宜|预算"
)

#: 产品/需求的「主题词」——拆开后每句都要带上的那个词。
#:
#: ★ 为什么需要补主题：
#:   「补钙的有什么推荐，有什么禁忌」拆开后第二句只剩「有什么禁忌」，
#:   召回范围会从「补钙类产品」扩到「全库禁忌」，答案就散了。
#:   所以要找出第一句的主题（「补钙的」），补到后面每一句上。
_TOPIC_HEAD = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9]{2,12}?的)(?=有什么|有哪些|都有|都可以|能|可以|适合|推荐)"
)

#: 一个片段是否像「一个完整的问题」。
_LOOKS_LIKE_QUESTION = re.compile(
    r"什么|哪些|哪个|怎么|如何|多少|能不能|可不可|是否|有没有|推荐|贵|便宜|"
    r"吗|呢|吧|？|\?"
)

#: 句首是这些词 → 这一句**没有自己的主语**，需要补主题。
#:
#: ★ 判据放在句首而不是「整句是否含主语」：
#:   「有什么禁忌」自身没有主语，补「鱼油」得「鱼油有什么禁忌」——正确
#:   「蛋白粉有什么功能」自己有主语，补「鱼油」得「鱼油蛋白粉有什么功能」——污染
_NO_OWN_SUBJECT = re.compile(
    r"^(?:有|能|可以|适合|推荐|怎么|如何|哪些|什么|多少|买|吃|要不要)"
)

#: 句首是「人群/状况」→ 也要补主题。
#: 「孕妇能吃吗」补上之后是「补钙的孕妇能吃吗」（问的是钙剂对这个人群的意思），
#: 不补就变成「孕妇能吃吗」——全库范围，答不出是哪个产品。
_TARGET_GROUP_HEAD = re.compile(
    r"^(?:孕妇|备孕|哺乳|产妇|儿童|小孩|宝宝|婴儿|青少年|老人|老年|"
    r"糖尿病|高血压|高血脂|痛风|肾病|肾不[好差]|肝[不好功能]|甲[亢减]|"
    r"心脏病|术后|经期|更年期|过敏|敏感体质)"
)


@dataclass(frozen=True, slots=True)
class SubQuery:
    """一个子查询（一步）。

    Attributes:
        question: 实际送进召回链路的问句（已补全主题）。
        kind: `simple` / `price_rank`。执行层按它选处理方式。
        depends_on_previous: 是否需要前一步的产品集合。

            ★ True 时执行层**必须**把前一步的 product_ids 作为候选集传入，
              不允许这一步自己重新全库检索。
        original_fragment: 原始片段，排障时对照用。
        order: 从 0 开始的序号。
    """

    question: str
    kind: str = HOP_SIMPLE
    depends_on_previous: bool = False
    original_fragment: str = ""
    order: int = 0

    @property
    def needs_candidates_from_previous(self) -> bool:
        return self.depends_on_previous


def plan_queries(question: str, *, max_steps: int = MAX_STEPS) -> tuple[SubQuery, ...]:
    """把问题拆成有序子查询。

    Args:
        question: 用户原始问题。
        max_steps: 拆解上限（含）。默认 3。

    Returns:
        至少 1 步、最多 `max_steps` 步。**永不返回空元组**：
        拆不出来时原样返回一条 `simple`，这样调用方不需要处理空列表。

    ★ 永不抛异常：它在链路最前面，抛出去等于问答不可用。
    """
    text = (question or "").strip()
    if not text:
        return (_simple("", "", 0),)

    try:
        return _plan(text, max_steps=max(1, max_steps))
    except Exception as exc:                        # noqa: BLE001 规划失败要能继续答
        logger.warning("查询规划失败，按单步处理: %s: %s", type(exc).__name__, exc)
        return (_simple(text, text, 0),)


# ---------------- 内部 ----------------
def _plan(text: str, *, max_steps: int) -> tuple[SubQuery, ...]:
    fragments = _split(text)

    # 只有一段 → 单步
    if len(fragments) <= 1:
        return (_simple(text, text, 0),)

    # 过滤掉「不像问题」的碎片（「其中那种最便宜」这种短尾巴要保留）
    parts = [f for f in fragments if _keep(f)]
    if len(parts) <= 1:
        return (_simple(text, text, 0),)

    parts = parts[:max_steps]

    topic = _extract_topic(parts[0])
    steps: list[SubQuery] = []
    for idx, frag in enumerate(parts):
        kind = HOP_PRICE_RANK if _PRICE.search(frag) else HOP_SIMPLE
        # ★ 依赖只可能从第 2 步开始：第 1 步没有「前面」
        depends = idx > 0 and bool(_ANAPHORA.search(frag))

        # 主题只补到「自己没有主语」的句子上。
        #   「有什么禁忌」      → 补（自身无主语）
        #   「孕妇能吃吗」      → 补（句首是人群）
        #   「蛋白粉有什么功能」→ 不补（自带了主语，补了就是污染）
        # 依赖前一步的步骤**一律不补**：候选集由前一步决定。
        if idx > 0 and not depends and _needs_topic(frag):
            q = _attach_topic(frag, topic)
        else:
            q = frag

        steps.append(SubQuery(
            question=q, kind=kind, depends_on_previous=depends,
            original_fragment=frag, order=idx,
        ))

    logger.info(
        "查询规划: %d 步 %s", len(steps),
        [(s.kind, s.depends_on_previous, s.question[:20]) for s in steps],
    )
    return tuple(steps)


def _split(text: str) -> list[str]:
    """按标点与问号切分，去掉空片段。"""
    raw = _SPLIT.split(text)
    out: list[str] = []
    for piece in raw:
        p = (piece or "").strip(" 　，,；;。？?")
        if p:
            out.append(p)
    return out


def dropped_fragments(question: str) -> tuple[str, ...]:
    """列出「因为不像问题而被丢掉、但可能承载了有效子问题」的片段。

    ★ 必须与 `plan_queries` 的实际行为**严格对齐**，否则会大面积误报。

      踩过的坑（2026-10-30）：
        「降血糖这个说法对不对」只有一个片段，`plan_queries` 走「只有一段→
        直接单步」的早退分支，**整句都被当成一步**（没有丢任何东西）。
        但本函数按「`_keep` 判否即为丢弃」来算，把整句报成丢弃 →
        触发条件 T3 几乎无条件成立，误触发率 7/59。

      所以这里必须先复现 `plan_queries` 的**第一个**早退判断：
        · 切出来 ≤1 段 → 整句就是那一步，没有丢弃

      ★ 但**不能**把「过滤后 ≤1 段」也当成无丢弃：
        那恰恰是「本该拆两步、却因为一片段被丢掉而退回单步」的场景，
        是最值得让 LLM 复核的情况。

        实测反例（这条修错过一次）：
          「改善体脂吃什么，这几样给我挑个划算的」
          切出 2 段，「这几样给我挑个划算的」被判非问题 → 只剩 1 段
          → `plan_queries` 退回整句单步（用户问了两件事只答一件）
          这里必须报告丢弃，才能让 LLM 补回第二步。
    """
    text = (question or "").strip()
    if not text:
        return ()
    try:
        fragments = _split(text)
        # 只有一段：整句就是唯一一步，不存在「丢弃」
        if len(fragments) <= 1:
            return ()
        # ≥2 段：所有被判「不像问题」的片段都是潜在漏拆
        return tuple(f for f in fragments if not _keep(f))
    except Exception:                                  # noqa: BLE001
        return ()


def _keep(fragment: str) -> bool:
    """这个片段要么像完整问题，要么是明确在说「前面那批」。

    ★ 两类的区别很重要：
      「其中那种最便宜」不像标准问句（没有「什么/怎么」），但它有回指词，
      是一条合法的第二跳 —— 必须留。
      而「太好了」这种语气词两者都不沾，丢掉。
    """
    if _LOOKS_LIKE_QUESTION.search(fragment):
        return True
    return bool(_ANAPHORA.search(fragment) or _PRICE.search(fragment))


def _extract_topic(first: str) -> str:
    """从第一句里抽主题词，用于补到后续「没有主语」的句子上。

    「补钙的有什么推荐」→「补钙的」
    「改善体脂都可以吃什么」→「改善体脂」
    「鱼油有什么功效」→「鱼油」        ← 没有「的」，靠疑问词前缀截断

    ★ 两种形式都必须能抽（红灯抓出来的）：
      「鱼油有什么功效，有什么禁忌」拆开后第二句变成「有什么禁忌」，
      主题丢了 → 召回范围从「鱼油」扩到全库，答案就散了。

    ★ 抽不出时返回空串，补主题的逻辑会跳过 —— 宁可范围大也不要拼错主语。
      实测反例：「鱼油有什么功能？蛋白粉有什么功能？」
      两句各有各的主语（这是**两个话题**不是一个主题的两个方面），
      硬把「鱼油」补到第二句会得到「鱼油蛋白粉有什么功能」，语义污染。
    """
    m = _TOPIC_HEAD.search(first)
    if m:
        return m.group(1)

    # 无「的」形式：截到第一个疑问词/判断词之前
    m2 = re.match(
        r"^([\u4e00-\u9fffA-Za-z0-9]{2,12}?)(?=有什么|有哪些|都有|都可以|能|可以|适合|推荐|好用)",
        first,
    )
    if m2:
        return m2.group(1)
    return ""


def _needs_topic(fragment: str) -> bool:
    """这个片段需不需要补主语。

    ★ 只看**句首**：中文里主语在句首，句首就是疑问词/动词说明没主语。
      看「整句是否含主语」会把「蛋白粉有什么功能」误判成无主语。
    """
    return bool(_NO_OWN_SUBJECT.match(fragment)
                or _TARGET_GROUP_HEAD.match(fragment))


def _attach_topic(fragment: str, topic: str) -> str:
    """把主题补到后续片段上，已经带了就不重复补。"""
    if not topic:
        return fragment
    base = topic[:-1] if topic.endswith("的") else topic
    if base and base in fragment:
        return fragment
    # 「有什么禁忌」+「补钙的」→「补钙的有什么禁忌」
    # 「有什么禁忌」+「鱼油」→「鱼油有什么禁忌」（无「的」的直接拼）
    return f"{topic}{fragment}"


def _simple(question: str, original: str, order: int) -> SubQuery:
    return SubQuery(
        question=question, kind=HOP_SIMPLE,
        depends_on_previous=False, original_fragment=original, order=order,
    )


__all__ = [
    "HOP_PRICE_RANK",
    "HOP_SIMPLE",
    "MAX_STEPS",
    "SubQuery",
    "dropped_fragments",
    "plan_queries",
]
