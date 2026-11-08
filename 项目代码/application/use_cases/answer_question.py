"""知识问答用例（P4 核心）。

串起三路召回 → 分区组装 → LLM 生成 → 输出护栏。

────────────────────────────────────────────────────────────────────
三条必须守住的约束（本用例是它们的落点）
────────────────────────────────────────────────────────────────────
1. **召回为空必须拒答，且不调 LLM**
   宁可说「暂无相关资料」，也不能让模型凭参数知识编。
   向量检索永远返回 Top-N，所以「空」要靠分数阈值判断（在 RetrieveUseCase 里）。

2. **护栏是必经节点**
   生成之后必须过 `OutputGuardrail`，不能因为「看起来挺规范」就跳过。
   LLM 每次输出都可能不同，规则必须每次都跑。

3. **普通食品不得有功能声称**
   由护栏的 `check_claim_allowed` 负责，但用例必须把产品类别传给它。
   类别判断错 → 护栏失效，所以类别取号必须走图谱（权威源）。

────────────────────────────────────────────────────────────────────
分区组装为什么不能省
────────────────────────────────────────────────────────────────────
混在一起喂给 LLM 会导致它把「员工经验」当成「公司规范」。
所以 prompt 里明确分区，并标注来源权威性。

国内合规场景还有一层考虑：LLM 看到大段培训文档，容易原文复述，
而培训文档里有「❌ 不能说：降血糖」这种反例教学。
必须在 prompt 里明确「不要把反例当成可用表述」。
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator

from application.types import StreamEvent, UseCaseResult
from application.use_cases.retrieve import RetrievalResult, RetrieveUseCase
from domain.ports import GraphQueryPort, LanguageModelPort
from domain.services.intent_classifier import (
    OPEN_INTENT,
    IntentClassifier,
    IntentResult,
)
from domain.services.output_guardrail import OutputGuardrail
from domain.services.rrf_fusion import INTENT_WEIGHTS
from domain.services.stream_guard import StreamGuard
from domain.value_objects import (
    ChunkMeta,
    PermissionLevel,
    ProductCategory,
    ProductFacts,
)

logger = logging.getLogger(__name__)

#: 拒答话术。
#: ★ 不能只说「无法回答」——要给下一步建议，否则客户体验极差。
REFUSAL_TEMPLATE = (
    "抱歉，知识库里暂时没有找到与这个问题直接相关的资料，"
    "我不能凭空给您答复。建议您：\n"
    "1. 换个说法再问一次（比如直接说产品名或具体场景）；\n"
    "2. 咨询您的门店主管，他能调阅内部资料；\n"
    "3. 如果涉及用药或身体状况，请先咨询医生。"
)

#: 护栏拦截后的说明。
#:
#: ★ 设计要点：不是把整个回答丢掉，而是**告诉用户被拦了什么**。
#:   实测问题：一个回答里 99% 是专业内容，只有一句用了「唯一」，
#:   整段被替换成拒答模板 —— 用户体验极差，也浪费了召回结果。
#:
#: ★ 关键：说明里**不复述禁用词原文**。
#:   拒答话术本身也不该出现「能降血糖」这样的字眼——
#:   否则一边说「我不能这么说」，一边又把它说了一遍。
#:   `reason` 由护栏提供，已是「类别 + 为什么」的描述性文字；
#:   这里再截断掉可能含原文的括号部分。
BLOCKED_TEMPLATE = (
    "关于这个问题，我不能按您期待的方式作答。\n\n"
    "原因：{reason}\n\n"
    "需要说明的是，如果涉及产品的批准功能、适宜人群、禁忌或用法用量，"
    "我可以如实告诉您；涉及疾病诊断、用药调整或效果承诺，请咨询医生或门店主管。"
)


def _sanitize_reason(reason: str) -> str:
    """把拦截原因里的「命中原文」部分去掉。

    护栏返回的 reason 形如：
        「命中禁用表述：[疾病治疗] 医疗用语…（命中「降血糖」）」
    括号里的原文要删掉——拒答话术自己不该复述禁用词。

    Args:
        reason: 护栏给出的原始原因。

    Returns:
        去掉原文引述后的说明。
    """
    import re

    # 去掉 （命中「xxx」） 与 （命中「xxx」；另有 N 处）
    cleaned = re.sub(r"（命中[^）]*）", "", reason)
    # 去掉 命中「xxx」 这种无括号形式
    cleaned = re.sub(r"命中「[^」]*」", "命中违规表述", cleaned)
    return cleaned.strip(" ；，")


#: 参考资料里需要剔除的「错误示范」标记。
_BAD_EXAMPLE_MARK = re.compile(
    r"(?:❌|✗|×|错误示范|错误说法|错误表述|反例|不建议这样|不要这样|别这么说)"
)

#: 「教学示例块」的块标题标记。
#:
#: ★ 为什么需要（2026-11-06 端到端实测）：
#:   问「在吃阿司匹林，能吃鱼油吗」，模型把 D012《客户档案记录规范》的
#:   「**填写示例**」代码块**整段照抄**当答案：
#:       后续安排：- 建议客户先咨询医生确认能否食用鱼油
#:   那段是「怎么填档案」的教学样例，不是产品事实。检索把它当相关资料召回，
#:   模型分不清「示例」和「事实」——3B 的能力上限，改 prompt 更可靠。
_EXAMPLE_BLOCK_TITLE = re.compile(
    r"(?:填写示例|填写范例|示例[：:]|范例[：:]|模板[：:]|"
    r"标准话术模板|沟通示例|改写示例|示范[：:])"
)

#: Markdown 代码围栏。
#:
#: ★ 必须容忍 2 个反引号（实测：源文档是 ```，但入库分块后变成 ``）。
#:   2026-11-06 端到端排查发现 prompt 里 D012 的围栏只剩 2 个反引号，
#:   只匹配 3 个的规则会漏掉——示例块剔不掉，模型就把它当答案抄。
#:   这里按「2 个及以上」匹配，对两种数据都成立。
_FENCE = re.compile(r"^\s*`{2,}")


def _strip_example_blocks(text: str) -> str:
    """剔除「块标题 + 紧跟代码围栏」组成的教学示例块。

    ★ 判据必须同时满足（避免误伤）：
      1. 某一行是**块标题**（命中 `_EXAMPLE_BLOCK_TITLE`）
      2. 从该行往后（跳过空行）遇到的第一个非空行是**代码围栏 ``` **

      只有 1 没有 2 → 不剔。实测依据：
        · `**填写示例**：` + ``` 围栏        → 要剔（D012 就栽在这）
        · `## 七、评分示例` + `> 引用块`     → 不剔（是评分标准，有用）
        · `### 5.2 八个改写示例` + 表格      → 不剔（改写对照表，有用）
        · 正文「可以参考示例，但别照抄」      → 不剔（只是提了一句）

      这个「标题 + 围栏」双条件，是把「教学样例」与「知识内容」分开的最小判据。

    Returns:
        剔除示例块后的文本。
    """
    lines = (text or "").split("\n")
    kept: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if _EXAMPLE_BLOCK_TITLE.search(line):
            # 从标题下一行开始找第一个非空行，看是不是围栏
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n and _FENCE.match(lines[j]):
                # 命中「标题 + 围栏」→ 整块丢弃：标题行 + 围栏内容 + 收尾围栏
                k = j + 1
                while k < n and not _FENCE.match(lines[k]):
                    k += 1
                # 跳过收尾围栏（若存在）
                i = k + 1 if k < n else n
                continue
        kept.append(line)
        i += 1

    return "\n".join(kept)


def _strip_bad_examples(text: str) -> str:
    """剔除参考资料里的错误示范。

    ★ 为什么必须剔除（端到端实测发现）：
      `06_正式内容/D006` 的「客户最常问的五个问题」里，Q1 同时给了：
          错误示范：「能，很多人吃了血脂都降了」
          正确示范：「批准的功能是『有助于维持血脂…』」
      模型看到这段后被带偏，实测输出了「降血脂」，被护栏拦下。

    ★ 剔除粒度：按行。以 ❌ 等标记开头的整行去掉。
      按段剔除太粗（会丢掉整节），按句剔除容易破坏语义。

    ★ 先剔「示例块」再剔「错误示范行」：
      示例块里通常混着 ✅/❌ 行，如果先按行剔，会留下一堆半截图（标题还在、
      内容没了）。先整体拿掉示例块，剩下的行再按标记清洗。

    Returns:
        剔除后的文本。可能为空（整段都是反例）。
    """
    # 1) 整块剔除教学示例（标题 + 代码围栏）
    cleaned_blocks = _strip_example_blocks(text)

    # 2) 按行剔除错误示范
    kept: list[str] = []
    for line in cleaned_blocks.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _BAD_EXAMPLE_MARK.search(stripped):
            continue          # 整行是反例，丢弃
        kept.append(line)

    out = "\n".join(kept)
    # 压缩多余空行
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


class AnswerQuestionUseCase:
    """知识问答用例。

    ★ 构造函数只接受 Port，不接受具体适配器——这样能用假实现测全部分支。
    """

    def __init__(
        self,
        *,
        graph: GraphQueryPort | None = None,
        vector=None,
        keyword=None,
        llm: LanguageModelPort | None = None,
        guardrail: OutputGuardrail | None = None,
        min_vector_score: float = 0.50,
        min_keyword_score: float = 14.0,
        guardrail_retries: int = 1,
        repository=None,
        intent_classifier: IntentClassifier | None = None,
        reranker=None,
        rerank_candidates: int = 20,
        rerank_top_k: int = 5,
    ) -> None:
        """
        Args:
            guardrail_retries: 护栏拦截后的重试次数。

            ★ 为什么需要重试（端到端实测发现）：
              即使 temperature=0，同一输入两次调用也可能得到不同措辞。
              实测「鱼油有什么功能」：
                第 1 次 → 自己加了引号「"降血脂"」，合规通过
                第 2 次 → 说「批准的唯一功能表述」，被绝对化规则拦下
              这不是规则错，也不是模型的错，是 LLM 输出的固有波动。

              处理方式：拦了就重试一次（prompt 里已明确禁止绝对化用语），
              比放宽规则安全得多——放宽会让真正的违规漏过。
              重试用尽仍被拦，才转拒答。

            intent_classifier: 意图分类器。为 None 时用默认规则分类器。

            ★ 为什么要内置分类器（2026-10-23）：
              P3 建好的 13 套意图权重表一直没被用上——接口层不传 `intent`，
              于是所有问题都走「开放问答」(0.1, 0.8, 0.1)。
              实测「在吃阿司匹林能吃鱼油吗」本该是禁忌咨询
              (0.6, 0.2, 0.2)，让图谱主导，实际却是向量主导 —— 权重结构反了。

              ★ 分类器**放在用例里**而不是接口层：这样流式、非流式、
                以及将来任何调用方（CLI / 批处理）都自动获得意图，
                不会再出现「某个入口忘了传」的情况。

            reranker: 候选重排端口（可选）。为 None 时按 RRF 原顺序取前
                `rerank_top_k` 条 —— 重排是**增强**，不是必需品，
                所以这里必须容忍 None，不能让它变成启动依赖。

            rerank_candidates: 送入重排的候选条数。
                取 20 而不是全部：重排是 cross-encoder，延迟随条数线性增长。

            rerank_top_k: 重排后进 prompt 的参考资料条数（原 `docs[:5]`）。
        """
        self._retriever = RetrieveUseCase(
            graph=graph, vector=vector, keyword=keyword,
            min_vector_score=min_vector_score,
            min_keyword_score=min_keyword_score,
        )
        self._graph = graph
        self._llm = llm
        self._guardrail = guardrail or OutputGuardrail()
        self._guardrail_retries = max(guardrail_retries, 0)
        #: 仓储（可选）。有值时，召回为空会写未命中日志（FR-047）
        self._repo = repository
        #: 意图分类器（默认纯规则，不调 LLM —— 见 `intent_classifier` 模块注释）
        self._intent_classifier = intent_classifier or IntentClassifier()
        #: 重排（可选）。None 时走 RRF 原顺序。
        self._reranker = reranker
        self._rerank_candidates = max(rerank_candidates, rerank_top_k)
        self._rerank_top_k = max(rerank_top_k, 1)

    # ---------------- 主流程 ----------------
    def execute(
        self,
        question: str,
        *,
        vector_collection: str = "",
        product_ids: tuple[str, ...] = (),
        allowed_permissions: tuple[str, ...] = (
            PermissionLevel.PUBLIC, PermissionLevel.INTERNAL,
        ),
        intent: str = "",
        top_k: int = 20,
        fact_hints: list[str] | None = None,
        history: str = "",
    ) -> UseCaseResult:
        """回答问题。

        Args:
            question: 用户问题。
            vector_collection: 向量检索目标 collection。
            product_ids: 已知的产品 ID（查询理解阶段抽出）。
            allowed_permissions: 权限级别（学员 / 主管）。
            intent: 意图标签。空则按「开放问答」处理。
            top_k: 每路召回条数。
            fact_hints: 上游（如多跳）已查实的**结构化事实**，
                以自然语言片段给出，直接进 prompt 的权威事实分区。

                ★ 为什么需要它（2026-10-23 实测暴露）：
                  多跳第一跳会先用图谱查明「哪些产品满足这个需求」，
                  但如果不把结果告诉生成层，模型只能靠文档段落作答：
                    · 「补钙的有什么推荐」被答成「先问客户几个问题」，
                      而图谱明明已查到一个产品
                    · 「改善体脂」：图谱查明目录第 9 项存在但无产品声称，
                      模型不知道这个事实，就编成「目录里没有这个功能」
                      —— **把法规说错了**，在合规培训里比不回答更糟

                所以：上一步查实的结论必须作为权威事实喂进去，
                不能只用于比价而不进 prompt。
            history: 前几轮的对话摘要（已渲染成文本，由会话存储给出）。

                ★ 为什么只进 prompt、**不进召回**：
                  追问的检索目标仍然是**当前这句话**。
                  把历史也拿去召回会把结果带偏
                  （问「第 1 点呢」，历史里有鱼油，召回就全跑鱼油去了）。
                  历史的作用只有两个：让模型知道指代、让答案接得上。

                ★ 为什么是字符串而不是对象：
                  应用层不该知道「会话怎么存、留几轮、怎么编号」。

        Returns:
            回答结果。`refused=True` 时 `answer` 是拒答话术。
        """
        t0 = time.perf_counter()
        text = (question or "").strip()
        if not text:
            return UseCaseResult(
                answer=REFUSAL_TEMPLATE,
                refused=True,
                refusal_reason="问题为空",
            )

        # ---- 0. 意图分类（★ 决定三路权重）----
        intent_result = self._resolve_intent(text, intent)

        # ---- 1. 三路召回 ----
        retrieval = self._retriever.execute(
            text,
            intent=intent_result.label,
            product_ids=product_ids,
            vector_collection=vector_collection,
            allowed_permissions=allowed_permissions,
            top_k=top_k,
        )

        # ---- 2. 召回为空 → 拒答（★ 不调 LLM）----
        #
        # ★ 例外：上游已查实结构化事实（`fact_hints`）时不拒答。
        #   理由：拒答的条件是「没有可依据的材料」。图谱查出来的
        #   「哪些产品满足这个需求」「目录里有没有这个功能」本身就是
        #   权威材料 —— 此时拒答等于把已经查到的事实丢掉，
        #   学员看到的是「不知道」，而系统其实知道。
        #   实测：「补钙的有什么推荐」召回段落够但没讲产品清单，
        #   图谱查到了 P011，若不把事实喂进去就只能让模型追问客户。
        hints = list(fact_hints or ())
        if (retrieval.failed or not retrieval.fused) and not hints:
            logger.info(
                "召回为空，直接拒答（不调 LLM）: query=%r, degraded=%s",
                text[:30], list(retrieval.degraded) or "无",
            )
            # ★ 记录未命中日志（FR-047）。
            #
            #   这是「补充知识库的唯一可靠依据」，必须留。
            #   写入失败不能影响拒答本身 —— 日志是运营素材，
            #   不该因为库写不进去就让学员看不到拒答话术。
            self._log_miss(text, retrieval)
            return UseCaseResult(
                answer=REFUSAL_TEMPLATE,
                refused=True,
                refusal_reason=self._refusal_reason(retrieval),
                meta=self._meta(retrieval, t0, llm_called=False,
                                intent_result=intent_result),
            )

        # ---- 3. 取产品事实（护栏要用类别判定）----
        facts = self._load_facts(product_ids, retrieval)

        # ---- 4. 分区组装 ----
        prompt = self._build_prompt(
            text, retrieval, facts, fact_hints=hints, history=history)

        # ---- 5~6. 生成 + 护栏（带重试）----
        category = self._category_of(facts, retrieval)
        gen = None
        verdict = None
        attempts = 0
        #: 最后一次生成的原文。
        #: ★ 重试用尽时要用它做「部分保留」分析——不能只看 verdict，
        #:   还得知道到底是哪几句违规。
        last_text = ""

        for attempt in range(self._guardrail_retries + 1):
            attempts = attempt + 1

            # ---- 生成 ----
            try:
                gen = self._llm.generate(
                    prompt,
                    system_prompt=self._system_prompt(),
                    temperature=0.0,
                )
            except Exception as exc:              # noqa: BLE001 生成失败要转拒答
                logger.error("LLM 生成失败: %s", exc, exc_info=True)
                return UseCaseResult(
                    answer=REFUSAL_TEMPLATE,
                    refused=True,
                    refusal_reason=f"生成服务不可用: {type(exc).__name__}",
                    meta=self._meta(retrieval, t0, llm_called=True,
                                    intent_result=intent_result),
                )

            # ---- 护栏（★ 必经节点，每次生成都要过）----
            last_text = gen.text
            verdict = self._guardrail.check(gen.text, category=category)
            if not verdict.blocked:
                break

            logger.warning(
                "护栏拦截（第 %d/%d 次）: rules=%s, reason=%s",
                attempts, self._guardrail_retries + 1,
                verdict.rule_ids, verdict.reason[:80],
            )

        # 重试用尽仍被拦 → 先尝试「剔除违规句、保留合规内容」
        if verdict is not None and verdict.blocked:
            annotation = self._guardrail.annotate(
                last_text, category=category)

            if annotation.has_safe_content:
                # ★ 部分保留：给出合规内容 + 标注明细供前端标红
                logger.info(
                    "护栏部分拦截：保留 %d 段，剔除 %d 段",
                    len(annotation.segments) - annotation.violation_count,
                    annotation.violation_count,
                )
                return UseCaseResult(
                    answer=annotation.safe_text,
                    refused=False,          # 有可用内容，不算拒答
                    refusal_reason=(
                        f"部分内容被拦截（{annotation.violation_count} 处）："
                        f"{verdict.reason}"
                    ),
                    cited_chunk_ids=tuple(h.chunk_id for h in retrieval.fused),
                    product_ids=tuple(f.product_id for f in facts),
                    meta=self._meta(
                        retrieval, t0, llm_called=True,
                        intent_result=intent_result,
                        blocked_rules=verdict.rule_ids,
                        llm_attempts=attempts,
                        partial_block=True,
                        removed_count=annotation.violation_count,
                        repaired_warning=annotation.repaired_warning,
                    ),
                    annotation=annotation,
                )

            # 无法产出安全内容 → 整段拒答
            return UseCaseResult(
                answer=BLOCKED_TEMPLATE.format(
                    reason=_sanitize_reason(verdict.reason)),
                refused=True,
                refusal_reason=f"输出护栏拦截: {verdict.reason}",
                cited_chunk_ids=tuple(h.chunk_id for h in retrieval.fused),
                product_ids=tuple(f.product_id for f in facts),
                meta=self._meta(retrieval, t0, llm_called=True,
                                intent_result=intent_result,
                                blocked_rules=verdict.rule_ids,
                                llm_attempts=attempts),
                annotation=annotation,      # 全违规时也要保留标注，供前端展示
            )

        # ---- 7. 正常返回 ----
        return UseCaseResult(
            answer=gen.text,
            refused=False,
            cited_chunk_ids=tuple(h.chunk_id for h in retrieval.fused),
            product_ids=tuple(f.product_id for f in facts),
            meta=self._meta(retrieval, t0, llm_called=True,
                            intent_result=intent_result,
                            finish_reason=gen.finish_reason,
                            llm_attempts=attempts),
        )

    # ---------------- 内部 ----------------
    # =================================================================
    # 流式输出
    # =================================================================
    def execute_stream(
        self,
        question: str,
        *,
        vector_collection: str = "",
        product_ids: tuple[str, ...] = (),
        allowed_permissions: tuple[str, ...] = (
            PermissionLevel.PUBLIC, PermissionLevel.INTERNAL,
        ),
        intent: str = "",
        top_k: int = 20,
        fact_hints: list[str] | None = None,
        history: str = "",
    ) -> Iterator[StreamEvent]:
        """流式回答问题。

        ★ 与非流式共用召回与组装逻辑，只有「生成 + 护栏」这一段不同：
          非流式：生成全文 → 护栏 → 整段返回
          流式：  逐句生成 → 逐句过护栏 → 逐句产出

        ★ 关键约束：**已产出给用户的每一句都过了护栏**。
          这是「输出护栏是必经节点」在流式下的实现方式。
          详见 `domain/services/stream_guard.py`。

        Yields:
            `StreamEvent(kind="text")` 正文增量；
            最后一个 `StreamEvent(kind="done")` 带 refused / meta。

        Note:
            缺警示语这类「必须看全文才知道」的问题，
            只能在全流结束后判定，此时已产出的内容无法撤回。
            处理：done 事件里标 refused=True，由调用方决定如何提示用户
            （通常是在已输出内容后追加一句「此外需要说明…」）。
        """
        t0 = time.perf_counter()
        text = (question or "").strip()

        if not text:
            yield from self._refusal_events(
                REFUSAL_TEMPLATE, reason="问题为空")
            return

        # ---- 0. 意图分类（★ 与非流式同一条路径，不能各判各的）----
        intent_result = self._resolve_intent(text, intent)

        # ---- 1. 三路召回（与非流式一致）----
        retrieval = self._retriever.execute(
            text,
            intent=intent_result.label,
            product_ids=product_ids,
            vector_collection=vector_collection,
            allowed_permissions=allowed_permissions,
            top_k=top_k,
        )

        # ---- 2. 召回为空 → 拒答，不调 LLM ----
        #   （有 fact_hints 时同非流式：不拒答，见 `execute` 的说明）
        hints = list(fact_hints or ())
        if (retrieval.failed or not retrieval.fused) and not hints:
            logger.info(
                "召回为空，直接拒答（不调 LLM，流式）: query=%r",
                text[:30],
            )
            yield from self._refusal_events(
                REFUSAL_TEMPLATE,
                reason=self._refusal_reason(retrieval),
                meta=self._meta(retrieval, t0, llm_called=False,
                                intent_result=intent_result),
            )
            return

        # ---- 3~4. 事实 + 组装（与非流式一致）----
        facts = self._load_facts(product_ids, retrieval)
        prompt = self._build_prompt(
            text, retrieval, facts, fact_hints=hints, history=history)
        category = self._category_of(facts, retrieval)

        # ---- 5. 流式生成 + 逐句护栏 ----
        guard = StreamGuard(guardrail=self._guardrail, category=category)
        llm_attempts = 0

        for attempt in range(self._guardrail_retries + 1):
            llm_attempts = attempt + 1
            guard.reset()

            try:
                for sentence in guard.filter(
                    self._llm.generate_stream(
                        prompt,
                        system_prompt=self._system_prompt(),
                        temperature=0.0,
                    )
                ):
                    yield StreamEvent(kind="text", text=sentence)
            except Exception as exc:              # noqa: BLE001
                logger.error("LLM 流式生成失败: %s", exc, exc_info=True)
                yield StreamEvent(
                    kind="done", refused=True,
                    refusal_reason=f"生成服务不可用: {type(exc).__name__}",
                    meta=self._meta(retrieval, t0, llm_called=True,
                                    intent_result=intent_result),
                )
                return

            # 中途被拦 → 无法重试（已产出的内容已经给用户了）
            if guard.blocked:
                logger.warning(
                    "流式护栏拦截（第 %d 次）: %s",
                    llm_attempts, guard.block_reason[:80],
                )
                yield StreamEvent(
                    kind="done", refused=True,
                    refusal_reason=f"输出护栏拦截: {guard.block_reason}",
                    meta=self._meta(
                        retrieval, t0, llm_called=True,
                        intent_result=intent_result,
                        blocked=True,
                        blocked_reason=guard.block_reason,
                        llm_attempts=llm_attempts),
                )
                return

            # 全流结束后的完整检查（补缺警示语）
            fv = guard.final_verdict
            if fv is not None and fv.blocked:
                # 已产出内容无法撤回，只能报 refused 让调用方提示
                logger.warning(
                    "流式结束后护栏拦截（第 %d 次）: %s",
                    llm_attempts, fv.reason[:80],
                )
                yield StreamEvent(
                    kind="done", refused=True,
                    refusal_reason=f"输出护栏拦截: {fv.reason}",
                    meta=self._meta(
                        retrieval, t0, llm_called=True,
                        intent_result=intent_result,
                        blocked=True,
                        blocked_rules=fv.rule_ids,
                        llm_attempts=llm_attempts),
                )
                return

            # 通过
            yield StreamEvent(
                kind="done", refused=False,
                meta={
                    **self._meta(retrieval, t0, llm_called=True,
                                 intent_result=intent_result,
                                 llm_attempts=llm_attempts),
                    "cited_chunk_ids": [h.chunk_id for h in retrieval.fused],
                    "product_ids": [f.product_id for f in facts],
                },
            )
            return

    # ---------------- 内部 ----------------
    def _resolve_intent(self, question: str, intent: str) -> IntentResult:
        """决定这次用哪个意图（直接决定三路权重）。

        ★ 三条取值规则（顺序不能换）：
          1. 调用方**显式**给了标签，就用它 —— 调用方可能比规则更清楚
             （比如演练侧从案例元数据带过来，或者前端让用户自己选了类别）。
          2. 否则用分类器判。
          3. 分类器给的标签若不在权重表里，**退回开放问答**，
             并打 warning —— 静默走兜底权重比报错更难查。

        ★ 为什么不能只靠「`intent or "开放问答"`」这一句：
          那正是改造前的写法，等于永远不分类。看起来「有兜底」，
          实际是 13 套权重表全部作废。
        """
        explicit = (intent or "").strip()
        if explicit:
            if explicit not in INTENT_WEIGHTS:
                logger.warning(
                    "调用方传入的意图「%s」不在权重表里，改用分类器", explicit)
            else:
                return IntentResult(
                    label=explicit, confidence=1.0, source="explicit")

        try:
            return self._intent_classifier.classify(question)
        except Exception as exc:                  # noqa: BLE001 分类失败不能拖垮问答
            logger.warning("意图分类失败，退回开放问答: %s: %s",
                           type(exc).__name__, exc)
            return IntentResult(label=OPEN_INTENT, confidence=0.0)

    @staticmethod
    def _refusal_events(
        text: str, *, reason: str = "", meta: dict | None = None
    ) -> Iterator[StreamEvent]:
        """把一段拒答话术切成 text 事件 + 一个 done 事件。

        切分是为了让前端有「逐字出现」的观感；
        拒答话术本身很短，其实一次性给也不影响体验，
        但保持与其他事件结构一致，前端逻辑更简单。
        """
        yield StreamEvent(kind="text", text=text)
        yield StreamEvent(
            kind="done", refused=True,
            refusal_reason=reason,
            meta=meta or {"llm_called": False},
        )

    @staticmethod
    def _system_prompt() -> str:
        """系统提示。把三条硬约束写进 prompt。

        注意：写进 prompt 只是「减少违规概率」，不能替代护栏。
        LLM 不保证听话，所以护栏必须独立成立。
        """
        return (
            "你是连锁健康食品门店的培训助教，回答新员工的问题。\n"
            "必须遵守：\n"
            "1. 只用【结构化事实】和【参考资料】里的内容作答，不要凭常识补充。\n"
            "2. 保健食品的回答必须包含「保健食品不是药物，不能代替药物治疗疾病」。\n"
            "3. 普通食品不得提及任何保健功能（增强免疫力、改善睡眠、补钙等）。\n"
            "4. 资料里出现的「❌ 不能说」「错误说法」是教学反例，"
            "不要把反例当成可以使用的表述。\n"
            "5. 资料不足时直接说「资料里没有这个信息」，不要推测。\n"
            "6. 不做诊断、不给用药建议、不承诺效果与时间。\n"
            "7. **禁止使用绝对化用语**：唯一、最有效、第一、100%、彻底、包治。\n"
            "   即使是陈述客观事实（如「目录里只有这一项功能」），"
            "也请换个说法（「该项是目录中对应此功能的唯一表述」→"
            "「该项对应目录中的第 19 项功能」）。\n"
            "8. **涉及停药/替代药物时必须用劝阻性表述**："
            "写「不可擅自停药」「请遵医嘱」，不要写「可以停药」。"
        )

    def _refusal_reason(self, retrieval: RetrievalResult) -> str:
        if retrieval.failed and retrieval.degraded:
            return "三路召回均无有效结果（" + "；".join(
                f"{k}: {v}" for k, v in retrieval.degraded.items()) + "）"
        if retrieval.degraded:
            return ("召回无有效结果（部分通路降级: "
                    + "、".join(retrieval.degraded) + "）")
        return "召回无有效结果（知识库无相关内容或相似度低于阈值）"

    def _load_facts(
        self, product_ids: tuple[str, ...], retrieval: RetrievalResult
    ) -> list[ProductFacts]:
        """加载产品完整事实。

        优先用入参 product_ids；没传时尝试从图谱命中里取。
        """
        if self._graph is None:
            return []

        ids = list(product_ids)
        if not ids:
            ids = [r.product_id for r in retrieval.graph_hits]

        out: list[ProductFacts] = []
        for pid in dict.fromkeys(ids):        # 去重保序
            facts = self._graph.get_product_facts(pid)
            if facts is not None:
                out.append(facts)
        return out

    def _category_of(
        self, facts: list[ProductFacts], retrieval: RetrievalResult
    ) -> ProductCategory | None:
        """判定回答涉及的产品类别。

        ★ 为什么必须判类别：
          普通食品不得有功能声称（约束 #2）。
          判不出类别时返回 None，护栏会只做禁用词检测（不误判为普通食品）。

          但注意：如果问题明确问的是某个产品，就必须判出类别，
          否则普通食品的违规声称会漏过。所以优先相信图谱事实。
        """
        if not facts:
            return None

        categories = {f.category for f in facts if f.category}
        if len(categories) == 1:
            cat = categories.pop()
            try:
                return ProductCategory(cat)
            except ValueError:
                # 未知类别：按普通食品处理（fail-safe），宁可多拦
                logger.warning("未知产品类别 %r，按普通食品处理", cat)
                return ProductCategory.ORDINARY_FOOD

        # 混合类别（保健食品 + 普通食品）→ 按最严格处理
        return ProductCategory.ORDINARY_FOOD

    def _build_prompt(
        self,
        question: str,
        retrieval: RetrievalResult,
        facts: list[ProductFacts],
        *,
        fact_hints: list[str] | None = None,
        history: str = "",
    ) -> str:
        """分区组装 prompt。

        分区顺序与 `retrieval-and-agent.md` §1.3 一致：
            结构化事实（权威）→ 相关文档（辅助）

        ★ 参考资料必须过滤「错误示范」（端到端实测发现）：
          培训文档里有大量反面教材，例如 D006 的 Q1 就同时给了
              错误示范：「能，很多人吃了血脂都降了」
              正确示范：「批准的功能是『有助于维持血脂…』」
          模型看到错误示范会被带偏——实测它复述了「降血脂」。

          处理：把以「❌」标记或在「错误示范/错误说法」上下文中的句子剔除。
          只保留可用的正面材料。

        Args:
            fact_hints: 上游查实的结构化结论（如「目录中有第 9 项
                『有助于控制体内脂肪』，但本店在售产品无此批准功能」）。
                这些放在**最前面**并标注为权威 —— 它们是图谱查出来的，
                比文档段落可靠，且直接决定「法规上能不能说」。
            history: 前几轮对话摘要。放在**参考资料之前、事实之后**：
                它只用来理解指代（「第 1 点」指什么），
                **优先级低于**已核实事实 —— 前几轮的说法可能不严谨，
                不能让它覆盖本轮查到的事实。
        """
        parts: list[str] = []

        # ---- 分区零：上游查实的结构化结论（最高权威）----
        #
        # ★ 为什么排在产品事实之前：
        #   「目录里有没有这个功能」是**法规层面**的结论。
        #   它必须在模型开口之前就摆明，否则模型会用文档段落自行推断，
        #   实测就编成了「目录中没有『改善体脂』等表述」（实际第 9 项就是）。
        if fact_hints:
            lines = [
                "【已核实的事实】（系统查询产品库与批准功能目录得出，"
                "权威，优先于参考资料，必须以此为准）",
            ]
            lines.extend(f"- {h}" for h in fact_hints)
            parts.append("\n".join(lines))

        # ---- 分区一：结构化事实（图谱，权威）----
        if facts:
            lines = ["【结构化事实】（来自产品库，权威，可直接引用）"]
            for f in facts:
                lines.append(f"产品：{f.name}（{f.product_id}）")
                lines.append(f"类别：{f.category}")
                if f.approval_no:
                    lines.append(f"批准文号：{f.approval_no}")
                if f.claims:
                    claims = "；".join(f"「{c.text}」" for c in f.claims)
                    lines.append(f"批准功能：{claims}")
                if f.suitable_for:
                    lines.append(f"适宜人群：{'、'.join(f.suitable_for)}")
                if f.contraindications:
                    contras = "、".join(
                        f"{c.name}" + ("（严重）" if c.severity.must_block else "")
                        for c in f.contraindications)
                    lines.append(f"不适宜人群：{contras}")
                if f.usage and f.usage.dosage:
                    lines.append(f"用法用量：{f.usage.dosage}")
                    if f.usage.daily_limit:
                        lines.append(f"每日上限：{f.usage.daily_limit}")
                lines.append("")
            parts.append("\n".join(lines).strip())

        # ---- 分区一点五：前几轮对话（只用于理解指代）----
        #
        # ★ 位置很关键：放在**已核实事实之后、参考资料之前**。
        #   理由：前几轮的回答可能有措辞不严谨之处，
        #   若放在事实之前或与之平级，模型会优先沿用旧说法，
        #   而本轮图谱/目录查到的事实才是最权威的。
        #
        # ★ 也绝不放进召回：追问的检索目标仍是当前这句话。
        if history and history.strip():
            parts.append(
                "【前几轮对话】（仅用于理解追问指代什么，"
                "若与上方已核实事实冲突，**一律以已核实事实为准**）\n"
                + history.strip()
            )

        # ---- 分区二：参考资料（向量 / 关键词）----
        docs = [h for h in retrieval.fused if h.chunk is not None]
        if docs:
            # ★ 重排（可选增强）：RRF 只看排名不看内容，重排按「问题-文档」
            #   相关性重新打分，补上这个盲区。
            #   取前 N 条送重排而不是全部：cross-encoder 延迟随条数线性增长。
            #   重排不可用（未启用/模型缺失/推理失败）时适配器内部已降级，
            #   这里无需再判断，直接拿到列表即可。
            picked = self._pick_docs(question, docs)
            lines = [
                "【参考资料】（来自培训文档，仅供参考）",
                "⚠️ 以下材料已剔除错误示范。写作时不要使用否认句（如「不能说X」），"
                "只做正面陈述。",
            ]
            for i, hit in enumerate(picked, 1):
                c: ChunkMeta = hit.chunk       # type: ignore[assignment]
                body = _strip_bad_examples(c.text)
                if not body:
                    continue
                lines.append(f"[{i}] 《{c.title}》{c.section}")
                lines.append(body[:600])
                lines.append("")
            if len(lines) > 2:
                parts.append("\n".join(lines).strip())

        # ---- 问题 ----
        parts.append(f"【员工的问题】\n{question}")

        return "\n\n".join(parts)

    def _pick_docs(self, question: str, docs: list) -> list:
        """选出进 prompt 的参考资料。

        分两种情况：
        - 未启用重排 / 重排不可用 → 取前 `rerank_top_k` 条（RRF 原顺序，与改造前一致）
        - 启用重排 → 取前 `rerank_candidates` 条送重排，返回重排后的 `rerank_top_k` 条

        ★ 为什么不在这里判断 reranker 是否「成功」：
          降级逻辑放在适配器内部（模型缺失、推理异常都返回原顺序），
          用例层只负责「调用或不调用」。这样职责单一，也避免两处各写一套降级规则。
        """
        if self._reranker is None:
            return docs[:self._rerank_top_k]

        candidates = docs[:self._rerank_candidates]
        try:
            return self._reranker.rerank(
                question, candidates, top_k=self._rerank_top_k)
        except Exception as exc:                       # noqa: BLE001
            # 适配器应已内部降级，这里是最后一道保险：
            # 重排出任何问题都不该让用户拿不到答案。
            logger.warning("重排调用异常，降级为 RRF 原顺序: %s", exc)
            return docs[:self._rerank_top_k]

    def _log_miss(self, question: str, retrieval: RetrievalResult) -> None:
        """记录未命中（FR-047）。

        ★ 这是「补充知识库的唯一可靠依据」。

        ★ 写入失败**不能影响拒答**：
          日志是运营素材，不该因为库写不进去就让学员看不到拒答话术。
          所以这里吞掉异常，只记 warning。
        """
        if self._repo is None:
            return

        try:
            from domain.ports.repository_port import MissLogRow

            self._repo.append_miss_log(MissLogRow(
                id="",
                query=question[:500],
                intent=retrieval.intent or "",
                trainee_id="",          # 用例层不感知调用者身份
                detail={
                    "graph_hits": len(retrieval.graph_hits),
                    "vector_hits": len(retrieval.vector_hits),
                    "keyword_hits": len(retrieval.keyword_hits),
                    "fused": len(retrieval.fused),
                    "degraded": list(retrieval.degraded),
                },
            ))
        except Exception as exc:                    # noqa: BLE001
            logger.warning("未命中日志写入失败（不影响拒答）: %s", exc)

    @staticmethod
    def _meta(retrieval: RetrievalResult, t0: float, *,
              llm_called: bool,
              intent_result: IntentResult | None = None,
              **extra) -> dict:
        """诊断信息。便于回答「为什么这条排第一」与「慢在哪一路」。"""
        meta = {
            "intent": retrieval.intent,
            # ★ 意图来源与置信度：分类器判错时，这两个字段是唯一线索。
            #   没有它们，只能看到「权重不对」却不知道为什么。
            "intent_source": intent_result.source if intent_result else "",
            "intent_confidence": (
                round(intent_result.confidence, 2) if intent_result else 0.0),
            "intent_matched": list(intent_result.matched) if intent_result else [],
            "weights": retrieval.weights,
            "graph_hits": len(retrieval.graph_hits),
            "vector_hits": len(retrieval.vector_hits),
            "keyword_hits": len(retrieval.keyword_hits),
            "fused": len(retrieval.fused),
            "degraded_paths": list(retrieval.degraded),
            "elapsed_ms": {
                **{k: round(v, 1) for k, v in retrieval.elapsed_ms.items()},
                "total": round((time.perf_counter() - t0) * 1000, 1),
            },
            "llm_called": llm_called,
        }
        meta.update(extra)
        return meta
