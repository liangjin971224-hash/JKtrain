"""三路召回用例（P3 核心）。

职责：并行调用图谱 / 向量 / 关键词三路，融合结果，处理降级。

────────────────────────────────────────────────────────────────────
三条必须守住的语义
────────────────────────────────────────────────────────────────────
1. **单路失败不影响整体**
   三路里任一路抛异常，记入 `degraded`，其余路继续。
   只有三路全失败才标记为彻底失败。

2. **「没找到」与「搜到了不相关内容」是两件事**
   向量检索永远返回 Top-N（哪怕问的是知识库里完全没有的东西），
   所以不能靠「结果为空」判断召回是否成功。
   必须用**分数阈值**：向量 Top1 低于阈值时，视为该路无有效召回。

3. **融合后仍要判可信度**
   RRF 分数是相对值（取决于路数与排名），不是绝对相关度。
   所以不能只看 RRF 分数，要结合各路原始分数判断。

第 2、3 条是 P4「召回为空则拒答，不调 LLM」能成立的前提。
没有阈值，P4 的拒答逻辑永远触发不了，LLM 会拿着无关内容编答案。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from domain.ports import (
    GraphQueryPort,
    KeywordSearchPort,
    RerankPort,
    SparseSearchPort,
    VectorSearchPort,
)
from domain.services.rrf_fusion import reciprocal_rank_fusion, weights_for_intent
from domain.value_objects import FusedHit, KeywordHit, ProductRef, VectorHit

logger = logging.getLogger(__name__)

#: 三路名称（顺序与 INTENT_WEIGHTS 一致：图谱 / 向量 / 关键词）
PATH_GRAPH = "graph"
PATH_VECTOR = "vector"
PATH_KEYWORD = "keyword"
PATH_SPARSE = "sparse"

#: ★ 向量检索的最低可信相似度。
#:
#: 依据：P0/P2 实测，真实相关内容的余弦相似度集中在 **0.55–0.78**；
#: 完全无关的查询 Top1 也能到 0.45 左右（向量检索的固有特性）。
#: 取 0.50 作为「这条内容是否值得给 LLM 看」的分界线。
#:
#: 阈值不能设太高：0.55 会误杀真实相关但表述不同的内容（P2 实测有 0.52 的正确命中）。
#: 也不能设太低：0.40 以下基本是噪声。
MIN_VECTOR_SCORE = 0.50

#: ★ 关键词路的**绝对**可信下限（本轮实测新增）。
#:
#: 问题背景：
#:     中文 2-gram 分词会产生大量偶然命中。
#:     实测「你们有没有量子能量水」（知识库里完全没有的东西）时：
#:         向量 Top1 = 0.472（被 0.50 阈值正确拦下）
#:         关键词 Top1 = 11.74（「能量」「量水」这类字组合在别处出现过）
#:     融合后 fused=10，「召回为空」的防线失效，系统开始编答案。
#:
#: 阈值怎么定的（44 条评测集 + 8 条噪声查询实测）：
#:     真实查询（有答案）  Top1 = 15.60 ~ 28.48
#:     噪声查询（库里没有）Top1 =  5.15 ~ 11.86
#:     分界落在 12~15 之间，取 **14** 留余量。
#:
#: ★ 为什么这条不能用相对阈值代替：
#:     噪声查询的 Top1 本身就很高（11.74），相对比例法拦不住——
#:     它的第 2、3 名分数也不低（都是"命中几个常见字"）。
#:     必须有绝对下限。
#:
#: ★ 已知限制：
#:     BM25 分数与语料规模相关。将来语料大幅扩张（比如 10 倍）后，
#:     同一查询的分数会变，这个阈值需要重新校准。
#:     所以它做成了 `RetrieveUseCase` 的构造参数，便于调整。
MIN_KEYWORD_SCORE = 14.0

#: 关键词路的**相对**阈值：低于 Top1 的这个比例视为噪声。
#:
#: 用于剪掉「勉强沾边」的尾部长尾，避免它们进入融合拉高 fused 数量。
#: 取 0.35：真实相关的第 2~5 名通常能到 Top1 的 40% 以上，
#: 而偶然命中通常在 20% 以下。
MIN_KEYWORD_RELATIVE = 0.35

#: 稀疏召回的绝对下限（IP 尺度）。
#:
#: ★ 不是相关性阈值，只是 sanity floor。实测（205 题）可答与不可答查询的
#:   稀疏 top1 分数几乎完全重叠（可答中位 0.209 / 不可答最大 0.222），
#:   任何能拦住噪声的值都会误拒 60%+ 正常查询。
#:   拒答仍由规则 + LLM「证据不足」路径负责，这里只拦"向量算坏了"。
MIN_SPARSE_SCORE = 0.02

#: 并集窗口里保留多少条 RRF 结果在最前面（其余位置由稀疏填充）。
#:
#: ★ 这个数字是被**提示词长度**倒推出来的，不是拍的：
#:   当时的 `answer_question` 只取 `fused[:5]`（`context_docs` 默认 5），
#:   所以 6 位往后放什么都不影响 LLM。案例库 34 题实测覆盖：
#:       app5 + sparse0（不做并集） 21/34
#:       app4 + sparse1             22/34
#:       app3 + sparse2             23/34
#:       app2 + sparse3             24/34   ← 取这个
#:       app1 + sparse4             25/34
#:       app0 + sparse5             25/34
#:   1+4 与 2+3 只差 1 题（噪声量级），选 2+3 是为了保留两条
#:   「向量+BM25 共识」结果，防止稀疏在某个查询上整体跑偏时窗口全空。
#:
#: ★ 前提变更注意：P1-b 若把 `context_docs` 调大（8/10），这个值要重新标定
#:   ——上面的分配实验只在 5 位窗口上做过。当前按 context_docs=5 保持 2。
SPARSE_UNION_HEAD = 2

#: 「每路头部进窗口」策略：每路先取前 N 条占住窗口头部，其余由 RRF 回填。
#:
#: ★ 为什么这条路对**文档库**有效（205 题离线模拟，2026-11-07）：
#:   文档库的 BM25 强项是精确串（编号、流程术语），但「开放问答」权重
#:   (0.1, 0.8, 0.1) 把 BM25 第 1 名压到 0.0016——只相当于向量第 20 名的
#:   六分之一，gold 永远进不了前 5。把「每路第 1 名」直接拎进窗口，
#:   等于绕开权重给精确匹配一个保底位。
#:
#:   离线模拟（文档库 131 题）：纯 RRF 105 → 每路头 1 = 111；
#:   案例库相反（21 → 22，不如稀疏并集 26），所以**策略必须按库选**。
#:   题库 40/40 无变化，不配置即保持纯 RRF。
PATH_HEAD = 1

#: 查询「可检索性」预检查。
#:
#: ★ 为什么需要（2026-10-23 实测发现）：
#:     `MIN_KEYWORD_SCORE=14.0` 只挡住了「中文噪声查询」
#:     （「量子能量水」这类 Top1≈11.7 的情况）。
#:     但**纯字母/数字乱码**反而得分更高：
#:
#:         查询                向量 Top1   关键词 Top1
#:         zzzzqqqqxxxx        0 条       **17.62**  ← 高于 14，被当有效召回
#:         1234567890abcdef    0 条       14.51
#:
#:     原因：BM25 的 IDF 惩罚「常见词」，而乱码分词后产生一堆
#:     「语料里几乎不出现」的 token —— 按公式反而给高分。
#:     这是 BM25 的固有特性，靠调阈值治不了（调高会误杀正常短查询）。
#:
#: ★ 修法：在召回前判断「这个查询有没有可检索的实词」。
#:    没有任何中文/英文实词 → 直接返回空，不进三路检索。
#:
#: ★ 判据只用「是否含实词」，不判断语义。
#:    因为语义判断必须靠 LLM，那又慢又贵，
#:    而这里要拦的是「压根不成词」的输入。
_MEANINGFUL_CHAR = re.compile(r"[\u4e00-\u9fffA-Za-z]{2,}")


def is_retrievable(query: str) -> bool:
    """查询里是否有**可检索的中文内容**。

    Args:
        query: 用户问题。

    Returns:
        True 表示值得检索；False 表示应当直接拒答。

    ★ 为什么必须有这个预检查（2026-10-23 实测发现）：
        知识库是**中文**的，但 BM25 对**纯英文/数字乱码**反而给高分：

            查询                向量      关键词 Top1
            zzzzqqqqxxxx        0 条      **17.62**   ← 高于阈值 14，被当有效召回
            1234567890abcdef    0 条      14.51

        原因：BM25 用 IDF 惩罚常见词，而乱码分词后产生的 token
        在语料里几乎不出现 → 按公式反而得分极高。
        这是 BM25 的固有特性，**调阈值治不了**
        （调高会误杀「鱼油」这类正常短查询）。

    ★ 判据：**必须含中文**。
        中文知识库里的有效查询必然含中文，纯英文乱码一定不含。

        例外：允许「含 3+ 字母英文词 且 长度 ≥ 6」的查询
        （覆盖 `CoQ10`、`Omega3` 这类真实产品名），
        但 `zzzzqqqqxxxx` 这种单块字母堆不算——
        它的特点是没有词边界，也没有数字/中文混合。
    """
    q = (query or "").strip()
    if not q:
        return False

    # 含中文 → 一定可检索（中文库）
    if any("\u4e00" <= ch <= "\u9fff" for ch in q):
        return True

    # 无中文：只接受「像术语」的形式
    #   · 有数字（CoQ10 / Omega3 / 维生素B2）
    #   · 或有空格分隔的多个英文词（"fish oil"）
    if re.search(r"[A-Za-z]+\d|\d+[A-Za-z]", q):
        return True
    if re.search(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}", q):
        return True

    # 其余（纯字母堆、纯数字、纯符号）视为不可检索
    return False


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """三路召回的完整结果。

    ★ `degraded` 与 `failed` 的区别：
       degraded: 部分路失败，但仍有结果可用 → 正常回答
       failed:   三路全失败 → 必须拒答（P4 据此不调 LLM）
    """

    query: str
    intent: str

    graph_hits: tuple[ProductRef, ...] = ()
    vector_hits: tuple[VectorHit, ...] = ()
    keyword_hits: tuple[KeywordHit, ...] = ()
    #: 稀疏路命中（IP 分数）。仅对启用了并集的 collection 有值。
    sparse_hits: tuple[VectorHit, ...] = ()

    fused: tuple[FusedHit, ...] = ()

    #: 失败的路：{路名: 失败原因}
    degraded: dict[str, str] = field(default_factory=dict)

    #: 各路耗时（毫秒），便于定位慢在哪一路
    elapsed_ms: dict[str, float] = field(default_factory=dict)

    #: 权重（图谱, 向量, 关键词），便于回放「为什么这条排第一」
    weights: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def failed(self) -> bool:
        """全部路都失败或全部无有效结果。

        ★ 必须把 `sparse_hits` 算进来：否则"向量路降级、只有稀疏命中"
          会被判成召回失败 → P4 直接拒答，稀疏收益全部作废。
        """
        return not (self.graph_hits or self.vector_hits
                    or self.keyword_hits or self.sparse_hits)

    @property
    def is_degraded(self) -> bool:
        """是否发生了降级（部分路失败）。"""
        return bool(self.degraded)

    @property
    def total_raw_hits(self) -> int:
        """三路原始命中总数（未融合）。"""
        return (len(self.graph_hits) + len(self.vector_hits)
                + len(self.keyword_hits) + len(self.sparse_hits))

    def top_doc_ids(self, n: int = 5) -> list[str]:
        """融合后的 Top-N doc_id，用于评测与日志。"""
        return [h.chunk_id for h in self.fused[:n]]


class RetrieveUseCase:
    """三路召回用例。

    构造函数只接受 Port，不碰具体适配器——这样才能用假实现测降级逻辑。
    """

    def __init__(
        self,
        *,
        graph: GraphQueryPort | None = None,
        vector: VectorSearchPort | None = None,
        keyword: KeywordSearchPort | None = None,
        rerank: RerankPort | None = None,
        sparse: SparseSearchPort | None = None,
        sparse_collections: frozenset[str] = frozenset(),
        min_sparse_score: float = MIN_SPARSE_SCORE,
        sparse_union_head: int = SPARSE_UNION_HEAD,
        path_head_collections: frozenset[str] = frozenset(),
        rerank_candidate_k: int = 20,
        min_vector_score: float = MIN_VECTOR_SCORE,
        min_keyword_score: float = MIN_KEYWORD_SCORE,
    ) -> None:
        """
        Args:
            graph: 图谱端口。None 表示该路不可用（P1 未完成时的降级）。
            vector: 向量端口。None 表示该路不可用。
            keyword: 关键词端口。None 表示该路不可用。
            rerank: 重排序端口（可选）。None = 不重排，融合序即最终序。
                ★ rerank 只重排「已过阈值与权限过滤」的文档块候选，
                  不参与判空决策——拒答语义与无 rerank 时完全一致
                  （docs/TDD_检索质量优化 §4.3）。
            sparse: 稀疏端口（可选）。None = 不做稀疏召回。
            sparse_collections: 启用「RRF 头部 + 稀疏候选」并集的 collection。
                ★ 只对需要的库开（实测案例库 +7 题、文档库无收益）。
            min_sparse_score: 稀疏召回的绝对下限（IP 尺度，仅 sanity floor）。
            sparse_union_head: 并集窗口里保留多少条 RRF 结果在最前（见常量注释）。
            path_head_collections: 启用「每路头部进窗口」的 collection。
                ★ 只对文档库开（离线模拟 105→111）；案例库走稀疏并集、
                  题库 40/40 无变化，都不配置。空集 = 全部走纯 RRF（回滚开关）。
                ★ 与 `sparse_collections` 重叠时**稀疏并集优先**（构造时告警）。
            rerank_candidate_k: 送入重排的融合候选窗口大小。
            min_vector_score: 向量召回的最低可信相似度。
            min_keyword_score: 关键词召回的绝对分数下限。
                ★ 与语料规模相关，语料大幅扩张后需重新校准。
        """
        self._graph = graph
        self._vector = vector
        self._keyword = keyword
        self._rerank = rerank
        self._sparse = sparse
        self._sparse_collections = frozenset(sparse_collections)
        self._min_sparse_score = min_sparse_score
        self._sparse_union_head = max(sparse_union_head, 0)
        self._path_head_collections = frozenset(path_head_collections)
        overlap = self._sparse_collections & self._path_head_collections
        if overlap:
            logger.warning(
                "collection %s 同时配置了稀疏并集与每路头部窗口，"
                "按「稀疏并集优先」处理", sorted(overlap))
        self._rerank_candidate_k = max(rerank_candidate_k, 1)
        self._min_vector_score = min_vector_score
        self._min_keyword_score = min_keyword_score

    # ---------------- 主流程 ----------------
    def execute(
        self,
        query: str,
        *,
        intent: str = "开放问答",
        product_ids: tuple[str, ...] = (),
        vector_collection: str = "",
        allowed_permissions: tuple[str, ...] | None = None,
        top_k: int = 20,
        fused_limit: int = 10,
    ) -> RetrievalResult:
        """执行三路召回并融合。

        Args:
            query: 用户问题。
            intent: 意图标签（决定权重）。
            product_ids: 查询理解阶段抽出的产品 ID（图谱精确查询用）。
            vector_collection: 向量检索的目标 collection。
            allowed_permissions: 权限级别（学员 / 主管）。
            top_k: 每路取多少条。
            fused_limit: 融合后保留多少条。

        Returns:
            召回结果。三路全失败时 `failed=True`。
        """
        text = (query or "").strip()
        weights = weights_for_intent(intent)
        degraded: dict[str, str] = {}
        elapsed: dict[str, float] = {}

        if not text:
            return RetrievalResult(query="", intent=intent, weights=weights)

        # ---- 0. ★ 可检索性预检查 ----
        #
        # 实测缺陷：乱码查询（`zzzzqqqqxxxx`）向量 0 条，
        # 但 BM25 给 17.62 分（高于阈值 14）→ 被当有效召回 → 系统开始编。
        # 这类输入压根不该进检索，详见 `is_retrievable` 的注释。
        if not is_retrievable(text):
            logger.info("查询无可检索内容，直接返回空: %r", text[:30])
            return RetrievalResult(
                query=text, intent=intent, weights=weights,
                degraded={"preflight": "查询无可检索的中文内容"},
            )

        # ---- 图谱路 ----
        graph_hits: list[ProductRef] = []
        graph_hits, err, ms = self._run_path(
            PATH_GRAPH,
            lambda: self._retrieve_graph(text, product_ids),
        )
        elapsed[PATH_GRAPH] = ms
        if err:
            degraded[PATH_GRAPH] = err

        # ---- 向量路 ----
        vector_hits, err, ms = self._run_path(
            PATH_VECTOR,
            lambda: self._retrieve_vector(
                text, vector_collection, allowed_permissions, top_k),
        )
        elapsed[PATH_VECTOR] = ms
        if err:
            degraded[PATH_VECTOR] = err

        # ---- 关键词路 ----
        keyword_hits, err, ms = self._run_path(
            PATH_KEYWORD,
            lambda: self._retrieve_keyword(text, allowed_permissions, top_k),
        )
        elapsed[PATH_KEYWORD] = ms
        if err:
            degraded[PATH_KEYWORD] = err

        # ---- 稀疏路（仅对配置的 collection）----
        sparse_hits: list[VectorHit] = []
        if self._sparse is not None and vector_collection in self._sparse_collections:
            sparse_hits, err, ms = self._run_path(
                PATH_SPARSE,
                lambda: self._retrieve_sparse(
                    text, vector_collection, allowed_permissions, top_k),
            )
            elapsed[PATH_SPARSE] = ms
            if err:
                degraded[PATH_SPARSE] = err

        # ---- 融合（+ 稀疏并集 + 可选 rerank）----
        # _fuse 返回全量融合序；rerank（若装配）在截断前对窗口内重排，
        # 最后才按 fused_limit 截断——让 rerank 有完整候选池可施展。
        fused_all = self._fuse(
            query=text,
            intent=intent,
            graph_hits=graph_hits,
            vector_hits=vector_hits,
            keyword_hits=keyword_hits,
            weights=weights,
        )

        # ★ 稀疏并集（P0-3）：把稀疏命中并入窗口，而不是让它们参与 RRF 排序。
        #   实测 RRF 会把「单路第 1、另一路第 15」的块稀释掉
        #   （案例库：稀疏单独 25/34 → 并进 RRF 后 21/34）。
        if sparse_hits:
            fused_all = self._union_sparse(fused_all, sparse_hits, fused_limit)
        elif vector_collection in self._path_head_collections:
            # ★ 每路头部进窗口（P1-a）：文档库的 BM25 精确串被意图权重压到
            #   20 名开外，直接把每路第 1 名拎进窗口头部（离线模拟 105→111）。
            fused_all = self._union_path_heads(
                fused_all,
                (
                    (PATH_VECTOR, vector_hits),
                    (PATH_KEYWORD, keyword_hits),
                ),
                fused_limit,
            )

        import time as _time

        t_r = _time.perf_counter()
        fused_ranked = self._rerank_fused(text, fused_all)
        if self._rerank is not None and fused_all:
            elapsed["rerank"] = (_time.perf_counter() - t_r) * 1000
        fused = fused_ranked[:fused_limit]

        result = RetrievalResult(
            query=text,
            intent=intent,
            graph_hits=tuple(graph_hits),
            vector_hits=tuple(vector_hits),
            keyword_hits=tuple(keyword_hits),
            sparse_hits=tuple(sparse_hits),
            fused=tuple(fused),
            degraded=degraded,
            elapsed_ms=elapsed,
            weights=weights,
        )

        logger.info(
            "三路召回完成: intent=%s, graph=%d, vector=%d, keyword=%d, sparse=%d, "
            "fused=%d, degraded=%s, elapsed_ms=%s",
            intent, len(graph_hits), len(vector_hits), len(keyword_hits),
            len(sparse_hits),
            len(fused), list(degraded) or "无",
            {k: round(v, 1) for k, v in elapsed.items()},
        )
        return result

    # ---------------- 各路实现 ----------------
    def _retrieve_graph(
        self, query: str, product_ids: tuple[str, ...]
    ) -> list[ProductRef]:
        """图谱召回：抽到产品 ID 时查完整画像，否则从查询里抽产品名再归一。

        ★ 离线的能力意识：图谱不擅长语义匹配，
          所以它只在「能定位到具体产品」时出力，不参与泛化召回。

        ★ 为什么要做产品名抽取（用户实测暴露的 bug）：
          直接把**整句**交给 `resolve_alias` 基本必然失败：
              resolve_alias("康源牌鱼油软胶囊有什么功能") → None
              resolve_alias("鱼油")                      → P001
          而事实上别名表里就有「鱼油」。

          后果非常严重：
            图谱没命中 → 产品事实没注入 prompt
              → 参考资料里没有批准功能原文
              → LLM 凭参数知识编出旧版功能名「辅助降血脂」
              → 被护栏拦（护栏是对的，但根因在事实没喂进去）

          所以先抽出候选产品名，再逐个解析。
        """
        if self._graph is None:
            raise RuntimeError("图谱端口未装配")

        refs: list[ProductRef] = []
        seen: set[str] = set()

        def _add(ref: ProductRef | None) -> None:
            if ref is not None and ref.product_id not in seen:
                seen.add(ref.product_id)
                refs.append(ref)

        # 1. 已抽出产品 ID → 直接取
        for pid in product_ids:
            facts = self._graph.get_product_facts(pid)
            if facts is not None:
                _add(ProductRef(
                    product_id=facts.product_id,
                    name=facts.name,
                    category=facts.category,
                    approval_no=facts.approval_no,
                ))

        # 2. 从查询里抽候选产品名，逐个解析别名
        if not refs:
            for candidate in self._extract_product_candidates(query):
                _add(self._graph.resolve_alias(candidate))
                if refs:
                    break            # 命中一个就够（问句通常只涉及一个产品）

        # 3. 兜底：整句当别名（覆盖「就问一个词」的情况）
        if not refs:
            _add(self._graph.resolve_alias(query))

        return refs

    @staticmethod
    def _extract_product_candidates(query: str) -> list[str]:
        """从查询里抽出候选产品名。

        策略（从长到短，长匹配优先）：
            ① 「品牌 + 品类」全称：康源牌鱼油软胶囊
            ② 「品牌 + 品类」短称：康源牌鱼油
            ③ 纯品类名：鱼油、辅酶Q10、蛋白粉…

        ★ 为什么按长度降序：
          「康源牌鱼油软胶囊」应该整体命中，而不是先被「鱼油」截胡——
          虽然当前两者指向同一产品，但若将来有多个鱼油产品就会错。

        Args:
            query: 用户原始问句。

        Returns:
            候选名列表（去重、按长度降序）。可能为空。
        """
        if not query:
            return []

        #: 品类词（与 `forbidden_claims._PRODUCT_NAME` 保持一致，
        #: 但这里只关心「能用来查图谱」的短名）
        category_words = (
            r"鱼油|辅酶Q10|辅酶|蛋白粉|乳清蛋白|大豆蛋白|钙维生素D|钙片|"
            r"褪黑素|益生菌|代餐奶昔|固体饮料|膳食纤维|胶原蛋白|维生素C|"
            r"叶黄素|氨糖|蜂胶|螺旋藻|破壁灵芝孢子粉|灵芝孢子粉|孢子粉|"
            r"黑芝麻丸|芝麻丸"
        )

        candidates: list[str] = []
        seen: set[str] = set()

        def _push(s: str) -> None:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                candidates.append(s)

        # ① 全称：康源牌 + 品类 + 可选剂型
        for m in re.finditer(
            rf"康源牌\s*(?:{category_words})\s*(?:软胶囊|胶囊|片|粉|丸)?",
            query,
        ):
            _push(m.group(0).replace(" ", ""))

        # ② 品牌 + 品类（不带剂型）
        for m in re.finditer(rf"康源牌\s*(?:{category_words})", query):
            _push(m.group(0).replace(" ", ""))

        # ③ 裸品类名
        for m in re.finditer(category_words, query):
            _push(m.group(0))

        # 按长度降序：长的先试，避免「鱼油」截胡「康源牌鱼油软胶囊」
        candidates.sort(key=len, reverse=True)
        return candidates

    def _retrieve_vector(
        self,
        query: str,
        collection: str,
        allowed_permissions: tuple[str, ...] | None,
        top_k: int,
    ) -> list[VectorHit]:
        """向量召回。

        ★ 关键：按分数阈值过滤。
          向量永远返回 Top-N，低分结果就是噪声，
          放进融合会污染排名，让 P4 误以为「召回到了」。
        """
        if self._vector is None:
            raise RuntimeError("向量端口未装配")
        if not collection:
            raise ValueError("未指定 vector_collection")

        hits = self._vector.search(
            query,
            collection=collection,
            top_k=top_k,
            allowed_permissions=allowed_permissions,
        )
        kept = [h for h in hits if h.score >= self._min_vector_score]
        if len(kept) < len(hits):
            logger.debug(
                "向量召回按阈值 %.2f 过滤: %d → %d",
                self._min_vector_score, len(hits), len(kept),
            )
        return kept

    def _retrieve_keyword(
        self,
        query: str,
        allowed_permissions: tuple[str, ...] | None,
        top_k: int,
    ) -> list[KeywordHit]:
        """关键词召回。

        ★ 两道阈值（缺一不可，实测都在真机上出过问题）：
          ① 绝对下限 `min_keyword_score`：拦掉「整条查询都不相关」的情况。
             中文 2-gram 噪声查询也能拿到 11 分，必须设绝对门槛。
          ② 相对比例 `MIN_KEYWORD_RELATIVE`：剪掉勉强沾边的长尾。
        """
        if self._keyword is None:
            raise RuntimeError("关键词端口未装配")

        hits = self._keyword.search(
            query, top_k=top_k, allowed_permissions=allowed_permissions)
        if not hits:
            return []

        # ① 绝对下限
        if hits[0].score < self._min_keyword_score:
            logger.debug(
                "关键词召回整体低于绝对阈值 %.1f（Top1=%.2f），视为无有效召回",
                self._min_keyword_score, hits[0].score,
            )
            return []

        # ② 相对比例
        floor = hits[0].score * MIN_KEYWORD_RELATIVE
        kept = [h for h in hits if h.score >= floor]

        if len(kept) < len(hits):
            logger.debug(
                "关键词召回按相对阈值 %.0f%% 剪枝: %d → %d",
                MIN_KEYWORD_RELATIVE * 100, len(hits), len(kept),
            )
        return kept

    def _retrieve_sparse(
        self,
        query: str,
        collection: str,
        allowed_permissions: tuple[str, ...] | None,
        top_k: int,
    ) -> list[VectorHit]:
        """稀疏召回（bge-m3 lexical_weights → IP）。

        ★ 阈值只做 sanity floor：实测可答/不可答查询的稀疏 top1 分数
          几乎完全重叠，任何能拦噪声的值都会误拒 60%+ 正常查询
          （详见 `MIN_SPARSE_SCORE` 注释）。
        """
        if self._sparse is None:
            raise RuntimeError("稀疏端口未装配")
        if not collection:
            raise ValueError("未指定 vector_collection")

        hits = self._sparse.search_sparse(
            query,
            collection=collection,
            top_k=top_k,
            allowed_permissions=allowed_permissions,
        )
        kept = [h for h in hits if h.score >= self._min_sparse_score]
        if len(kept) < len(hits):
            logger.debug(
                "稀疏召回按 floor %.2f 过滤: %d → %d",
                self._min_sparse_score, len(hits), len(kept),
            )
        return kept

    def _union_sparse(
        self,
        fused_all: list[FusedHit],
        sparse_hits: list[VectorHit],
        fused_limit: int,
    ) -> list[FusedHit]:
        """把稀疏命中并入融合窗口（**覆盖优先**，不参与 RRF 重排）。

        构图（实测口径，205 题案例库）：
            窗口 = RRF 头部(sparse_union_head 条) + 稀疏未重复命中 + RRF 尾部回填
        ★ 头部条数由**提示词只取前 5 条**倒推（见 `SPARSE_UNION_HEAD`）；
          若将来提示词扩容，该常量要跟着改，否则稀疏仍进不了 LLM。

        ★ 为什么不把稀疏当作第 4 路丢进 RRF：
          RRF 只用名次，1/(60+rank) 会把「单路第 1、另一路第 15」的块
          稀释成"两边都中等"的水平。实测案例库：稀疏单独 25/34，
          并进 RRF 后回落到 21/34，而并集窗口能到 28/34。

        ★ `fused_score` 语义：稀疏来源的条目恒为 0.0，并在
          `sources` 里标 `sparse`。因此**并集生效时 fused 的顺序是
          「覆盖优先」而非「分数降序」**，下游不得按 fused_score 重排。
        """
        head = min(self._sparse_union_head, fused_limit)
        window: list[FusedHit] = list(fused_all[:head])
        seen = {h.chunk_id for h in window}

        for hit in sparse_hits:
            if len(window) >= fused_limit:
                break
            cid = hit.chunk.chunk_id if hit.chunk is not None else ""
            if not cid or cid in seen:
                continue
            seen.add(cid)
            window.append(FusedHit(
                chunk_id=cid,
                fused_score=0.0,
                chunk=hit.chunk,
                sources=(PATH_SPARSE,),
            ))

        for hit in fused_all[head:]:
            if len(window) >= fused_limit:
                break
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            window.append(hit)

        return window

    def _union_path_heads(
        self,
        fused_all: list[FusedHit],
        path_hits: tuple[tuple[str, list], ...],
        fused_limit: int,
    ) -> list[FusedHit]:
        """把每路头部若干条提到窗口最前面（**覆盖优先**，其余按 RRF 回填）。

        构图（205 题离线模拟口径）：
            窗口 = 向量头 PATH_HEAD 条 + 关键词头 PATH_HEAD 条 + RRF 回填

        ★ 为什么需要（文档库 131 题，2026-11-07）：
          「开放问答」权重 (0.1, 0.8, 0.1) 下 BM25 第 1 名只有 0.0016，
          约等于向量第 20 名的六分之一——「BM25 精确命中、向量没命中」
          的题 gold 被压到 20 名开外。每路头 1 相当于给精确匹配保底一个位。
          模拟：文档库纯 RRF 105 → 111；案例库 21 → 22（不如稀疏并集 26，
          所以本策略只按库配置，不做全局默认）。题库 40/40 无变化。

        ★ 与 `_union_sparse` 一样，`fused_score` 是 RRF 原分而非窗口顺序分；
          下游不得按 `fused_score` 重排（窗口顺序 = 覆盖优先）。

        Args:
            fused_all: 完整 RRF 融合序（含全部候选）。
            path_hits: (路名, 该路命中列表)，按优先级排列。
            fused_limit: 窗口大小。

        Returns:
            至多 `fused_limit` 条、跨路去重的窗口。
        """
        by_id = {h.chunk_id: h for h in fused_all}
        window: list[FusedHit] = []
        seen: set[str] = set()

        for path_name, hits in path_hits:
            for hit in hits[:PATH_HEAD]:
                if len(window) >= fused_limit:
                    return window
                cid = hit.chunk.chunk_id if hit.chunk is not None else ""
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                # 复用 RRF 条目的分数与 sources；理论上头部命中必在 fused_all 里
                # （向量/关键词权重恒 > 0），兜底构造是为了不因权重误配丢块。
                window.append(by_id.get(cid) or FusedHit(
                    chunk_id=cid,
                    fused_score=0.0,
                    chunk=hit.chunk,
                    sources=(path_name,),
                ))

        for hit in fused_all:
            if len(window) >= fused_limit:
                break
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            window.append(hit)

        return window

    # ---------------- 降级包装 ----------------
    @staticmethod
    def _run_path(
        name: str, fn: Callable[[], list]
    ) -> tuple[list, str, float]:
        """执行一路召回，捕获异常。

        用 try/except 包住单路，但**不吞掉信息**：
        异常类型与消息要记进 degraded，供上层展示与排障。

        Returns:
            (结果列表, 错误描述或空串, 耗时毫秒)
        """
        import time

        t0 = time.perf_counter()
        try:
            result = fn()
            return result, "", (time.perf_counter() - t0) * 1000
        except Exception as exc:              # noqa: BLE001 单路失败必须降级
            ms = (time.perf_counter() - t0) * 1000
            msg = f"{type(exc).__name__}: {exc}"
            logger.warning("召回路 %s 失败，降级继续: %s", name, msg)
            return [], msg, ms

    # ---------------- 融合 ----------------
    def _fuse(
        self,
        *,
        query: str,
        intent: str,
        graph_hits: list[ProductRef],
        vector_hits: list[VectorHit],
        keyword_hits: list[KeywordHit],
        weights: tuple[float, float, float],
    ) -> list[FusedHit]:
        """RRF 融合**文档块**两路（向量 + 关键词）。

        ★ 为什么图谱路不参与排名融合：
          图谱返回的是**产品事实**（ProductRef，权威结构化数据），
          向量/关键词返回的是**文档块**（chunk_id，供 LLM 参考的文本）。
          两者是不同类别的东西：
            · 产品事实要走「事实卡」直接呈现，不该和文档块抢排名
            · 事实卡没有正文，混进 fused 会产生 `chunk=None` 的条目，
              下游按 fused 取内容时会踩空
            · 用一个排名同时表达「这个产品对不对」和「这段文档相关不相关」，
              语义是混乱的

          所以：图谱结果单独放在 `graph_hits` 里给上层用，
          融合只管文档块。图谱的权重体现在**上层怎么用事实卡**，
          而不是体现在这个排名里。

          这也是 P3 验收里「三路提升」的判定方式：
          文档召回看 R@5，图谱正确性看排除法/禁忌命中的 100%。
        """
        rankings = [
            [h.chunk.chunk_id for h in vector_hits],
            [h.chunk.chunk_id for h in keyword_hits],
        ]
        # 权重只取后两路（向量、关键词）；图谱权重不参与文档排名
        doc_weights = [weights[1], weights[2]]

        if not any(rankings):
            return []

        fused_pairs = reciprocal_rank_fusion(rankings, weights=doc_weights)

        # 回填 chunk 元数据，并记录该结果来自哪几路
        chunk_index: dict[str, object] = {}
        sources: dict[str, list[str]] = {}

        for name, hits in ((PATH_VECTOR, vector_hits),
                           (PATH_KEYWORD, keyword_hits)):
            for h in hits:
                chunk_index.setdefault(h.chunk.chunk_id, h.chunk)
                sources.setdefault(h.chunk.chunk_id, []).append(name)

        return [
            FusedHit(
                chunk_id=doc_id,
                fused_score=score,
                chunk=chunk_index.get(doc_id),      # type: ignore[arg-type]
                sources=tuple(sources.get(doc_id, ())),
            )
            for doc_id, score in fused_pairs
        ]

    # ---------------- 重排序（可选） ----------------
    def _rerank_fused(self, query: str, fused: list[FusedHit]) -> list[FusedHit]:
        """对融合结果做 CrossEncoder 重排（装配了 rerank 端口时）。

        语义约束（docs/TDD_检索质量优化 §4.3）：
          · 只调整顺序，不增删条目——下游拿到的集合与无 rerank 时完全相同；
          · rerank 异常 → 原样返回（记 warning），问答链路不中断；
          · 无正文的候选（chunk=None）不送模型，按原序排在窗口尾部。

        Returns:
            与输入**同集合**的 FusedHit 列表（顺序可能不同）。
        """
        if self._rerank is None or len(fused) < 2:
            return fused

        window = fused[: self._rerank_candidate_k]
        tail = fused[len(window):]

        pairs: list[tuple[str, str]] = []
        no_text: list[FusedHit] = []
        for h in window:
            text = (h.chunk.text if h.chunk is not None else "").strip()
            if text:
                pairs.append((h.chunk_id, text))
            else:
                no_text.append(h)
        if len(pairs) < 2:
            return fused                            # 无可重排的有效文本

        try:
            ranked = self._rerank.rerank(query, pairs)
        except Exception as exc:              # noqa: BLE001 rerank 失败必须降级
            logger.warning("rerank 失败，保持融合原序: %s: %s",
                           type(exc).__name__, exc)
            return fused

        by_id = {h.chunk_id: h for h in window}
        out: list[FusedHit] = []
        seen: set[str] = set()
        for rh in ranked:
            h = by_id.get(rh.chunk_id)
            if h is not None and h.chunk_id not in seen:
                out.append(h)
                seen.add(h.chunk_id)
        # 防御性兜底（端口契约是不增不减）：端口没返回的候选——
        # 含全部 no_text 项——按窗口原序补回尾部。★ 不能再单独 += no_text，
        # 否则空正文候选会被加两次（测试抓过这个重复 bug）。
        for h in window:
            if h.chunk_id not in seen:
                out.append(h)
                seen.add(h.chunk_id)
        out += tail
        return out
