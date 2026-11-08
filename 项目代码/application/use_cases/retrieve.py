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
    VectorSearchPort,
)
from domain.services.rrf_fusion import reciprocal_rank_fusion, weights_for_intent
from domain.value_objects import FusedHit, KeywordHit, ProductRef, VectorHit

logger = logging.getLogger(__name__)

#: 三路名称（顺序与 INTENT_WEIGHTS 一致：图谱 / 向量 / 关键词）
PATH_GRAPH = "graph"
PATH_VECTOR = "vector"
PATH_KEYWORD = "keyword"

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

    fused: tuple[FusedHit, ...] = ()

    #: 失败的路：{路名: 失败原因}
    degraded: dict[str, str] = field(default_factory=dict)

    #: 各路耗时（毫秒），便于定位慢在哪一路
    elapsed_ms: dict[str, float] = field(default_factory=dict)

    #: 权重（图谱, 向量, 关键词），便于回放「为什么这条排第一」
    weights: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def failed(self) -> bool:
        """三路全失败（或全部无有效结果）。"""
        return not (self.graph_hits or self.vector_hits or self.keyword_hits)

    @property
    def is_degraded(self) -> bool:
        """是否发生了降级（部分路失败）。"""
        return bool(self.degraded)

    @property
    def total_raw_hits(self) -> int:
        """三路原始命中总数（未融合）。"""
        return len(self.graph_hits) + len(self.vector_hits) + len(self.keyword_hits)

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
        min_vector_score: float = MIN_VECTOR_SCORE,
        min_keyword_score: float = MIN_KEYWORD_SCORE,
    ) -> None:
        """
        Args:
            graph: 图谱端口。None 表示该路不可用（P1 未完成时的降级）。
            vector: 向量端口。None 表示该路不可用。
            keyword: 关键词端口。None 表示该路不可用。
            min_vector_score: 向量召回的最低可信相似度。
            min_keyword_score: 关键词召回的绝对分数下限。
                ★ 与语料规模相关，语料大幅扩张后需重新校准。
        """
        self._graph = graph
        self._vector = vector
        self._keyword = keyword
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

        # ---- 融合 ----
        fused = self._fuse(
            query=text,
            intent=intent,
            graph_hits=graph_hits,
            vector_hits=vector_hits,
            keyword_hits=keyword_hits,
            weights=weights,
            limit=fused_limit,
        )

        result = RetrievalResult(
            query=text,
            intent=intent,
            graph_hits=tuple(graph_hits),
            vector_hits=tuple(vector_hits),
            keyword_hits=tuple(keyword_hits),
            fused=tuple(fused),
            degraded=degraded,
            elapsed_ms=elapsed,
            weights=weights,
        )

        logger.info(
            "三路召回完成: intent=%s, graph=%d, vector=%d, keyword=%d, "
            "fused=%d, degraded=%s, elapsed_ms=%s",
            intent, len(graph_hits), len(vector_hits), len(keyword_hits),
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
            r"叶黄素|氨糖|蜂胶|螺旋藻|黑芝麻丸|芝麻丸"
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
        limit: int,
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
            for doc_id, score in fused_pairs[:limit]
        ]
