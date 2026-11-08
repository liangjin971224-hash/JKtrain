"""BM25 关键词检索适配器。

补的是一路「字面精确匹配」——图谱和向量都做不了
「国食健注G20250012」「康源牌鱼油软胶囊」这类编号/型号查询。

────────────────────────────────────────────────────────────────────
实现选择
────────────────────────────────────────────────────────────────────
不引入 rank_bm25 / jieba 等外部依赖，原因：
1. 语料只有 580 条、单机运行，纯 Python 足够快（实测毫秒级）
2. 少一个依赖就少一个版本冲突与部署风险
3. 排序逻辑自己写，行为完全可预期、可测试

中文分词策略：
    没有词典时，按 **2-gram + 单字数字/字母** 切分。
    2-gram 对中文短查询够用（"保健食品" → 保健/健食/食品），
    且对「国食健注G20250012」这类混排也能切出 `G2`/`20`/`00` 之类片段。
    代价是索引体积大，但 580 条的规模无所谓。
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter

from domain.ports import KeywordSearchPort
from domain.value_objects import ChunkMeta, KeywordHit

logger = logging.getLogger(__name__)

#: BM25 参数（业界默认值，未经本项目调优 —— 已在验收记录里标注「待确认」）
BM25_K1 = 1.5
BM25_B = 0.75

#: 字段权重：标题/章节命中的价值高于正文命中
#: 客户问「体脂仪怎么用」，标题里带「体脂仪」的段落比正文提一句的更相关
FIELD_WEIGHTS = {
    "title": 3.0,
    "section": 2.0,
    "text": 1.0,
}


# ===================================================================
# 分词
# ===================================================================
_CJK = re.compile(r"[\u4e00-\u9fff]")
_ASCII_TOKEN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """轻量分词。

    规则：
      · 中文字符 → 2-gram（单字若孤立出现也保留）
      · 连续字母/数字 → 整体保留，再补 2-gram（保证「G20250012」也能部分命中）

    不做去重，BM25 需要词频。
    """
    if not text:
        return []

    tokens: list[str] = []

    # 1. 字母/数字块整体保留 + 内部 2-gram
    for chunk in _ASCII_TOKEN.findall(text):
        low = chunk.lower()
        tokens.append(low)
        if len(low) > 2:
            tokens.extend(low[i:i + 2] for i in range(len(low) - 1))

    # 2. 中文 2-gram
    cjk_chars = [c for c in text if _CJK.match(c)]
    if len(cjk_chars) == 1:
        tokens.append(cjk_chars[0])
    else:
        tokens.extend(cjk_chars[i] + cjk_chars[i + 1]
                      for i in range(len(cjk_chars) - 1))

    return tokens


# ===================================================================
# 适配器
# ===================================================================
class Bm25SearchAdapter(KeywordSearchPort):
    """内存 BM25 索引。

    索引需要显式构建（`build_index`），因为关键词检索要预先统计
    文档频率与平均长度。构建一次即可复用。
    """

    def __init__(self) -> None:
        self._chunks: list[ChunkMeta] = []
        # 每条文档的分字段词频
        self._field_tfs: list[dict[str, Counter[str]]] = []
        # 每条文档的加权长度
        self._doc_lens: list[float] = []
        # 每个词出现在多少条文档里
        self._doc_freq: Counter[str] = Counter()
        self._avg_len: float = 0.0
        self._built: bool = False

    # ---------------- 索引 ----------------
    def build_index(self, chunks: list[object]) -> int:
        metas = [c for c in chunks if isinstance(c, ChunkMeta)]
        if not metas:
            logger.warning("build_index 收到空输入，索引为空")
            self._chunks = []
            self._field_tfs = []
            self._doc_lens = []
            self._doc_freq = Counter()
            self._avg_len = 0.0
            self._built = True
            return 0

        self._chunks = metas
        self._field_tfs = []
        self._doc_lens = []
        self._doc_freq = Counter()

        for meta in metas:
            per_field: dict[str, Counter[str]] = {}
            weighted_len = 0.0
            all_terms: set[str] = set()

            for field_name, weight in FIELD_WEIGHTS.items():
                raw = getattr(meta, field_name, "") or ""
                tf = Counter(tokenize(raw))
                per_field[field_name] = tf

                n = sum(tf.values())
                weighted_len += n * weight
                all_terms.update(tf.keys())

            self._field_tfs.append(per_field)
            self._doc_lens.append(weighted_len)
            self._doc_freq.update(all_terms)

        self._avg_len = (
            sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0
        )
        self._built = True
        logger.info(
            "BM25 索引构建完成: %d 条文档, %d 个词, 平均长度 %.1f",
            len(self._chunks), len(self._doc_freq), self._avg_len,
        )
        return len(self._chunks)

    # ---------------- 检索 ----------------
    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        allowed_permissions: tuple[str, ...] | None = None,
    ) -> list[KeywordHit]:
        text = (query or "").strip()
        if not text or top_k <= 0:
            return []

        if not self._built:
            logger.warning("BM25 索引尚未构建，返回空结果")
            return []

        # ★ 权限过滤在**打分之前**，与 VectorSearchAdapter 同语义：
        #   先筛出可见文档再算分，而不是算完分再筛。
        allowed: set[str] | None = None
        if allowed_permissions is not None:
            allowed = {str(p) for p in allowed_permissions}

        q_terms = Counter(tokenize(text))
        if not q_terms:
            return []

        n_docs = len(self._chunks)
        hits: list[KeywordHit] = []

        for idx, meta in enumerate(self._chunks):
            if allowed is not None and str(meta.permission_level) not in allowed:
                continue

            score = self._score(idx, q_terms, n_docs)
            if score > 0:
                hits.append(KeywordHit(chunk=meta, score=score))

        hits.sort(key=lambda h: h.score, reverse=True)
        out = hits[:top_k]

        logger.debug(
            "BM25 检索: query_len=%d, 命中=%d, 返回=%d",
            len(text), len(hits), len(out),
        )
        return out

    def _score(self, doc_idx: int, q_terms: Counter[str], n_docs: int) -> float:
        """单文档 BM25 得分（分字段加权）。"""
        doc_len = self._doc_lens[doc_idx] or 1.0
        per_field = self._field_tfs[doc_idx]

        score = 0.0
        for term, qtf in q_terms.items():
            df = self._doc_freq.get(term, 0)
            if df == 0:
                continue

            # IDF：标准 BM25 变体，加 1 防止负值（df > n/2 时 log 为负）
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

            # 分字段累加词频，标题权重更高
            tf = sum(
                per_field.get(field, Counter()).get(term, 0) * weight
                for field, weight in FIELD_WEIGHTS.items()
            )
            if tf == 0:
                continue

            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / self._avg_len)
            score += idf * (tf * (BM25_K1 + 1) / denom) * qtf

        return score
