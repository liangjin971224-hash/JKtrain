"""RRF（Reciprocal Rank Fusion）多路召回融合。

纯领域服务：**零框架、零 IO、零外部依赖**，可秒级全测。

────────────────────────────────────────────────────────────────────
公式
────────────────────────────────────────────────────────────────────
    score(d) = Σ_i  w_i / (k + rank_i(d))

    rank_i(d) 是文档 d 在第 i 路的排名（从 1 开始，不是 0）
    k         是平滑常数，默认 60（业界惯例，来自 Cormack et al. 2009）
    w_i       是第 i 路的权重，按查询意图动态调整

为什么用 RRF 而不是分数加权：
    三路的分数**不同量纲**——图谱是布尔/离散，向量是余弦相似度（0–1），
    BM25 是无上界实数。直接加权求和需要先归一化，而归一化本身会引入偏差
    （比如某一路这次最高分只有 0.4，归一化后会被放大到 1.0）。
    RRF 只用**排名**，天然规避量纲问题，且对异常分数鲁棒。

────────────────────────────────────────────────────────────────────
置信度标注
────────────────────────────────────────────────────────────────────
- 公式与 k=60：**已确认**（业界标准做法）
- 三路权重表：**稳定惯例**（来自 `开发指导/03_项目开发指导.doc` 的多路召回章节，
  按本项目业务适配）。权重值可调，但「安全类提高图谱权重」这个方向不变。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: 平滑常数。业界惯例 60，作用是让排名靠后的文档分数不至于衰减过快。
DEFAULT_K = 60


# ===================================================================
# 意图 → 三路权重
# ===================================================================
#: 权重顺序固定为 (图谱, 向量, 关键词)。
#:
#: 为什么权重必须按意图调整：
#:     客户问「在服抗凝药能吃鱼油吗」，答案在**图谱的禁忌边**上，
#:     向量可能召回一堆讲鱼油好处的段落 —— 此时图谱权重高才不会答错。
#:     而问「接待流程第一步」，答案只在**培训文档正文**里，图谱帮不上忙。
#:
#: 每组权重和必须为 1.0，否则不同意图的融合分数不可比。
INTENT_WEIGHTS: dict[str, tuple[float, float, float]] = {
    # 产品功效 / 禁忌：图谱有精确关系
    "产品功效": (0.5, 0.3, 0.2),
    "禁忌咨询": (0.6, 0.2, 0.2),
    # 安全相关：宁可不精确召回，也不能漏禁忌
    "特殊人群": (0.6, 0.2, 0.2),
    "要停药": (0.5, 0.3, 0.2),
    "要诊断": (0.5, 0.2, 0.3),
    "要处方": (0.5, 0.3, 0.2),
    "超量服用": (0.5, 0.3, 0.2),
    # 业务流程咨询：全靠文本
    "业务流程": (0.1, 0.7, 0.2),
    "设备使用": (0.2, 0.7, 0.1),
    # 产品名 / 文号查询：编号靠精确匹配
    "产品名查询": (0.3, 0.2, 0.5),
    "文号查询": (0.2, 0.2, 0.6),
    # 开放式问答：语义为主
    "开放问答": (0.1, 0.8, 0.1),
    # 应急预案：文档为主，图谱有禁忌线索
    "应急处理": (0.3, 0.6, 0.1),
}

#: 未知意图的兜底权重。
#:
#: ★ 2026-10-23 改为「开放问答」(0.1, 0.8, 0.1)。
#:   原先取「产品功效」(0.5, 0.3, 0.2)，但主链路里空串会先被转成
#:   「开放问答」，两处兜底不一致 —— 看代码的人会以为默认是 (0.5,0.3,0.2)，
#:   实际跑的是 (0.1,0.8,0.1)。**同一件事只能有一个兜底。**
#:
#:   为什么选开放问答而不是产品功效：
#:   兜底场景是「不知道用户问什么」，此时应让语义（向量）主导；
#:   强行提高图谱权重等于在不知道问什么的情况下假设「问的是产品」。
DEFAULT_WEIGHTS: tuple[float, float, float] = INTENT_WEIGHTS["开放问答"]


def weights_for_intent(intent: str) -> tuple[float, float, float]:
    """按意图取三路权重。

    Args:
        intent: 意图标签。意图分类可能出错，所以**未知意图不抛异常**，
            而是回退到默认权重并打日志。

    Returns:
        (图谱权重, 向量权重, 关键词权重)，和为 1.0。
    """
    weights = INTENT_WEIGHTS.get(intent)
    if weights is None:
        logger.debug("未知意图「%s」，使用默认权重 %s", intent, DEFAULT_WEIGHTS)
        return DEFAULT_WEIGHTS
    return weights


# ===================================================================
# RRF 融合
# ===================================================================
def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    weights: list[float] | tuple[float, ...] | None = None,
    k: int = DEFAULT_K,
) -> list[tuple[str, float]]:
    """融合多路召回结果。

    Args:
        rankings: 每路的 doc_id 有序列表（已按相关度**降序**，最相关的在索引 0）。
            允许某些路为空列表（该路无结果或降级）。
        weights: 各路权重，长度须与 rankings 一致。None 则等权。
        k: 平滑常数，默认 60。

    Returns:
        (doc_id, score) 降序列表。三路全空时返回空列表
        （P4 据此判定「召回为空」并拒答，不调 LLM）。

    Raises:
        ValueError: 权重数量不符 / 含负值 / 全零 / k 非正。

    Note:
        同一路内重复出现的文档只按**首次出现**的排名计分。
        否则一路里重复 10 次就能把分数刷上去。
    """
    if not rankings:
        return []

    if k <= 0:
        raise ValueError(f"平滑常数 k 必须为正数，当前 {k}")

    n_paths = len(rankings)
    if weights is None:
        w = [1.0] * n_paths
    else:
        w = [float(x) for x in weights]
        if len(w) != n_paths:
            raise ValueError(
                f"权重数量({len(w)})与召回路径数量({n_paths})不一致。"
                f"静默按最短对齐会导致某一路权重被丢弃，必须显式报错。"
            )
        if any(x < 0 for x in w):
            raise ValueError(f"权重不能为负（负权重会让排名反转）: {w}")
        if all(x == 0 for x in w):
            raise ValueError(f"权重不能全为 0（等于没有信号）: {w}")

    scores: dict[str, float] = {}

    for path_idx, ranking in enumerate(rankings):
        weight = w[path_idx]
        if weight == 0:
            continue                       # 权重为 0 的路跳过，省一次遍历

        seen_in_this_path: set[str] = set()
        for rank0, doc_id in enumerate(ranking):
            if doc_id in seen_in_this_path:
                # 同一路内重复：只认首次排名，不重复计分
                continue
            seen_in_this_path.add(doc_id)

            # rank 从 1 开始，与公式一致
            rank = rank0 + 1
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)

    # 按分数降序；分数相同时按 doc_id 保证输出稳定（可复现）
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
