"""组卷引擎（P6）。

★ 这是一个**纯函数**：输入题库 + 参数，输出试卷。不碰 IO、不依赖随机全局状态。
  好处是规则可以 100% 单测，而且给定 seed 完全可复现（补考、复核都需要）。

────────────────────────────────────────────────────────────────────
两个约束的优先级（这是设计的核心）
────────────────────────────────────────────────────────────────────
需求给的是两个约束，但它们会冲突：

    「组卷结果应满足知识点全覆盖」              ← 硬约束（文档明确写「应满足」）
    「难度配比：各难度等级的题目数量」          ← 软约束（配比可调）

冲突场景：要求 6 题、知识点 3 个、难度 1/2/3 各 2 题，
         但难度 1 只有 1 道。
若强行满足难度配比 → 某个知识点可能 0 题 → 违反覆盖。
若强行满足覆盖     → 难度配比凑不齐。

**取舍：保覆盖，难度降级并写进 warnings。**

理由：覆盖不到的知识点 = 这次考试没考那个模块，是**实质缺陷**；
      难度配比不准 = 卷子偏难或偏易，是**可接受的偏差**。

────────────────────────────────────────────────────────────────────
算法（分三步，每步都有明确目的）
────────────────────────────────────────────────────────────────────
第一步 **覆盖优先**：每个知识点先保 1 题。
    这一步保证硬约束，且优先挑「高优先级」的题（有产品关联 > 纯知识点）。

第二步 **填难度配额**：按 difficulty_quota 补齐，优先从已覆盖的知识点里挑，
    保持知识点分布均衡。

第三步 **补足题量**：还有剩余名额就继续抽，轮转知识点，避免某点过量。

把「覆盖」放第一步而不是最后校验，是因为最后校验发现问题时已经无法补救
（题目已选完，只能报错重来）。
"""
from __future__ import annotations

import logging
import random

from domain.exceptions import ExamBuildError, InvalidExamTransition
from domain.value_objects.exam import (
    ExamPaper,
    ExamSpec,
    ExamStatus,
    QuestionSnapshot,
)

logger = logging.getLogger(__name__)

#: 难度等级（1 易 / 2 中 / 3 难）
DIFFICULTIES = (1, 2, 3)


def build_paper(
    bank: list[QuestionSnapshot],
    spec: ExamSpec,
    *,
    seed: int | None = None,
    paper_no: str = "",
    created_at: str = "",
    created_by: str = "",
) -> ExamPaper:
    """按规则从题库组一份试卷。

    Args:
        bank: 候选题目（应已过滤掉不可用题，如未审核、权限不符）。
        spec: 组卷参数。
        seed: 随机种子。给定则结果可复现（补考/复核/测试需要）。
        paper_no: 卷号。为空时用占位符，由调用方补。
        created_at: 创建时间（ISO 字符串）。
        created_by: 创建人。

    Returns:
        试卷（草稿状态，题目为快照）。

    Raises:
        ExamBuildError: 题量不够、知识点无题可用。

    ★ 失败时一定说清缺什么 —— 见 `_fail` 的实现。
    """
    rng = random.Random(seed)

    # ---- 0. 过滤去重窗口内的题 ----
    available = [q for q in bank
                 if q.question_id not in spec.exclude_question_ids]

    if not available:
        raise ExamBuildError(
            "题库中没有可用题目"
            + (f"（已排除 {len(spec.exclude_question_ids)} 道窗口内用过的题）"
               if spec.exclude_question_ids else "")
        )

    # ---- 按知识点分组 ----
    by_kp: dict[str, list[QuestionSnapshot]] = {}
    for q in available:
        by_kp.setdefault(q.knowledge_point or "", []).append(q)

    # ---- 1. ★ 覆盖优先：每个知识点至少 1 题 ----
    required_kps = list(spec.knowledge_points) or list(by_kp)

    missing_kps = [kp for kp in required_kps if not by_kp.get(kp)]
    if missing_kps:
        raise ExamBuildError(_missing_kp_message(missing_kps, by_kp, spec))

    if len(required_kps) > spec.total:
        raise ExamBuildError(
            f"要求覆盖 {len(required_kps)} 个知识点，但题量只有 {spec.total} 题。"
            f"知识点全覆盖无法满足，请提高题量或减少覆盖要求。"
        )

    # ★★ 封闭候选池：只从**要求覆盖的知识点**里抽题
    #
    #   实测缺陷：原来从全库抽「补足题量」的题，
    #   结果试卷里出现了未被要求考察的知识点（如要求 KP_00/01，
    #   却抽到了 KP_02）。这会让考试范围失控 ——
    #   主管明明只要求考「功能声称边界」，卷子里却混进「接待流程」。
    #
    #   所以：一旦显式指定了 knowledge_points，候选池就**锁死在这些点内**。
    if spec.knowledge_points:
        required_set = set(required_kps)
        available = [q for q in available
                     if q.knowledge_point in required_set]
        by_kp = {k: v for k, v in by_kp.items() if k in required_set}

    picked: list[QuestionSnapshot] = []
    picked_ids: set[str] = set()
    warnings: list[str] = []

    def _take(q: QuestionSnapshot) -> None:
        picked.append(q)
        picked_ids.add(q.question_id)

    # 每个知识点先抽 1 题。
    #
    # ★ 挑选策略（实测两次修正）：
    #
    #   第一版：有产品关联优先，其余随机。
    #     问题：覆盖阶段随机把某难度抽光，导致配额凑不齐。
    #     实测：要求难度 3 出 7 道，14 个知识点各抽 1 题时随机到 3 道难度 3，
    #          配额只剩 4 道 → 报「难度不足」，而题库其实有 28 道难度 3。
    #
    #   第二版：按「题库量/配额」比例挑最紧缺的。
    #     问题：修反了。难度 3 比例 28/7=4、难度 1 比例 14/2=7，
    #          于是难度 3 被认为「最紧缺」→ 覆盖阶段全抽难度 3，
    #          结果 14 题全是难度 3，难度 1/2 一道没抽。
    #
    #   第三版（当前）：按**剩余配额**动态挑，且覆盖阶段只消耗「还差得多」的那档。
    #     每抽一题就更新剩余配额，天然不会把某一档吃光。
    #     做法见下方循环里的 `_pick_for_coverage`。
    for kp in required_kps:
        pool = by_kp.get(kp, [])
        if not pool:
            continue
        chosen = _pick_for_coverage(
            pool, rng, spec, picked, picked_ids)
        if chosen is not None:
            _take(chosen)

    # ---- 2. 按难度配额补齐 ----
    if spec.difficulty_quota:
        for diff in DIFFICULTIES:
            need = spec.difficulty_quota.get(diff, 0)
            have = sum(1 for q in picked if q.difficulty == diff)
            shortage = need - have

            if shortage <= 0:
                continue

            # 从全库找同难度的未选题，轮转知识点保持均衡
            candidates = [q for q in available
                          if q.difficulty == diff
                          and q.question_id not in picked_ids]
            candidates = _rotate_by_kp(candidates, picked, rng)

            for q in candidates[:shortage]:
                if len(picked) >= spec.total:
                    break
                _take(q)

            # 没凑够 → 记录警告（软约束，不失败）
            still = need - sum(1 for q in picked if q.difficulty == diff)
            if still > 0:
                warnings.append(
                    f"难度 {diff} 的题量不足：需要 {need} 道，"
                    f"实际只凑到 {need - still} 道（题库同难度仅 "
                    f"{sum(1 for q in available if q.difficulty == diff)} 道）"
                )

    # ---- 3. 补足题量（轮转知识点，避免某点过量） ----
    if len(picked) < spec.total:
        rest = [q for q in available if q.question_id not in picked_ids]
        rest = _rotate_by_kp(rest, picked, rng)
        for q in rest:
            if len(picked) >= spec.total:
                break
            _take(q)

    if len(picked) < spec.total:
        raise ExamBuildError(
            f"题量不足：需要 {spec.total} 道，可用仅 {len(picked)} 道。"
            f"题库共 {len(bank)} 道"
            + (f"，其中 {len(spec.exclude_question_ids)} 道在防重复窗口内已排除"
               if spec.exclude_question_ids else "")
            + "。请补充题库或缩短防重复窗口。"
        )

    # ---- 4. 产品覆盖检查（软约束） ----
    if spec.product_ids:
        covered_products = {q.product_id for q in picked if q.product_id}
        lack = [p for p in spec.product_ids if p not in covered_products]
        if lack:
            warnings.append(
                f"以下产品未覆盖到题目：{', '.join(lack)}"
                f"（题库中这些产品可能没有可用题）"
            )

    # ---- 5. 打乱顺序 + 选项乱序 ----
    questions = _finalize(picked, spec, rng)

    paper = ExamPaper(
        paper_no=paper_no or "(待分配)",
        stage=spec.stage,
        spec=spec,
        questions=tuple(questions),
        status=ExamStatus.DRAFT,
        created_at=created_at,
        created_by=created_by,
        warnings=tuple(warnings),
    )

    logger.info(
        "组卷完成: %s 题=%d 知识点=%d 难度=%s 警告=%d",
        paper.paper_no, paper.count,
        len(paper.covered_knowledge_points),
        paper.difficulty_distribution, len(warnings),
    )
    return paper


# ===================================================================
# 内部辅助
# ===================================================================
def _pick_for_coverage(
    pool: list[QuestionSnapshot],
    rng: random.Random,
    spec: ExamSpec,
    picked: list[QuestionSnapshot],
    picked_ids: set[str],
) -> QuestionSnapshot | None:
    """覆盖阶段选一道题，兼顾难度配额。

    ★ 核心思路：**优先消耗还剩得最多的那档配额**。

    例：配额 {1:2, 2:6, 3:7}，已选 {3:3}
        → 剩余 {1:2, 2:6, 3:4}
        → 难度 2 剩最多，优先抽难度 2

    这样覆盖阶段的 14 道题会自然按 2:6:4 的比例摊开，
    既不会把某一档吃光，也顺手完成大部分配额。
    """
    if not pool:
        return None

    shuffled = list(pool)
    rng.shuffle(shuffled)

    if not spec.difficulty_quota:
        # 无配额要求：有产品关联的优先，其次随机
        return sorted(shuffled, key=lambda q: 0 if q.product_id else 1)[0]

    # 算各难度的剩余配额
    taken: dict[int, int] = {}
    for q in picked:
        taken[q.difficulty] = taken.get(q.difficulty, 0) + 1

    remaining = {
        d: quota - taken.get(d, 0)
        for d, quota in spec.difficulty_quota.items()
    }

    def key(q: QuestionSnapshot) -> tuple[int, int]:
        # 剩余配额多的优先（取负数让它排前面）
        rem = remaining.get(q.difficulty, 0)
        # 配额已用完的难度排最后（但不会硬排除，题库不够时仍可用）
        exhausted = 0 if rem > 0 else 1
        return (exhausted, -rem, 0 if q.product_id else 1)

    return sorted(shuffled, key=key)[0]


def _sorted_for_pick(
    pool: list[QuestionSnapshot],
    rng: random.Random,
) -> list[QuestionSnapshot]:
    """把候选池排序（仅用于无难度配额的场景）。

    优先级：有产品关联 > 无产品关联；同级随机。
    """
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return sorted(shuffled, key=lambda q: 0 if q.product_id else 1)


def _rotate_by_kp(
    candidates: list[QuestionSnapshot],
    picked: list[QuestionSnapshot],
    rng: random.Random,
) -> list[QuestionSnapshot]:
    """按「当前知识点已选数量」排序，少选的优先。

    ★ 为什么要轮转，而不是纯随机：
      纯随机会导致某知识点抽到 5 题、另一个只 1 题，
      考完一看「全是功能声称」，分布很不合理。
      按已选数量升序，天然均衡。
    """
    counts: dict[str, int] = {}
    for q in picked:
        counts[q.knowledge_point] = counts.get(q.knowledge_point, 0) + 1

    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return sorted(shuffled, key=lambda q: counts.get(q.knowledge_point, 0))


def _finalize(
    picked: list[QuestionSnapshot],
    spec: ExamSpec,
    rng: random.Random,
) -> list[QuestionSnapshot]:
    """打乱题序 + 按需打乱选项。"""
    ordered = list(picked)
    rng.shuffle(ordered)

    if not spec.shuffle_options:
        return ordered

    return [_shuffle_options(q, rng) for q in ordered]


def _shuffle_options(
    q: QuestionSnapshot, rng: random.Random
) -> QuestionSnapshot:
    """打乱选项顺序，并**同步更新答案**。

    ★ 这是最容易写错的地方：打乱了选项却忘了更新答案，
      会导致学员选对了被判错。

    做法：
        1. 记住正确答案的**文本**（不是字母）
        2. 打乱选项
        3. 按文本找回新位置，重新算字母
    """
    if len(q.options) <= 1:
        return q

    correct_text = q.answer_text
    if not correct_text:
        return q                     # 答案无效，不打乱（避免制造错误）

    opts = list(q.options)
    rng.shuffle(opts)

    try:
        new_idx = opts.index(correct_text)
    except ValueError:
        return q                     # 答案文本不在选项里，不动

    if q.is_multi:
        # 多选：逐个答案文本找新位置，排序后拼字母
        idxs = []
        for token in _split_answer(q.answer):
            text = q.options[_answer_idx(token)]
            if text in opts:
                idxs.append(opts.index(text))
        new_answer = "".join(sorted(chr(ord("A") + i) for i in sorted(idxs)))
    else:
        new_answer = chr(ord("A") + new_idx)

    return QuestionSnapshot(
        question_id=q.question_id,
        knowledge_point=q.knowledge_point,
        kp_name=q.kp_name,
        question_type=q.question_type,
        difficulty=q.difficulty,
        stem=q.stem,
        options=tuple(opts),
        answer=new_answer,
        explanation=q.explanation,
        product_id=q.product_id,
    )


def _split_answer(answer: str) -> list[str]:
    """把 "ABD" 拆成 ["A","B","D"]。"""
    return [c for c in (answer or "").upper() if c.isalpha()]


def _answer_idx(token: str) -> int:
    return ord(token.upper()) - ord("A")


def _missing_kp_message(
    missing: list[str],
    by_kp: dict[str, list[QuestionSnapshot]],
    spec: ExamSpec,
) -> str:
    """组装「缺哪些知识点」的详细报错。

    ★ 需求要求「明确提示原因」，所以要说清：
      · 缺哪几个知识点
      · 现有题库有哪些知识点（便于对比发现是编码写错还是真没题）
    """
    have = sorted(k for k in by_kp if k)
    return (
        f"以下知识点在题库中没有可用题目：{', '.join(missing)}。\n"
        f"要求覆盖：{', '.join(spec.knowledge_points)}；"
        f"题库现有知识点：{', '.join(have) if have else '（空）'}。\n"
        f"请检查知识点编码是否写错，或补充对应题目。"
    )


# ===================================================================
# 状态流转
# ===================================================================
def transition(current: ExamStatus, target: ExamStatus) -> ExamStatus:
    """试卷状态流转（FR-040：单向，不可回退）。

    Args:
        current: 当前状态。
        target: 目标状态。

    Returns:
        目标状态。

    Raises:
        InvalidExamTransition: 流转不被允许。
    """
    if not current.can_transition_to(target):
        raise InvalidExamTransition(
            f"试卷状态不能从「{current.label}」变为「{target.label}」。"
            f"允许的流转：草稿 → 生效 → 归档（单向，不可回退）。"
        )
    return target


__all__ = ["build_paper", "transition", "DIFFICULTIES"]
