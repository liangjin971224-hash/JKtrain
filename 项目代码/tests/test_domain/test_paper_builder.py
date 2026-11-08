"""组卷规则测试（P6）。

★ 需求来源：FR-037 / FR-038 / FR-039

────────────────────────────────────────────────────────────────────
四条硬指标（文档 391 行明确列为验收标准）
────────────────────────────────────────────────────────────────────
1. **知识点覆盖 100%** —— 每个知识点至少抽到 1 题
2. **30 天防重复生效** —— 同一学员窗口内不重复抽到同一题
3. **试卷快照可追溯** —— 改题库不影响历史试卷
4. **卷号唯一且可读** —— 形如 `S1-20261105-001`

────────────────────────────────────────────────────────────────────
设计上的三个关键取舍
────────────────────────────────────────────────────────────────────
① 组卷是**纯函数**（输入题库 + 参数 → 输出试卷），不碰 IO。
   持久化由 RepositoryPort 负责，这样规则可 100% 测试。

② 知识点覆盖**优先于**难度配比。
   需求原文：「组卷结果应满足知识点全覆盖」——覆盖是硬约束，
   难度是可调配的。冲突时保覆盖，并在 warnings 里报告。

③ 组卷失败必须**说清缺什么**。
   需求原文：「组卷失败时应明确提示原因（缺少哪个知识点、哪个难度的题目）」。
   只返回「失败」等于没说，管理员无法补题。
"""
from __future__ import annotations

import pytest

from domain.value_objects import (
    ExamSpec,
    ExamStatus,
    QuestionSnapshot,
    make_paper_no,
)


# ===================================================================
# 夹具：构造题库
# ===================================================================
def _q(
    qid: str,
    kp: str = "KP_CLAIM_BOUNDARY",
    difficulty: int = 2,
    qtype: str = "单选",
    product_id: str = "",
) -> QuestionSnapshot:
    return QuestionSnapshot(
        question_id=qid,
        knowledge_point=kp,
        kp_name=f"知识点{kp}",
        question_type=qtype,
        difficulty=difficulty,
        stem=f"{qid} 的题干内容",
        options=("A 选项", "B 选项", "C 选项", "D 选项"),
        answer="A",
        explanation=f"{qid} 的解析",
        product_id=product_id,
    )


def _bank(n_per_kp: int = 5, kps: int = 3) -> list[QuestionSnapshot]:
    """构造 n_per_kp × kps 的题库，难度 1/2/3 均分。"""
    out: list[QuestionSnapshot] = []
    for k in range(kps):
        for i in range(n_per_kp):
            out.append(_q(
                qid=f"Q{k:02d}{i:02d}",
                kp=f"KP_{k:02d}",
                difficulty=(i % 3) + 1,
            ))
    return out


# ===================================================================
# 1. ★ 知识点全覆盖
# ===================================================================
class TestKnowledgeCoverage:
    def test_every_kp_covered(self) -> None:
        """每个知识点至少抽到 1 题。"""
        bank = _bank(n_per_kp=5, kps=4)
        spec = ExamSpec(total=8, knowledge_points=("KP_00", "KP_01",
                                                   "KP_02", "KP_03"))

        from domain.services.paper_builder import build_paper

        paper = build_paper(bank, spec)

        covered = {q.knowledge_point for q in paper.questions}
        assert covered == set(spec.knowledge_points), (
            f"知识点未全覆盖，缺 {set(spec.knowledge_points) - covered}"
        )

    def test_coverage_reported(self) -> None:
        """试卷要能报告覆盖率（主管要看「这次考了哪些点」）。"""
        from domain.services.paper_builder import build_paper

        paper = build_paper(_bank(kps=3),
                            ExamSpec(total=6, knowledge_points=("KP_00", "KP_01")))
        assert paper.covered_knowledge_points
        assert len(paper.covered_knowledge_points) == 2

    def test_missing_kp_raises_with_detail(self) -> None:
        """★ 某知识点题库为空 → 必须报出**具体缺哪个知识点**。

        需求：「组卷失败时应明确提示原因」。
        只说「失败」等于没说，管理员不知道补什么。
        """
        from domain.exceptions import ExamBuildError
        from domain.services.paper_builder import build_paper

        bank = _bank(kps=2)          # 只有 KP_00 / KP_01
        spec = ExamSpec(total=5, knowledge_points=("KP_00", "KP_99"))

        with pytest.raises(ExamBuildError) as ei:
            build_paper(bank, spec)

        msg = str(ei.value)
        assert "KP_99" in msg, f"错误信息没指出缺哪个知识点: {msg}"


# ===================================================================
# 2. 难度配比
# ===================================================================
class TestDifficultyMix:
    def test_difficulty_quota_respected(self) -> None:
        """指定的难度配比要被满足。"""
        from domain.services.paper_builder import build_paper

        bank = _bank(n_per_kp=9, kps=2)
        spec = ExamSpec(
            total=6,
            knowledge_points=("KP_00", "KP_01"),
            difficulty_quota={1: 2, 2: 2, 3: 2},
        )
        paper = build_paper(bank, spec)

        dist: dict[int, int] = {}
        for q in paper.questions:
            dist[q.difficulty] = dist.get(q.difficulty, 0) + 1

        assert dist.get(1, 0) == 2, f"难度 1 应有 2 题，实际 {dist}"
        assert dist.get(2, 0) == 2, f"难度 2 应有 2 题，实际 {dist}"
        assert dist.get(3, 0) == 2, f"难度 3 应有 2 题，实际 {dist}"

    def test_difficulty_shortage_warns_not_fails(self) -> None:
        """★ 难度配比不足 → **降级并警告**，不直接失败。

        知识点覆盖是硬约束，难度是软约束。
        难度凑不齐时应该先用题库里有的题顶上，同时在 warnings 里说明。
        """
        from domain.services.paper_builder import build_paper

        # 只有难度 2 的题
        bank = [_q(f"Q{i:02d}", kp="KP_00", difficulty=2) for i in range(6)]
        spec = ExamSpec(
            total=3,
            knowledge_points=("KP_00",),
            difficulty_quota={1: 1, 2: 1, 3: 1},
        )
        paper = build_paper(bank, spec)

        assert len(paper.questions) == 3, "应凑够题量"
        assert paper.warnings, "难度缺口应产生警告"
        assert any("难度" in w for w in paper.warnings)


# ===================================================================
# 3. ★ 防重复窗口
# ===================================================================
class TestDedup:
    def test_recently_used_excluded(self) -> None:
        """窗口内用过的题不能再抽。"""
        from domain.services.paper_builder import build_paper

        bank = _bank(n_per_kp=5, kps=1)     # Q0000..Q0004
        used = {"Q0000", "Q0001", "Q0002"}

        spec = ExamSpec(
            total=2, knowledge_points=("KP_00",), exclude_question_ids=used)
        paper = build_paper(bank, spec)

        picked = {q.question_id for q in paper.questions}
        assert not (picked & used), f"抽到了窗口内已用的题: {picked & used}"

    def test_insufficient_after_dedup_raises(self) -> None:
        """★ 去重后题量不足 → 报错并指出原因。"""
        from domain.exceptions import ExamBuildError
        from domain.services.paper_builder import build_paper

        bank = _bank(n_per_kp=3, kps=1)     # Q0000..Q0002
        used = {"Q0000", "Q0001"}           # 只剩 1 题

        spec = ExamSpec(
            total=3, knowledge_points=("KP_00",), exclude_question_ids=used)

        with pytest.raises(ExamBuildError) as ei:
            build_paper(bank, spec)
        msg = str(ei.value)
        assert "不足" in msg or "不够" in msg
        # 报错要说明「有题被去重排除了」，否则管理员不知道是题库少还是窗口太长
        assert "防重复" in msg or "排除" in msg


# ===================================================================
# 4. 选项乱序
# ===================================================================
class TestOptionShuffle:
    def test_options_shuffled_but_answer_follows(self) -> None:
        """★ 选项顺序打乱，但正确答案必须跟着走。

        这是最容易写错的地方：打乱了选项却忘了更新 answer，
        会导致学员选对了被判错。
        """
        from domain.services.paper_builder import build_paper

        bank = [_q(f"Q{i:02d}", kp="KP_00") for i in range(3)]
        spec = ExamSpec(total=2, knowledge_points=("KP_00",), shuffle_options=True)
        paper = build_paper(bank, spec, seed=42)

        for q in paper.questions:
            # 答案文本必须仍在选项中
            assert q.answer_text in q.options, (
                f"打乱后正确答案不在选项里: {q.answer_text} not in {q.options}"
            )
            # 答案字母要对上新位置
            assert q.options[q.answer_index] == q.answer_text

    def test_shuffle_is_deterministic_with_seed(self) -> None:
        """给定 seed 必须可复现（否则补考/复核时对不上）。"""
        from domain.services.paper_builder import build_paper

        bank = _bank(n_per_kp=5, kps=1)
        spec = ExamSpec(total=3, knowledge_points=("KP_00",), shuffle_options=True)

        p1 = build_paper(bank, spec, seed=7)
        p2 = build_paper(bank, spec, seed=7)

        assert [q.question_id for q in p1.questions] == \
               [q.question_id for q in p2.questions]
        assert [q.options for q in p1.questions] == \
               [q.options for q in p2.questions]


# ===================================================================
# 5. ★ 卷号
# ===================================================================
class TestPaperNo:
    def test_format(self) -> None:
        """卷号形如 S1-20261105-001（阶段-日期-序号）。"""
        no = make_paper_no(stage="S1", date="20261105", seq=1)
        assert no == "S1-20261105-001"

    def test_seq_padded(self) -> None:
        assert make_paper_no("S1", "20261105", 42) == "S1-20261105-042"
        assert make_paper_no("S1", "20261105", 999) == "S1-20261105-999"

    def test_invalid_seq(self) -> None:
        with pytest.raises(ValueError):
            make_paper_no("S1", "20261105", 0)
        with pytest.raises(ValueError):
            make_paper_no("S1", "20261105", 1000)

    def test_empty_stage_rejected(self) -> None:
        with pytest.raises(ValueError, match="阶段"):
            make_paper_no("", "20261105", 1)


# ===================================================================
# 6. ★ 快照（改题库不影响历史试卷）
# ===================================================================
class TestSnapshot:
    def test_paper_holds_copy_not_reference(self) -> None:
        """★ 试卷必须存题目**副本**，不是引用。

        这是 FR-039 的硬要求：
        「即使题库中的题目被修改或删除，历史试卷的题目内容不得变化」
        """
        from domain.services.paper_builder import build_paper

        bank = [_q(f"Q{i:02d}", kp="KP_00") for i in range(3)]
        spec = ExamSpec(total=2, knowledge_points=("KP_00",))
        paper = build_paper(bank, spec)

        # 模拟：题库里的题目被改
        original_stem = paper.questions[0].stem
        # QuestionSnapshot 是 frozen 的，改不了 —— 这本身就是保障
        with pytest.raises(Exception):
            paper.questions[0].stem = "被篡改的题干"  # type: ignore[misc]

        assert paper.questions[0].stem == original_stem

    def test_snapshot_has_full_content(self) -> None:
        """快照要含题干、选项、答案、解析的完整副本。"""
        from domain.services.paper_builder import build_paper

        bank = [_q("Q00", kp="KP_00")]
        paper = build_paper(bank, ExamSpec(total=1, knowledge_points=("KP_00",)))
        q = paper.questions[0]

        assert q.stem
        assert q.options
        assert q.answer
        assert q.explanation


# ===================================================================
# 7. 试卷状态流转
# ===================================================================
class TestExamStatus:
    def test_forward_only(self) -> None:
        """状态单向：草稿 → 生效 → 归档，不可回退。"""
        assert ExamStatus.DRAFT.can_transition_to(ExamStatus.ACTIVE)
        assert ExamStatus.ACTIVE.can_transition_to(ExamStatus.ARCHIVED)
        assert not ExamStatus.ARCHIVED.can_transition_to(ExamStatus.ACTIVE)
        assert not ExamStatus.ACTIVE.can_transition_to(ExamStatus.DRAFT)

    def test_active_not_editable(self) -> None:
        """★ 生效后不可改题目。"""
        assert ExamStatus.DRAFT.is_editable
        assert not ExamStatus.ACTIVE.is_editable
        assert not ExamStatus.ARCHIVED.is_editable

    def test_active_can_export(self) -> None:
        """只有生效/归档可导出（草稿还在变，导出没意义）。"""
        assert not ExamStatus.DRAFT.can_export
        assert ExamStatus.ACTIVE.can_export
        assert ExamStatus.ARCHIVED.can_export

    def test_invalid_transition_raises(self) -> None:
        from domain.exceptions import InvalidExamTransition
        from domain.services.paper_builder import transition

        with pytest.raises(InvalidExamTransition):
            transition(ExamStatus.ARCHIVED, ExamStatus.ACTIVE)


# ===================================================================
# 8. ★ 不得越界抽题（实测缺陷）
# ===================================================================
class TestNoOutOfScopeQuestions:
    """指定了知识点就**只从这些知识点里抽**，不能夹带别的。

    ★ 实测缺陷：要求覆盖 KP_00/KP_01，组出来的卷子里却出现了 KP_02。
      原因：「补足题量」那一步从全库抽题，没限制在要求范围内。

    后果很实际：主管只要求考「功能声称边界」，卷子里却混进「接待流程」，
    考出来的结果无法反映他关心的模块。
    """

    def test_only_required_kps_selected(self) -> None:
        from domain.services.paper_builder import build_paper

        bank = _bank(n_per_kp=5, kps=3)      # KP_00 / KP_01 / KP_02
        spec = ExamSpec(total=6, knowledge_points=("KP_00", "KP_01"))
        paper = build_paper(bank, spec, seed=1)

        got = {q.knowledge_point for q in paper.questions}
        assert got <= {"KP_00", "KP_01"}, (
            f"抽到了未要求的知识点: {got - {'KP_00', 'KP_01'}}"
        )
        assert paper.count == 6

    def test_no_kp_specified_uses_whole_bank(self) -> None:
        """不指定知识点时（自动全覆盖）可以用全部题库。"""
        from domain.services.paper_builder import build_paper

        bank = _bank(n_per_kp=3, kps=3)
        spec = ExamSpec(total=6)             # 不传 knowledge_points
        paper = build_paper(bank, spec, seed=1)

        assert paper.count == 6
        # 应该覆盖到全部 3 个知识点（自动全覆盖语义）
        assert len(paper.covered_knowledge_points) == 3

    def test_insufficient_in_scope_raises(self) -> None:
        """要求的范围内题量不够 → 报错，不能从范围外补。"""
        from domain.exceptions import ExamBuildError
        from domain.services.paper_builder import build_paper

        # KP_00 只有 2 题，KP_01 有 5 题；要求只在 KP_00 出 4 题
        bank = [_q(f"A{i}", kp="KP_00") for i in range(2)]
        bank += [_q(f"B{i}", kp="KP_01") for i in range(5)]
        spec = ExamSpec(total=4, knowledge_points=("KP_00",))

        with pytest.raises(ExamBuildError) as ei:
            build_paper(bank, spec)
        assert "不足" in str(ei.value)


# ===================================================================
# 9. ★ 覆盖阶段不得吃掉难度配额（实测连错两次）
# ===================================================================
class TestCoverageDoesNotEatQuota:
    """「每知识点保 1 题」不能把某个难度抽光。

    ★ 实测两次修错：
      第一版：覆盖阶段随机抽 → 随机到 3 道难度 3，配额只剩 4 道，
             报「难度 3 不足」，而题库有 28 道难度 3。
      第二版：按「题库量/配额」比例挑最紧缺 → 修反了，
             难度 3 比例 28/7=4 被当成最紧缺，14 题全抽成难度 3，
             难度 1/2 一道没抽。
      第三版：按**剩余配额**动态挑（剩得多的优先消耗）→ 正确。

    这个测试锁住第三版的行为。
    """

    def _bank(self):
        bank = []
        for i in range(14):
            kp = f"KP_{i:02d}"
            for j in range(5):
                bank.append(_q(f"{kp}_d2_{j}", kp, 2))
            for j in range(2):
                bank.append(_q(f"{kp}_d3_{j}", kp, 3))
            bank.append(_q(f"{kp}_d1", kp, 1))
        return bank

    def test_quota_respected_with_full_coverage(self) -> None:
        from domain.services.paper_builder import build_paper

        bank = self._bank()
        kps = tuple(f"KP_{i:02d}" for i in range(14))
        spec = ExamSpec(
            total=15,
            knowledge_points=kps,
            difficulty_quota={1: 2, 2: 6, 3: 7},
        )

        for seed in (99, 1, 7, 2026):
            paper = build_paper(bank, spec, seed=seed)
            dist = paper.difficulty_distribution

            assert dist.get(1, 0) == 2, f"seed={seed} 难度1={dist}"
            assert dist.get(2, 0) == 6, f"seed={seed} 难度2={dist}"
            assert dist.get(3, 0) == 7, f"seed={seed} 难度3={dist}"
            assert len(paper.covered_knowledge_points) == 14
            assert not paper.warnings, f"seed={seed} 不该有警告: {paper.warnings}"

    def test_no_difficulty_starved(self) -> None:
        """★ 不能出现「某一档一道都没抽」的极端情况。"""
        from domain.services.paper_builder import build_paper

        bank = self._bank()
        kps = tuple(f"KP_{i:02d}" for i in range(14))
        spec = ExamSpec(total=15, knowledge_points=kps,
                        difficulty_quota={1: 2, 2: 6, 3: 7})

        paper = build_paper(bank, spec, seed=99)
        for d in (1, 2, 3):
            assert paper.difficulty_distribution.get(d, 0) > 0, (
                f"难度 {d} 一道都没抽到: {paper.difficulty_distribution}"
            )
