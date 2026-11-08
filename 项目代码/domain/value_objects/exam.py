"""试卷与题目快照（P6）。

★ 核心设计：**试卷存快照，不存引用**。

需求 FR-039 原文：
    「必须保存题目快照，而非仅保存题目编号；
      即使题库中的题目被修改或删除，历史试卷的题目内容不得变化。」

所以 `QuestionSnapshot` 是 `frozen=True` 的完整副本（题干、选项、答案、解析全带），
不做「按 question_id 回查题库」这种事 —— 那样题库一改，历史试卷就变了。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


# ===================================================================
# 题目快照
# ===================================================================
@dataclass(frozen=True, slots=True)
class QuestionSnapshot:
    """一道题的完整快照。

    ★ 全部字段 frozen：试卷一旦生成，内容不可变。
      这是「快照可追溯」的技术保障——不是靠流程约束，是类型系统保证。
    """

    question_id: str
    #: 知识点编码（组卷覆盖的依据）
    knowledge_point: str = ""
    kp_name: str = ""
    #: 题型：单选 / 多选
    question_type: str = "单选"
    #: 难度 1–3
    difficulty: int = 2

    stem: str = ""
    #: 选项（打乱后的顺序）
    options: tuple[str, ...] = ()
    #: 正确答案（字母，如 "A" / "AB"）
    answer: str = ""
    explanation: str = ""
    #: 关联产品（用于「产品覆盖」规则）
    product_id: str = ""

    def __post_init__(self) -> None:
        if not self.question_id:
            raise ValueError("题目编号不能为空")
        if not 1 <= self.difficulty <= 3:
            raise ValueError(f"难度必须在 1–3 之间，实际 {self.difficulty}")

    @property
    def answer_text(self) -> str:
        """正确答案对应的**选项文本**。

        ★ 为什么要这个：选项打乱后，「答案 A」这个字母本身没意义了，
          要按字母找回原文再重新定位。
        """
        idx = _answer_index(self.answer)
        if 0 <= idx < len(self.options):
            return self.options[idx]
        return ""

    @property
    def answer_index(self) -> int:
        """正确答案对应的**当前选项下标**。"""
        return _answer_index(self.answer)

    @property
    def is_multi(self) -> bool:
        return self.question_type == "多选" or len(self.answer) > 1


def _answer_index(answer: str) -> int:
    """把答案字母转成下标（A→0, B→1…）。"""
    a = (answer or "").strip().upper()
    if not a:
        return -1
    return ord(a[0]) - ord("A")


# ===================================================================
# 试卷状态
# ===================================================================
class ExamStatus(StrEnum):
    """试卷状态（FR-040）。

    ★ 单向流转，不可回退：
        草稿 → 生效 → 归档
    """

    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"

    @property
    def label(self) -> str:
        return {"draft": "草稿", "active": "生效", "archived": "归档"}[self.value]

    def can_transition_to(self, target: ExamStatus) -> bool:
        """是否允许流转到目标状态。"""
        allowed: dict[ExamStatus, tuple[ExamStatus, ...]] = {
            ExamStatus.DRAFT: (ExamStatus.ACTIVE,),
            ExamStatus.ACTIVE: (ExamStatus.ARCHIVED,),
            ExamStatus.ARCHIVED: (),
        }
        return target in allowed[self]

    @property
    def is_editable(self) -> bool:
        """★ 只有草稿可改题目。

        需求原文：「生效后修改题目应被拒绝并提示原因」。
        """
        return self is ExamStatus.DRAFT

    @property
    def can_export(self) -> bool:
        """是否可导出。草稿还在变，导出没意义。"""
        return self in (ExamStatus.ACTIVE, ExamStatus.ARCHIVED)

    @property
    def can_upload_answer_sheet(self) -> bool:
        """是否可上传答卷。"""
        return self in (ExamStatus.ACTIVE, ExamStatus.ARCHIVED)


# ===================================================================
# 组卷参数
# ===================================================================
@dataclass(frozen=True, slots=True)
class ExamSpec:
    """组卷参数（FR-037）。

    Attributes:
        total: 题目总数。
        knowledge_points: 要求覆盖的知识点编码。
        difficulty_quota: 难度配比，如 {1: 2, 2: 2, 3: 2}。
            ★ 这是**软约束**：凑不齐时降级并在 warnings 里说明，
              因为「知识点覆盖」才是硬约束。
        product_ids: 要求覆盖的产品（主推产品必考）。
        exclude_question_ids: 要排除的题目（防重复窗口内已抽过的）。
        shuffle_options: 是否打乱选项顺序。
        stage: 所属阶段（用于卷号前缀）。
    """

    total: int
    knowledge_points: tuple[str, ...] = ()
    difficulty_quota: dict[int, int] = field(default_factory=dict)
    product_ids: tuple[str, ...] = ()
    exclude_question_ids: frozenset[str] = frozenset()
    shuffle_options: bool = True
    stage: str = "S1"

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError(f"题目总数必须为正，实际 {self.total}")
        bad = [d for d in self.difficulty_quota if d not in (1, 2, 3)]
        if bad:
            raise ValueError(f"难度只能是 1/2/3，实际 {bad}")


# ===================================================================
# 试卷
# ===================================================================
@dataclass(frozen=True, slots=True)
class ExamPaper:
    """一份试卷（含题目快照）。

    ★ `questions` 是**快照副本**，不是题库引用。
    """

    paper_no: str
    stage: str
    spec: ExamSpec
    questions: tuple[QuestionSnapshot, ...]
    status: ExamStatus = ExamStatus.DRAFT
    created_at: str = ""
    created_by: str = ""
    #: 试卷标题（主管可自定义，如「S1 期中考核」）
    title: str = ""
    #: 组卷过程中的降级说明（如「难度 1 题量不足，已用难度 2 补」）
    warnings: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.questions)

    @property
    def covered_knowledge_points(self) -> tuple[str, ...]:
        """实际覆盖的知识点（去重、保序）。"""
        seen: list[str] = []
        for q in self.questions:
            if q.knowledge_point and q.knowledge_point not in seen:
                seen.append(q.knowledge_point)
        return tuple(seen)

    @property
    def difficulty_distribution(self) -> dict[int, int]:
        dist: dict[int, int] = {}
        for q in self.questions:
            dist[q.difficulty] = dist.get(q.difficulty, 0) + 1
        return dict(sorted(dist.items()))

    @property
    def knowledge_point_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for q in self.questions:
            if q.knowledge_point:
                dist[q.knowledge_point] = dist.get(q.knowledge_point, 0) + 1
        return dist

    def covers_all(self) -> bool:
        """是否满足知识点全覆盖。"""
        required = set(self.spec.knowledge_points)
        return required <= set(self.covered_knowledge_points)


# ===================================================================
# 卷号
# ===================================================================
_PAPER_NO_RE = re.compile(r"^[A-Za-z0-9]{1,8}-\d{8}-\d{3}$")


def make_paper_no(stage: str, date: str, seq: int) -> str:
    """生成可读卷号，形如 `S1-20261105-001`。

    Args:
        stage: 阶段标识，如 "S1"。不能为空。
        date: 日期 `YYYYMMDD`。
        seq: 同日序号 1–999。

    Raises:
        ValueError: 阶段为空，或序号越界。

    ★ 为什么序号只到 999：一天出 999 份试卷在培训场景足够；
      真超过说明流程有问题，应该报错而不是静默截断。
    """
    if not stage or not stage.strip():
        raise ValueError("阶段标识不能为空")
    if not 1 <= seq <= 999:
        raise ValueError(f"同一天序号必须在 1–999 之间，实际 {seq}")

    no = f"{stage.strip().upper()}-{date}-{seq:03d}"
    if not _PAPER_NO_RE.match(no):
        raise ValueError(f"卷号格式不合法: {no}")
    return no


__all__ = [
    "ExamPaper",
    "ExamSpec",
    "ExamStatus",
    "QuestionSnapshot",
    "make_paper_no",
]
