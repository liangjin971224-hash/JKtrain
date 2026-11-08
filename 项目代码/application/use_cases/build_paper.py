"""组卷用例（P6）。

把「读题库 → 组卷 → 落库 → 记录使用」串成一个事务性流程。

★ 为什么需要一个用例层，而不是直接在接口里调 build_paper：
  组卷不是单个函数调用，它包含四步协作：
    ① 从题库取候选（可能来自 Milvus）
    ② 取防重复窗口内已用的题
    ③ 调纯函数组卷
    ④ 落库 + 记录使用
  这四步的顺序和事务边界是业务知识，必须集中在一处。

★ 一条重要顺序：**先组卷成功，才记录使用**。
  反过来会导致「组卷失败但题被标记已用」，
  下次组卷这些题就被排除，越试可用题越少。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from domain.exceptions import ExamBuildError, InvalidExamTransition
from domain.services.paper_builder import build_paper, transition
from domain.value_objects.exam import (
    ExamPaper,
    ExamSpec,
    ExamStatus,
    QuestionSnapshot,
    make_paper_no,
)

logger = logging.getLogger(__name__)


class BuildPaperUseCase:
    """组卷。"""

    def __init__(
        self,
        *,
        repository: Any = None,
        bank_loader: Any = None,
    ) -> None:
        """
        Args:
            repository: `RepositoryPort` 实现（可为 None，此时不落库）。
            bank_loader: 取题库的函数 `(stage) -> list[QuestionSnapshot]`。
        """
        self._repo = repository
        self._bank_loader = bank_loader

    def execute(
        self,
        spec: ExamSpec,
        *,
        title: str = "",
        created_by: str = "",
        trainee_id: str = "",
        seed: int | None = None,
        persist: bool = True,
        bank: list[QuestionSnapshot] | None = None,
    ) -> ExamPaper:
        """组一份试卷。

        Args:
            spec: 组卷参数。
            title: 试卷标题。
            created_by: 创建人。
            trainee_id: 学员编号（防重复按学员隔离；空表示全局去重）。
            seed: 随机种子；不传则用当前时间，保证每次不同。
            persist: 是否落库。False 用于「先看效果再决定」。
            bank: 直接给题库（测试用）；为空则走 bank_loader。

        Returns:
            组好的试卷。

        Raises:
            ExamBuildError: 组卷规则无法满足。
        """
        # ---- 1. 取题库 ----
        if bank is None:
            if self._bank_loader is None:
                raise ExamBuildError("未配置题库来源，无法组卷")
            bank = list(self._bank_loader(spec.stage))

        if not bank:
            raise ExamBuildError(
                f"阶段 {spec.stage} 没有可用题目，请先导入题库")

        # ---- 2. 取防重复窗口内已用的题 ----
        exclude = set(spec.exclude_question_ids)
        if self._repo is not None and spec.exclude_question_ids is not None:
            try:
                used = self._repo.recent_used_question_ids(
                    trainee_id=trainee_id, days=DEDUP_WINDOW_DAYS)
                exclude |= used
                logger.info(
                    "防重复窗口（%d 天）内已用 %d 题，本次排除",
                    DEDUP_WINDOW_DAYS, len(used))
            except Exception as exc:                # noqa: BLE001
                # ★ 去重查询失败**不阻断组卷**：
                #   宁可这次可能重复，也不要让主管出不了卷。
                #   但要警告——重复抽题是体验问题，不是合规问题。
                logger.warning("查询去重记录失败，本次不去重: %s", exc)

        spec = _with_exclude(spec, frozenset(exclude))

        # ---- 3. 组卷（纯函数）----
        now = datetime.now()
        paper_no = self._next_paper_no(spec.stage, now)

        paper = build_paper(
            bank, spec,
            seed=seed,
            paper_no=paper_no,
            created_at=now.strftime("%Y-%m-%d %H:%M:%S"),
            created_by=created_by,
        )

        if title:
            paper = _with_title(paper, title)

        # ---- 4. 落库 + 记录使用（★ 顺序不能反）----
        if persist and self._repo is not None:
            self._repo.save_exam_paper(_to_dict(paper))
            self._repo.record_question_usage(
                paper.paper_no,
                [q.question_id for q in paper.questions],
                trainee_id=trainee_id,
            )
            logger.info("试卷落库完成: %s", paper.paper_no)

        return paper

    def _next_paper_no(self, stage: str, now: datetime) -> str:
        """生成当天下一个可用卷号。

        同日已有卷子时顺延序号。
        """
        date_str = now.strftime("%Y%m%d")

        if self._repo is None:
            return make_paper_no(stage, date_str, 1)

        try:
            existing = self._repo.list_exam_papers(stage=stage, limit=200)
        except Exception:                          # noqa: BLE001
            existing = []

        prefix = f"{stage.upper()}-{date_str}-"
        used = {
            int(p["paper_no"].rsplit("-", 1)[-1])
            for p in existing
            if p.get("paper_no", "").startswith(prefix)
            and p["paper_no"].rsplit("-", 1)[-1].isdigit()
        }
        seq = 1
        while seq in used:
            seq += 1
        return make_paper_no(stage, date_str, seq)


class ChangePaperStatusUseCase:
    """试卷状态流转（草稿 → 生效 → 归档）。"""

    def __init__(self, *, repository: Any) -> None:
        self._repo = repository

    def execute(self, paper_no: str, target: ExamStatus) -> ExamStatus:
        """变更状态。

        Raises:
            InvalidExamTransition: 流转非法（如归档回生效）。
            ValueError: 试卷不存在。
        """
        paper = self._repo.get_exam_paper(paper_no)
        if paper is None:
            raise ValueError(f"试卷不存在: {paper_no}")

        current = _to_status(paper["status"])
        new = transition(current, target)       # 领域层校验，非法则抛
        self._repo.update_exam_status(paper_no, str(new))

        logger.info("试卷状态: %s %s → %s",
                    paper_no, current.label, new.label)
        return new


#: 防重复窗口（天）。需求 FR-037 明确 30 天。
DEDUP_WINDOW_DAYS = 30


# ===================================================================
# 内部转换
# ===================================================================
def _with_exclude(spec: ExamSpec, exclude: frozenset[str]) -> ExamSpec:
    """替换 spec 的排除集合（frozen 值对象，只能重建）。"""
    return ExamSpec(
        total=spec.total,
        knowledge_points=spec.knowledge_points,
        difficulty_quota=dict(spec.difficulty_quota),
        product_ids=spec.product_ids,
        exclude_question_ids=exclude,
        shuffle_options=spec.shuffle_options,
        stage=spec.stage,
    )


def _with_title(paper: ExamPaper, title: str) -> ExamPaper:
    """给试卷补标题（frozen，重建）。"""
    return ExamPaper(
        paper_no=paper.paper_no,
        stage=paper.stage,
        spec=paper.spec,
        questions=paper.questions,
        status=paper.status,
        created_at=paper.created_at,
        created_by=paper.created_by,
        warnings=paper.warnings,
        title=title,
    )


def _to_status(v: Any) -> ExamStatus:
    """字符串/枚举 → ExamStatus。"""
    if isinstance(v, ExamStatus):
        return v
    return ExamStatus(str(v))


def _to_dict(paper: ExamPaper) -> dict[str, Any]:
    """试卷 → 仓储要的字典（含题目快照）。"""
    return {
        "paper_no": paper.paper_no,
        "stage": paper.stage,
        "title": getattr(paper, "title", "") or "",
        "status": str(paper.status),
        "created_by": paper.created_by,
        # ★ created_at 必须显式写入。
        #   数据库列有 DEFAULT CURRENT_TIMESTAMP，但那是「入库时间」，
        #   而组卷时间可能更早（比如先预览、几分钟后才落库）。
        #   不传的话读回来的时间会偏，主管对不上账。
        "created_at": paper.created_at,
        "spec": {
            "total": paper.spec.total,
            "knowledge_points": list(paper.spec.knowledge_points),
            "difficulty_quota": paper.spec.difficulty_quota,
            "product_ids": list(paper.spec.product_ids),
            "shuffle_options": paper.spec.shuffle_options,
            "stage": paper.spec.stage,
            # ★ 警告也存进 spec_json：
            #   主管事后看试卷要能知道「当时为什么这道题难度不对」，
            #   否则 warnings 只在组卷那一刻存在，落库就丢了。
            "warnings": list(paper.warnings),
        },
        "questions": [
            {
                "question_id": q.question_id,
                "knowledge_point": q.knowledge_point,
                "kp_name": q.kp_name,
                "question_type": q.question_type,
                "difficulty": q.difficulty,
                "stem": q.stem,
                "options": list(q.options),
                "answer": q.answer,
                "explanation": q.explanation,
                "product_id": q.product_id,
            }
            for q in paper.questions
        ],
    }


__all__ = [
    "BuildPaperUseCase",
    "ChangePaperStatusUseCase",
    "DEDUP_WINDOW_DAYS",
]
