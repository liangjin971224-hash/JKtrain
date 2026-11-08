"""演练会话（P5 对话环节）。

★ 演练是**状态机**，不是 LLM 自由对话。

────────────────────────────────────────────────────────────────────
为什么客户台词必须来自脚本
────────────────────────────────────────────────────────────────────
案例文档已经把对话顺序和考察点写死了：

    开场「你们有鱼油吗？」
      → F1「血脂有点高，吃这个管用吗」（考功效边界）
      → F2「要吃多久」（考不给疗程承诺）
      → F3「在吃降压药」（考禁忌识别）
      → F4「怎么知道是真的」（考信任建立）
      → F5「该不该吃」（考不替客户判断）

如果让 LLM 自由扮演客户：
    · 同一份答卷两次演练难度不同 → 学员之间不公平，无法横向比较
    · 追问可能跑偏，漏掉关键考察点（漏了 F3 就测不到禁忌识别）
    · 无法复现，学员申诉时说不清「当时客户到底问了什么」

所以：**台词固定，只有「什么时候进下一句」需要判断**。
而那部分也能用规则做（按对话进度）。

────────────────────────────────────────────────────────────────────
LLM 在演练里的位置
────────────────────────────────────────────────────────────────────
不在「扮演客户」，而在：
    · 学员答完后给即时反馈（可选，需要 LLM）
    · 生成模拟答卷做演示（可选）

客户台词永远是脚本。
────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import time

from domain.value_objects import CaseSpec, TraineeTurn

logger = logging.getLogger(__name__)


class DrillSession:
    """一次演练会话。

    用法::

        s = DrillSession(spec, trainee_id="T1")
        while not s.is_finished:
            print("客户：", s.current_question)
            s.answer(学员输入)

        card = ReviewDrillUseCase().execute(
            spec, s.to_trainee_turns(),
            trainee_id=s.trainee_id, duration_sec=s.duration_sec)
    """

    def __init__(self, spec: CaseSpec, *, trainee_id: str = "") -> None:
        """
        Args:
            spec: 案例规格（含开场白与追问脚本）。
            trainee_id: 学员编号。

        Raises:
            ValueError: 案例没有开场白（无法开始演练）。
        """
        if not spec.opening_line or not spec.opening_line.strip():
            raise ValueError(
                f"案例 {spec.case_id} 缺少开场白，无法开始演练")

        self._spec = spec
        self._trainee_id = trainee_id
        self._follow_ups = list(spec.follow_ups)

        #: 已答的轮次（学员发言）
        self._answers: list[str] = []
        #: 当前提问游标：-1 表示还在开场白，0 表示在 F1…
        self._cursor = -1
        #: 会话是否已结束。
        #: ★ 构造时恒为 False——即使案例没有追问，
        #:   开场白也是一个问题，必须学员答了才算完成。
        self._finished = False

        self._started_at = time.perf_counter()
        self._frozen_duration: int | None = None

    # ---------------- 状态 ----------------
    @property
    def trainee_id(self) -> str:
        return self._trainee_id

    @property
    def case_id(self) -> str:
        return self._spec.case_id

    @property
    def turn_count(self) -> int:
        """已答轮次。"""
        return len(self._answers)

    @property
    def total_questions(self) -> int:
        """总问题数 = 开场白 1 条 + 追问 N 条。"""
        return 1 + len(self._follow_ups)

    @property
    def current_question(self) -> str:
        """当前该问的问题。已结束返回空串。"""
        if self._finished:
            return ""
        if self._cursor < 0:
            return self._spec.opening_line
        return self._follow_ups[self._cursor].question

    @property
    def current_follow_up_id(self) -> str | None:
        """当前追问的编号。开场白阶段返回 None。"""
        if self._finished or self._cursor < 0:
            return None
        return self._follow_ups[self._cursor].follow_up_id

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def progress(self) -> tuple[int, int]:
        """(当前第几问, 总问数)，都从 1 开始。"""
        if self._finished:
            return (self.total_questions, self.total_questions)
        return (self._cursor + 2, self.total_questions)

    @property
    def remaining(self) -> int:
        """还剩几问（含当前这问）。"""
        if self._finished:
            return 0
        return self.total_questions - (self._cursor + 1)

    @property
    def duration_sec(self) -> int:
        """已用时长（秒）。结束后冻结。"""
        if self._frozen_duration is not None:
            return self._frozen_duration
        return int(time.perf_counter() - self._started_at)

    # ---------------- 主流程 ----------------
    def answer(self, text: str) -> None:
        """学员回答当前问题。

        Args:
            text: 学员的回答。

        Raises:
            ValueError: 会话已结束，或回答为空。

        Note:
            会话结束后再答**必须报错**，不能静默忽略——
            静默忽略会让前端以为答成功了，但内容没进评阅。
        """
        if self._finished:
            raise ValueError(
                f"会话已结束（共 {self.turn_count} 轮），不能再作答")

        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("回答不能为空（空回答不计入轮次）")

        self._answers.append(cleaned)

        # 游标前进
        self._cursor += 1
        if self._cursor >= len(self._follow_ups):
            self._finished = True
            self._frozen_duration = int(
                time.perf_counter() - self._started_at)
            logger.info(
                "演练结束: case=%s trainee=%s 轮次=%d 用时=%ds",
                self.case_id, self._trainee_id,
                self.turn_count, self._frozen_duration,
            )

    # ---------------- 导出 ----------------
    def to_trainee_turns(self) -> list[TraineeTurn]:
        """导出为评阅用例需要的轮次列表。"""
        return [
            TraineeTurn(turn_index=i, text=t)
            for i, t in enumerate(self._answers, start=1)
        ]

    def messages(self) -> list[dict[str, str]]:
        """导出完整对话（客户台词 + 学员回答），供前端展示。

        ★ 必须含客户台词：评阅时人要看「是在什么提问下这么答的」。
          「能不能吃」和「该不该吃」下的同一个回答，合规性不同。

        Returns:
            `[{"role": "customer"|"trainee", "text": ...}, ...]`
        """
        out: list[dict[str, str]] = [
            {"role": "customer", "text": self._spec.opening_line}]

        for i, answer in enumerate(self._answers):
            out.append({"role": "trainee", "text": answer})
            if i < len(self._follow_ups):
                out.append({
                    "role": "customer",
                    "text": self._follow_ups[i].question,
                })
        return out

    def transcript(self) -> str:
        """导出可读对话记录（存档用）。

        ★ 含提问和回答，因为评阅时人要看「是在什么提问下这么答的」。
          「能不能吃」和「该不该吃」下的同一个回答，合规性不同。
        """
        lines: list[str] = []
        lines.append(f"【案例 {self.case_id}】{self._spec.title}")
        lines.append(f"学员：{self._trainee_id or '未填'}")
        lines.append("")

        # 开场白
        lines.append(f"客户：{self._spec.opening_line}")

        for i, ans in enumerate(self._answers):
            lines.append(f"学员：{ans}")
            # 这轮答完后客户下一句（若还有）
            if i < len(self._follow_ups):
                lines.append(f"客户：{self._follow_ups[i].question}")

        lines.append("")
        lines.append(f"（共 {self.turn_count} 轮，用时 {self.duration_sec} 秒）")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """导出结构化状态（供接口层返回前端）。"""
        return {
            "case_id": self.case_id,
            "trainee_id": self._trainee_id,
            "case_title": self._spec.title,
            "current_question": self.current_question,
            "current_follow_up_id": self.current_follow_up_id,
            "progress": {
                "current": self.progress[0],
                "total": self.progress[1],
            },
            "remaining": self.remaining,
            "turn_count": self.turn_count,
            "is_finished": self.is_finished,
            "duration_sec": self.duration_sec,
        }


__all__ = ["DrillSession"]
