"""演练会话测试（P5 对话环节）。

★ 演练的骨架是**状态机**，不是 LLM 自由对话。

原因：
    案例文档已经规定死了「开场白 → F1 → F2 → F3 → F4 → F5」的顺序，
    以及每个追问的考察点。如果用 LLM 自由生成客户台词：
      · 同一份答卷，两次演练难度不同 → 不公平
      · 追问可能跑偏，漏掉关键考察点（比如漏了 F3 就测不到禁忌识别）
      · 无法复现，学员申诉时说不清

所以：**台词来自脚本，只有「什么时候该问下一句」需要判断**。
    而这部分也能用规则做（按对话进度 + 可选的内容条件）。

LLM 在演练里的位置是「学员写回答」（人是学员）或
    「自动生成一份模拟答卷用于演示」，不是扮演客户。
"""
from __future__ import annotations

import pytest

from application.use_cases.drill_session import DrillSession
from domain.value_objects import (
    CaseSpec,
    ElementSpec,
    ForbiddenRule,
    RiskLevel,
    ScriptedFollowUp,
    TraineeTurn,
)


# ===================================================================
# 夹具
# ===================================================================
def _c01_spec() -> CaseSpec:
    """C01 精简版（结构与真实案例一致）。"""
    return CaseSpec(
        case_id="C01",
        title="客户询问是否有鱼油",
        opening_line="你们有鱼油吗？",
        elements=(
            ElementSpec("E1", "明确回答有 / 没有", 20, True),
            ElementSpec("E2", "给出产品名称", 10, True),
            ElementSpec("E5", "加强制警示语", 15, True),
        ),
        forbidden_rules=(
            ForbiddenRule("违规宣称", r"降血脂|治疗", RiskLevel.CRITICAL),
        ),
        follow_ups=(
            ScriptedFollowUp("F1", "我体检说血脂有点高，吃这个管用吗？",
                             "学员报了产品信息后", "功效边界"),
            ScriptedFollowUp("F2", "要吃多久啊？", "学员回答了 F1", "不给疗程承诺"),
            ScriptedFollowUp("F3", "我平时还在吃降压药，能一起吃吗？",
                             "学员回答了 F2", "禁忌识别"),
        ),
    )


# ===================================================================
# 1. 开场
# ===================================================================
class TestOpening:
    def test_session_starts_with_opening(self) -> None:
        """会话创建后第一条就应是客户开场白。"""
        s = DrillSession(_c01_spec(), trainee_id="T1")
        assert s.current_question == "你们有鱼油吗？"
        assert s.turn_count == 0
        assert s.is_finished is False

    def test_opening_not_repeated(self) -> None:
        """学员答完后不该再返回开场白。"""
        s = DrillSession(_c01_spec(), trainee_id="T1")
        first = s.current_question
        s.answer("有的，康源牌鱼油软胶囊。")
        assert s.current_question != first

    def test_empty_opening_validation(self) -> None:
        spec = CaseSpec("C99", "t", opening_line="")
        with pytest.raises(ValueError, match="开场"):
            DrillSession(spec, trainee_id="T1")


# ===================================================================
# 2. 按序追问
# ===================================================================
class TestFollowUpOrder:
    def test_advances_to_f1_after_first_answer(self) -> None:
        s = DrillSession(_c01_spec(), trainee_id="T1")
        s.answer("有的，康源牌鱼油软胶囊，每粒1000mg。")
        assert "血脂有点高" in s.current_question

    def test_advances_in_order(self) -> None:
        """必须按 F1 → F2 → F3 顺序，不能跳。"""
        s = DrillSession(_c01_spec(), trainee_id="T1")
        seen: list[str] = []
        for _ in range(3):
            seen.append(s.current_question)
            s.answer("合规回答，保健食品不是药物，不能代替药物治疗疾病。")

        assert "血脂有点高" in seen[1], f"第2问应是 F1，实际 {seen}"
        assert "要吃多久" in seen[2], f"第3问应是 F2，实际 {seen}"

    def test_finishes_after_last_follow_up(self) -> None:
        """追问跑完 → 会话结束。"""
        s = DrillSession(_c01_spec(), trainee_id="T1")
        # 开场 + 3 个追问 = 4 个问题，需答 4 次
        for _ in range(4):
            s.answer("有的，康源牌鱼油软胶囊。保健食品不是药物，不能代替药物治疗疾病。")

        assert s.is_finished is True
        assert s.current_question == ""

    def test_no_follow_ups_finishes_after_one_answer(self) -> None:
        spec = CaseSpec("C99", "t", "开场问一句？",
                        follow_ups=())
        s = DrillSession(spec, trainee_id="T1")
        s.answer("答完了。")
        assert s.is_finished is True


# ===================================================================
# 3. 轮次与历史
# ===================================================================
class TestTurns:
    def test_turns_accumulate(self) -> None:
        s = DrillSession(_c01_spec(), trainee_id="T1")
        s.answer("第一答。")
        s.answer("第二答。")
        assert s.turn_count == 2

    def test_turns_as_trainee_turns(self) -> None:
        """能导出成 TraineeTurn 列表，直接喂给评阅用例。"""
        s = DrillSession(_c01_spec(), trainee_id="T1")
        s.answer("有的，康源牌鱼油软胶囊。")
        s.answer("保健食品不是药物，不能代替药物治疗疾病。")

        turns = s.to_trainee_turns()
        assert len(turns) == 2
        assert turns[0].turn_index == 1
        assert turns[1].turn_index == 2
        assert "康源牌" in turns[0].text

    def test_turn_index_one_based(self) -> None:
        s = DrillSession(_c01_spec(), trainee_id="T1")
        s.answer("随便答一句。")
        assert s.to_trainee_turns()[0].turn_index == 1

    def test_transcript_readable(self) -> None:
        """要能导出可读对话记录（存档用）。"""
        s = DrillSession(_c01_spec(), trainee_id="T1")
        s.answer("有的。")
        text = s.transcript()
        assert "客户" in text
        assert "学员" in text
        assert "你们有鱼油吗" in text


# ===================================================================
# 4. 答完就不再接受
# ===================================================================
class TestFinishedSession:
    def test_answer_after_finished_raises(self) -> None:
        """会话结束后再答应报错，而不是静默忽略。

        ★ 静默忽略会让前端以为答成功了，但内容没进评阅。
        """
        spec = CaseSpec("C99", "t", "问？", follow_ups=())
        s = DrillSession(spec, trainee_id="T1")
        s.answer("答。")

        with pytest.raises(ValueError, match="已结束"):
            s.answer("再答一次。")

    def test_empty_answer_raises(self) -> None:
        """空回答不该算一轮——否则评阅会把空轮次算进去。"""
        s = DrillSession(_c01_spec(), trainee_id="T1")
        with pytest.raises(ValueError, match="不能为空"):
            s.answer("")

    def test_whitespace_answer_raises(self) -> None:
        s = DrillSession(_c01_spec(), trainee_id="T1")
        with pytest.raises(ValueError, match="不能为空"):
            s.answer("   \n  ")


# ===================================================================
# 5. 进度信息
# ===================================================================
class TestProgress:
    def test_progress_reported(self) -> None:
        """要能告诉学员「第几问 / 共几问」。"""
        s = DrillSession(_c01_spec(), trainee_id="T1")
        assert s.progress == (1, 4)      # 开场 + 3 追问

    def test_progress_advances(self) -> None:
        s = DrillSession(_c01_spec(), trainee_id="T1")
        s.answer("答。")
        assert s.progress == (2, 4)

    def test_current_follow_up_id(self) -> None:
        """当前问的是哪个追问（开场返回 None）。"""
        s = DrillSession(_c01_spec(), trainee_id="T1")
        assert s.current_follow_up_id is None

        s.answer("答。")
        assert s.current_follow_up_id == "F1"

    def test_remaining_count(self) -> None:
        s = DrillSession(_c01_spec(), trainee_id="T1")
        assert s.remaining == 4
        s.answer("答。")
        assert s.remaining == 3


# ===================================================================
# 6. 时长
# ===================================================================
class TestDuration:
    def test_duration_recorded(self) -> None:
        """用时用于评阅摘要卡。"""
        s = DrillSession(_c01_spec(), trainee_id="T1")
        s.answer("答。")
        assert s.duration_sec >= 0

    def test_finish_freezes_duration(self) -> None:
        """结束后时长不再变（避免前端轮询时数字一直涨）。"""
        spec = CaseSpec("C99", "t", "问？", follow_ups=())
        s = DrillSession(spec, trainee_id="T1")
        s.answer("答。")
        d1 = s.duration_sec
        d2 = s.duration_sec
        assert d1 == d2


# ===================================================================
# 7. 与评阅衔接
# ===================================================================
class TestReviewHandoff:
    def test_can_review_after_finish(self) -> None:
        """会话结束后能直接调评阅，产出摘要卡。"""
        from application.use_cases.review_drill import ReviewDrillUseCase

        s = DrillSession(_c01_spec(), trainee_id="T1")
        for _ in range(4):
            s.answer("有的，康源牌鱼油软胶囊。保健食品不是药物，不能代替药物治疗疾病。")

        card = ReviewDrillUseCase().execute(
            _c01_spec(), s.to_trainee_turns(),
            trainee_id=s.trainee_id, duration_sec=s.duration_sec)

        assert card.case_id == "C01"
        assert card.trainee_id == "T1"
        assert len(card.element_coverage) == 3

    def test_review_before_finish_allowed(self) -> None:
        """中途也能评（「练到一半想看看」的场景）。

        但要如实反映「只答了部分」——缺的要素会显示为缺失。
        """
        from application.use_cases.review_drill import ReviewDrillUseCase

        s = DrillSession(_c01_spec(), trainee_id="T1")
        s.answer("有的。")

        card = ReviewDrillUseCase().execute(
            _c01_spec(), s.to_trainee_turns(), trainee_id="T1")

        assert card.case_id == "C01"
        # 只说了「有的」，E2/E5 应缺失
        missing = {c.element_id for c in card.missing_required_elements}
        assert "E5" in missing
