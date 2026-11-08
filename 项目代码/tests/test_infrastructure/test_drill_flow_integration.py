"""演练对话链路集成测试（P5）。

★ 用**真实案例文件**验证「演练 → 评阅 → 摘要卡」全链路。

这些测试的作用是防止「改了某处导致某个案例跑不通」——
16 个案例的脚本长度、开场白格式、追问数量都不同，
单元测试用假数据盖不住这些差异。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from application.use_cases.drill_session import DrillSession
from application.use_cases.review_drill import ReviewDrillUseCase
from infrastructure.content.case_loader import load_cases_from_dir

CASE_DIR = Path(r"D:\项目\某健康公司培训系统\06_正式内容\02_情景案例")

#: 一份「什么都说了」的合规回答，用于跑通链路
SAFE_ANSWER = (
    "有的，康源牌鱼油软胶囊，每粒含鱼油1000mg。"
    "这是保健食品，批准功能是有助于维持血脂健康水平。"
    "保健食品不是药物，不能代替药物治疗疾病。"
    "不适宜人群包括出血倾向者、孕妇乳母。"
    "如果您在服药，建议先咨询医生。"
)


@pytest.fixture(scope="module")
def cases():
    if not CASE_DIR.exists():
        pytest.skip(f"案例目录不存在: {CASE_DIR}")
    return load_cases_from_dir(CASE_DIR)


def _run_full(case, answer: str = SAFE_ANSWER) -> DrillSession:
    """跑完一个案例的完整演练。"""
    s = DrillSession(case, trainee_id="T001")
    guard = 0
    while not s.is_finished and guard < 50:
        s.answer(answer)
        guard += 1
    return s


# ===================================================================
# 1. 全部 16 个案例能跑完
# ===================================================================
class TestAllCasesRunnable:
    def test_every_case_finishes(self, cases) -> None:
        """每个案例的演练都必须能正常结束（不死循环、不提前结束）。"""
        for case in cases:
            s = _run_full(case)
            assert s.is_finished, f"{case.case_id} 演练未结束"
            assert s.turn_count == 1 + len(case.follow_ups), (
                f"{case.case_id} 轮次不符："
                f"期望 {1 + len(case.follow_ups)}，实际 {s.turn_count}"
            )

    def test_every_case_produces_card(self, cases) -> None:
        """每个案例都能产出摘要卡。"""
        uc = ReviewDrillUseCase()
        for case in cases:
            s = _run_full(case)
            card = uc.execute(
                case, s.to_trainee_turns(),
                trainee_id=s.trainee_id, duration_sec=s.duration_sec)

            assert card.case_id == case.case_id
            assert card.element_coverage, f"{case.case_id} 无要素判定"
            assert card.trainee_id == "T001"

    def test_every_case_has_opening(self, cases) -> None:
        for case in cases:
            assert case.opening_line, f"{case.case_id} 缺开场白"

    def test_questions_not_empty(self, cases) -> None:
        """所有追问文案都不能为空。"""
        for case in cases:
            for f in case.follow_ups:
                assert f.question.strip(), (
                    f"{case.case_id} 的 {f.follow_up_id} 追问为空")


# ===================================================================
# 2. C01 完整链路（重点案例）
# ===================================================================
class TestC01FullFlow:
    def test_six_questions(self, cases) -> None:
        """C01 是开场 + 5 个追问 = 6 问。"""
        c = {x.case_id: x for x in cases}["C01"]
        s = DrillSession(c, trainee_id="T1")
        assert s.total_questions == 6

    def test_question_sequence(self, cases) -> None:
        """按案例文档规定的顺序提问。"""
        c = {x.case_id: x for x in cases}["C01"]
        s = DrillSession(c, trainee_id="T1")
        seq = [s.current_question]
        for _ in range(5):
            s.answer("合规回答。")
            if s.current_question:
                seq.append(s.current_question)

        assert "鱼油" in seq[0]
        assert "血脂" in seq[1]        # F1
        assert "多久" in seq[2]        # F2
        assert "降压药" in seq[3]      # F3
        assert "真的" in seq[4]        # F4
        assert "该不该" in seq[5]      # F5

    def test_bad_answer_produces_critical(self, cases) -> None:
        """★ 说「一起吃没问题」必须报 critical（禁忌场景越界判断）。"""
        c = {x.case_id: x for x in cases}["C01"]
        s = DrillSession(c, trainee_id="T1")

        bad_answers = [
            "有的，康源牌鱼油软胶囊。",
            "有助于维持血脂健康水平。",
            "吃三个月就行。",
            "可以的，一起吃没问题。",       # ← 越界判断
            "我觉得您应该吃。",            # ← 替客户决定
        ]
        for a in bad_answers:
            if s.is_finished:
                break
            s.answer(a)

        card = ReviewDrillUseCase().execute(
            c, s.to_trainee_turns(), trainee_id="T1",
            duration_sec=s.duration_sec)

        assert card.has_critical_violation, (
            f"应报 critical，实际风险: "
            f"{[(r.rule_type, r.matched) for r in card.risk_items]}"
        )
        # 必须能定位到具体轮次
        for r in card.critical_items:
            assert r.turn_index >= 1
            assert r.location.startswith("第")

    def test_good_answer_no_critical(self, cases) -> None:
        """★ 完整合规的回答不应报 critical。

        ★ 注意：这里可能会因为「缺警示语」被拦，
          但那属于 final_verdict 而不是 risk_items ——
          风险提示区只放违禁表述，缺警示语归要素覆盖（E5）。
        """
        c = {x.case_id: x for x in cases}["C01"]
        s = _run_full(c)
        card = ReviewDrillUseCase().execute(
            c, s.to_trainee_turns(), trainee_id="T1")

        assert not card.has_critical_violation, (
            f"合规回答被误报 critical: "
            f"{[(r.rule_type, r.matched, r.location) for r in card.critical_items]}"
        )


# ===================================================================
# 3. 摘要卡在各案例上都不含分数
# ===================================================================
class TestNoScoreAcrossCases:
    def test_no_score_in_any_card(self, cases) -> None:
        """★ 约束 #3 在全部 16 个案例上都成立。"""
        import json

        uc = ReviewDrillUseCase()
        for case in cases:
            s = _run_full(case)
            card = uc.execute(case, s.to_trainee_turns(),
                              trainee_id="T1", duration_sec=s.duration_sec)
            text = uc.to_json(card).lower()

            assert '"score"' not in text, f"{case.case_id} 摘要卡出现 score"
            assert '"verdict"' not in text, f"{case.case_id} 摘要卡出现 verdict"

    def test_render_has_four_sections(self, cases) -> None:
        uc = ReviewDrillUseCase()
        case = cases[0]
        s = _run_full(case)
        card = uc.execute(case, s.to_trainee_turns(), trainee_id="T1")
        text = uc.render_text(card)

        for section in ("【基础信息】", "【风险提示】",
                        "【要素覆盖】", "【待人工确认】"):
            assert section in text, f"缺区块: {section}"


# ===================================================================
# 4. 会话不可重入
# ===================================================================
class TestSessionIntegrity:
    def test_cannot_answer_twice_after_finish(self, cases) -> None:
        s = _run_full(cases[0])
        with pytest.raises(ValueError, match="已结束"):
            s.answer("再答一次")

    def test_turns_match_answers(self, cases) -> None:
        s = _run_full(cases[0])
        turns = s.to_trainee_turns()
        assert len(turns) == s.turn_count
        assert [t.turn_index for t in turns] == list(
            range(1, s.turn_count + 1))

    def test_transcript_contains_both_sides(self, cases) -> None:
        """对话记录必须含客户和学员双方，否则无法复核。"""
        s = _run_full(cases[0])
        text = s.transcript()
        assert text.count("客户：") >= 1
        assert text.count("学员：") >= 1
