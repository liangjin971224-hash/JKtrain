"""输出护栏「按句标注 + 部分保留」测试。

★ 产品决策（2026-10-23）：
    原策略：有一句违规 → 整段拒答。
    新策略：违规句标红展示并说明原因，其余合规内容照常给出。

原因：这是**培训系统**。整段拒答的话，学员连「配料是什么」都看不到，
      也永远学不到「刚才那句话为什么不能说」。

★ 但有一个必须守住的底线：
    接口返回给外部消费方的正文（safe_text）**必须是剔除违规句后的版本**。
    违规原文只出现在 `segments` 里，供前端渲染标红用。
    否则「标红展示违规内容」就变成了「对外传播违法宣传」。
"""
from __future__ import annotations

import pytest

from domain.services.output_guardrail import (
    AnswerSegment,
    AnnotationResult,
    OutputGuardrail,
    split_sentences,
)


@pytest.fixture()
def guard() -> OutputGuardrail:
    return OutputGuardrail()


# ===================================================================
# 1. 分句
# ===================================================================
class TestSplitSentences:
    def test_basic(self) -> None:
        parts = split_sentences("第一句。第二句！第三句？")
        assert parts == ["第一句。", "第二句！", "第三句？"]

    def test_no_enumeration_comma_split(self) -> None:
        """★ 顿号不能当分句点（与 stream_guard 同一个坑）。

        「疾病预防、治疗功能」是一个整体概念，切开就断义。
        """
        parts = split_sentences("不能宣称疾病预防、治疗功能。")
        assert len(parts) == 1
        assert "疾病预防、治疗功能" in parts[0]

    def test_drops_blank(self) -> None:
        parts = split_sentences("第一句。\n\n\n第二句。")
        assert all(p.strip() for p in parts)
        assert len(parts) == 2

    def test_empty_input(self) -> None:
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_single_sentence_no_period(self) -> None:
        assert split_sentences("没有标点的一句话") == ["没有标点的一句话"]


# ===================================================================
# 2. ★ 核心：部分保留
# ===================================================================
class TestPartialRetention:
    def test_keeps_compliant_drops_violation(self, guard) -> None:
        """用户实测场景：黑芝麻丸答了三句，只有最后一句违规。

        期望：前两句保留，最后一句标红。
        """
        text = ("康源牌黑芝麻丸是普通食品。"
                "配料为黑芝麻和蜂蜜，每丸10g。"
                "黑芝麻有乌发养颜的功效。")
        r = guard.annotate(text, category="普通食品")

        assert r.has_safe_content is True
        assert r.violation_count == 1

        # safe_text 必须不含违规句
        assert "乌发" not in r.safe_text
        # 必须含合规句
        assert "普通食品" in r.safe_text
        assert "黑芝麻和蜂蜜" in r.safe_text

    def test_segment_marked_with_reason(self, guard) -> None:
        """违规段必须带类别标签和原因，否则前端没法解释「为什么不能说」。"""
        r = guard.annotate(
            "配料为黑芝麻和蜂蜜。黑芝麻有乌发养颜的功效。",
            category="普通食品")

        bad = [s for s in r.segments if not s.ok]
        assert len(bad) == 1
        assert bad[0].rule_types, "违规段缺类别标签"
        assert bad[0].reason, "违规段缺原因说明"
        assert "乌发" in bad[0].text

    def test_all_clean_no_annotation(self, guard) -> None:
        text = ("配料为黑芝麻和蜂蜜，每丸10g。"
                "可作为日常零食食用。")
        r = guard.annotate(text, category="普通食品")

        assert r.has_safe_content is True
        assert r.violation_count == 0
        assert r.safe_text == text

    def test_removed_text_available_for_audit(self, guard) -> None:
        """被剔除的原文要能取到（培训记录/审计用）。"""
        r = guard.annotate(
            "配料为黑芝麻和蜂蜜。黑芝麻有乌发养颜的功效。",
            category="普通食品")
        assert r.removed_text
        assert "乌发" in "".join(r.removed_text)


# ===================================================================
# 3. ★ 底线：全违规仍须拒答
# ===================================================================
class TestAllViolatingStillRefused:
    def test_single_violating_sentence(self, guard) -> None:
        r = guard.annotate("这个能降血糖。", category="保健食品")
        assert r.has_safe_content is False
        assert r.reason

    def test_all_sentences_violating(self, guard) -> None:
        r = guard.annotate(
            "这个能降血糖。还能治疗糖尿病。", category="保健食品")
        assert r.has_safe_content is False


# ===================================================================
# 4. ★★ 关键底线：safe_text 必须真的合规
# ===================================================================
class TestSafeTextIsActuallySafe:
    """这是整个功能最重要的测试组。

    「剔除违规句」不等于「结果合规」——拼回去可能出现新问题。
    safe_text 必须过一遍完整护栏，否则就是漏放。
    """

    @pytest.mark.parametrize(("text", "category"), [
        ("配料为黑芝麻和蜂蜜。黑芝麻有乌发养颜的功效。", "普通食品"),
        ("这款鱼油有助于维持血脂健康水平。这个能降血糖。", "保健食品"),
        ("本品不是药物。但可以治疗糖尿病。", "保健食品"),
    ])
    def test_safe_text_passes_full_guardrail(self, guard, text, category) -> None:
        r = guard.annotate(text, category=category)
        if not r.has_safe_content:
            pytest.skip("无可保留内容，不适用")

        verdict = guard.check(r.safe_text, category=category)
        assert verdict.passed, (
            f"safe_text 未通过护栏！这是漏放违规输出。\n"
            f"safe_text={r.safe_text!r}\n"
            f"原因={verdict.reason}"
        )

    def test_warning_repaired_when_deleted(self, guard) -> None:
        """★ 警示语若随违规句被删，必须补回。

        否则 safe_text 会变成「保健食品缺警示语」→ 二次检测不通过。
        """
        text = ("这款鱼油有助于维持血脂健康水平。"
                "这个还能降血糖。"
                "保健食品不是药物，不能代替药物治疗疾病。")
        r = guard.annotate(text, category="保健食品")

        assert r.has_safe_content is True
        assert "不能代替药物治疗疾病" in r.safe_text

    def test_missing_warning_not_silently_passed(self, guard) -> None:
        """★★ 回归：单句全通过时也必须整段验一次。

        踩过的坑：逐句检测用 `check_warning=False`（警示语是整段属性），
        如果逐句全通过就直接返回，会**完全跳过警示语检查**——
        一个缺法定警示语的保健食品回答被判定为「全部合规」。
        这是合规底线被绕过。
        """
        r = guard.annotate(
            "这款鱼油有助于维持血脂健康水平。", category="保健食品")

        # 要么补回警示语，要么判不可用 —— 绝不能「原样返回且判定合规」
        assert r.repaired_warning is True or not r.has_safe_content
        if r.has_safe_content:
            assert "不能代替药物治疗疾病" in r.safe_text


# ===================================================================
# 5. ★ 否定依赖：不能因为删除而让语义变危险
# ===================================================================
class TestDanglingNegationGuard:
    def test_safe_text_never_starts_with_turn_word(self, guard) -> None:
        """★ 残留内容不能以转折词开头。

        「但能降血糖」单独出现比原句更危险——它没有了前面的否定约束，
        读起来就是一句直白的功效宣称。
        """
        r = guard.annotate(
            "本品不是药物。但可以治疗糖尿病。", category="保健食品")

        if r.has_safe_content:
            first = split_sentences(r.safe_text)[0]
            assert not first.startswith(("但", "但是", "不过", "然而", "可是")), (
                f"safe_text 以转折词开头，语义不安全: {r.safe_text!r}"
            )

    def test_dangling_negation_rejects_partial_retention(self, guard) -> None:
        """★ 真正会残留转折句的场景：转折句本身**不含禁用词**。

        这种最难处理——转折句单看没违规，但它依赖前文的否定才有意义。
        如果前文被删、转折句留下，语义就反了。

        实测构造：前一句含禁用词被删，后一句是「但效果好」。
        """
        r = guard.annotate(
            "这个能降血糖。但效果是温和的。", category="保健食品")

        if r.has_safe_content:
            # 若保留了内容，不能是「但效果是温和的」这种悬空转折
            assert not r.safe_text.strip().startswith("但"), (
                f"残留了悬空的转折句: {r.safe_text!r}"
            )

    def test_normal_sentences_retained(self, guard) -> None:
        """正常句（不以转折词开头）不受此限制。"""
        r = guard.annotate(
            "配料为黑芝麻和蜂蜜。黑芝麻有乌发养颜的功效。",
            category="普通食品")
        assert r.has_safe_content is True


# ===================================================================
# 6. 标注内容质量
# ===================================================================
class TestAnnotationQuality:
    def test_reason_is_human_readable(self, guard) -> None:
        """原因里不该出现内部措辞（「命中禁用表述：」「[类别]」）。"""
        r = guard.annotate(
            "这个能降血糖，很多人吃了都说好。", category="保健食品")
        bad = [s for s in r.segments if not s.ok]
        assert bad

        for seg in bad:
            assert "命中禁用表述" not in seg.reason
            assert not seg.reason.startswith("[")
            assert seg.reason.endswith(("。", "！", "）")) or len(seg.reason) > 5

    def test_reason_not_duplicated(self, guard) -> None:
        """同一原因说两遍会让学员困惑（实测见过）。"""
        r = guard.annotate("可以的，一起吃没问题。", category="保健食品")
        bad = [s for s in r.segments if not s.ok]
        if bad:
            reason = bad[0].reason
            parts = [p for p in reason.split("；") if p.strip()]
            assert len(parts) == len(set(parts)), (
                f"原因里有重复片段: {reason}"
            )

    def test_rule_types_populated(self, guard) -> None:
        """类别标签要能区分不同违规类型。"""
        r = guard.annotate("这个能降血糖。", category="保健食品")
        bad = [s for s in r.segments if not s.ok]
        assert bad
        assert bad[0].rule_types
        assert "违规表述" not in bad[0].rule_types, (
            f"类别标签过于笼统: {bad[0].rule_types}"
        )


# ===================================================================
# 7. 数据结构
# ===================================================================
class TestDataStructures:
    def test_segment_is_frozen(self) -> None:
        seg = AnswerSegment(text="x", ok=True)
        with pytest.raises(Exception):
            seg.text = "y"  # type: ignore[misc]

    def test_result_defaults(self) -> None:
        r = AnnotationResult()
        assert r.has_safe_content is False
        assert r.has_violation is False
        assert r.violation_count == 0
        assert r.removed_text == ()

    def test_empty_input(self, guard) -> None:
        r = guard.annotate("", category="保健食品")
        assert r.has_safe_content is False
        assert r.segments == ()


# ===================================================================
# 8. ★ 指称语境 vs 使用（2026-10-23 实测暴露）
# ===================================================================
class TestReferentialVsUsage:
    """培训系统最该说话，却最容易被自己拦掉。

    实测：问「黑芝麻丸能乌发吗」，LLM 答
        「乌发属于功能暗示，不符合普通食品的法规要求，因此不能这样表述。」
    —— 这是在**解释法规**，完全合规，却因含「乌发」被拦。
    """

    @pytest.mark.parametrize("text", [
        "乌发属于功能暗示，不符合普通食品的法规要求，因此不能这样表述。",
        "「乌发」是功能暗示，普通食品不能说。",
        "不能宣称乌发功效。",
        "不得使用「降血脂」这类医疗用语。",
        "「治疗」属于医疗用语，保健食品不能用。",
    ])
    def test_referential_allowed(self, guard, text) -> None:
        v = guard.check(text, category="普通食品")
        assert not v.blocked, (
            f"指称/反例教学被误拦: {text}\n原因: {v.reason}"
        )

    @pytest.mark.parametrize("text", [
        "黑芝麻有乌发养颜的功效。",
        "乌发养颜的功效",
        "可以补钙。",
        "有助于增强免疫力。",
    ])
    def test_usage_blocked(self, guard, text) -> None:
        """在实际使用这些词时必须拦。"""
        v = guard.check(text, category="普通食品")
        assert v.blocked, f"违规使用被漏放: {text}"


# ===================================================================
# 9. ★ 词表缺口：「治好」类
# ===================================================================
class TestCureVerbs:
    def test_cure_verbs_blocked(self, guard) -> None:
        """「治好」「药到病除」是最典型的违规话术，必须在词表里。"""
        for text in ("彻底治好", "能治好病", "药到病除", "可以治好"):
            v = guard.check(text, category="保健食品")
            assert v.blocked, f"漏放: {text}"

    def test_cure_verb_after_reference_still_blocked(self, guard) -> None:
        """★ 关键：前半句是指称、后半句是真违规，必须拦。

        实测漏放事故：「❌ 不能说：治疗，我们这个确实能治好」
        前半句是反例教学（应放行），后半句是真实宣称（必须拦）。
        挖除时只能挖指称部分，不能连带后半句。
        """
        v = guard.check("❌ 不能说：治疗，我们这个确实能治好",
                        category="保健食品")
        assert v.blocked, (
            "指称片段被挖掉时连带吞掉了后半句的真违规 —— 安全事故级漏放"
        )


# ===================================================================
# 10. ★ 类别隔离：批准功能只对保健食品合规
# ===================================================================
class TestCategoryIsolation:
    """同一句话，保健食品合规、普通食品违规。

    实测漏放：「有助于增强免疫力」在普通食品下被判合规，
    因为「有助于 X」的放行名单没按类别区分。
    """

    @pytest.mark.parametrize("text", [
        "有助于增强免疫力。",
        "有助于维持血脂健康水平。",
        "有助于改善睡眠。",
    ])
    def test_health_food_claims_blocked_for_ordinary_food(self, guard, text) -> None:
        v = guard.check(text, category="普通食品")
        assert v.blocked, f"保健功能声称在普通食品下被漏放: {text}"

    def test_same_text_allowed_for_health_food(self, guard) -> None:
        text = ("这款鱼油有助于维持血脂（胆固醇/甘油三酯）健康水平。"
                "保健食品不是药物，不能代替药物治疗疾病。")
        v = guard.check(text, category="保健食品")
        assert not v.blocked, f"保健食品的批准功能原文被误拦: {v.reason}"

    @pytest.mark.parametrize("text", [
        "根据国家规定，普通食品不能宣称任何保健功能或疾病预防、治疗作用。",
        "该产品不具有疾病预防、治疗功能。",
    ])
    def test_prohibition_statements_allowed_for_ordinary_food(self, guard, text) -> None:
        """★★ 回归：普通食品类别的判定曾用另一套正则，没共用合规放行逻辑。

        结果「普通食品不能宣称任何保健功能或疾病预防、治疗作用」
        被判「命中治疗」—— 而这是《广告法》原文式的禁令陈述。
        """
        v = guard.check(text, category="普通食品")
        assert not v.blocked, f"禁令陈述被误拦: {text}\n原因: {v.reason}"
