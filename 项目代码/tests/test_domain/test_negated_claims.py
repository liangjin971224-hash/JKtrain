"""否定陈述放行与「双否/转折」拦截的回归测试。

★ 全部来自真实端到端使用暴露的问题，不是假想场景。

背景（用户实测截图）：
    问「康源牌黑芝麻丸有什么功能」，同一问题问了两次：
        第一次 → 正常回答
        第二次 → 已拒答，内容被截断在「或疾病预防、」

    拦截命中的是 `治疗`，而完整句子是：
        普通食品不能宣称任何保健功能或疾病预防、治疗功能。

    **这是《广告法》原文表述，完全合规。**

根因有两层：
    ① `_NEGATED_CLAIM` 的否定词表只有「不涉及」，实际文本是「不能涉及」
    ② 中间插入语 `{0,14}` 排除了顿号，而「疾病预防、治疗」中间有顿号

修这两处时又踩了三次：
    v1 改成「(不能|不) + 任意动词」→ 漏放「不可以治疗，但能缓解」
    v2 加入双否排除但先挖掉 → 功效词一起被挖，漏放
    v3 改为「重叠则跳过挖除」→ 正确

本文件把这三轮都固化成测试。
"""
from __future__ import annotations

import pytest

from domain.rules.forbidden_claims import check_forbidden_claims


# ===================================================================
# 1. ★ 「疾病预防、治疗功能」这一整类必须放行
# ===================================================================
class TestDiseasePreventionTreatment:
    """《广告法》标准表述：「不得涉及疾病预防、治疗功能」。

    这类句子含「治疗」，但整句是**禁令陈述**，不是功效宣称。
    """

    @pytest.mark.parametrize("text", [
        "普通食品不能宣称任何保健功能或疾病预防、治疗功能。",
        "不能宣称疾病预防、治疗功能",
        "不得明示或者暗示具有疾病预防、治疗功能",
        "不能涉及疾病预防、治疗功能",
        "该产品不具有疾病预防、治疗功能",
        "不能声称具有疾病预防、治疗功能",
        "不得作疾病预防、治疗功能的宣传",
    ])
    def test_allowed(self, text: str) -> None:
        result = check_forbidden_claims(text)
        assert not result.has_critical, (
            f"合规的禁令陈述被误拦: {text}\n"
            f"命中: {[(v.rule_id, v.matched) for v in result.violations]}"
        )

    def test_punctuation_does_not_break_match(self) -> None:
        """★ 顿号不能断开否定词与功效词的联系。

        实测：「疾病预防、治疗」中间的顿号让 `{0,14}` 的排除组匹配失败。
        顿号是**并列**（一个整体概念），逗号才是**转折**。
        """
        assert not check_forbidden_claims(
            "不能涉及疾病预防、治疗功能").has_critical
        # 对照：逗号是转折，必须拦
        assert check_forbidden_claims(
            "不能这个，但可以治疗糖尿病").has_critical


# ===================================================================
# 2. ★ 双重否定与转折必须拦（形式否定、语义肯定）
# ===================================================================
class TestDoubleNegationBlocked:
    """这些句子**形式是否定，语义是肯定**。

    若被「否定陈述」规则放行，就是漏放。
    """

    @pytest.mark.parametrize("text", [
        "不可否认它能降血糖",
        "不能不说这个能降血脂",
        "不可以治疗，但能缓解",
        "不能这个，但可以治疗糖尿病",
        "不能在柜台上说，但确实能降血糖",
    ])
    def test_blocked(self, text: str) -> None:
        result = check_forbidden_claims(text)
        assert result.has_critical, (
            f"双重否定/转折被漏放: {text}\n"
            f"这是安全事故级别的问题——话术会直接说给客户。"
        )

    def test_double_negation_without_banned_word_not_flagged(self) -> None:
        """双重否定本身不是禁用词，只有含功效词才拦。

        「不得不说这个确实有效」里「有效」不在六类禁用词表内
        （它太宽泛，正常描述也会用到），所以不判 critical。
        这是**有意为之**：禁用词表只放明确违规的表述。
        """
        assert not check_forbidden_claims(
            "不得不说这个确实有效").has_critical

    def test_normal_negation_still_allowed(self) -> None:
        """加了双否排除后，普通否定陈述不能被误伤。"""
        assert not check_forbidden_claims(
            "它不用于治疗高血脂症").has_critical


# ===================================================================
# 3. 否定词与动词的各种搭配（组合式，不靠枚举）
# ===================================================================
class TestNegationVerbCombinations:
    """否定前缀 × 动词的组合都要能识别。

    实测教训：原先枚举 `不涉及`，但真实文本是「不能涉及」。
    逐条枚举永远漏，所以改成组合式。
    """

    @pytest.mark.parametrize("prefix", ["不能", "不得", "不可", "不应", "不"])
    @pytest.mark.parametrize("verb", [
        "用于", "具备", "具有", "涉及", "属于", "声称", "宣称", "明示", "暗示",
    ])
    def test_combination_allowed(self, prefix: str, verb: str) -> None:
        text = f"{prefix}{verb}治疗作用"
        result = check_forbidden_claims(text)
        assert not result.has_critical, (
            f"否定搭配被误拦: {text}"
        )

    def test_fixed_phrases(self) -> None:
        """固定短语也要覆盖。"""
        for text in ("本品无治疗作用", "不属于疾病治疗范畴", "不视为治疗效果"):
            assert not check_forbidden_claims(text).has_critical, (
                f"固定短语被误拦: {text}")


# ===================================================================
# 4. 真违规不能因为否定规则而漏掉
# ===================================================================
class TestRealViolationsStillBlocked:
    @pytest.mark.parametrize("text", [
        "这个能治疗糖尿病",
        "能降血脂",
        "这个可以治疗高血脂",
        "虽然不是药，但能治疗",
        "该产品有治疗作用",
    ])
    def test_blocked(self, text: str) -> None:
        assert check_forbidden_claims(text).has_critical, f"漏检: {text}"


# ===================================================================
# 5. 综合场景（来自真实 LLM 输出）
# ===================================================================
class TestRealLLMOutput:
    def test_black_sesame_ball_answer_allowed(self) -> None:
        """★ 用户截图里被误拦的那段真实回答。

        补全后的完整句子应整体放行（它是合规的普通食品说明）。
        """
        answer = (
            "康源牌黑芝麻丸（产品编号：P019）是普通食品，不是保健食品，"
            "也不属于药品。\n"
            "根据《D008 产品知识手册（第三册：普通食品类）》，"
            "它的配料为黑芝麻和蜂蜜，建议零售价为 88 元，每丸 10g，"
            "可作为日常零食食用。\n\n"
            "需要说明的是：普通食品不能宣称任何保健功能或疾病预防、"
            "治疗功能，也不能替代药物。"
        )
        result = check_forbidden_claims(answer)
        assert not result.has_critical, (
            f"合规回答被误拦\n"
            f"命中: {[(v.rule_id, v.matched) for v in result.violations]}"
        )

    def test_stream_does_not_split_at_enumeration_comma(self) -> None:
        """★ 顿号不能作为流式切分点（用户实测暴露的根本原因）。

        原实现把「、」列为分句标点，导致：
            「…或疾病预防、」+「治疗功能，也不能替代药物。」
        第二段单独检测命中「治疗」→ 误拦。

        用户看到的现象：同一问题问两次，一次正常、一次拒答
        （LLM 每次措辞略有不同，有的版本恰好在顿号处凑够切分长度）。

        修法：`_CLAUSE_END` 去掉顿号。顿号是**并列**，切开会把概念劈开；
        逗号/冒号是**分句**，切开语义完整。
        """
        from domain.services.stream_guard import SentenceBuffer

        text = ("普通食品不能宣称任何保健功能或疾病预防、治疗功能，"
                "也不能替代药物。")
        buf = SentenceBuffer()
        pieces: list[str] = []
        for ch in text:
            pieces.extend(buf.feed(ch))
        pieces.extend(buf.flush())

        # 不应出现以顿号结尾的片段
        for p in pieces:
            assert not p.rstrip().endswith("、"), (
                f"流式在顿号处断开了: {p!r}，"
                f"这会让后半段单独检测时误拦"
            )

        # 整段应完整通过
        from domain.services.output_guardrail import OutputGuardrail
        from domain.services.stream_guard import StreamGuard

        guard = StreamGuard(guardrail=OutputGuardrail())
        out = "".join(guard.filter(iter(text)))
        assert not guard.blocked, f"合规内容被误拦: {guard.block_reason}"
        assert out == text


# ===================================================================
# 6. ★ 指代医生/医院已有诊疗（第二轮用户实测暴露）
# ===================================================================
class TestMedicalContextAllowed:
    """「以医生的诊断和治疗方案为准」这类表述必须放行。

    ★ 用户第二轮实测：问「辅酶Q10对心脏好不好」，LLM 答得完全正确，
      结尾「请以医生的诊断和治疗方案为准」却被 `DC-01` 拦了。

    根因：`DC-01` 的正则 `(治疗|治愈|根治)\\s*(?!疾病)`
      在「诊断和治疗方案」里命中「治疗」（后面是「方案」不是「疾病」）。

    而这句话是 D005 要求必须说的（提示就医），主语是**医生**，
    不是本品在宣称治疗效果。

    区分靠主语：
        主语 = 医生/医院/医嘱/诊疗方案 → 合规（指代第三方诊疗）
        主语 = 本品/这个/它          → 违规（宣称自己的治疗作用）
    """

    @pytest.mark.parametrize("text", [
        "请以医生的诊断和治疗方案为准",
        "遵医嘱，按医生的治疗方案执行",
        "以医院的诊断结论和治疗方案为准",
        "不要擅自改变医生的治疗方案",
        "配合医生的治疗",
        "正在接受医生的治疗",
        "如果您有健康问题，请以医生的诊断和治疗方案为准。",
        "这款不能替代医生的治疗方案",
        "是否调整用药请遵从医嘱",
    ])
    def test_allowed(self, text: str) -> None:
        result = check_forbidden_claims(text)
        assert not result.has_critical, (
            f"指代医生诊疗的合规表述被误拦: {text}\n"
            f"命中: {[(v.rule_id, v.matched) for v in result.violations]}"
        )

    @pytest.mark.parametrize("text", [
        "这个能治疗糖尿病",
        "本品可以治疗高血脂",
        "这个产品的治疗作用很好",
        "治疗效果很好",
    ])
    def test_own_product_claim_still_blocked(self, text: str) -> None:
        """★ 主语是本品时必须拦——不能因为放行「医生的治疗」就漏放。"""
        assert check_forbidden_claims(text).has_critical, (
            f"本品治疗宣称被漏放: {text}"
        )

    def test_real_coq10_answer_allowed(self) -> None:
        """★ 用户实测里那段真实回答（删去编造部分）。"""
        answer = (
            "这款产品是康源牌辅酶Q10软胶囊（P002），属于保健食品，"
            "批准的功能是「有助于增强免疫力」和「有助于抗氧化」，"
            "没有批准与心脏相关的功能声称，所以我不能说它对心脏有什么作用。\n\n"
            "保健食品不是药物，不能代替药物治疗疾病。"
            "如果您有心脏方面的健康问题，请以医生的诊断和治疗方案为准。"
        )
        result = check_forbidden_claims(answer)
        assert not result.has_critical, (
            f"合规回答被误拦\n"
            f"命中: {[(v.rule_id, v.matched) for v in result.violations]}"
        )
