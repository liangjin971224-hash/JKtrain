"""引述语境识别回归测试（P4 真实链路暴露）。

★ 全部来自真实端到端测试的失败，不是假想场景。

背景：
    问「鱼油有什么功能」，千问给出的回答**完全合规而且很专业**：

        康源牌鱼油软胶囊是保健食品，其批准功能为：
        「有助于维持血脂（胆固醇/甘油三酯）健康水平」。
        需要特别说明：
        ✅ 这是国家市场监管总局批准的唯一功能表述；
        ❌ 不能说"降血脂""调节血脂""治疗高血脂"等；
        ❌ 不能说"护心""保护血管""预防心梗"等；
        保健食品不是药物，不能代替药物治疗疾病。

    但护栏把它拦了 —— 因为它把**引号里的反例**当成了违规命中。

这是一个典型的两难：
    放行引述  → 合规回答能过，但可能漏放「说了不能说之后又违规」的情况
    不放行    → 合规回答全被拦，业务不可用

本文件的每个测试都在守这个边界：
    **只放行「否定词 + 引号片段」本身，引号外的一律照常检测。**
"""
from __future__ import annotations

import pytest

from domain.rules.forbidden_claims import check_forbidden_claims


# ===================================================================
# 1. 引述语境必须放行（否则合规回答被误拦）
# ===================================================================
class TestQuotedNegationAllowed:
    """「不能说『降血脂』」这类反例教学是合规内容。"""

    @pytest.mark.parametrize("text", [
        "❌ 不能说「降血脂」",
        '❌ 不能说"降血脂"等表述',
        "❌ 不能说“降血脂”“调节血脂”“治疗高血脂”等；",
        "❌ 不能说“护心”“保护血管”“预防心梗”等",
        "- ❌ 不能说：治疗",
        "禁止使用「唯一」这类绝对化用语",
        "❌ 不能说“治疗”“降血糖”等",
        "不能说「降血脂」「治疗」等医疗术语",
        "❌ 不能将“维持血脂健康水平”等同于“降低血脂数值”",
        "禁止使用“最有效”“第一”等绝对化用语",
        "应改为「有助于维持血脂健康水平」",
        # 劝阻性否定表述（D005 要求的标准话术，同样含被禁词根）
        "不可擅自停药或替代治疗",
        "请遵医嘱，不可自行停药或改变治疗方案",
        "提醒客户遵医嘱用药，不可擅自停药",
    ])
    def test_quoted_examples_pass(self, text: str) -> None:
        result = check_forbidden_claims(text)
        assert not result.has_critical, (
            f"引述/反例被误判为违规: {text}\n"
            f"命中: {[v.matched for v in result.violations]}"
        )

    def test_real_llm_output_passes(self) -> None:
        """千问的真实回答（删去「唯一」那句）必须整体放行。

        这条是最直接的回归：它就是端到端跑出来的原文。
        """
        answer = (
            "康源牌鱼油软胶囊（P001）是保健食品，其批准功能为："
            "**有助于维持血脂（胆固醇/甘油三酯）健康水平**。\n\n"
            "需要特别说明：  \n"
            "❌ 不能说“降血脂”“调节血脂”“治疗高血脂”等；  \n"
            "❌ 不能说“护心”“保护血管”“预防心梗”等与疾病防治相关的内容；  \n"
            "❌ 不能将“维持血脂健康水平”等同于“降低血脂数值”。\n\n"
            "同时必须强调：  \n"
            "**保健食品不是药物，不能代替药物治疗疾病。**  \n\n"
            "如客户有血脂异常相关诊断或正在服用降脂药，请提醒其遵医嘱用药，"
            "不可擅自停药或替代治疗。"
        )
        result = check_forbidden_claims(answer)
        assert not result.has_critical, (
            f"合规的真实 LLM 输出被误拦\n"
            f"命中: {[(v.rule_id, v.matched) for v in result.violations]}"
        )


# ===================================================================
# 2. ★ 引述放行的边界：不能漏放真违规
# ===================================================================
class TestNegationBoundary:
    """放行引述的最大风险是「把真违规也放掉」。这里逐条守住。"""

    @pytest.mark.parametrize("text", [
        # 没有引号 → 不是引述
        "这个能降血脂，效果很好",
        "今儿个能治疗糖尿病",
        # 引号后还有转折 → 后半句是真的
        "❌ 不能说“降血脂”，我们这款确实降血脂",
        "❌ 不能说「降血脂」，但这个确实降血脂",
        # 无引号且后面还有内容 → 无法判断，按可疑处理
        "❌ 不能说：治疗，我们这个确实能治好",
        # 肯定语境 + 引号（引号里是产品名，不是反例）
        "客户问能不能治疗，我说可以的",
        # 普通违规
        "很多人吃了都说好",
        "保证有效，没效果退款",
        "这是行业第一品牌",
    ])
    def test_still_blocked(self, text: str) -> None:
        result = check_forbidden_claims(text)
        assert result.has_critical, (
            f"真违规被漏放: {text}\n"
            f"这是安全事故级别的问题——违规话术会直接输出给客户。"
        )

    def test_comma_after_quote_does_not_swallow_rest(self) -> None:
        """★ 引号后面的逗号不能被吃进匹配范围。

        实测教训：正则写成 `[、，,\\s]*` 时，
        「❌ 不能说“降血脂”，我们这款确实降血脂」的整个前半段
        连逗号一起被挖掉，后半句真违规被移位漏检。

        现在只允许引号间用顿号/空白，逗号留给后续检测。
        """
        text = '❌ 不能说"降血脂"，我们这款确实降血脂'
        result = check_forbidden_claims(text)
        assert result.has_critical, (
            "逗号后的真违规被漏放了。引号之间只能用顿号，不能用逗号。"
        )

    def test_etc_suffix_does_not_swallow_rest(self) -> None:
        """★ 「等…」后缀不能无限延伸。

        实测教训：加了 `等[^。！？\\n]{0,10}` 后，
        「❌ 不能说“降血脂”等，我们这款确实降血脂」的后半句被吞。
        """
        text = '❌ 不能说"降血脂"等，我们这款确实降血脂'
        result = check_forbidden_claims(text)
        assert result.has_critical, "「等」后的真违规被漏放了"

    def test_bare_negation_whole_line_only(self) -> None:
        """「❌ 不能说：治疗」整行成立，加逗号续写就不成立。"""
        assert not check_forbidden_claims("❌ 不能说：治疗").has_critical
        assert check_forbidden_claims(
            "❌ 不能说：治疗，我们这个确实能治好").has_critical

    def test_unjquoted_table_form_still_blocked(self) -> None:
        """★ 无引号的表格对照写法，**按可疑处理（拦）**。

        例如培训材料里常见的：
            错误说法：降血脂；正确说法：有助于维持血脂健康水平

        这里「降血脂」没有引号，与真违规「这个能降血脂」在文本上无法区分。
        按既定原则（无引号不放行），宁可误拦。

        ★ 影响评估：护栏只对**模型输出**生效。
          模型的自然输出不会出现「错误说法：X；正确说法：Y」这种表格化结构
          （prompt 里已要求「不要原样照抄参考资料」）。
          若真出现，误拦的代价是人工复核一条，远小于漏放的代价。
        """
        result = check_forbidden_claims(
            "错误说法：降血脂；正确说法：有助于维持血脂健康水平")
        assert result.has_critical, "无引号的对照写法应按可疑处理"


# ===================================================================
# 3. 不得因引述处理而破坏原本正常的能力
# ===================================================================
class TestNoRegression:
    """引述规则不能把「本来就该拦」的规则弄坏。"""

    @pytest.mark.parametrize("text", [
        "这个能降血糖",
        "可以治疗糖尿病",
        "彻底根治失眠",
        "有很好的疗效",
        "能抗炎消炎",
        "可以停降压药了",
        "比药还管用",
        "最有效的产品",
        "100% 有效",
        "包治百病",
        "三天见效",
        "保证有效",
        "很多人吃了都说好",
        "我朋友吃了就有效",
        "增强免疫力",
    ])
    def test_plain_violations_still_detected(self, text: str) -> None:
        assert check_forbidden_claims(text).has_critical, f"漏检: {text}"

    @pytest.mark.parametrize("text", [
        "有助于维持血糖健康水平",
        "保健食品不是药物，不能代替药物治疗疾病",
        "个体差异较大，不能承诺具体效果",
        "建议咨询医生",
    ])
    def test_compliant_still_passes(self, text: str) -> None:
        assert not check_forbidden_claims(text).has_critical, f"误拦: {text}"
