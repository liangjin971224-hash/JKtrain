"""报告建议校验器的测试。

★ 必须**双向测**：
  只测「引述能放行」会做出一个「什么都放行」的实现（真违规就漏了）；
  只测「违规能拦住」会做出一个「报告建议永远被拦」的实现（功能不可用）。
  两者同等重要 —— 这是项目一贯的教训（误报与漏报同级）。
"""
from __future__ import annotations

import pytest

from domain.services.report_guardrail import ReportGuardrail, strip_quoted


@pytest.fixture()
def g() -> ReportGuardrail:
    return ReportGuardrail()


# ===================================================================
# 一、必须放行（否则报告功能不可用）
# ===================================================================
class TestShouldPass:
    """这些是报告的正常写法，绝不能拦。"""

    @pytest.mark.parametrize("text", [
        # 引号内引述违规话术 —— 报告的核心内容
        "C07 第1轮第1句说「降血脂」属于疾病治疗，需重练。",
        "第2轮第1句「血脂降下来」属于经验背书。",
        "学员说了『治疗糖尿病』，这是明确违规。",
        '第1句说"降血脂"，属于功效声称。',
        "案例 C03 中「增强免疫力」的表述需要纠正。",
        # 多个引述
        "出现了「降血脂」「调节血脂」两处违规。",
        # 否定语境（现有护栏本来就放行）
        "不能对客户说降血脂。",
        "禁止使用「唯一」这类绝对化表述。",
        # 纯训练建议，无违规词
        "建议重练 C07，并复习 D005。",
        "本次表现稳定，建议增加案例覆盖广度。",
    ])
    def test_passes(self, g: ReportGuardrail, text: str) -> None:
        v = g.check(text)
        assert v.blocked is False, f"不该拦: {text} ← {v.reason}"


# ===================================================================
# 二、必须拦住（否则真违规漏过）
# ===================================================================
class TestShouldBlock:
    """这些是真正的违规表述，必须拦——哪怕混在报告里。"""

    @pytest.mark.parametrize("text", [
        # 引号外的声称
        "建议这样讲：这款产品能降血脂。",
        "可以告诉客户本产品能治疗高血压。",
        "你只要说能降血糖就行。",
        # 引号内引述 + 引号外也声称（最危险的一类）
        # ★ 用「降血脂」而不是「调节血脂」：后者不在护栏词表里
        #   （护栏的既有缺口，见下面 test_known_gap 的说明）
        "第1句说「降血糖」是错的，但你可以说能降血脂。",
        # 教违规话术
        "下次遇到这种情况，直接说产品能治好。",
    ])
    def test_blocks(self, g: ReportGuardrail, text: str) -> None:
        v = g.check(text)
        assert v.blocked is True, f"该拦却没拦: {text}"
        assert v.reason


# ===================================================================
# 三、边界：引号内外混合时必须拦住
# ===================================================================
class TestMixed:
    def test_quoted_plus_unquoted_claim_blocks(
            self, g: ReportGuardrail) -> None:
        """引号内是引述、引号外是声称 → 必须拦。

        ★ 这是最容易漏的一类：
          去引号后还剩「但你可以说能调节血脂」，仍是声称。
        """
        text = "「降血糖」这句不能说，但你可以说能降血脂。"
        v = g.check(text)
        assert v.blocked is True

    def test_quoted_only_passes(self, g: ReportGuardrail) -> None:
        text = "「降血脂」这句不能说。"
        assert g.check(text).blocked is False

    def test_known_gap_not_smooth_over(self, g: ReportGuardrail) -> None:
        """★ 记录一个**已知的护栏缺口**，不掩饰。

        「调节血脂」不在现有护栏的禁用词表里，所以：
            「「降血脂」不能说，但你可以说能调节血脂」
        会被放行 —— 而它后半句其实是违规的。

        这不是报告校验器的错，是**护栏词表的覆盖问题**。
        要不要补词表属于在线链路的改动（会影响学员回答判定），
        需要单独评估，不在 P1 范围内。

        这条测试的存在意义：**不要把已知缺口当成「已验证安全」。**
        """
        text = "「降血脂」这句不能说，但你可以说能调节血脂。"
        # 当前行为：放行（因为「调节血脂」不在词表）
        assert g.check(text).blocked is False

    def test_empty_text(self, g: ReportGuardrail) -> None:
        assert g.check("").blocked is False
        assert g.check("   ").blocked is False


# ===================================================================
# 四、strip_quoted 工具
# ===================================================================
class TestStripQuoted:
    @pytest.mark.parametrize("text,expected", [
        ("说「降血脂」属于违规", "说属于违规"),
        ("『治疗』和「降血糖」都要删", "和都要删"),
        ('"降血脂" 是违规', " 是违规"),
        ("没有引号就直接返回", "没有引号就直接返回"),
        ("", ""),
    ])
    def test_strip(self, text: str, expected: str) -> None:
        assert strip_quoted(text) == expected

    def test_multiline_quote_not_matched(self) -> None:
        """跨行的引号不配对（避免误吞大段文本）。"""
        text = "说「降血脂\n这句话」属于违规"
        # 引号内不含换行 → 不匹配，内容保留
        assert "降血脂" in strip_quoted(text)


# ===================================================================
# 五、与现有护栏的关系
# ===================================================================
class TestCompatibility:
    def test_uses_base_guardrail(self) -> None:
        """报告校验器复用现有护栏，不是另写一套规则。

        ★ 这点很重要：合规规则只能有一份实现，
          否则两边漂移（项目踩过）。
        """
        from domain.services.output_guardrail import OutputGuardrail

        base = OutputGuardrail()
        g = ReportGuardrail(base)
        # 纯违规文本在两边都该被拦
        assert base.check("这个能降血脂", category=None).blocked is True
        assert g.check("这个能降血脂").blocked is True
