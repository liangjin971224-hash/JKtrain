"""数字核实测试（NUM-01）。

★ 为什么单独成文件：
  「回答里出现资料里没有的数字」是实测四轮微调都压不住的编造模式
  （问价格答 128 元、问体脂答参考范围 18% 到 25%）。
  这道检查是确定性的纯函数，不依赖 LLM，必须能在任何机器上秒级跑绿。

规则来源：
    eval_questions.jsonl 的 ev-01 / ev-04 实测失败样本
"""
from __future__ import annotations

import pytest

from domain.services.output_guardrail import (
    strip_ungrounded_sentences,
    ungrounded_numbers,
)

EVIDENCE_PRICE_ABSENT = (
    "【结构化事实】\n产品：康源牌乳清蛋白粉（P004）\n"
    "批准文号：食健备G202544000124\n"
    "【参考资料】\n规格 400g/罐（约 20 份，每份 20g）。"
)

EVIDENCE_PRICE_PRESENT = (
    "产品编号 P001，康源牌鱼油软胶囊，规格 1.0g/粒 × 60 粒/瓶，建议零售价 298 元。"
)


class TestUngroundedNumbers:
    def test_编造价格被检出(self) -> None:
        assert ungrounded_numbers("售价是 128 元。", EVIDENCE_PRICE_ABSENT) == ("128",)

    def test_有依据的价格放行(self) -> None:
        assert ungrounded_numbers("建议零售价是 298 元。", EVIDENCE_PRICE_PRESENT) == ()

    def test_编造参考范围被检出(self) -> None:
        got = ungrounded_numbers("参考范围是 18% 到 25%。", EVIDENCE_PRICE_ABSENT)
        assert got == ("18", "25")

    def test_问题里的数字允许出现(self) -> None:
        # 客户说「体脂率 32%」，模型复述 32% 不算编造
        evidence = "员工的问题：客户体脂率测出来 32%，正常吗？"
        assert ungrounded_numbers("数值是 32%。", evidence) == ()

    def test_百分号与区间不误报(self) -> None:
        evidence = "每日用量上限：30–50mg。参考范围 1–9。"
        assert ungrounded_numbers("上限是 30–50mg，参考范围 1–9。", evidence) == ()

    def test_空回答不报错(self) -> None:
        assert ungrounded_numbers("", EVIDENCE_PRICE_ABSENT) == ()

    def test_数字顺序稳定(self) -> None:
        got = ungrounded_numbers("198 和 128 和 6", "")
        assert got == ("6", "128", "198")


class TestStripUngroundedSentences:
    def test_删掉编造句保留其余(self) -> None:
        text = "售价是 128 元。价格以门店标价为准，我不做推测。"
        kept, dropped = strip_ungrounded_sentences(text, EVIDENCE_PRICE_ABSENT)
        assert "128" not in kept
        assert "以门店标价为准" in kept
        assert dropped == ("128",)

    def test_整段都编则返回空(self) -> None:
        kept, dropped = strip_ungrounded_sentences("售价 128 元。", EVIDENCE_PRICE_ABSENT)
        assert kept == ""
        assert dropped == ("128",)

    def test_有依据时原样返回(self) -> None:
        text = "建议零售价是 298 元。规格 1.0g/粒 × 60 粒/瓶。"
        kept, dropped = strip_ungrounded_sentences(text, EVIDENCE_PRICE_PRESENT)
        assert kept == text
        assert dropped == ()

    @pytest.mark.parametrize("text", ["", "   "])
    def test_空文本(self, text: str) -> None:
        assert strip_ungrounded_sentences(text, EVIDENCE_PRICE_PRESENT) == ("", ())
