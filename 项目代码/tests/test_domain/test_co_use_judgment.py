"""同服安全判断（JD-02）——2026-10-31 实测补充。

★ 背景：
  客户在吃其他保健品时，模型回了「可以一起吃」。
  鱼油与抗凝药同服有出血风险，门店无权做这个判断。
  这类回答不含数字，数字护栏拦不住，只能靠这条规则。

★ 同时修了一个老误报：
  JD-02 形态② 原本允许「可以 + 任意 0–8 字 + 一起服用」，
  把「可以和饭一起吃」「和牛奶一起服用」也当成同服判断拦掉。
  随餐服用是正常说法，必须放行。
"""
from __future__ import annotations

import pytest

from domain.rules.forbidden_claims import check_forbidden_claims


class TestCoUseBlocked:
    @pytest.mark.parametrize("text", [
        "可以一起吃，但建议先告诉我他现在吃的是什么。",
        "可以同时服用，没有问题。",
        "可以和抗凝药同时服用。",
        "可以和华法林一起吃。",
    ])
    def test_肯定同服要拦(self, text: str) -> None:
        assert check_forbidden_claims(text).has_critical


class TestMealContextAllowed:
    @pytest.mark.parametrize("text", [
        "本品可以和饭一起吃，随餐服用。",
        "可以和食物一起服用。",
        "可以和牛奶一起服用。",
        "不能一起吃，请先咨询医生。",
    ])
    def test_餐饮或否定放行(self, text: str) -> None:
        assert not check_forbidden_claims(text).has_critical
