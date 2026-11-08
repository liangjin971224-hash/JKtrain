"""教学语境下的引述放行（2026-10-31 实测补充）。

★ 背景：
  本项目要求「只做正面陈述」，于是合规答案写成
      「防癌」不在国家批准的 24 项功能目录内
      药物替代类里最危险的是「可以停药」这一条
  这两句都在**讨论禁用词**，却因引号形式（直引号 / 单引号）
  未被 `_QUOTED_TERM` 覆盖而被整段拦住，
  实测把 ev2-04、ev2-05 两道规则题答成了拒答模板。

★ 放行边界不变的底线：
  引号外说明「本品有某功效」仍必须拦。
"""
from __future__ import annotations

import pytest

from domain.rules.forbidden_claims import check_forbidden_claims


class TestQuotedTeachingAllowed:
    @pytest.mark.parametrize("text", [
        # 全角直角引号
        "「防癌」不在国家批准的 24 项保健食品功能目录中。",
        # 直引号（实测模型更常用）
        "目前国家批准的保健功能中没有'防癌'这一项。",
        # 弯引号
        "“防癌”不属于可声称的范围。",
        # 引号在前、判类在后
        "药物替代类里最危险的是「可以停药」这一条。",
        # 停药风险的客观描述
        "该表述可能导致客户延误治疗，对健康造成实质性风险。",
    ])
    def test_不拦(self, text: str) -> None:
        assert not check_forbidden_claims(text).has_critical


class TestRealViolationStillBlocked:
    @pytest.mark.parametrize("text", [
        "这个能降血脂，吃了就不用吃降脂药了。",
        "本品可以治疗糖尿病。",
        "吃我们的产品能防癌抗癌。",
    ])
    def test_仍拦(self, text: str) -> None:
        assert check_forbidden_claims(text).has_critical
