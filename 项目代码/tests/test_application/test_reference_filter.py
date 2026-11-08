"""参考资料过滤测试：错误示范 + 示例块剔除。

★ 为什么要有这组测试
  2026-11-06 端到端实测（Ollama 3B）暴露：问「在吃阿司匹林，能吃鱼油吗」，
  模型把 D012《客户档案记录规范》里的「**填写示例**」代码块**整段照抄**当成答案：

      后续安排：
      - 建议客户先咨询医生确认能否食用鱼油
      - 3 天后（10-24）电话回访

  根因不在模型也不在清洗器，而在 **prompt 组装**：
  检索把「教学示例」当成相关资料喂进去了，模型又分不清「示例」与「事实」。

★ 判据设计（避免误伤）
  只剔除「**块标题命中** + 紧跟代码围栏」的组合：
      命中：**填写示例**：\n\n``` … ```
      不剔：## 七、评分示例\n\n**答卷 A**\n\n> 引用块   ← 用 > 不是围栏
      不剔：正文里出现「示例」二字但不是块标题
"""
from __future__ import annotations

import pytest

from application.use_cases.answer_question import _strip_bad_examples


class TestBadExamples:
    """原有行为：按行剔除错误示范（不能回退）。"""

    def test_strips_line_with_x_mark(self) -> None:
        text = "❌ 错误：客户有高血脂\n✅ 正确：客户自述血脂偏高"
        out = _strip_bad_examples(text)
        assert "错误：客户有高血脂" not in out
        assert "客户自述血脂偏高" in out

    def test_keeps_normal_lines(self) -> None:
        text = "客户自述在吃阿司匹林\n不适宜人群包括孕妇"
        out = _strip_bad_examples(text)
        assert "阿司匹林" in out
        assert "孕妇" in out


class TestExampleBlocks:
    """新增：块标题命中时剔除整块。"""

    def test_strips_fill_example_block(self) -> None:
        """D012 真实结构：**填写示例**：+ 代码围栏。"""
        text = (
            "| 字段 | 要求 |\n"
            "| 记录时间 | 标注是哪次到店说的 |\n\n"
            "**填写示例**：\n\n"
            "```\n"
            "✅ 正确：客户自述在吃阿司匹林（未核实）（2026-10-21）\n"
            "✅ 正确：客户自述在吃降压药（未说剂量）\n"
            "```\n\n"
            "── 沟通记录 ──\n"
            "客户关注：血脂"
        )
        out = _strip_bad_examples(text)
        # 示例块内容必须消失
        assert "客户自述在吃阿司匹林（未核实）" not in out
        # 前后正文必须保留
        assert "标注是哪次到店说的" in out
        assert "客户关注：血脂" in out

    def test_strips_multiple_fill_examples(self) -> None:
        """D012 里有两个「填写示例」块，都要剔。"""
        text = (
            "**填写示例**：\n\n```\n甲内容\n```\n\n"
            "正文保留一\n\n"
            "**填写示例**：\n\n```\n乙内容\n```\n\n"
            "正文保留二"
        )
        out = _strip_bad_examples(text)
        assert "甲内容" not in out
        assert "乙内容" not in out
        assert "正文保留一" in out
        assert "正文保留二" in out

    def test_keeps_scoring_example_with_blockquote(self) -> None:
        """「评分示例」用 > 引用块，不是围栏 → 不能剔。

        这是最容易被误伤的一类：标题里也有「示例」二字。
        """
        text = (
            "## 七、评分示例\n\n"
            "**答卷 A（正确，95 分）**\n\n"
            "> 这是保健食品，批准功能是「有助于增强免疫力」。\n\n"
            "覆盖 E1✅ E2✅，无违禁。"
        )
        out = _strip_bad_examples(text)
        assert "有助于增强免疫力" in out
        assert "答卷 A" in out

    def test_keeps_plain_mention_of_example_word(self) -> None:
        """正文里出现「示例」二字，但不是块标题 → 不能剔。"""
        text = "写记录时可以参考示例，但不要照抄。\n客户关注血脂。"
        out = _strip_bad_examples(text)
        assert "可以参考示例" in out
        assert "客户关注血脂" in out

    def test_heading_without_fence_is_kept(self) -> None:
        """块标题命中但没有代码围栏 → 不剔（避免误杀）。

        例：`### 5.2 八个改写示例` 下面跟的是表格或列表，不是围栏。
        """
        text = (
            "### 5.2 八个改写示例\n\n"
            "| 原说法 | 改写后 |\n"
            "| 降血压 | 有助于维持血压健康水平 |"
        )
        out = _strip_bad_examples(text)
        assert "有助于维持血压健康水平" in out

    def test_strips_block_with_two_backtick_fence(self) -> None:
        """★ 回归：入库分块会把 ``` 变成 ``（实测 2026-11-06）。

        只匹配 3 个反引号的规则会漏掉这种，导致示例块剔不掉、
        模型把示例当答案抄（真实故障现象）。
        """
        text = (
            "**填写示例**：\n\n"
            "``\n"
            "后续安排：\n"
            "- 建议客户先咨询医生确认能否食用鱼油\n"
            "- 3 天后电话回访\n"
            "``\n\n"
            "── 沟通记录 ──\n"
            "客户关注：血脂"
        )
        out = _strip_bad_examples(text)
        assert "建议客户先咨询医生确认能否食用鱼油" not in out
        assert "客户关注：血脂" in out

    def test_empty_input(self) -> None:
        assert _strip_bad_examples("") == ""
        assert _strip_bad_examples(None) == ""  # type: ignore[arg-type]
