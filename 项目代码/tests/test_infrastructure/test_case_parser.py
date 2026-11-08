"""案例解析器测试（P5）。

★ 全部针对**真实文件**（`06_正式内容/02_情景案例/`）跑。
  解析器最容易犯的错是「在某个特定案例上能跑，遇到格式变体就崩」——
  而格式变体恰恰是真实文档的常态。

本文件记录的格式差异（都是实测踩到的）：
    ① 两个文件的标题层级不同：`###` vs `##`
    ② 章节名不同：「违禁表述检测」vs「违禁检测」
    ③ 违禁表格列数不同：4 列（含级别）vs 3 列（级别在标题里）
    ④ 正则单元格内含 `|`，与表格分隔符冲突
"""
from __future__ import annotations

from pathlib import Path

import pytest

from domain.services.case_parser import (
    dedupe_cases,
    parse_case_text,
)
from infrastructure.content.case_loader import load_cases_from_dir

CASE_DIR = Path(
    r"D:\项目\某健康公司培训系统\06_正式内容\02_情景案例")


@pytest.fixture(scope="module")
def cases():
    if not CASE_DIR.exists():
        pytest.skip(f"案例目录不存在: {CASE_DIR}")
    return load_cases_from_dir(CASE_DIR)


@pytest.fixture(scope="module")
def by_id(cases):
    return {c.case_id: c for c in cases}


# ===================================================================
# 1. 覆盖完整性
# ===================================================================
class TestCoverage:
    def test_all_16_cases_parsed(self, cases) -> None:
        """16 个案例必须全部解析出来。"""
        assert len(cases) == 16, (
            f"应解析 16 个案例，实际 {len(cases)}: "
            f"{[c.case_id for c in cases]}"
        )

    def test_case_ids_complete(self, cases) -> None:
        ids = {c.case_id for c in cases}
        expected = {f"C{i:02d}" for i in range(1, 17)}
        assert ids == expected, f"缺失: {expected - ids}"

    def test_every_case_has_elements(self, cases) -> None:
        """每个案例都必须解析出必答要素——没有要素就无法评阅。"""
        empty = [c.case_id for c in cases if not c.elements]
        assert empty == [], f"这些案例没解析出要素: {empty}"

    def test_every_case_has_forbidden_rules(self, cases) -> None:
        """每个案例都必须解析出违禁规则。

        ★ 这条曾经失败过：C01/C04 因为章节名是「违禁表述检测」、
          另外 8 个因为表格只有 3 列，规则全部丢失。
        """
        empty = [c.case_id for c in cases if not c.forbidden_rules]
        assert empty == [], (
            f"这些案例没解析出违禁规则: {empty}。"
            f"规则缺失意味着评阅会漏掉全部违规。"
        )

    def test_every_case_has_opening_line(self, cases) -> None:
        """开场白缺失就没法开始演练。"""
        empty = [c.case_id for c in cases if not c.opening_line]
        assert empty == [], f"这些案例缺开场白: {empty}"

    def test_total_rules_reasonable(self, cases) -> None:
        total = sum(len(c.forbidden_rules) for c in cases)
        assert total >= 50, f"违禁规则总数偏少（{total}），可能有解析遗漏"


# ===================================================================
# 2. 关键字段正确性
# ===================================================================
class TestC01Detail:
    def test_opening_line(self, by_id) -> None:
        assert by_id["C01"].opening_line == "你们有鱼油吗？"

    def test_seven_elements(self, by_id) -> None:
        c = by_id["C01"]
        assert len(c.elements) == 7
        assert [e.element_id for e in c.elements] == [
            "E1", "E2", "E3", "E4", "E5", "E6", "E7"]

    def test_required_vs_optional(self, by_id) -> None:
        """E1–E5 必答，E6/E7 建议（案例文档明确说明）。"""
        c = by_id["C01"]
        required = {e.element_id for e in c.required_elements}
        assert required == {"E1", "E2", "E3", "E4", "E5"}

    def test_weights_parsed(self, by_id) -> None:
        c = by_id["C01"]
        weights = {e.element_id: e.weight for e in c.elements}
        assert weights["E1"] == 20
        assert weights["E2"] == 10
        assert weights["E5"] == 15

    def test_critical_rules_present(self, by_id) -> None:
        c = by_id["C01"]
        assert len(c.critical_rules) >= 2

    def test_regex_with_pipe_parsed(self, by_id) -> None:
        """★ 正则里的 `|` 不能被当成表格分隔符切碎。"""
        c = by_id["C01"]
        patterns = [r.pattern for r in c.forbidden_rules]
        # 「降血脂|降胆固醇|治疗」这种必须完整保留
        assert any("降血脂" in p and "降胆固醇" in p for p in patterns), (
            f"含 | 的正则被切碎了，实际 patterns: {patterns}"
        )

    def test_follow_ups_parsed(self, by_id) -> None:
        assert len(by_id["C01"].follow_ups) == 5


class TestOtherCases:
    def test_c03_three_column_table(self, by_id) -> None:
        """★ C03 的违禁表只有 3 列，级别写在章节标题里。"""
        c = by_id["C03"]
        assert c.forbidden_rules, "3 列表格没能解析出规则"
        # 标题写「（全部 critical）」，所以默认级别是 critical
        assert any(r.level.is_blocking for r in c.forbidden_rules)

    def test_c04_has_six_rules(self, by_id) -> None:
        """C04 是核心案例，规则应较全。"""
        assert len(by_id["C04"].forbidden_rules) >= 4


# ===================================================================
# 3. 格式变体处理
# ===================================================================
class TestFormatVariants:
    def test_both_heading_levels(self) -> None:
        """两个源文件标题层级不同，解析器都要能处理。"""
        text_h2 = "## 案例 C01：测试\n\n## 一、基本设定\n| case_id | C01 |\n"
        text_h3 = "## 案例 C02：测试\n\n### 一、案例基本设定\n| case_id | C02 |\n"

        assert parse_case_text(text_h2, "h2")
        assert parse_case_text(text_h3, "h3")

    def test_no_case_head_returns_empty(self) -> None:
        assert parse_case_text("这是纯文本，没有案例。", "x") == []

    def test_invalid_regex_skipped(self) -> None:
        """案例文档里写错正则时，跳过该条而不是整体失败。

        ★ 为什么要这样：案例是人工维护的，难免写错。
          一条正则写错就让整个案例不可评阅，代价太大。
        """
        text = (
            "## 案例 C99：测试\n"
            "## 一、基本设定\n| case_id | C99 |\n"
            "## 四、必答要素\n"
            "| ID | 要素 | 权重 | 类型 |\n|---|---|---|---|\n"
            "| E1 | 明确回答 | 20 | 必须 |\n"
            "## 五、违禁检测\n"
            "| 类型 | 正则 | 级别 |\n|---|---|---|\n"
            "| 坏正则 | `[未闭合` | **critical** |\n"
            "| 好正则 | `降血脂` | **critical** |\n"
        )
        cases = parse_case_text(text, "test")
        assert len(cases) == 1
        # 坏正则被跳过，好正则保留
        patterns = [r.pattern for r in cases[0].forbidden_rules]
        assert "降血脂" in patterns
        assert "[未闭合" not in patterns


# ===================================================================
# 4. 去重
# ===================================================================
class TestDedupe:
    def test_prefers_richer_spec(self) -> None:
        """C01 在两个文件都出现，保留要素更多的那份。"""
        from domain.value_objects import CaseSpec, ElementSpec

        thin = CaseSpec("C01", "thin", "o",
                        elements=(ElementSpec("E1", "x", 10, True),))
        rich = CaseSpec("C01", "rich", "o", elements=(
            ElementSpec("E1", "x", 10, True),
            ElementSpec("E2", "y", 10, True),
            ElementSpec("E3", "z", 10, True),
        ))
        merged = dedupe_cases([thin, rich])
        assert len(merged) == 1
        assert len(merged[0].elements) == 3

    def test_sorted_by_id(self, cases) -> None:
        ids = [c.case_id for c in cases]
        assert ids == sorted(ids)


# ===================================================================
# 4. 表格粗体键（2026-10-31 回归）
# ===================================================================
class TestBoldKeyParsing:
    """两批案例的表格写法不一致，键可能带 `**` 粗体。

    ★ 真实 bug（2026-10-31）：
      C01/C04 出自单独文件，表格写成 `| **提问类型** | ... |`，
      其余 14 个写成 `| 提问类型 | ... |`。
      原实现直接拿 key 当字典键 → C01/C04 的键变成 `**提问类型**`
      → `question_type` 解析为空串
      → 月度报告里这两个案例归不进任何能力项（C04 是唯一的红线案例）。
    """

    def test_all_cases_have_question_type(self, cases) -> None:
        """16 个案例都必须解析出提问类型 —— 它是能力项归并的唯一依据。"""
        empty = [c.case_id for c in cases if not c.question_type]
        assert empty == [], f"这些案例的 question_type 为空: {empty}"

    def test_c01_c04_question_type(self, by_id) -> None:
        """C01/C04 的表格带粗体键，必须能正常取到值。"""
        assert by_id["C01"].question_type == "产品信息询问（product_existence）"
        assert by_id["C04"].question_type == "替代药物询问（drug_replacement）"

    def test_all_cases_have_difficulty(self, cases) -> None:
        """难度也必须解析出来（同为粗体键问题）。"""
        zero = [c.case_id for c in cases if c.difficulty == 0]
        assert zero == [], f"这些案例难度解析为 0: {zero}"

    def test_all_cases_have_stage(self, cases) -> None:
        empty = [c.case_id for c in cases if not c.stage]
        assert empty == [], f"这些案例 stage 为空: {empty}"

    def test_bold_key_stripped_directly(self) -> None:
        """直接验证解析函数：粗体键与普通键结果一致。"""
        from domain.services.case_parser import _parse_kv_table

        plain = _parse_kv_table("| 提问类型 | X |\n| 难度 | ★☆☆ |")
        bold = _parse_kv_table("| **提问类型** | X |\n| **难度** | ★☆☆ |")
        assert plain == bold == {"提问类型": "X", "难度": "★☆☆"}
