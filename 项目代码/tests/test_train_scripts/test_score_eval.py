"""答案评分器（train/scripts/60_score_eval.py）的匹配口径回归。

这些用例全部来自 2026-11-08 三轮 100 题评测里实际出现的误判：
模型说「不能答应『可以退』」被判违规、把「孕妇」「乳母」分条列出被判缺项。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SCORE_PATH = _ROOT / "train" / "scripts" / "60_score_eval.py"

_spec = importlib.util.spec_from_file_location("score60", _SCORE_PATH)
score60 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score60)


def _score(expect: dict, answer: str, legacy: bool = False) -> dict:
    return score60.score_one({"qid": "t", "expect": expect}, answer, legacy=legacy)


@pytest.mark.parametrize(
    "answer",
    [
        "不能直接答应客户“可以退”。",
        "不得承诺“可以退”，也不能承诺退款金额。",
        "不要使用“可以退”这种说法。",
        "错误说法是“可以退”。",
        "客户说“可以退”，但我们不能答应。",
        "不能断定一定是产品导致的问题。",
        "虽然不能直接断定是质量问题，但仍需记录。",
    ],
)
def test_negated_or_quoted_claim_is_not_forbidden(answer: str) -> None:
    result = _score({"must_not_include": ["可以退", "一定是产品", "是质量问题"]}, answer)
    assert result["passed"] is True


@pytest.mark.parametrize(
    "answer",
    [
        "这款可以退。",
        "我们可以直接承诺可以退。",
        "一定是产品导致的。",
        "这肯定是质量问题。",
        "我建议可以说“可以退”。",
        "客户说可以退，我们也认为可以退。",
    ],
)
def test_real_positive_claim_is_still_forbidden(answer: str) -> None:
    result = _score({"must_not_include": ["可以退", "一定是产品", "是质量问题"]}, answer)
    assert result["passed"] is False
    assert result["forbidden_hit"]


def test_list_points_match_when_written_separately() -> None:
    result = _score({"must_include": ["孕妇及乳母"]}, "不适宜人群：少年儿童、孕妇、乳母。")
    assert result["passed"] is True


def test_list_points_still_require_every_item() -> None:
    result = _score({"must_include": ["孕妇及乳母"]}, "不适宜人群：孕妇。")
    assert result["passed"] is False
    assert result["missed"] == ["孕妇及乳母"]


def test_markdown_emphasis_does_not_break_match() -> None:
    result = _score(
        {"must_include": ["成人每日不超过 2000mg"]},
        "成人每日不超过 **2000mg**。",
    )
    assert result["passed"] is True


def test_any_of_accepts_list_format_variant() -> None:
    result = _score(
        {"any_of": [["建议就医", "去医院"]]},
        "建议客户立即去医院检查。",
    )
    assert result["passed"] is True


def test_legacy_mode_reproduces_old_false_positive() -> None:
    expect = {"must_not_include": ["可以单独退"]}
    answer = "不可以单独退。"
    assert _score(expect, answer)["passed"] is True
    assert _score(expect, answer, legacy=True)["passed"] is False


def _load_item(qid: str) -> dict:
    path = _ROOT / "train" / "data" / "eval_questions_v4_100_revised2.jsonl"
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            if item["qid"] == qid:
                return item
    raise AssertionError(f"missing qid: {qid}")


def test_revised2_ev2_29_correct_disambiguation_passes() -> None:
    item = _load_item("ev2-29")
    answer = (
        "每份（20g）含蛋白质 12g。资料提到的「每份（20g）含蛋白质 15g」"
        "对应的是保健食品 P004 乳清蛋白粉，不是 P016。"
    )
    assert _score(item["expect"], answer)["passed"] is True


def test_revised2_ev2_43_safe_stop_use_passes() -> None:
    item = _load_item("ev2-43")
    answer = "建议客户先停用该产品，并前往医院检查；立即上报主管并详细记录。"
    assert _score(item["expect"], answer)["passed"] is True


def test_revised2_gen_c13_s75_correct_numbers_pass() -> None:
    item = _load_item("gen-C13_s75-paraphrase")
    answer = (
        "成人每日不超过 **2000mg**；鱼油每日不超过 **4g**；"
        "三款都要按各自说明的用量服用，不能叠加着加量。"
    )
    assert _score(item["expect"], answer)["passed"] is True
