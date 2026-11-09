"""按 expect 给评测结果打分。

    python scripts/60_score_eval.py \
        --eval data/eval_questions_v3.jsonl \
        --pred /mnt/workspace/output/adapter_eval_v4.jsonl

pred 就是 20_baseline.py / 50_compare.py 的输出（含 qid 与 answer）。

★ 判定口径（与审核报告一致）：
  must_include      —— 归一化后做**子串**匹配。归一化含：NFKC、去空白、
                       统一破折号、克→g、％→%、全角括号转半角。
                       列表连接词（、，,和及与/）视为可选，兼容
                       「孕妇及乳母 / 孕妇、乳母 / 孕妇和乳母」。
  must_not_include  —— 同样归一化后匹配。**必须写成短语**，
                       不要写「护心」这类裸词，否则会误伤
                       「资料没有批准保护心脏的功能」这种正确表述。
                       否定/引述语境不计命中：同一句内出现
                       不/没/禁止/不得/不要/无法等否定词，或
                       「客户说/错误说法/例如」等提及提示时，放行。

★ 这是个粗筛工具，不是语义判定器：
  它能稳定抓住「该说的没说」「不该出现的编造出现了」，
  但抓不住「说反了」「数字配错了对象」。最终仍需人工判读。

★ 兼容开关：`score_one(..., legacy=True)` 还原旧的纯子串口径，
  用于对比与新口径的差异；命令行 `--legacy` 同样生效。
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import unicodedata
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

_DASHES = "–—−－~〜"


def normalize(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    for d in _DASHES:
        s = s.replace(d, "-")
    # Markdown 强调/代码符不应打断「成人每日不超过 **2000mg**」这类匹配
    s = s.replace("**", "").replace("*", "").replace("`", "")
    s = s.replace("克", "g")
    s = "".join(ch for ch in s if not ch.isspace())
    return s


_AND_SEPARATORS = "、，,和及与"
_SEPARATORS = _AND_SEPARATORS + "/／"

# 同一句内出现这些词，说明模型在否定/劝阻，而不是在作肯定宣称。
_NEG_RE = re.compile(
    r"(?:不|没|禁止|避免|严禁|无法|并非|不得|不能|不可|不要|无需|无须|勿|"
    r"别(?:把|说|用|承诺)|未(?:曾|能|可|予|获))"
)

# 引述/教学语境：命中词只是被提到，不是被推荐。
_MENTION_RE = re.compile(
    r"(?:客户说|顾客说|客户认为|顾客认为|客户觉得|顾客觉得|客户以为|顾客以为|"
    r"听说|据说|所谓|禁用词|错误说法|反例|称为|例如|比如|表述|这句话|这种说法|"
    r"这样的说法|写成|写作|资料提到|文档提到|原文提到)"
)

# 命中词后面紧跟这些评价，同样说明是在指出它是错误说法。
_POST_MENTION_RE = re.compile(r"(?:是错误的|不对|不允许|禁止|违规|不能说|不可以|不得)")
_SENT_SPLIT_RE = re.compile(r"[。！？；!?;\n]")


@functools.lru_cache(maxsize=4096)
def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """把列表连接词变成可选，兼容「孕妇及乳母 / 孕妇、乳母」。"""
    normalized = normalize(phrase)
    parts = [re.escape(p) for p in re.split(f"[{re.escape(_SEPARATORS)}]", normalized) if p]
    if not parts:
        return re.compile(re.escape(normalized))
    return re.compile(f"[{re.escape(_SEPARATORS)}]?".join(parts))


def _contains(text: str, phrase: str) -> bool:
    """列表型要点按「每项都出现」判定，不要求相邻。

    这样「孕妇及乳母」在「孕妇」「乳母」分条列出时也算命中；
    斜杠仍按字面处理，同义表达请写进 any_of。
    """
    normalized = normalize(phrase)
    parts = [p for p in re.split(f"[{re.escape(_AND_SEPARATORS)}]", normalized) if p]
    if len(parts) > 1:
        return all(_phrase_pattern(part).search(text) is not None for part in parts)
    return _phrase_pattern(normalized).search(text) is not None


def _prefix(text: str, start: int, limit: int) -> str:
    return text[max(0, start - limit):start]


def _sentence_prefix(text: str, start: int, limit: int = 40) -> str:
    prefix = _prefix(text, start, limit)
    cut = 0
    for match in _SENT_SPLIT_RE.finditer(prefix):
        cut = match.end()
    return prefix[cut:]


def _is_excused(text: str, start: int, end: int) -> bool:
    """命中词是否处在否定/引述语境，因而不算违规。"""
    if _NEG_RE.search(_sentence_prefix(text, start)):
        return True
    if _MENTION_RE.search(_prefix(text, start, 8)):
        return True
    return _POST_MENTION_RE.search(text[end:end + 12]) is not None


def _has_forbidden(text: str, phrase: str) -> bool:
    """同一短语可能出现多次，只要有一次不是否定/引述语境就算命中。"""
    pattern = _phrase_pattern(phrase)
    return any(not _is_excused(text, m.start(), m.end()) for m in pattern.finditer(text))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def score_one(item: dict, answer: str, legacy: bool = False) -> dict:
    expect = item.get("expect") or {}
    text = normalize(answer)
    if legacy:
        missed = [m for m in expect.get("must_include", []) if normalize(m) not in text]
        for group in expect.get("any_of", []):
            if not any(normalize(g) in text for g in group):
                missed.append("／".join(group))
        hit_bad = [m for m in expect.get("must_not_include", []) if normalize(m) in text]
    else:
        missed = [m for m in expect.get("must_include", []) if not _contains(text, m)]
        # any_of: 每组至少命中一个；用于「结论由医生出具」这类同义表达
        for group in expect.get("any_of", []):
            if not any(_contains(text, g) for g in group):
                missed.append("／".join(group))
        hit_bad = [m for m in expect.get("must_not_include", []) if _has_forbidden(text, m)]
    return {
        "qid": item.get("qid"),
        "passed": not missed and not hit_bad,
        "missed": missed,
        "forbidden_hit": hit_bad,
        "answer_chars": len(answer or ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="逐题明细写这里（可选）")
    ap.add_argument("--show-fail", action="store_true", default=True)
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="还原旧口径（纯子串匹配），用于对比新旧评分器差异",
    )
    args = ap.parse_args()

    items = {i["qid"]: i for i in load_jsonl(args.eval)}
    preds = {p["qid"]: p for p in load_jsonl(args.pred)}

    results: list[dict] = []
    missing = [q for q in items if q not in preds]
    for qid, item in items.items():
        pred = preds.get(qid)
        if pred is None:
            continue
        rec = score_one(
            item,
            pred.get("answer") or pred.get("output") or "",
            legacy=args.legacy,
        )
        rec["question"] = item["question"]
        results.append(rec)

    passed = sum(r["passed"] for r in results)
    print(f"评测 {len(results)} 题 | 通过 {passed} | 未通过 {len(results) - passed}")
    if missing:
        print(f"⚠️ 预测里缺这些题：{missing}")

    if args.show_fail:
        for r in results:
            if r["passed"]:
                continue
            print(f"\n--- {r['qid']}  {r['question']}")
            if r["missed"]:
                print(f"    缺少要点: {r['missed']}")
            if r["forbidden_hit"]:
                print(f"    命中禁用: {r['forbidden_hit']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="\n") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n明细 -> {args.out}")


if __name__ == "__main__":
    main()
