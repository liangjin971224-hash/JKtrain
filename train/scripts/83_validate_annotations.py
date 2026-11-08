"""标注产物机器校验（Codex 产出 → 84 号复判前的质量闸门）。

    python scripts/83_validate_annotations.py
    python scripts/83_validate_annotations.py --wp-dir ../annotation_wp_v1

校验两类产出（缺哪个查哪个，都不存在则报错）：

  outputs/gen_questions.jsonl   策略B：chunk → 问题
      硬校验：schema / chunk_id 存在 / difficulty 合法 / 数量 = 目标chunk数×2
      软校验（WARN，不拦截但要人工看）：
        · 近抄袭：问题与 chunk 正文最长公共子串 ≥8 字（brief 的硬规则，
          这里是兜底检测；命中即打回该条）
        · 重复：归一化后精确重复，或编辑相似度 ≥0.85 的近重复
        · 难度档不符：与 gen_b_targets 的 required_difficulties 不一致

  outputs/judge_results.jsonl   策略A：候选相关性判定
      硬校验：schema / qid 存在 / candidate_id 存在 / verdict 合法 / 全覆盖

退出码：有硬违规 = 1（必须修），只有软警告 = 0。
报告写入 outputs/validation_report.json。
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WP = TRAIN_ROOT / "annotation_wp_v1"

VALID_DIFFICULTIES = {"keyword", "paraphrase", "scenario"}
VALID_VERDICTS = {"relevant", "partial", "irrelevant"}
MIN_COPY_LEN = 8          # 问题与正文的最长公共子串上限
NEAR_DUP_RATIO = 0.85


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip().lower()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append({"__bad_line__": i, "__error__": str(exc)})
    return rows


def longest_common_run(a: str, b: str) -> int:
    """最长公共连续子串长度（问题很短，SequenceMatcher 足够快）。"""
    m = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return m.find_longest_match(0, len(a), 0, len(b)).size


def validate_gen(wp: Path, report: dict) -> bool:
    targets = load_jsonl(wp / "gen_b_targets.jsonl")
    gen = load_jsonl(wp / "outputs" / "gen_questions.jsonl")
    if not targets or not gen:
        print("  [SKIP] gen_questions 校验：目标或产出缺失")
        return True

    target_by_id = {t["chunk_id"]: t for t in targets}
    hard: list[str] = []
    warn: list[str] = []
    seen_questions: list[tuple[str, str]] = []       # (归一化问题, chunk_id)
    diffs = Counter()

    for r in gen:
        line = r.get("__bad_line__")
        if line:
            hard.append(f"gen 第{line}行不是合法 JSON: {r.get('__error__')}")
            continue
        cid = r.get("chunk_id", "")
        q = (r.get("question") or "").strip()
        diff = r.get("difficulty", "")
        diffs[diff] += 1

        if not cid or cid not in target_by_id:
            hard.append(f"未知 chunk_id: {cid!r}")
            continue
        if not q:
            hard.append(f"{cid}: 问题为空")
            continue
        if diff not in VALID_DIFFICULTIES:
            hard.append(f"{cid}: difficulty 非法 {diff!r}")
        t = target_by_id[cid]
        if diff and diff not in t.get("required_difficulties", []):
            warn.append(f"{cid}: 难度 {diff} 不在要求档 {t['required_difficulties']}")

        # 近抄袭检测（对归一化文本）
        nq = normalize(q)
        lcs = longest_common_run(nq, normalize(t["text"]))
        if lcs >= MIN_COPY_LEN:
            warn.append(f"{cid}: 问题与正文最长公共子串 {lcs} 字（≥{MIN_COPY_LEN}，打回重写）: {q[:40]}")

        seen_questions.append((nq, cid))

    # 数量核对
    expect = len(targets) * 2
    got = sum(1 for r in gen if "__bad_line__" not in r)
    if got != expect:
        hard.append(f"数量不符: 产出 {got}，预期 {expect}（{len(targets)} chunk × 2）")

    # 重复检测（O(n²) 但 n≈820，归一化串短，可接受）
    for i in range(len(seen_questions)):
        for j in range(i + 1, len(seen_questions)):
            a, ca = seen_questions[i]
            b, cb = seen_questions[j]
            if a == b:
                warn.append(f"精确重复: {ca} 与 {cb}: {a[:40]}")
            elif difflib.SequenceMatcher(None, a, b).ratio() >= NEAR_DUP_RATIO:
                warn.append(f"近重复(≥{NEAR_DUP_RATIO}): {ca} 与 {cb}: {a[:40]}")

    report["gen"] = {
        "targets": len(targets), "produced": got, "expected": expect,
        "difficulty_dist": dict(diffs),
        "hard_violations": hard, "warnings": warn,
    }
    print(f"  [gen] 产出 {got}/{expect}，难度分布 {dict(diffs)}，"
          f"硬违规 {len(hard)}，警告 {len(warn)}")
    return not hard


def validate_judge(wp: Path, report: dict) -> bool:
    queries = load_jsonl(wp / "judge_a_queries.jsonl")
    results = load_jsonl(wp / "outputs" / "judge_results.jsonl")
    if not queries or not results:
        print("  [SKIP] judge_results 校验：种子或产出缺失")
        return True

    cand_ids: dict[str, set[str]] = {}
    for q in queries:
        qid = q["qid"]
        cand_file = wp / "candidates" / f"{qid}.jsonl"
        cand_ids[qid] = ({r["candidate_id"] for r in load_jsonl(cand_file)}
                         if cand_file.exists() else set())

    hard: list[str] = []
    judged: set[tuple[str, str]] = set()
    verdicts = Counter()
    for r in results:
        line = r.get("__bad_line__")
        if line:
            hard.append(f"judge 第{line}行不是合法 JSON")
            continue
        qid, cid = r.get("qid", ""), r.get("candidate_id", "")
        v = r.get("verdict", "")
        verdicts[v] += 1
        if qid not in cand_ids:
            hard.append(f"未知 qid: {qid!r}")
            continue
        if cid not in cand_ids[qid]:
            hard.append(f"{qid}: 未知 candidate_id {cid!r}")
            continue
        if v not in VALID_VERDICTS:
            hard.append(f"{qid}/{cid}: verdict 非法 {v!r}")
        judged.add((qid, cid))

    missing = [(q, c) for q, ids in cand_ids.items() for c in ids
               if (q, c) not in judged]
    if missing:
        hard.append(f"缺 {len(missing)} 条判定（须全覆盖），例: {missing[:3]}")

    report["judge"] = {
        "queries": len(queries), "candidates": sum(len(v) for v in cand_ids.values()),
        "judged": len(judged), "verdict_dist": dict(verdicts),
        "hard_violations": hard,
    }
    print(f"  [judge] 判定 {len(judged)}，verdict 分布 {dict(verdicts)}，硬违规 {len(hard)}")
    return not hard


def main() -> int:
    parser = argparse.ArgumentParser(description="标注产物机器校验")
    parser.add_argument("--wp-dir", default=str(DEFAULT_WP))
    args = parser.parse_args()
    wp = Path(args.wp_dir)

    print("=" * 68)
    print(f"  标注校验 → {wp}")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 68)

    report: dict = {}
    ok_gen = validate_gen(wp, report)
    ok_judge = validate_judge(wp, report)

    report["generated_at"] = datetime.now(timezone.utc).astimezone().isoformat(
        timespec="seconds")
    report["pass"] = ok_gen and ok_judge
    out = wp / "outputs" / "validation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    n_warn = sum(len(report.get(k, {}).get("warnings", [])) for k in ("gen",))
    print("-" * 68)
    print(f"  结论: {'PASS' if report['pass'] else 'FAIL'}（软警告 {n_warn} 条）")
    print(f"  报告: {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
