"""校验 BGE-M3 微调标注产物（98 号工作包的 outputs/ft_questions.jsonl）。

    python train/scripts/99_validate_ft_annotations.py
    python train/scripts/99_validate_ft_annotations.py --wp-dir ../annotation_wp_v2

硬校验（任一失败 exit 1，必须修）：
    · 合法 JSON、字段恰好 {chunk_id, collection, question, difficulty}
    · chunk_id 存在、collection 一致
    · 每个 target 恰好 2 题，难度集合与 required_difficulties 一致
    · 问题与 chunk 正文、以及 existing_questions 的最大连续公共子串 <8 字
    · 问题归一化后不得重复；长度 8~120 字

软校验（WARN，人工抽检）：
    · 问题之间编辑相似度 ≥0.85 的近重复
    · 单题与正文公共子串 6~7 字（贴线）

★ 复用语 83 号的 normalize / longest_common_run，保证与 v1 标注同一口径。
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WP = TRAIN_ROOT / "annotation_wp_v2"
VALID_DIFFICULTIES = {"keyword", "paraphrase", "scenario"}
MAX_COPY_LEN = 8            # 连续 ≥8 字视为抄袭
NEAR_DUP_RATIO = 0.85
MIN_Q_CHARS = 8
MAX_Q_CHARS = 120
FIELDS = {"chunk_id", "collection", "question", "difficulty"}


def _load_83():
    path = Path(__file__).with_name("83_validate_annotations.py")
    spec = importlib.util.spec_from_file_location("validate83", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            rows.append({"__bad_line__": i, "__error__": str(exc)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="校验微调标注产物")
    ap.add_argument("--wp-dir", default=str(DEFAULT_WP))
    args = ap.parse_args()
    wp = Path(args.wp_dir)
    v = _load_83()

    targets = load_jsonl(wp / "targets.jsonl")
    pred_path = wp / "outputs" / "ft_questions.jsonl"
    preds = load_jsonl(pred_path)
    if not targets:
        print(f"[FAIL] 找不到 targets.jsonl：{wp}")
        return 1
    if not preds:
        print(f"[FAIL] 找不到产出：{pred_path}")
        return 1

    # 全量评测题：新增问题必须与任何一道评测题都不同（防跨 chunk 撞题）
    eval_path = TRAIN_ROOT / "data" / "retrieval_eval_v1_remap.jsonl"
    eval_questions = [
        (r["qid"], v.normalize(r["question"]))
        for r in load_jsonl(eval_path) if r.get("question")
    ]

    target_by_id = {t["chunk_id"]: t for t in targets}
    hard: list[str] = []
    warn: list[str] = []
    by_chunk: dict[str, list[dict]] = {}
    seen: list[tuple[str, str]] = []          # (归一化问题, chunk_id)

    for r in preds:
        line = r.get("__bad_line__")
        if line:
            hard.append(f"第{line}行不是合法 JSON: {r.get('__error__')}")
            continue
        extra = set(r) - FIELDS
        missing = FIELDS - set(r)
        if extra or missing:
            hard.append(f"字段不符：多 {sorted(extra)} 缺 {sorted(missing)} | {r}")
            continue
        cid, coll = r["chunk_id"], r["collection"]
        q, diff = (r.get("question") or "").strip(), r.get("difficulty")
        if cid not in target_by_id:
            hard.append(f"未知 chunk_id: {cid}")
            continue
        t = target_by_id[cid]
        if coll != t["collection"]:
            hard.append(f"{cid} collection 不一致: {coll} != {t['collection']}")
        if diff not in VALID_DIFFICULTIES:
            hard.append(f"{cid} 非法难度: {diff}")
        if not (MIN_Q_CHARS <= len(q) <= MAX_Q_CHARS):
            hard.append(f"{cid} 问题长度 {len(q)} 不在 {MIN_Q_CHARS}~{MAX_Q_CHARS}: {q[:30]}")
        # 8 字重叠：与正文、与已有评测问题
        run = v.longest_common_run(v.normalize(q), v.normalize(t["text"]))
        if run >= MAX_COPY_LEN:
            hard.append(f"{cid} 与正文连续 {run} 字重合: {q[:40]}")
        elif run >= MAX_COPY_LEN - 2:
            warn.append(f"{cid} 与正文连续 {run} 字重合（贴线）: {q[:40]}")
        for ex in t.get("existing_questions") or []:
            ex_run = v.longest_common_run(
                v.normalize(q), v.normalize(ex["question"]))
            if ex_run >= MAX_COPY_LEN:
                hard.append(
                    f"{cid} 与已有评测题连续 {ex_run} 字重合（{ex['qid']}）: {q[:40]}")
        # 与全量 205 评测题比对（防跨 chunk 撞题/近似）
        nq = v.normalize(q)
        for eqid, eq in eval_questions:
            if abs(len(nq) - len(eq)) > 12:
                continue
            if v.longest_common_run(nq, eq) >= MAX_COPY_LEN:
                hard.append(f"{cid} 与评测题 {eqid} 连续 ≥{MAX_COPY_LEN} 字重合: {q[:40]}")
                break
            ratio = difflib.SequenceMatcher(None, nq, eq).ratio()
            if ratio >= NEAR_DUP_RATIO:
                warn.append(f"{cid} 与评测题 {eqid} 近似 {ratio:.2f}: {q[:40]}")
                break
        # 重复
        if any(nq == prev for prev, _ in seen):
            hard.append(f"{cid} 问题重复: {q[:40]}")
        seen.append((nq, cid))
        by_chunk.setdefault(cid, []).append(r)

    # 覆盖与数量
    for cid, t in target_by_id.items():
        got = by_chunk.get(cid, [])
        need = len(t["required_difficulties"])
        if len(got) != need:
            hard.append(f"{cid} 数量 {len(got)} != {need}")
            continue
        got_diffs = Counter(x["difficulty"] for x in got)
        if got_diffs != Counter(t["required_difficulties"]):
            hard.append(
                f"{cid} 难度 {dict(got_diffs)} != {t['required_difficulties']}")
    for cid in by_chunk:
        if cid not in target_by_id:
            hard.append(f"多余 chunk: {cid}")

    # 近重复（软）
    nqs = [v.normalize(x["question"]) for x in preds if "question" in x]
    for i in range(len(nqs)):
        for j in range(i + 1, len(nqs)):
            if abs(len(nqs[i]) - len(nqs[j])) > 12:
                continue
            ratio = difflib.SequenceMatcher(None, nqs[i], nqs[j]).ratio()
            if ratio >= NEAR_DUP_RATIO:
                warn.append(f"近重复 {ratio:.2f}: {preds[i].get('question','')[:30]}")
                break

    report = {
        "wp_dir": str(wp),
        "n_targets": len(targets),
        "n_expected": sum(len(t["required_difficulties"]) for t in targets),
        "n_pred": len([p for p in preds if "chunk_id" in p]),
        "hard": hard,
        "warn": warn,
        "verdict": "PASS" if not hard else "FAIL",
        "generated_at": datetime.now(timezone.utc)
        .astimezone().isoformat(timespec="seconds"),
    }
    out = wp / "outputs" / "validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("=" * 68)
    print(f"  目标 {report['n_targets']} 块 / 期望 {report['n_expected']} 题 / "
          f"实收 {report['n_pred']} 题")
    print(f"  硬违规 {len(hard)} | 软警告 {len(warn)} | 判定 {report['verdict']}")
    for msg in hard[:20]:
        print(f"  [HARD] {msg}")
    if len(hard) > 20:
        print(f"  ... 其余 {len(hard) - 20} 条见报告")
    for msg in warn[:10]:
        print(f"  [WARN] {msg}")
    print(f"  报告 -> {out}")
    print("=" * 68)
    return 0 if not hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
