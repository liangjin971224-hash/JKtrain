"""把主数据转成阿里百炼的 SFT 格式（messages JSONL）。

    python scripts/70_to_bailian.py \
        --input data/samples_seed.jsonl data/samples_seed.add9.jsonl ... \
        --out-dir data/bailian

输出：
  bailian_train.jsonl / bailian_val.jsonl
      每行 {"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}
  dataset_report.json   条数、字符长度分布、分组切分、来源哈希

★ 与 convert_to_alpaca.py 的区别：
  百炼要的是**对话消息数组**，不是 instruction/input/output 三字段；
  而且系统提示词直接写进 messages，不用 LF 的模板。
  所以这里的 user 消息 = 证据块 + 问题（与线上 _build_prompt 同源，见 prompt_template）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_template import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REQUIRED = ("sample_id", "question", "output", "group_id", "review_status", "data_origin")


def stable_bucket(group_id: str, modulo: int = 10) -> int:
    return int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % modulo


def load(paths: list[Path]) -> tuple[list[dict], dict[str, int]]:
    rows, seen, counts = [], {}, {}
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                miss = [k for k in REQUIRED if not r.get(k)]
                if miss:
                    raise SystemExit(f"{p.name}:{lineno} 缺字段 {miss}")
                if r["sample_id"] in seen:
                    raise SystemExit(f"sample_id 重复：{r['sample_id']}")
                seen[r["sample_id"]] = p.name
                counts[r["review_status"]] = counts.get(r["review_status"], 0) + 1
                rows.append(r)
    return rows, counts


def to_messages(row: dict) -> dict:
    user = build_user_prompt(
        row["question"],
        fact_hints=row.get("fact_hints"),
        facts=row.get("facts"),
        references=row.get("references"),
    )
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["output"]},
    ]}


def write(path: Path, records: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, nargs="+")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--max-chars", type=int, default=3000,
                    help="超过这个字符数的样本给出警告（百炼按 token 限长，以控制台为准）")
    args = ap.parse_args()

    rows, counts = load(args.input)
    approved = [r for r in rows if r["review_status"] == "approved"]
    if not approved:
        raise SystemExit(f"没有 approved 样本（统计 {counts}）")

    splits: dict[str, list[dict]] = {"train": [], "val": []}
    group_split: dict[str, str] = {}
    for r in approved:
        gid = r["group_id"]
        want = "val" if stable_bucket(gid) < round(args.val_ratio * 10) else "train"
        prev = group_split.setdefault(gid, want)
        if prev != want:
            raise SystemExit(f"group_id {gid} 同时落入 train 与 val，同源样本必须同侧")
        splits[prev].append(r)

    report = {"sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in args.input],
              "status_counts_in_source": counts, "exported": {}}

    for name in ("train", "val"):
        recs = [to_messages(r) for r in splits[name]]
        chars = [sum(len(m["content"]) for m in rec["messages"]) for rec in recs]
        path = args.out_dir / f"bailian_{name}.jsonl"
        digest = write(path, recs)
        report["exported"][name] = {
            "file": path.name, "sha256": digest, "records": len(recs),
            "chars_median": sorted(chars)[len(chars) // 2] if chars else 0,
            "chars_max": max(chars) if chars else 0,
            "over_max_chars": sum(1 for c in chars if c > args.max_chars),
        }
        print(f"{path.name}: {len(recs)} 条 | 字符 中位 {report['exported'][name]['chars_median']} "
              f"最大 {report['exported'][name]['chars_max']} | 超 {args.max_chars} 字 {report['exported'][name]['over_max_chars']} 条")

    (args.out_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"报告 -> {args.out_dir / 'dataset_report.json'}")


if __name__ == "__main__":
    main()
