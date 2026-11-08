"""把主数据转成火山方舟（Ark）精调格式，并做方舟侧的格式校验。

    python scripts/71_to_ark.py \
        --input data/samples_seed.jsonl ... --out-dir data/ark139

输出：
  ark_train.jsonl / ark_val.jsonl
      每行 {"messages":[{"role":"system"|"user"|"assistant","content":"..."}]}
  ark_dataset_report.json  条数、字符/token 估算、校验结果、来源哈希

★ 方舟与百炼的数据格式都是 messages JSONL，差别在平台侧：
  方舟要把文件传 TOS → 导入为数据集 → 校验 → 建精调任务 → 部署推理接入点。
  这里只负责「产出格式正确、能过校验」的文件。

★ 校验项（对应方舟导入时的常见拒绝原因）：
  1. 每行必须是 {"messages":[...]}
  2. role 只能是 system / user / assistant
  3. system 最多一条且在首位
  4. content 必须是非空字符串（不能是数组、不能为 null）
  5. 同一 group_id 不能跨 train/val
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
ROLES = {"system", "user", "assistant"}


def stable_bucket(group_id: str, modulo: int = 10) -> int:
    return int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % modulo


def approx_tokens(text: str) -> int:
    """粗估 token：中文约 1.5 字符/token，英文约 4 字符/token。只用于预警，不精确。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4) + 1


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


def validate(rec: dict, where: str) -> list[str]:
    errs = []
    if set(rec) != {"messages"}:
        errs.append(f"{where}: 顶层字段应只有 messages，实际 {sorted(rec)}")
        return errs
    msgs = rec["messages"]
    if not isinstance(msgs, list) or len(msgs) < 2:
        errs.append(f"{where}: messages 至少要有 user + assistant")
        return errs
    if msgs[0]["role"] == "system":
        if sum(1 for m in msgs if m["role"] == "system") > 1:
            errs.append(f"{where}: system 多于一条")
        body = msgs[1:]
    else:
        body = msgs
    for i, m in enumerate(msgs):
        if m.get("role") not in ROLES:
            errs.append(f"{where}: 第 {i} 条 role 非法 {m.get('role')!r}")
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            errs.append(f"{where}: 第 {i} 条 content 不是非空字符串")
    if [m["role"] for m in body] != ["user", "assistant"]:
        errs.append(f"{where}: 单轮样本应为 user → assistant，实际 {[m['role'] for m in body]}")
    return errs


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
    ap.add_argument("--max-tokens", type=int, default=8000,
                    help="估算 token 超过这个值就预警（方舟单条上限以控制台为准）")
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

    report = {
        "platform": "volcengine-ark",
        "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in args.input],
        "status_counts_in_source": counts,
        "exported": {},
        "errors": [],
    }

    for name in ("train", "val"):
        recs = [to_messages(r) for r in splits[name]]
        for i, rec in enumerate(recs):
            report["errors"] += validate(rec, f"{name}[{i}]")
        toks = [approx_tokens("".join(m["content"] for m in rec["messages"])) for rec in recs]
        path = args.out_dir / f"ark_{name}.jsonl"
        digest = write(path, recs)
        report["exported"][name] = {
            "file": path.name, "sha256": digest, "records": len(recs),
            "tokens_median": sorted(toks)[len(toks) // 2] if toks else 0,
            "tokens_max": max(toks) if toks else 0,
            "over_max_tokens": sum(1 for t in toks if t > args.max_tokens),
        }
        e = report["exported"][name]
        print(f"{e['file']}: {e['records']} 条 | token 估算 中位 {e['tokens_median']} "
              f"最大 {e['tokens_max']} | 超 {args.max_tokens} {e['over_max_tokens']} 条")

    if report["errors"]:
        print(f"\n校验失败 {len(report['errors'])} 处：")
        for err in report["errors"][:10]:
            print("  -", err)
    else:
        print("\n格式校验：通过")

    (args.out_dir / "ark_dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"报告 -> {args.out_dir / 'ark_dataset_report.json'}")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
