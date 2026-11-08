"""主数据审核工具：按条查看、批量标 approved/rejected、导出前体检。

    python scripts/review_cli.py stats  --input data/samples_*.jsonl
    python scripts/review_cli.py list   --input data/samples_*.jsonl --tier redline
    python scripts/review_cli.py show   --input data/samples_*.jsonl --id sft-0001
    python scripts/review_cli.py set    --input data/samples_*.jsonl \
        --id sft-0001 sft-0002 --status approved --reviewer zhangsan
    python scripts/review_cli.py check  --input data/samples_*.jsonl

`set` 直接改写源 JSONL 的 review_status / reviewer_id，保留其余字段与原始顺序。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# 按风险排序审：审错一条的后果越重，排在越前
TIER_BY_DOC = {
    "D005": "redline",   # 功能声称规范、绝对化用语、疾病关联
    "D008": "redline",   # 普通食品三条铁律
    "D013": "redline",   # 退换货与上报
    "D014": "redline",   # 急救红旗征
    "D010": "device",    # 体脂仪解读
    "D011": "device",    # 骨密度仪解读
    "D006": "product",   # 产品参数
    "D007": "product",   # 目录内产品
    "D009": "dialog",    # 异议处理
}
TIER_ORDER = ["redline", "device", "product", "dialog", "meta"]


def tier_of(row: dict) -> str:
    """多份来源取风险最高的那层，不能只看第一个来源。"""
    found = []
    for sid in row.get("source_ids", []):
        doc = str(sid).split("#", 1)[0]
        if doc in TIER_BY_DOC:
            found.append(TIER_BY_DOC[doc])
    if not found:
        return "meta"
    return min(found, key=TIER_ORDER.index)


def read_rows(paths: list[Path]) -> list[tuple[Path, int, dict]]:
    out = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                out.append((path, lineno, json.loads(line)))
    return out


def write_rows(path: Path, rows: list[tuple[int, dict]]) -> None:
    """整文件重写；先留一份 .bak，覆盖前可回退。"""
    if path.exists():
        path.with_suffix(path.suffix + ".bak").write_bytes(path.read_bytes())
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for _, row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_stats(rows) -> None:
    total = len(rows)
    by_status: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    for _, _, row in rows:
        by_status[row.get("review_status", "?")] = by_status.get(row.get("review_status", "?"), 0) + 1
        t = tier_of(row)
        by_tier[t] = by_tier.get(t, 0) + 1
    print(f"总数 {total}")
    print("按状态:", json.dumps(by_status, ensure_ascii=False))
    print("按风险层:", json.dumps({k: by_tier.get(k, 0) for k in TIER_ORDER}, ensure_ascii=False))


def cmd_list(rows, tier: str, status: str, limit: int) -> None:
    picked = [
        row for _, _, row in rows
        if (tier == "all" or tier_of(row) == tier) and (status == "all" or row.get("review_status") == status)
    ]
    picked.sort(key=lambda r: (TIER_ORDER.index(tier_of(r)), r["sample_id"]))
    if limit:
        picked = picked[:limit]
    for row in picked:
        print(f"{row['sample_id']}\t{tier_of(row):8s}\t{row.get('review_status'):9s}\t"
              f"{row.get('group_id','')}\t{row['question'][:40]}")
    print(f"\n共 {len(picked)} 条")


def cmd_show(rows, ids: list[str]) -> None:
    index = {row["sample_id"]: row for _, _, row in rows}
    for sid in ids:
        row = index.get(sid)
        if row is None:
            print(f"!! 找不到 {sid}")
            continue
        print(f"=== {sid} | {tier_of(row)} | {row.get('review_status')} | {row.get('group_id')} ===")
        print("【问题】", row.get("question"))
        if row.get("fact_hints"):
            print("【事实提示】", " / ".join(row["fact_hints"]))
        for ref in row.get("references", []):
            print(f"【资料 {ref.get('doc_id')} {ref.get('section')}】{ref.get('body')}")
        print("【目标答案】", row.get("output"))
        print()


def cmd_set(rows, ids: list[str], status: str, reviewer: str, note: str,
            files: list[Path] | None = None) -> None:
    target = set(ids)
    if files:
        # 整文件批量标记：命令行里不用自己拼 161 个 sample_id
        wanted = {f.resolve() for f in files}
        target |= {row["sample_id"] for path, _, row in rows if path.resolve() in wanted}
    found: set[str] = set()
    per_file: dict[Path, list[tuple[int, dict]]] = {}
    for path, lineno, row in rows:
        # 每个文件的全部行都要收回来，只写命中的行会把其余行抹掉
        per_file.setdefault(path, []).append((lineno, row))
        if row["sample_id"] in target:
            found.add(row["sample_id"])
            row["review_status"] = status
            if status == "approved":
                row["reviewer_id"] = reviewer
            if note:
                row["review_note"] = note
    missing = target - found
    if missing:
        raise SystemExit(f"找不到这些 sample_id：{sorted(missing)}")
    for path, file_rows in per_file.items():
        before = len(read_rows([path]))
        write_rows(path, file_rows)
        after = len(read_rows([path]))
        if before != after:
            raise SystemExit(f"{path} 写回后行数 {before} -> {after}，已中止")
        print(f"已写回 {path}：{len(file_rows)} 行，其中命中 {len(target & {r['sample_id'] for _, r in file_rows})} 条")
    print(f"共 {len(found)} 条 -> {status}，reviewer_id={reviewer or '（未填）'}")


def cmd_check(rows) -> None:
    problems: list[str] = []
    ids: dict[str, str] = {}
    groups: dict[str, str] = {}
    approved = 0
    for _, _, row in rows:
        sid = row["sample_id"]
        if sid in ids:
            problems.append(f"sample_id 重复：{sid}")
        ids[sid] = sid
        st = row.get("review_status")
        if st == "approved":
            approved += 1
            if not row.get("reviewer_id"):
                problems.append(f"{sid} 标了 approved 但 reviewer_id 为空")
        elif st not in ("pending", "rejected"):
            problems.append(f"{sid} 的 review_status 非法：{st!r}")
        gid = row.get("group_id")
        if not gid:
            problems.append(f"{sid} 缺 group_id")
        groups.setdefault(gid, st)
    for gid, st in groups.items():
        if st != groups.get(gid):
            problems.append(f"group_id {gid} 状态不一致")
    print(f"总数 {len(rows)} | approved {approved} | 问题 {len(problems)}")
    for p in problems:
        print(" -", p)
    if not problems:
        print("体检通过。")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str):
        p = sub.add_parser(name)
        p.add_argument("--input", required=True, type=Path, nargs="+")
        return p

    add("stats")

    p_list = add("list")
    p_list.add_argument("--tier", default="all", choices=TIER_ORDER + ["all"])
    p_list.add_argument("--status", default="pending", choices=("pending", "approved", "rejected", "all"))
    p_list.add_argument("--limit", type=int, default=0)

    p_show = add("show")
    p_show.add_argument("--id", nargs="+", required=True)

    p_set = add("set")
    p_set.add_argument("--id", nargs="+", default=[])
    p_set.add_argument("--file", nargs="+", type=Path, default=[],
                       help="整文件批量标记（与 --id 可同时用）")
    p_set.add_argument("--status", required=True, choices=("approved", "rejected", "pending"))
    p_set.add_argument("--reviewer", default="")
    p_set.add_argument("--note", default="")

    add("check")

    args = ap.parse_args()
    rows = read_rows(args.input)

    if args.cmd == "stats":
        cmd_stats(rows)
    elif args.cmd == "list":
        cmd_list(rows, args.tier, args.status, args.limit)
    elif args.cmd == "show":
        cmd_show(rows, args.id)
    elif args.cmd == "set":
        if not args.id and not args.file:
            raise SystemExit("--id 与 --file 至少给一个")
        cmd_set(rows, args.id, args.status, args.reviewer, args.note, args.file)
    elif args.cmd == "check":
        cmd_check(rows)


if __name__ == "__main__":
    main()
