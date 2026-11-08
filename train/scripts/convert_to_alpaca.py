"""审核主数据 → LLaMA-Factory alpaca 数据集。

主数据保持结构化的「问题 + 结构化事实 + 参考资料」，训练用的用户提示由
`prompt_template.build_user_prompt` 渲染 —— 与线上推理走同一个函数，避免训练/推理
输入分布不一致。

输出（全部落在 --out-dir，可直接作为 LLaMA-Factory 的 dataset_dir）：
  train.jsonl / val.jsonl   alpaca：system / instruction / input / output（仅官方字段）
  dataset_info.json         数据集注册；本文件必须与数据同目录
  index.jsonl               sample_id / group_id / 来源 / 提示哈希，训练文件之外的可追溯信息
  manifest.json             来源哈希、导出计数、分组清单

主数据每条必需字段：sample_id / question / output / group_id / review_status / data_origin
可选：references[{title,section,body}] / facts[dict] / fact_hints[list] / source_ids /
source_version / permission_level / task_type / split / reviewer_id
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_template import SYSTEM_PROMPT, build_evidence, build_user_prompt  # noqa: E402

# Windows 控制台默认 GBK，预览里的 ⚠️ 等字符会直接抛 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REQUIRED = (
    "sample_id",
    "question",
    "output",
    "group_id",
    "review_status",
    "data_origin",
)
ALLOWED_SPLIT = ("train", "val", "test")


def stable_bucket(group_id: str, modulo: int = 10) -> int:
    digest = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def load_master(paths: list[Path]) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    seen: dict[str, str] = {}
    status_counts: dict[str, int] = {}
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{lineno} JSON 解析失败：{exc}") from exc
                missing = [k for k in REQUIRED if not row.get(k)]
                if missing:
                    raise SystemExit(f"{path}:{lineno} 缺字段 {missing}")
                sid = row["sample_id"]
                here = f"{path.name}:{lineno}"
                if sid in seen:
                    raise SystemExit(f"sample_id 重复：{sid}（{seen[sid]} 与 {here}）")
                seen[sid] = here
                status_counts[row["review_status"]] = status_counts.get(row["review_status"], 0) + 1
                rows.append(row)
    return rows, status_counts


def render(row: dict, style: str) -> tuple[str, str]:
    """返回 (instruction, input)。

    question_first（默认，对齐 LLaMA-Factory 官方 alpaca 习惯）：
        instruction = 员工的原话问法，input = 证据块（可为空）
    production（对齐线上 `_build_prompt` 的现状）：
        instruction = 证据 + 问题合成的一条消息，input = 空
    """
    evidence = build_evidence(
        fact_hints=row.get("fact_hints"),
        facts=row.get("facts"),
        references=row.get("references"),
    )
    if style == "production":
        return build_user_prompt(
            row["question"],
            fact_hints=row.get("fact_hints"),
            facts=row.get("facts"),
            references=row.get("references"),
        ), ""
    return row["question"], evidence


def assign_splits(rows: list[dict], val_ratio: float) -> dict[str, list[dict]]:
    groups: dict[str, str] = {}
    for row in rows:
        gid = row["group_id"]
        explicit = row.get("split")
        if explicit and explicit not in ALLOWED_SPLIT:
            raise SystemExit(f"{row['sample_id']} 的 split 取值非法：{explicit!r}")
        chosen = explicit or ("val" if stable_bucket(gid) < round(val_ratio * 10) else "train")
        prev = groups.setdefault(gid, chosen)
        if prev != chosen:
            raise SystemExit(f"group_id {gid!r} 同时落入 {prev} 与 {chosen}，同源样本必须同侧")

    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(groups[row["group_id"]], []).append(row)
    return out


def write_jsonl(path: Path, records: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preview(rows: list[dict], n: int, style: str) -> None:
    approved = [r for r in rows if r["review_status"] == "approved"]
    pool = approved or rows
    scope = "已审核样本" if approved else "未审核样本（仅预览，不会被导出）"
    print(f"—— 预览 {min(n, len(pool))} 条{scope} | 排布方式：{style} ——\n")
    for row in pool[:n]:
        instruction, evidence = render(row, style)
        print(f"=== {row['sample_id']} | {row['review_status']} | {row['group_id']} ===")
        print("【instruction：用户说的话】")
        print(instruction)
        print("【input：随问题一起给出的资料】" if evidence else "【input：空】")
        if evidence:
            print(evidence)
        print("【output：目标答案】")
        print(row["output"])
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, nargs="+",
                    help="可给多个文件，按顺序合并；sample_id 全局去重")
    ap.add_argument("--out-dir", type=Path, help="导出目录；仅 --preview 时可省略")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--preview", type=int, default=0, help="按渲染结果预览前 N 条，不导出")
    ap.add_argument(
        "--style",
        choices=("question_first", "production"),
        default="question_first",
        help="question_first：instruction=问题、input=证据（官方 alpaca 习惯）；"
             "production：instruction=证据+问题、input 为空（对齐线上现状）",
    )
    args = ap.parse_args()

    rows, status_counts = load_master(args.input)

    if args.preview:
        if not rows:
            raise SystemExit("主数据为空")
        preview(rows, args.preview, args.style)
        if not any(r["review_status"] == "approved" for r in rows):
            print("注意：没有任何 approved 样本，本次不导出训练集。")
            return

    approved = [r for r in rows if r["review_status"] == "approved"]
    if not approved:
        raise SystemExit(
            f"没有 review_status=approved 的样本（统计：{status_counts}），拒绝生成训练集。"
            f"先用 --preview 检查渲染结果。"
        )

    if args.out_dir is None:
        ap.error("存在可导出样本时必须指定 --out-dir")

    splits = assign_splits(approved, args.val_ratio)
    unknown = set(splits) - set(ALLOWED_SPLIT)
    if unknown:
        raise SystemExit(f"出现未预期的 split 名：{sorted(unknown)}")

    manifest = {
        "sources": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in args.input
        ],
        "status_counts_in_source": status_counts,
        "exported": {},
        "group_ids": {name: sorted({r["group_id"] for r in recs})
                      for name, recs in splits.items()},
    }

    dataset_info: dict[str, dict] = {}
    index_records: list[dict] = []
    for name in ("train", "val"):
        recs = splits.get(name, [])
        alpaca = []
        for r in recs:
            instruction, evidence = render(r, args.style)
            alpaca.append({
                "system": SYSTEM_PROMPT,
                "instruction": instruction,
                "input": evidence,
                "output": r["output"],
            })
            index_records.append({
                "sample_id": r["sample_id"],
                "group_id": r["group_id"],
                "split": name,
                "source_ids": r.get("source_ids", []),
                "source_version": r.get("source_version"),
                "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
                "input_sha256": hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
            })

        alpaca_path = args.out_dir / f"{name}.jsonl"
        dataset_info[f"health_qa_{name}"] = {
            "file_name": alpaca_path.name,
            "formatting": "alpaca",
            "columns": {
                "system": "system",
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
        manifest["exported"][name] = {
            "file": alpaca_path.name,
            "sha256": write_jsonl(alpaca_path, alpaca),
            "records": len(recs),
            "groups": len(manifest["group_ids"].get(name, [])),
        }

    (args.out_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(args.out_dir / "index.jsonl", index_records)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "source_status_counts": status_counts,
        "exported": {k: v["records"] for k, v in manifest["exported"].items()},
        "dataset_dir": str(args.out_dir),
        "manifest": str(args.out_dir / "manifest.json"),
    }, ensure_ascii=False, indent=2))

    if manifest["exported"]["val"]["records"] == 0:
        print("警告：验证集 0 条，训练时无法按验证表现挑检查点；40_train.sh 会自动去掉 eval_dataset。")


if __name__ == "__main__":
    main()
