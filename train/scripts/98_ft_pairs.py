"""构建 BGE-M3 微调训练对 v2（现有 150 + 新增 450 ≈ 600 对）。

    python train/scripts/98_ft_pairs.py
    python train/scripts/98_ft_pairs.py --annotations ../annotation_wp_v2/outputs/ft_questions.jsonl

流程：
    ① 读 99 号校验通过的标注 + 98 号 targets（pos 用 chunk 线上正文）；
    ② 用**当前模型**对每个新问题检索 top-20，去掉 gold 后取 top-N 作难负例；
    ③ split 按 chunk_id 确定性 9:1（同一 chunk 的问题不跨 train/val）；
    ④ 合并 `embed_pairs_v1_train150.jsonl`（150 对，已排除 holdout 55）。

输出：`data/embed_pairs_v2.jsonl` + `data/embed_pairs_v2_summary.json`
    {"qid","query","pos":[text],"neg":[text...],"split","source"}
    source = existing | ft

★ 只读脚本：不写库、不改应用代码；但会加载 embedding 模型做难负例挖掘。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAIN_ROOT.parent
PROJECT_CODE = PROJECT_ROOT / "项目代码"
DATA_DIR = TRAIN_ROOT / "data"
DEFAULT_TARGETS = TRAIN_ROOT / "annotation_wp_v2" / "targets.jsonl"
DEFAULT_ANN = TRAIN_ROOT / "annotation_wp_v2" / "outputs" / "ft_questions.jsonl"
DEFAULT_EXISTING = DATA_DIR / "embed_pairs_v1_train150.jsonl"
DEFAULT_OUT = DATA_DIR / "embed_pairs_v2.jsonl"

#: 新对的难负例数量
NEG_K = 5
#: 难负例检索深度
TOP_K = 20


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def stable_split(key: str) -> str:
    """确定性 9:1；同一 chunk 的问题永远落在同一侧。"""
    return "val" if (hash(key) % 10) == 0 else "train"


def main() -> int:
    ap = argparse.ArgumentParser(description="构建微调训练对 v2")
    ap.add_argument("--targets", default=str(DEFAULT_TARGETS))
    ap.add_argument("--annotations", default=str(DEFAULT_ANN))
    ap.add_argument("--existing", default=str(DEFAULT_EXISTING))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--neg-k", type=int, default=NEG_K)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    targets = {t["chunk_id"]: t for t in load_jsonl(Path(args.targets))}
    ann = load_jsonl(Path(args.annotations))
    existing = load_jsonl(Path(args.existing))
    if not targets:
        raise SystemExit("[FAIL] targets 为空，先跑 98_ft_workpackage.py")
    if not ann:
        raise SystemExit("[FAIL] 标注为空，先让 Codex 产出 ft_questions.jsonl")

    sys.path.insert(0, str(PROJECT_CODE))
    from config.assembly import build_container
    from config.settings import Settings

    container = build_container(Settings())
    if container.vector is None:
        raise SystemExit("[FAIL] 向量端口不可用")

    rows: list[dict] = []
    seen_queries: set[str] = set()
    missing_neg = 0

    for i, a in enumerate(ann, 1):
        cid = a["chunk_id"]
        t = targets.get(cid)
        if t is None:
            print(f"  ⚠️ 未知 chunk {cid}，跳过")
            continue
        q = a["question"].strip()
        key = q.lower()
        if key in seen_queries:
            continue
        seen_queries.add(key)

        hits = container.vector.search(
            q, collection=t["collection"], top_k=args.top_k,
            allowed_permissions=None)          # 训练对挖掘不做权限过滤
        negs = [
            h.chunk.text for h in hits
            if h.chunk is not None and h.chunk.chunk_id != cid
            and (h.chunk.text or "").strip()
        ][:args.neg_k]
        if not negs:
            missing_neg += 1

        rows.append({
            "qid": f"ft-{cid}-{len(rows):03d}",
            "query": q,
            "pos": [t["text"]],
            "neg": negs,
            "split": stable_split(cid),
            "source": "ft",
        })
        if i % 50 == 0:
            print(f"    {i}/{len(ann)} ...")

    # 合并现有 150 对（保留原 split；source=existing）
    for r in existing:
        key = r["query"].strip().lower()
        if key in seen_queries:
            continue
        seen_queries.add(key)
        rows.append({
            "qid": r["qid"], "query": r["query"], "pos": r["pos"],
            "neg": r["neg"], "split": r.get("split", "train"),
            "source": "existing",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "out": str(out),
        "n_total": len(rows),
        "n_ft": sum(1 for r in rows if r["source"] == "ft"),
        "n_existing": sum(1 for r in rows if r["source"] == "existing"),
        "split": dict(Counter(r["split"] for r in rows)),
        "source_split": {
            src: dict(Counter(r["split"] for r in rows if r["source"] == src))
            for src in ("ft", "existing")
        },
        "no_negative_rows": missing_neg,
        "neg_k": args.neg_k,
        "generated_at": datetime.now(timezone.utc)
        .astimezone().isoformat(timespec="seconds"),
    }
    (out.parent / (out.stem + "_summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 68)
    print(f"  训练对 -> {out}")
    print(f"  合计 {summary['n_total']}（ft {summary['n_ft']} / "
          f"existing {summary['n_existing']}）")
    print(f"  split {summary['split']}")
    if missing_neg:
        print(f"  ⚠️ {missing_neg} 条没有负例（仍会写入，训练时需注意）")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
