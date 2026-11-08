"""从 Milvus 导出 4 个 collection 的全量 chunk，作为检索评测集标注的原始数据。

    python scripts/80_export_chunks.py
    python scripts/80_export_chunks.py --milvus-uri http://localhost:19530

输出（每个 collection 一个 JSONL）：
    train/annotation_wp_v1/chunks/<collection>.jsonl
        每行 {"chunk_id","doc_id","text","title","section","category",
              "permission_level","collection", [...题库专属字段]}

★ 这是 82 号标注工作包的输入。Codex 标注只需要 text/title/section，
   但这里保留完整元数据：85 号合成评测集时要用 permission_level 生成
   allowed_permissions 字段。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = TRAIN_ROOT / "annotation_wp_v1" / "chunks"

#: 4 个 collection 与各自要导出的字段（与 MilvusSearchAdapter 的 schema 探测同源，
#: 见 项目代码/infrastructure/vector/milvus_search_adapter.py 的 _COMMON_FIELDS）
COMMON_FIELDS = (
    "chunk_id", "text", "doc_id", "title", "section",
    "category", "permission_level", "review_status", "data_origin",
    "source_file", "chunk_index",
)
QUESTION_ONLY_FIELDS = ("knowledge_point", "question_type", "difficulty")

COLLECTIONS: dict[str, tuple[str, ...]] = {
    "health_training_docs": COMMON_FIELDS,
    "health_case_library": COMMON_FIELDS,
    "health_question_bank": COMMON_FIELDS + QUESTION_ONLY_FIELDS,
    "health_self_test": COMMON_FIELDS,
}


def export_collection(client, name: str, fields: tuple[str, ...], out_path: Path) -> int:
    """导出单个 collection，返回条数。空库直接报错退出（空库说明环境不对）。"""
    rows = client.query(
        collection_name=name,
        filter="",
        output_fields=list(fields),
        limit=2000,
    )
    if not rows:
        raise SystemExit(
            f"[FAIL] {name} 是空的 —— Milvus 还没灌数据。"
            "先跑 08_数据入库/milvus/build_collections.py，再回来。")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            row = {"collection": name}
            for k in fields:
                v = r.get(k)
                row[k] = "" if v is None else v
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 Milvus 全量 chunk")
    parser.add_argument("--milvus-uri", default="http://localhost:19530",
                        help="Milvus 地址（默认 http://localhost:19530）")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT),
                        help=f"输出目录（默认 {DEFAULT_OUT}）")
    args = parser.parse_args()

    from pymilvus import MilvusClient

    client = MilvusClient(uri=args.milvus_uri)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"  chunk 导出 → {out_dir}")
    print(f"  Milvus: {args.milvus_uri}")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 68)

    total = 0
    for name, fields in COLLECTIONS.items():
        n = export_collection(client, name, fields, out_dir / f"{name}.jsonl")
        total += n
        print(f"  [OK] {name:<24} {n:>4} 条")

    print("-" * 68)
    print(f"  合计 {total} 条（V2 预期 580 = 293+117+140+30）")
    if total != 580:
        print("  [WARN] 条数与 V2 定稿不一致 —— 若刚重建过库属正常（以本脚本输出为准），"
              "否则先跑 verify_v2.py 排查。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
