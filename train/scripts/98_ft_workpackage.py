"""生成 BGE-M3 微调标注工作包 v2（600 对计划中的 450 道新题）。

    python train/scripts/98_ft_workpackage.py
    python train/scripts/98_ft_workpackage.py --out-dir ../annotation_wp_v2

选择规则（见 docs/实施方案_微调BGE-M3_600对_2026-11-07.md §3）：
    ① 单一问法块（被 205 题覆盖 1 次）：每块补 2 个新问法；难度避开已有难度；
    ② 零覆盖块：案例库 26 块全取；文档库排除正文 <50 字后按 doc_id 轮转取 100 块；
    ③ 共 225 块 × 2 = 450 题；加上现有 150 对 = 600 对。

产物（annotation_wp_v2/）：
    targets.jsonl             225 个 chunk（含正文、已有问题、要求难度）
    chunks/*.jsonl            本次使用的块正文快照（案例块从 Milvus 取，保证与线上一致）
    ANNOTATION_BRIEF_FT.md    给 Codex 的任务书
    manifest.json             数量/规则/源文件/生成时间

★ 只读脚本：不写库、不改应用代码。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAIN_ROOT.parent
PROJECT_CODE = PROJECT_ROOT / "项目代码"
DATA_DIR = TRAIN_ROOT / "data"
WP_V1 = TRAIN_ROOT / "annotation_wp_v1"
DEFAULT_OUT = TRAIN_ROOT / "annotation_wp_v2"

EVAL_PATH = DATA_DIR / "retrieval_eval_v1_remap.jsonl"
CASE_MAP = PROJECT_ROOT / "08_数据入库" / "milvus" / "case_chunk_map.json"
CASE_COLLECTION = "health_case_library"
DOC_COLLECTION = "health_training_docs"

#: 零覆盖文档块的选取数量（配合 26 个案例块，凑满 126 个零覆盖块）
ZERO_DOC_TARGETS = 100
#: 正文过短的块不标注（多为标题/空段）
MIN_CHUNK_CHARS = 50
#: 单一问法块的新增问法数
NEW_PER_CHUNK = 2

BRIEF = """# 微调数据标注任务书（BGE-M3 对比学习，v2）

> 由 `98_ft_workpackage.py` 于 {generated_at} 生成。
> 执行者：Codex。产出写入本目录 `outputs/ft_questions.jsonl`，逐行合法 JSON，UTF-8。

## 背景

为 bge-m3 的对比学习准备「问题 ↔ 知识块」正例。输入 `targets.jsonl` 每行一个知识块：
`chunk_id / collection / title / section / text / existing_questions / required_difficulties / bucket`。

## 任务

对**每个 chunk 恰好生成 {new_per_chunk} 个新问题**，难度按 `required_difficulties`
各取一个（顺序不限），写入 `outputs/ft_questions.jsonl`：

    {{"chunk_id": "...", "collection": "...", "question": "...", "difficulty": "keyword|paraphrase|scenario"}}

难度定义：

- `keyword`   关键词式：像搜索框输入，短、直给，但必须是完整问句。
- `paraphrase` 转述式：换一种说法问同一个事实，尽量不出现原文的连续措辞。
- `scenario`  情景式：门店场景口吻（员工问培训助教，或转述客户的问题）。

## 硬性规则

1. 问题必须**仅凭该 chunk 的 `text` 就能独立回答**；`title/section` 只辅助理解主题。
2. 问题与 `text`、以及 `existing_questions` 里的任何问题，
   **都不得有连续 ≥8 字相同的片段**（99 号校验会打回）。
3. **不得复述或改写 `existing_questions`**：它们已用于评测，必须换角度/换事实点。
   若某块只有 1 个已有问题，新问题要问该块里**不同的事实点**。
4. 不编造 chunk 里没有的事实；不问跨产品对比（chunk 只讲一个产品/主题）。
5. 问题长度建议 8~80 字；不要用「请告诉我」「帮我看看」这类无信息前缀。
6. 每行一个 JSON 对象，字段名固定，不要加注释或多余字段。
7. 目标数 {n_targets} 块 × {new_per_chunk} = **{n_questions} 行**，不多不少。

## 产出

- `outputs/ft_questions.jsonl`（{n_questions} 行）
- 不要修改 `targets.jsonl`；不要动其他目录。

校验命令（交付前自跑）：

    python train/scripts/99_validate_ft_annotations.py
"""


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case_chunks(out_dir: Path) -> list[dict]:
    """案例库 49 块：优先读已有快照，否则从 Milvus 取（与线上正文一致）。"""
    dump = out_dir / "chunks" / f"{CASE_COLLECTION}.jsonl"
    if dump.exists():
        return load_jsonl(dump)

    case_map = json.loads(CASE_MAP.read_text(encoding="utf-8"))
    ids = [k for k, v in case_map.items() if isinstance(v, dict)]

    sys.path.insert(0, str(PROJECT_CODE))
    from config.assembly import build_container
    from config.settings import Settings

    container = build_container(Settings())
    if container.vector is None:
        raise SystemExit("[FAIL] 向量端口不可用，无法取案例块正文")
    chunks = container.vector.get_chunks_by_ids(ids, collection=CASE_COLLECTION)
    rows = [
        {
            "collection": CASE_COLLECTION,
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "title": c.title,
            "section": c.section,
            "text": c.text,
            "permission_level": str(c.permission_level),
        }
        for c in chunks
    ]
    dump.parent.mkdir(parents=True, exist_ok=True)
    with dump.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def existing_difficulty() -> dict[tuple[str, str], str]:
    """v1 生成问题的 (chunk_id, question) → difficulty。"""
    out: dict[tuple[str, str], str] = {}
    for r in load_jsonl(WP_V1 / "outputs" / "gen_questions.jsonl"):
        cid, q = r.get("chunk_id"), r.get("question")
        if cid and q:
            out[(cid, q)] = r.get("difficulty", "")
    return out


def required_difficulties(existing: list[str], known: str) -> list[str]:
    """在三种难度里挑 2 个，尽量避开该块已有问法的难度。"""
    order = ["paraphrase", "scenario", "keyword"]
    if known in order:
        order = [d for d in order if d != known]
    chosen: list[str] = []
    for d in order:
        if d not in chosen:
            chosen.append(d)
        if len(chosen) == NEW_PER_CHUNK:
            break
    return chosen


def select_zero_docs(docs: list[dict], covered: set[str]) -> list[dict]:
    """零覆盖文档块：排除过短，按 doc_id 轮转取满 ZERO_DOC_TARGETS。"""
    pool = [
        c for c in docs
        if c["chunk_id"] not in covered
        and len((c.get("text") or "").strip()) >= MIN_CHUNK_CHARS
    ]
    by_doc: dict[str, list[dict]] = {}
    for c in pool:
        by_doc.setdefault(c["doc_id"], []).append(c)
    for items in by_doc.values():
        items.sort(key=lambda x: x.get("chunk_index", 0))
    selected: list[dict] = []
    round_idx = 0
    while len(selected) < ZERO_DOC_TARGETS:
        added = False
        for doc_id in sorted(by_doc):
            items = by_doc[doc_id]
            if round_idx < len(items):
                selected.append(items[round_idx])
                added = True
                if len(selected) >= ZERO_DOC_TARGETS:
                    break
        if not added:
            break
        round_idx += 1
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 BGE-M3 微调标注工作包")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = load_jsonl(EVAL_PATH)
    questions_by_chunk: dict[str, list[dict]] = {}
    for r in eval_rows:
        if r["collection"] not in (DOC_COLLECTION, CASE_COLLECTION):
            continue
        for cid in r["gold_chunk_ids"]:
            questions_by_chunk.setdefault(cid, []).append({
                "qid": r["qid"], "question": r["question"],
                "origin": r.get("origin", ""),
            })
    covered = set(questions_by_chunk)

    docs = load_jsonl(WP_V1 / "chunks" / f"{DOC_COLLECTION}.jsonl")
    cases = load_case_chunks(out_dir)
    if not docs or not cases:
        raise SystemExit("[FAIL] 块正文缺失")

    # 快照（文档库直接复用 v1 的块文件）
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    with (chunks_dir / f"{DOC_COLLECTION}.jsonl").open(
            "w", encoding="utf-8", newline="\n") as fh:
        for c in docs:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    known = existing_difficulty()
    targets: list[dict] = []

    # ① 单一问法块（docs + case）
    for c in docs + cases:
        cid = c["chunk_id"]
        qs = questions_by_chunk.get(cid, [])
        if len(qs) != 1:
            continue
        known_diff = known.get((cid, qs[0]["question"]), "")
        targets.append({
            "chunk_id": cid,
            "collection": c["collection"],
            "doc_id": c.get("doc_id", ""),
            "title": c.get("title", ""),
            "section": c.get("section", ""),
            "text": c["text"],
            "existing_questions": qs,
            "required_difficulties": required_difficulties(qs, known_diff),
            "bucket": "single",
        })

    # ② 零覆盖块：案例库全取 + 文档库按规则取 100
    for c in cases:
        if c["chunk_id"] in covered:
            continue
        targets.append({
            "chunk_id": c["chunk_id"],
            "collection": c["collection"],
            "doc_id": c.get("doc_id", ""),
            "title": c.get("title", ""),
            "section": c.get("section", ""),
            "text": c["text"],
            "existing_questions": [],
            "required_difficulties": ["keyword", "paraphrase"],
            "bucket": "zero",
        })
    for c in select_zero_docs(docs, covered):
        targets.append({
            "chunk_id": c["chunk_id"],
            "collection": c["collection"],
            "doc_id": c.get("doc_id", ""),
            "title": c.get("title", ""),
            "section": c.get("section", ""),
            "text": c["text"],
            "existing_questions": [],
            "required_difficulties": ["keyword", "paraphrase"],
            "bucket": "zero",
        })

    targets.sort(key=lambda t: (t["collection"], t["chunk_id"]))
    with (out_dir / "targets.jsonl").open(
            "w", encoding="utf-8", newline="\n") as fh:
        for t in targets:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    n = len(targets)
    (out_dir / "ANNOTATION_BRIEF_FT.md").write_text(
        BRIEF.format(
            generated_at=datetime.now(timezone.utc).astimezone()
            .isoformat(timespec="seconds"),
            new_per_chunk=NEW_PER_CHUNK,
            n_targets=n,
            n_questions=n * NEW_PER_CHUNK,
        ),
        encoding="utf-8",
    )

    manifest = {
        "purpose": "BGE-M3 contrastive fine-tuning annotation package v2",
        "n_targets": n,
        "n_questions": n * NEW_PER_CHUNK,
        "counts": {
            "single": sum(1 for t in targets if t["bucket"] == "single"),
            "zero": sum(1 for t in targets if t["bucket"] == "zero"),
            "case": sum(1 for t in targets if t["collection"] == CASE_COLLECTION),
            "docs": sum(1 for t in targets if t["collection"] == DOC_COLLECTION),
        },
        "rules": {
            "single_question_chunks": "被 205 题覆盖 1 次的块，每块补 2 问",
            "zero_coverage": f"案例库全取 + 文档库排除 <{MIN_CHUNK_CHARS} 字后按 doc_id 轮转取 {ZERO_DOC_TARGETS}",
            "new_per_chunk": NEW_PER_CHUNK,
        },
        "sources": {
            "eval": str(EVAL_PATH), "eval_sha256": sha256(EVAL_PATH),
            "wp_v1_docs": str(WP_V1 / "chunks" / f"{DOC_COLLECTION}.jsonl"),
            "case_map": str(CASE_MAP),
        },
        "generated_at": datetime.now(timezone.utc)
        .astimezone().isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 68)
    print(f"  工作包 -> {out_dir}")
    print(f"  目标块 {n}（single {manifest['counts']['single']} / "
          f"zero {manifest['counts']['zero']}；docs {manifest['counts']['docs']} / "
          f"case {manifest['counts']['case']}）")
    print(f"  待标注问题 {n * NEW_PER_CHUNK}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
