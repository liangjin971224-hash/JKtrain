"""合成检索评测集：把多路来源合并、去重、分层抽样成 retrieval_eval_v1.jsonl。

    python scripts/85_merge_eval_set.py      # 有多少合成多少，可重复跑（幂等）

来源组成（配额见 QUOTA，防 auto_exact 抬虚高整体 R@5）：

  1. eval_v3 映射   —— eval_questions_v3 的 references 按 doc_id+section 归一化映射到
                       chunk_id（确定性，无需标注）。section 对不上时回退同 doc_id，
                       但仅当该 doc 的 chunk 数 ≤3 才收（避免 gold 过泛抬高 R@5）。
  2. gen_b          —— Codex 在工作包生成的两问（outputs/gen_questions.jsonl），
                       gold = 该 chunk。
  3. auto_exact     —— question_bank 题干原句（截到「答案：」之前，避免答案泄漏），
                       gold = 该题 chunk。确定性底线样本，单独分层、配额封顶。

★ 若 Codex 输出缺失，脚本仍产出 v0 临时集（来源 1+3）并打印 PROVISIONAL 警告：
   基线可先用 v0 跑通流程，但 M1 验收必须等标注合入后重跑本脚本。

★ miss_log 种子问题（ml-*）不在此合成：其 gold 由 Codex 策略A判定后经 84 号回填，
   回填结果作为 judge_a origin 的行注入本脚本（--judge-gold 参数，见 84 号输出）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
WP_DIR = TRAIN_ROOT / "annotation_wp_v1"
DATA_DIR = TRAIN_ROOT / "data"
OUT_PATH = DATA_DIR / "retrieval_eval_v1.jsonl"
PROVISIONAL_PATH = DATA_DIR / "retrieval_eval_v0_provisional.jsonl"

#: 各来源配额；不在表里的来源全保留
QUOTA = {"auto_exact": 40, "gen_b": 120}
SHUFFLE_SEED = 20261106


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip().lower()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- v3 映射
def map_v3_references(v3_rows: list[dict], chunks: list[dict]) -> tuple[list[dict], dict]:
    """eval_questions_v3 的 references → gold_chunk_ids（确定性映射）。"""
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        if c["collection"] == "health_training_docs":
            by_doc.setdefault(c["doc_id"], []).append(c)

    stats = {"queries": 0, "section_matched": 0, "doc_fallback": 0, "unmapped": 0}
    out: list[dict] = []
    for r in v3_rows:
        stats["queries"] += 1
        golds: list[str] = []
        used_fallback = False
        for ref in r.get("references") or []:
            doc_chunks = by_doc.get(ref.get("doc_id", ""), [])
            sec = normalize(ref.get("section", ""))
            matched = []
            if sec:
                for c in doc_chunks:
                    c_sec = normalize(c.get("section", ""))
                    if sec in c_sec or c_sec in sec:
                        matched.append(c)
            if not matched and sec and len(doc_chunks) <= 3:
                matched, used_fallback = doc_chunks, True
            golds += [c["chunk_id"] for c in matched]
        golds = list(dict.fromkeys(golds))          # 去重保序
        if not golds:
            stats["unmapped"] += 1
            continue
        if used_fallback:
            stats["doc_fallback"] += 1
        else:
            stats["section_matched"] += 1
        out.append({
            "qid": r["qid"], "origin": "eval_v3", "question": r["question"],
            "intent": "", "collection": "health_training_docs",
            "allowed_permissions": ["public", "internal"],
            "gold_chunk_ids": golds, "notes": "",
        })
    return out, stats


def auto_exact_from_question_bank(chunks: list[dict]) -> list[dict]:
    """题库原句查询：题干截到「答案：」之前。"""
    out = []
    for c in chunks:
        if c["collection"] != "health_question_bank":
            continue
        stem = (c["text"] or "").split("答案：")[0].split("答案:")[0].strip()
        if len(stem) < 6:
            continue
        out.append({
            "qid": f"auto-{c['chunk_id']}", "origin": "auto_exact",
            "question": stem, "intent": c.get("knowledge_point", ""),
            "collection": "health_question_bank",
            "allowed_permissions": ["restricted"],
            "gold_chunk_ids": [c["chunk_id"]], "notes": "",
        })
    return out


def gen_b_rows(chunks_by_id: dict[str, dict], gen_rows: list[dict]) -> list[dict]:
    out = []
    for g in gen_rows:
        chunk = chunks_by_id.get(g.get("chunk_id", ""))
        if chunk is None or not (g.get("question") or "").strip():
            continue
        out.append({
            "qid": f"gen-{g['chunk_id']}-{g.get('difficulty', 'x')}",
            "origin": "gen_b", "question": g["question"].strip(), "intent": "",
            "collection": chunk["collection"],
            "allowed_permissions": ["public", "internal"]
            if chunk["collection"] == "health_training_docs" else ["restricted"],
            "gold_chunk_ids": [g["chunk_id"]], "notes": "",
        })
    return out


def stratified_sample(rows: list[dict]) -> list[dict]:
    """按 origin 配额抽样；gen_b 内部再按 collection 分层。"""
    rng = random.Random(SHUFFLE_SEED)
    by_origin: dict[str, list[dict]] = {}
    for r in rows:
        by_origin.setdefault(r["origin"], []).append(r)

    picked: list[dict] = []
    for origin, group in by_origin.items():
        cap = QUOTA.get(origin)
        if cap is None:
            picked += group
            continue
        if origin == "gen_b":                    # gen_b 按 collection 分层分配配额
            by_coll: dict[str, list[dict]] = {}
            for r in group:
                by_coll.setdefault(r["collection"], []).append(r)
            docs = by_coll.get("health_training_docs", [])
            cases = by_coll.get("health_case_library", [])
            n_docs = round(cap * len(docs) / max(1, len(docs) + len(cases)))
            rng.shuffle(docs)
            rng.shuffle(cases)
            picked += docs[:n_docs] + cases[: cap - n_docs]
        else:
            g = list(group)
            rng.shuffle(g)
            picked += g[:cap]
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description="合成检索评测集 v1")
    args = parser.parse_args()

    chunks = []
    for coll in ("health_training_docs", "health_case_library",
                 "health_question_bank", "health_self_test"):
        chunks += load_jsonl(WP_DIR / "chunks" / f"{coll}.jsonl")
    if not chunks:
        raise SystemExit("[FAIL] 找不到 chunk 导出。先跑 80_export_chunks.py。")
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    print("=" * 68)
    print(f"  合成 retrieval_eval_v1 → {OUT_PATH}")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 68)

    rows: list[dict] = []

    v3_rows = load_jsonl(TRAIN_ROOT / "data" / "eval_questions_v3.jsonl")
    mapped, map_stats = map_v3_references(v3_rows, chunks)
    rows += mapped
    print(f"  [v3映射] {map_stats}")

    # 复判 gold 回填（84 号产出）：给种子问题补 Codex×复判一致的 gold；
    # miss_log 种子（ml-*）在这里第一次成行
    gold_rows = load_jsonl(WP_DIR / "outputs" / "judge_gold.jsonl")
    if gold_rows:
        by_qid: dict[str, list[str]] = {}
        for g in gold_rows:
            by_qid.setdefault(g["qid"], []).append(g["chunk_id"])
        seeds = {s["qid"]: s for s in load_jsonl(
            WP_DIR / "judge_a_queries.jsonl")}
        added = 0
        for qid, chunk_ids in by_qid.items():
            valid = [c for c in chunk_ids if c in chunks_by_id]
            if not valid:
                continue
            seed = seeds.get(qid, {})
            hit = next((r for r in rows if r["qid"] == qid), None)
            if hit is not None:                    # v3 种子：合并 gold
                hit["gold_chunk_ids"] = list(dict.fromkeys(
                    hit["gold_chunk_ids"] + valid))
            elif qid in seeds:
                # 无映射行的新种子：origin 跟随种子来源（ml-* 是 miss_log；
                # v3 映射未中但拿到复判 gold 的 ev2-* 保持 eval_v3，
                # ★ 不能硬编码——2026-11-07 实测 13 条 ev2 被错标成 miss_log）
                rows.append({
                    "qid": qid, "origin": seed.get("origin", "miss_log"),
                    "question": seed["question"], "intent": seed.get("intent", ""),
                    "collection": "health_training_docs",
                    "allowed_permissions": ["public", "internal"],
                    "gold_chunk_ids": valid, "notes": "策略A复判一致 gold",
                })
            added += len(valid)
        print(f"  [复判gold] 回填 {added} 条 gold（来自 {len(by_qid)} 题）")

    gen_rows = load_jsonl(WP_DIR / "outputs" / "gen_questions.jsonl")
    gen = gen_b_rows(chunks_by_id, gen_rows)
    rows += gen
    has_codex = bool(gen)

    rows += auto_exact_from_question_bank(chunks)

    # 去重：同一 (问题归一化, collection) 保留先到的来源
    seen: set[tuple[str, str]] = set()
    deduped = []
    for r in rows:
        key = (normalize(r["question"]), r["collection"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    final = [r for r in stratified_sample(deduped) if r["gold_chunk_ids"]]

    composition = {o: sum(1 for r in final if r["origin"] == o)
                   for o in sorted({r["origin"] for r in final})}
    print(f"  [组成] {composition}")
    print(f"  [合计] {len(final)} 条")

    if has_codex:
        dump_jsonl(OUT_PATH, final)
        print(f"  [OK] 已写入 {OUT_PATH}")
    else:
        dump_jsonl(PROVISIONAL_PATH, final)
        print("  [PROVISIONAL] Codex 标注未就绪：v0 临时集（v3映射+auto_exact）")
        print(f"                → {PROVISIONAL_PATH}")
        print("                流程可用、可跑基线；M1 验收必须等标注合入后重跑本脚本。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
