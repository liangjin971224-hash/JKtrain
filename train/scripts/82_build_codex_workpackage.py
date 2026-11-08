"""构建 Codex 标注工作包：任务书 + 策略A候选预生成 + 策略B目标清单。

    python scripts/82_build_codex_workpackage.py
    python scripts/82_build_codex_workpackage.py --skip-retrieval   # 不连 Milvus/模型，只生成任务书骨架

产出（train/annotation_wp_v1/ 下）：
    ANNOTATION_BRIEF.md          给 Codex 的任务书（口径/格式/禁止事项）
    gen_b_targets.jsonl          策略B：待生成问题的 chunk 清单
    judge_a_queries.jsonl        策略A：种子问题（v3 评测题 + miss_log 真实未命中）
    candidates/<qid>.jsonl       策略A：每题的 Top-30 候选（已打乱、已脱敏）
    manifest.json                本次数（可复现）

分工约定（与 PRD/TDD 一致）：
    策略B（chunk→问题）  只覆盖 training_docs + case_library（知识语料）；
    question_bank        由 85 号脚本自动生成「题干原句」查询（origin=auto_exact）；
    self_test            不进评测集（全 restricted，问答链路检索不到，无业务意义）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WP = TRAIN_ROOT / "annotation_wp_v1"
PROJECT_CODE = TRAIN_ROOT.parent / "项目代码"

#: 策略B目标 collection
GEN_B_COLLECTIONS = ("health_training_docs", "health_case_library")
#: 策略A候选来源（问答主语料，权限与线上学员路径一致）
JUDGE_COLLECTION = "health_training_docs"
CANDIDATE_K = 30

BRIEF_TEMPLATE = """# 标注任务书（检索评测集 v1）

> 由 82_build_codex_workpackage.py 于 {generated_at} 生成。
> 执行者：Codex。产出写入本目录 `outputs/`，逐行合法 JSON，UTF-8，无多余字段。

## 背景

为「健康顾问培训系统」构建检索级评测集：给定问题，判断正确知识块是否被检回。
你负责两类标注，对应两个输出文件。

## 任务 1：从知识块生成问题（策略 B）

输入：`gen_b_targets.jsonl`（每行一个 chunk：chunk_id / collection / title / section / text）。

对**每个 chunk 恰好生成 2 个问题**，写入 `outputs/gen_questions.jsonl`：

    {{"chunk_id": "...", "collection": "...", "question": "...", "difficulty": "..."}}

难度取值（每个 chunk 的两问各取其一，`required_difficulties` 字段指定了要哪两档）：

- `keyword`   关键词式：像搜索框输入，短、直给（如「鱼油 成分」风格但要是完整问句）
- `paraphrase` 转述式：换一种说法问同一个事实，不出现原文措辞
- `scenario`  情景式：门店场景口吻，参考「客户问这款能不能改善视力，我该怎么答？」

**硬性规则**：

1. 问题必须**仅凭该 chunk 的 text 就能独立回答**（title/section 只辅助理解主题）。
2. **禁止问题与原文存在连续 ≥8 字相同的片段**（校验脚本 83 号会打回）。
3. 情景式问题口吻遵循培训语境：员工向培训助教提问，或转述客户的问题。
4. 不编造 chunk 里没有的事实；不问跨产品对比（chunk 只讲一个产品/主题）。
5. 每行一个 JSON 对象，字段名固定，不要加注释或多余字段。

## 任务 2：候选相关性判定（策略 A）

输入：`judge_a_queries.jsonl`（qid / question）+ `candidates/<qid>.jsonl`
（每行 candidate_id / chunk_id / title / section / text，顺序已随机打乱）。

对每个 qid 的**每条候选**判一次，写入 `outputs/judge_results.jsonl`：

    {{"qid": "...", "candidate_id": "...", "verdict": "..."}}

verdict 取值与判定口径：

- `relevant`    候选含回答该问题所需的核心事实（去掉它答案就立不住）→ 计入 gold
- `partial`     只沾边（同主题但不含核心事实）→ 不计入 gold
- `irrelevant`  与问题无关 → 不计入 gold

**判定纪律**：

1. 只依据候选文本内容判断，不臆测「库里可能有更好的块」。
2. 候选顺序已打乱，与检索排名无关，不要猜测排序意图。
3. 一个 qid 通常有 0~3 个 relevant；如果超过 5 个，重新审视口径是否放太宽。

## 完成定义

- `outputs/gen_questions.jsonl` 行数 = gen_b_targets.jsonl 行数 × 2
- `outputs/judge_results.jsonl` 行数 = Σ 每个qid的候选数
- 两个文件都能通过 `python ../../scripts/83_validate_annotations.py` 校验
"""

BRIEF_TEMPLATE += """## 已知范围裁剪（不必疑问，均有人工评审兜底）

- `question_bank` 的 140 条由脚本自动生成「题干原句」查询，**不在你的任务里**。
- `self_test` 的 30 条不进评测集（全 restricted，问答链路检索不到）。
- heldout2 的 12 条纯人工金标准不参与任何标注。
"""


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_miss_log_queries(database_url: str | None, limit: int) -> list[dict]:
    """从 miss_log 捞真实未命中问题。连不上或表空都不阻塞（策略A可只有v3题）。"""
    if not database_url:
        return []
    try:
        import pymysql

        parsed = _parse_database_url(database_url)
        conn = pymysql.connect(
            host=parsed["host"], port=parsed["port"], user=parsed["user"],
            password=parsed["password"], database=parsed["database"], charset="utf8mb4",
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, query, intent FROM miss_log "
                    "WHERE CHAR_LENGTH(TRIM(query)) >= 4 "
                    "ORDER BY created_at DESC LIMIT %s", (limit,))
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as exc:                      # noqa: BLE001 标注数据源尽力而为
        print(f"  [WARN] miss_log 不可用（{type(exc).__name__}: {exc}），"
              "策略A只用 v3 评测题")
        return []

    out = []
    for rid, query, intent in rows:
        out.append({"qid": f"ml-{rid}", "origin": "miss_log",
                    "question": query, "intent": intent or ""})
    return out


def _parse_database_url(url: str) -> dict[str, str]:
    """解析 mysql+pymysql://user:pass@host:port/db?... （init_db.py 同款约定）。"""
    from urllib.parse import urlparse, unquote

    p = urlparse(url)
    return {
        "host": p.hostname or "localhost",
        "port": p.port or 3306,
        "user": unquote(p.username or "root"),
        "password": unquote(p.password or ""),
        "database": (p.path or "/").lstrip("/"),
    }


def read_database_url_from_env() -> str | None:
    """从 项目代码/.env 读 DATABASE_URL（不打印值，避免密钥进终端）。"""
    env_path = PROJECT_CODE / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.upper().startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def pregenerate_candidates(
    queries: list[dict], wp_dir: Path, model_dir: str, milvus_uri: str,
) -> int:
    """对每道种子问题取 Top-30 候选，打乱后写入 candidates/<qid>.jsonl。

    ★ 与线上检索路径对齐：只取 health_training_docs，
      filter 权限 public/internal + review_status approved（同 MilvusSearchAdapter）。
    """
    from pymilvus import MilvusClient
    from sentence_transformers import SentenceTransformer

    import os
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    print(f"  加载 bge-m3（CPU）：{model_dir}")
    model = SentenceTransformer(model_dir, device="cpu")
    model.max_seq_length = 4096          # 与线上 embed 配置一致

    client = MilvusClient(uri=milvus_uri)
    texts = [q["question"] for q in queries]
    print(f"  编码 {len(texts)} 条问题…")
    vectors = model.encode(texts, batch_size=8, normalize_embeddings=True,
                           show_progress_bar=False)

    cand_dir = wp_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261106)
    total = 0
    for q, vec in zip(queries, vectors):
        hits = client.search(
            collection_name=JUDGE_COLLECTION,
            data=[vec.tolist()],
            limit=CANDIDATE_K,
            filter='permission_level in ["public", "internal"] '
                   'and review_status == "approved"',
            output_fields=["chunk_id", "title", "section", "text"],
        )[0]
        rows = []
        for h in hits:
            e = h["entity"]
            rows.append({
                "chunk_id": e.get("chunk_id", ""),
                "title": e.get("title") or "",
                "section": e.get("section") or "",
                "text": e.get("text", ""),
            })
        rng.shuffle(rows)
        for i, r in enumerate(rows):
            r["candidate_id"] = f"c{i + 1:02d}"       # 打乱后重编号：codex 看不到真实排名
        with (cand_dir / f"{q['qid']}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(rows)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 Codex 标注工作包")
    parser.add_argument("--wp-dir", default=str(DEFAULT_WP))
    parser.add_argument("--embed-model-dir",
                        default=str(TRAIN_ROOT.parent / "models" / "bge-m3"))
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--database-url", default=None,
                        help="miss_log 连接串；默认读 项目代码/.env")
    parser.add_argument("--miss-log-limit", type=int, default=30)
    parser.add_argument("--skip-retrieval", action="store_true",
                        help="跳过候选预生成（不加载模型不连 Milvus）")
    args = parser.parse_args()

    wp_dir = Path(args.wp_dir)
    chunks_dir = wp_dir / "chunks"
    out_dir = wp_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"  Codex 标注工作包 → {wp_dir}")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 68)

    # 1) 策略B目标清单：training_docs + case_library 全量
    targets = []
    for coll in GEN_B_COLLECTIONS:
        rows = load_jsonl(chunks_dir / f"{coll}.jsonl")
        for r in rows:
            targets.append({
                "chunk_id": r["chunk_id"],
                "collection": coll,
                "title": r.get("title", ""),
                "section": r.get("section", ""),
                "text": r["text"],
                "required_difficulties": ["paraphrase", "scenario"]
                if coll == "health_case_library" else ["paraphrase", "keyword"],
            })
    with (wp_dir / "gen_b_targets.jsonl").open("w", encoding="utf-8") as f:
        for t in targets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"  [OK] gen_b_targets.jsonl          {len(targets)} chunk × 2 问")

    # 2) 策略A种子问题：v3 评测题 + miss_log
    seeds = []
    v3_path = TRAIN_ROOT / "data" / "eval_questions_v3.jsonl"
    if v3_path.exists():
        for r in load_jsonl(v3_path):
            seeds.append({"qid": r["qid"], "origin": "eval_v3",
                          "question": r["question"], "intent": ""})
    db_url = args.database_url or read_database_url_from_env()
    seeds += load_miss_log_queries(db_url, args.miss_log_limit)
    # 去重（同一问题文本可能同时出现在 v3 与 miss_log）
    seen, deduped = set(), []
    for s in seeds:
        key = s["question"].strip()
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    with (wp_dir / "judge_a_queries.jsonl").open("w", encoding="utf-8") as f:
        for s in deduped:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  [OK] judge_a_queries.jsonl        {len(deduped)} 题"
          f"（v3 {sum(1 for s in deduped if s['origin'] == 'eval_v3')}"
          f" + miss_log {sum(1 for s in deduped if s['origin'] == 'miss_log')}）")

    # 3) 候选预生成
    cand_total = 0
    if args.skip_retrieval:
        print("  [SKIP] --skip-retrieval：未生成候选（离线模式）")
    else:
        cand_total = pregenerate_candidates(
            deduped, wp_dir, args.embed_model_dir, args.milvus_uri)
        print(f"  [OK] candidates/                  {len(deduped)} 题 × ≤{CANDIDATE_K}"
              f" = {cand_total} 候选（已打乱脱敏）")

    # 4) 任务书 + manifest
    brief = BRIEF_TEMPLATE.format(
        generated_at=datetime.now(timezone.utc).astimezone()
        .isoformat(timespec="seconds"))
    (wp_dir / "ANNOTATION_BRIEF.md").write_text(brief, encoding="utf-8")
    print(f"  [OK] ANNOTATION_BRIEF.md")

    manifest = {
        "generated_at": datetime.now(timezone.utc)
        .astimezone().isoformat(timespec="seconds"),
        "gen_b_chunks": len(targets),
        "gen_b_expected_questions": len(targets) * 2,
        "judge_a_queries": len(deduped),
        "judge_a_candidates": cand_total,
        "candidate_k": CANDIDATE_K,
        "judge_collection": JUDGE_COLLECTION,
        "shuffle_seed": 20261106,
        "skip_retrieval": args.skip_retrieval,
    }
    (wp_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 68)
    print("  把本目录交给 Codex，按 ANNOTATION_BRIEF.md 执行；")
    print("  完成后跑 83_validate_annotations.py 校验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
