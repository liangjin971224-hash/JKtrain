"""生成 20% 人工抽检工作表（M1 人工门禁的最小化执行工具）。

    python scripts/90_sample_spotcheck.py                 # 抽 41 条 + DeepSeek 预审
    python scripts/90_sample_spotcheck.py --no-llm        # 不调 LLM（离线抽样）

产出：train/data/SPOT_CHECK_41.csv
    列：sp_id, qid, origin, question, gold_chunk_id, gold_text_head,
        deepseek_opinion(预审,仅供参考), user_verdict(★人工填 OK 或 BAD)

抽样口径：按 origin 分层等比（205 条 → 41 条 = 20%）：
    eval_v3 45→9, gen_b 120→24, auto_exact 40→8；种子固定可复现。
一条评测集行有多个 gold 时抽第 1 个（抽检目的是验「gold 对不对」，
不是验「gold 全不全」）。

人工流程：打开 CSV → 只填 user_verdict 列（OK=gold 正确 / BAD=gold 错误）
→ 跑 91 号算一致率。PRD 门禁：一致率 ≥90%（41 条里 BAD ≤4）。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = TRAIN_ROOT / "data"
WP_DIR = TRAIN_ROOT / "annotation_wp_v1"
PROJECT_CODE = TRAIN_ROOT.parent / "项目代码"

EVAL_PATH = DATA_DIR / "retrieval_eval_v1.jsonl"
OUT_CSV = DATA_DIR / "SPOT_CHECK_41.csv"
TOTAL = 41
STRATA = {"eval_v3": 9, "gen_b": 24, "auto_exact": 8}
SEED = 20261107


def read_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def deepseek_check(question: str, gold_text: str, api_key: str) -> str:
    """DeepSeek 预审：该块能否独立回答该问题。仅参考，不替代人工。"""
    import requests

    prompt = (
        "判断「知识块」是否包含回答「问题」所需的核心事实。\n"
        "只答一个词：能 / 不能\n\n问题：" + question[:300] +
        "\n\n知识块：" + gold_text[:1200])
    try:
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "deepseek-chat", "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return "能" if "能" in content[:10] else "不能"
    except Exception as exc:              # noqa: BLE001 预审失败不阻塞抽样
        return f"(预审失败:{type(exc).__name__})"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成抽检工作表")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(l) for l in EVAL_PATH.open(encoding="utf-8") if l.strip()]
    chunks_by_id: dict[str, dict] = {}
    for coll in ("health_training_docs", "health_case_library",
                 "health_question_bank", "health_self_test"):
        f = WP_DIR / "chunks" / f"{coll}.jsonl"
        if f.exists():
            for c in (json.loads(l) for l in f.open(encoding="utf-8") if l.strip()):
                chunks_by_id[c["chunk_id"]] = c

    by_origin: dict[str, list[dict]] = {}
    for r in rows:
        by_origin.setdefault(r["origin"], []).append(r)

    rng = random.Random(SEED)
    sampled: list[dict] = []
    for origin, n in STRATA.items():
        group = list(by_origin.get(origin, []))
        rng.shuffle(group)
        sampled += group[:n]

    api_key = ""
    if not args.no_llm:
        api_key = read_env(PROJECT_CODE / ".env").get("DEEPSEEK_API_KEY", "")

    print("=" * 68)
    print(f"  抽检工作表 → {OUT_CSV}")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 68)

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sp_id", "qid", "origin", "question", "gold_chunk_id",
                    "gold_text_head", "deepseek_opinion", "user_verdict"])
        for i, r in enumerate(sampled, 1):
            cid = r["gold_chunk_ids"][0]
            text = chunks_by_id.get(cid, {}).get("text", "")
            opinion = ""
            if api_key:
                opinion = deepseek_check(r["question"], text, api_key)
                print(f"  SP-{i:02d} [{r['origin']}] 预审={opinion} "
                      f"{r['question'][:36]}")
            w.writerow([f"SP-{i:02d}", r["qid"], r["origin"], r["question"],
                        cid, text[:180].replace("\n", " "), opinion, ""])

    n_llm = 41 if api_key else 0
    print("-" * 68)
    print(f"  抽样 {len(sampled)} 条（{STRATA}），预审 {n_llm} 条")
    print("  下一步：用 Excel 打开，只填 user_verdict 列（OK/BAD），")
    print("  存盘后跑：python scripts/91_apply_spotcheck.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
