"""结算抽检结果：算一致率，输出 M1 门禁结论；不达标时可剥离坏 gold。

    python scripts/91_apply_spotcheck.py            # 只算一致率
    python scripts/91_apply_spotcheck.py --strip    # 一并剥离 BAD 的 gold（写备份）

门禁（PRD §6）：人工一致率 ≥90%（41 条中 BAD ≤4）→ M1 人工项闭环。
--strip 的处置逻辑：
    BAD 的 (qid, chunk_id) 从 retrieval_eval_v1.jsonl 对应行剥离；
    行因此失去全部 gold 则整行删除；
    原文件备份为 retrieval_eval_v1.jsonl.bak_<时间戳>。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = TRAIN_ROOT / "data"
EVAL_PATH = DATA_DIR / "retrieval_eval_v1.jsonl"
CSV_PATH = DATA_DIR / "SPOT_CHECK_41.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="结算抽检一致率")
    parser.add_argument("--strip", action="store_true",
                        help="剥离 BAD 的 gold（自动备份原文件）")
    parser.add_argument("--csv", default=str(CSV_PATH),
                        help="工作表路径（默认 SPOT_CHECK_41.csv；测试用）")
    parser.add_argument("--eval", default=str(EVAL_PATH),
                        help="评测集路径（默认 retrieval_eval_v1.jsonl；测试用）")
    args = parser.parse_args()

    csv_path, eval_path = Path(args.csv), Path(args.eval)
    if not csv_path.exists():
        raise SystemExit(f"[FAIL] 找不到 {csv_path}，先跑 90 号。")
    with csv_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    unfilled = [r["sp_id"] for r in rows if r["user_verdict"].strip() == ""]
    if unfilled:
        print(f"[FAIL] 还有 {len(unfilled)} 条未填 user_verdict："
              f"{unfilled[:5]}{'…' if len(unfilled) > 5 else ''}")
        return 1
    bad = {r["user_verdict"].strip().upper() for r in rows} - {"OK", "BAD"}
    if bad:
        print(f"[FAIL] user_verdict 只能填 OK 或 BAD，发现：{sorted(bad)}")
        return 1

    bad_rows = [r for r in rows if r["user_verdict"].strip().upper() == "BAD"]
    agreement = 1 - len(bad_rows) / len(rows)
    verdict = "PASS" if agreement >= 0.90 else "FAIL"

    print("=" * 68)
    print(f"  抽检结算（{len(rows)} 条）")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 68)
    print(f"  一致率 = {agreement:.1%}（BAD {len(bad_rows)} 条）")
    print(f"  M1 人工门禁（≥90%）: {verdict}")
    for r in bad_rows:
        print(f"    BAD {r['sp_id']} {r['qid']} gold={r['gold_chunk_id']} "
              f"{r['question'][:40]}")
    if verdict == "PASS":
        print("  M1 全部门禁闭环。可启动云端训练（见 train/configs/embed_lora_v1.yaml）。")
        return 0

    if not args.strip:
        print("  未达标。修复标注后重跑 85 号；或 --strip 剥离坏 gold 后重跑 81 号基线。")
        return 1

    # ---- --strip：剥离坏 gold ----
    bad_pairs = {(r["qid"], r["gold_chunk_id"]) for r in bad_rows}
    backup = eval_path.with_suffix(
        ".jsonl.bak_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy(eval_path, backup)

    eval_rows = [json.loads(l) for l in eval_path.open(encoding="utf-8") if l.strip()]
    out, stripped_golds, dropped_rows = [], 0, 0
    for r in eval_rows:
        golds = [g for g in r["gold_chunk_ids"] if (r["qid"], g) not in bad_pairs]
        removed = len(r["gold_chunk_ids"]) - len(golds)
        if removed:
            stripped_golds += removed
            if not golds:
                dropped_rows += 1
                continue                       # 整行失去 gold → 删
            r["gold_chunk_ids"] = golds
            r["notes"] = (r.get("notes", "") +
                          f"｜抽检剥离 {removed} gold").strip("｜")
        out.append(r)

    with eval_path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  [OK] 剥离 {stripped_golds} 个 gold，删除 {dropped_rows} 条空行；"
          f"评测集剩余 {len(out)} 条")
    print(f"  [OK] 原文件备份：{backup}")
    print(f"  [NOTE] --strip 会影响可复现性：81 号基线需重跑并重新入档。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
