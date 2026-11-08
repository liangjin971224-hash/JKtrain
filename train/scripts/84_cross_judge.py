"""策略A判定交叉复判：用第二家模型独立重判，分歧项导出人工队列。

    python scripts/84_cross_judge.py                      # 用 Qwen（读 项目代码/.env 的 key）
    python scripts/84_cross_judge.py --provider deepseek
    python scripts/84_cross_judge.py --limit 20           # 试跑前 20 条

输入：工作包 judge_a_queries.jsonl + candidates/<qid>.jsonl
      + outputs/judge_results.jsonl（Codex 的判定）
输出：outputs/cross_judge_report.json  一致率、verdict 分布
      outputs/HUMAN_REVIEW.md          分歧项清单（人工裁决后回填）
      outputs/judge_gold.jsonl         一致项汇总 gold（85 号 --judge-gold 的输入）

★ 为什么只复判策略A：策略B的 gold 来自 chunk 本身（构造即正确），
  风险在「问题太像原文」——那是 83 号的抄袭检测职责，不需要第二意见。
  策略A是对「相关不相关」的主观判定，才需要交叉。

★ 成本：默认每条候选 1 次调用（qwen-plus），~1400 条 ≈ 几元。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WP = TRAIN_ROOT / "annotation_wp_v1"
PROJECT_CODE = TRAIN_ROOT.parent / "项目代码"

JUDGE_PROMPT = """你是培训知识库的检索质量评审。判断「候选块」是否包含回答「问题」所需的核心事实。

判定口径（三选一）：
- relevant：候选含回答该问题所需的核心事实，去掉它答案就立不住
- partial：同主题但不含核心事实，只是沾边
- irrelevant：与问题无关

只输出一个 JSON 对象：{{"verdict": "relevant|partial|irrelevant", "reason": "不超过30字"}}

问题：{question}

候选块：
{candidate}"""


def read_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


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


def ask_llm(base_url: str, api_key: str, model: str, prompt: str,
            timeout: int = 30) -> str:
    """OpenAI 兼容 chat.completions，返回 verdict（解析失败抛异常）。"""
    import requests

    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "temperature": 0,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=timeout,
    )
    if resp.status_code != 200:
        # 带上响应体：欠费(Arrearage)/限流/内容审查的原因都藏在 body 里，
        # 只有 HTTP 码会让排障变成猜谜（2026-11-07 Qwen 欠费实测教训）
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    content = resp.json()["choices"][0]["message"]["content"]
    m = re.search(r'"verdict"\s*:\s*"(relevant|partial|irrelevant)"', content)
    if not m:
        raise ValueError(f"无法从回复解析 verdict: {content[:120]}")
    return m.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="策略A交叉复判")
    parser.add_argument("--wp-dir", default=str(DEFAULT_WP))
    parser.add_argument("--provider", default="qwen",
                        choices=["qwen", "deepseek"])
    parser.add_argument("--limit", type=int, default=0, help="只复判前 N 条（试跑）")
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    args = parser.parse_args()
    wp = Path(args.wp_dir)

    env = read_env(PROJECT_CODE / ".env")
    if args.provider == "qwen":
        api_key = env.get("QWEN_API_KEY", "")
        base_url = env.get("QWEN_BASE_URL",
                           "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = env.get("QWEN_MODEL", "qwen-plus")
    else:
        api_key = env.get("DEEPSEEK_API_KEY", "")
        base_url = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = env.get("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        raise SystemExit(f"[FAIL] .env 里没有 {args.provider.upper()}_API_KEY")

    codex_rows = load_jsonl(wp / "outputs" / "judge_results.jsonl")
    if not codex_rows:
        raise SystemExit("[FAIL] 没有 Codex 判定产出（outputs/judge_results.jsonl）")
    codex = {(r["qid"], r["candidate_id"]): r["verdict"] for r in codex_rows}

    queries = {q["qid"]: q["question"] for q in load_jsonl(wp / "judge_a_queries.jsonl")}
    candidates: dict[str, dict[str, str]] = {}
    for qid in queries:
        cfile = wp / "candidates" / f"{qid}.jsonl"
        if cfile.exists():
            candidates[qid] = {r["candidate_id"]: r["text"] for r in load_jsonl(cfile)}

    todo = [(qid, cid) for (qid, cid) in codex
            if qid in queries and cid in candidates.get(qid, {})]
    if args.limit:
        todo = todo[: args.limit]

    print("=" * 68)
    print(f"  交叉复判 provider={args.provider}({model})，共 {len(todo)} 条")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 68)

    agree, disagree, failed = 0, [], 0
    gold: list[dict] = []
    for i, (qid, cid) in enumerate(todo, 1):
        prompt = JUDGE_PROMPT.format(question=queries[qid],
                                     candidate=candidates[qid][cid][:1500])
        try:
            v2 = ask_llm(base_url, api_key, model, prompt)
        except Exception as exc:              # noqa: BLE001 单条失败不断全局
            failed += 1
            print(f"  [{i}/{len(todo)}] {qid}/{cid} 调用失败: {exc}")
            continue
        v1 = codex[(qid, cid)]
        if v2 == v1:
            agree += 1
            if v2 == "relevant":
                gold.append({"qid": qid, "chunk_id": _chunk_of(wp, qid, cid)})
        else:
            disagree.append({"qid": qid, "candidate_id": cid,
                             "codex": v1, "cross": v2,
                             "question": queries[qid]})
        if i % 50 == 0:
            print(f"  进度 {i}/{len(todo)}，一致率 {agree / i:.1%}")
        time.sleep(args.sleep_sec)

    total = agree + len(disagree)
    report = {
        "provider": args.provider, "model": model,
        "judged": total, "agree": agree,
        "agreement_rate": round(agree / total, 4) if total else None,
        "failed_calls": failed,
        "disagreements": disagree,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"),
    }
    (wp / "outputs" / "cross_judge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (wp / "outputs" / "judge_gold.jsonl").write_text(
        "".join(json.dumps(g, ensure_ascii=False) + "\n" for g in gold),
        encoding="utf-8")

    lines = ["# 人工裁决队列（交叉复判分歧项）", "",
             f"复判模型：{args.provider}/{model}｜一致 {agree}/{total}"
             f"（{report['agreement_rate']:.1%}）｜失败 {failed}", ""]
    for d in disagree:
        lines += [f"## {d['qid']} / {d['candidate_id']}",
                  f"- 问题：{d['question']}",
                  f"- Codex 判：**{d['codex']}**｜复判：**{d['cross']}**",
                  "- 人工裁决（二选一填入）：________", ""]
    (wp / "outputs" / "HUMAN_REVIEW.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print("-" * 68)
    if total:
        print(f"  一致率 {report['agreement_rate']:.1%}（{agree}/{total}），"
              f"分歧 {len(disagree)} → HUMAN_REVIEW.md")
    else:
        print("  无可复判条目")
    print(f"  gold 汇总 {len(gold)} 条 → judge_gold.jsonl")
    return 0


def _chunk_of(wp: Path, qid: str, cid: str) -> str:
    for r in load_jsonl(wp / "candidates" / f"{qid}.jsonl"):
        if r["candidate_id"] == cid:
            return r["chunk_id"]
    return ""


if __name__ == "__main__":
    sys.exit(main())
