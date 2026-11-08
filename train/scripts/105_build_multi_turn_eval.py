"""生成多轮问答评测工作包 v4（20 组对话 + 最终轮参考块）。

    python train/scripts/105_build_multi_turn_eval.py

产物（annotation_wp_v4/）：
    groups.jsonl    20 组：turns / expected_standalone / expected_product_ids
    targets.jsonl   给 Codex：每组 + 最终轮 top-5 参考块
    ANNOTATION_BRIEF_MULTI_TURN.md
    manifest.json

★ 只读脚本：不写库、不改应用代码。
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
PROJECT_ROOT = TRAIN_ROOT.parent
PROJECT_CODE = PROJECT_ROOT / "项目代码"
DEFAULT_OUT = TRAIN_ROOT / "annotation_wp_v4"
COLLECTION = "health_training_docs"
PERMS = ("public", "internal")
TOP_K = 5

#: 20 组多轮对话。`standalone` 是「理想改写结果」，用于取参考块与人工核对；
#: `pids` 是最终轮应继承/切换到的产品 ID（空表示无产品，考场景继承）。
GROUPS: list[dict] = [
    {"id": "mt-01", "turns": ["鱼油每天吃几粒？", "那孕妇能吃吗？"],
     "standalone": "康源牌鱼油软胶囊 孕妇能吃吗", "pids": ["P001"]},
    {"id": "mt-02", "turns": ["鱼油每天吃几粒？", "那和阿司匹林一起吃行吗？"],
     "standalone": "康源牌鱼油软胶囊 阿司匹林 同服", "pids": ["P001"]},
    {"id": "mt-03", "turns": ["辅酶Q10 怎么吃？", "那孕妇能吃吗？"],
     "standalone": "康源牌辅酶Q10软胶囊 孕妇能吃吗", "pids": ["P002"]},
    {"id": "mt-04", "turns": ["蛋白粉适合哪些人？", "那肾不好能吃吗？"],
     "standalone": "蛋白粉 肾功能不全 能吃吗", "pids": []},
    {"id": "mt-05", "turns": ["破壁灵芝孢子粉能增强免疫力吗？", "那儿童能吃吗？"],
     "standalone": "康源牌破壁灵芝孢子粉 儿童能吃吗", "pids": []},
    {"id": "mt-06", "turns": ["鱼油的禁忌有哪些？", "那哺乳期呢？"],
     "standalone": "康源牌鱼油软胶囊 哺乳期 禁忌", "pids": ["P001"]},
    {"id": "mt-07", "turns": ["客户吃了头晕怎么办？", "那要退货吗？"],
     "standalone": "头晕 要退货吗", "pids": []},
    {"id": "mt-08", "turns": ["客户说吃了拉肚子", "那要赔偿吗？"],
     "standalone": "拉肚子 要赔偿吗", "pids": []},
    {"id": "mt-09", "turns": ["客户想停药改吃鱼油", "那能停吗？"],
     "standalone": "鱼油 能停药吗", "pids": ["P001"]},
    {"id": "mt-10", "turns": ["客户对海鲜过敏", "那能吃鱼油吗？"],
     "standalone": "海鲜过敏 能吃鱼油吗", "pids": ["P001"]},
    {"id": "mt-11", "turns": ["鱼油怎么吃？", "辅酶Q10 呢？"],
     "standalone": "康源牌辅酶Q10软胶囊 怎么吃？", "pids": ["P002"]},
    {"id": "mt-12", "turns": ["蛋白粉适合谁？", "鱼油呢？"],
     "standalone": "康源牌鱼油软胶囊 适合谁？", "pids": ["P001"]},
    {"id": "mt-13", "turns": ["孢子粉怎么吃？", "蛋白粉呢？"],
     "standalone": "蛋白粉 怎么吃？", "pids": []},
    {"id": "mt-14", "turns": ["鱼油的禁忌？", "辅酶Q10 呢？"],
     "standalone": "康源牌辅酶Q10软胶囊 禁忌？", "pids": ["P002"]},
    {"id": "mt-15", "turns": ["P001 的批准功能？", "不是这款，是 P002"],
     "standalone": "P002 的批准功能", "pids": ["P002"]},
    {"id": "mt-16", "turns": ["P004 多少钱？", "不是这款，是 P005"],
     "standalone": "P005 多少钱", "pids": ["P005"]},
    {"id": "mt-17", "turns": ["鱼油的禁忌？", "换成 P006"],
     "standalone": "P006 的禁忌", "pids": ["P006"]},
    {"id": "mt-18", "turns": ["客户问能不能改善视力", "那怎么回答才合规？"],
     "standalone": "改善视力 怎么回答才合规", "pids": []},
    {"id": "mt-19", "turns": ["客户问能不能降血糖", "那辅助降血糖能说吗？"],
     "standalone": "辅助降血糖 能不能说", "pids": []},
    {"id": "mt-20", "turns": ["客户要退货，已经开封", "那还能退吗？"],
     "standalone": "退货 已经开封 还能退吗", "pids": []},
]

BRIEF = """# 多轮问答评测 rubric 任务书（20 组）

> 由 `105_build_multi_turn_eval.py` 于 {generated_at} 生成。
> 执行者：Codex。产出写入本目录 `outputs/multi_turn_rubrics.jsonl`。

## 任务

输入 `targets.jsonl`（20 组，每组含 `turns`、`final_question`、`expected_standalone`
与 `reference_chunks`）。为**每组最后一轮**写机器可判的评分要点：

    {{"group_id": "...", "expect": {{"must_include": [...], "any_of": [...], "must_not_include": [...]}}}}

- `must_include`：2~5 条，必须是 reference_chunks 里能逐字找到的短语（编号/数值/术语/步骤），
  长度 2~20 字；
- `any_of`：同义表达分组，每组 2~5 个，至少命中一个；
- `must_not_include`：0~3 条明确错误表述；**不得**是正确否定句的子串
  （「不可以单独退」→ 不要写「可以单独退」）。

## 硬性规则

1. 20 行，group_id 与 targets 一一对应。
2. 只评**最后一轮**的答案（前面几轮用于建立上下文，不评分）。
3. 不编造 reference_chunks 里没有的数字/结论。
4. 输出 UTF-8 无 BOM，一行一个 JSON；字段恰好 `group_id` 与 `expect`。

## 交付后自检

```bash
项目代码/.venv/Scripts/python.exe train/scripts/106_validate_multi_turn_rubrics.py
```

要求硬违规 0、判定 PASS；有 HARD 按提示修，改完重跑。
"""


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="生成多轮评测工作包")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(PROJECT_CODE))
    from config.assembly import build_container
    from config.settings import Settings

    container = build_container(Settings())
    if container.vector is None:
        raise SystemExit("[FAIL] 向量端口不可用")

    targets: list[dict] = []
    for g in GROUPS:
        hits = container.vector.search(
            g["standalone"], collection=COLLECTION, top_k=TOP_K,
            allowed_permissions=PERMS)
        refs = [
            {
                "chunk_id": h.chunk.chunk_id,
                "title": h.chunk.title,
                "section": h.chunk.section,
                "text": h.chunk.text,
            }
            for h in hits if h.chunk is not None
        ]
        targets.append({
            "group_id": g["id"],
            "turns": g["turns"],
            "final_question": g["turns"][-1],
            "expected_standalone": g["standalone"],
            "expected_product_ids": g["pids"],
            "reference_chunks": refs,
        })

    with (out / "targets.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for t in targets:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    with (out / "groups.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for g in GROUPS:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    (out / "ANNOTATION_BRIEF_MULTI_TURN.md").write_text(
        BRIEF.format(generated_at=datetime.now(timezone.utc).astimezone()
                     .isoformat(timespec="seconds")), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "n_groups": len(targets),
        "collection": COLLECTION,
        "top_k": TOP_K,
        "generated_at": datetime.now(timezone.utc).astimezone()
        .isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 68)
    print(f"  工作包 -> {out}")
    print(f"  20 组；每组参考块 top-{TOP_K}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
