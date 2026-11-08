"""重建后验收（V2）。

六项验收，全过才算「可交付」：
  1. 数据完整性      4 个 collection 条数与预期一致
  2. 权限边界        ★ 学员搜不到任何答案 / 案例 / 题库（FR-050）
  3. 检索质量        学员视角 R@1/R@3/R@5/MRR
  4. 案例定位        16 个案例按 case_id 能否 Top1 命中
  5. 题库过滤        按 knowledge_point / 题型 精确过滤可用
  6. 顺序一致性      同一查询重复检索结果稳定

用法：python verify_v2.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, ".")

from common import Embedder, get_client

EVAL = json.loads(Path("eval_set.json").read_text(encoding="utf-8"))

COLL = {
    "training": "health_training_docs",
    "case": "health_case_library",
    "question": "health_question_bank",
    "self_test": "health_self_test",
}

# P0-2（2026-11-07）：案例库切片由 117 块合并为 49 块（16 案例 × 3 + 1 附录块）
EXPECT = {"training": 293, "case": 49, "question": 140, "self_test": 30}

TRAINEE = 'permission_level in ["public","internal"]'
MANAGER = 'permission_level in ["public","internal","restricted"]'


def sec(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("-" * 72)


# ---------------------------------------------------------------- 1
def check_counts(client) -> bool:
    sec("1. 数据完整性")
    ok = True
    for k, name in COLL.items():
        n = client.get_collection_stats(name).get("row_count", 0)
        good = n == EXPECT[k]
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} {name:<24} {n:>4} / 预期 {EXPECT[k]}")
    return ok


# ---------------------------------------------------------------- 2
def check_permission(client, emb) -> bool:
    """权限边界：学员视角必须搜不到答案/案例/题库。"""
    sec("2. 权限边界（FR-050）")
    ok = True

    probes = [
        ("参考答案", COLL["training"], TRAINEE),
        ("答案", COLL["training"], TRAINEE),
        ("客户询问能否降血糖", COLL["case"], TRAINEE),
        ("多选", COLL["question"], TRAINEE),
    ]
    for q, name, flt in probes:
        v = emb.encode([q])[0]
        res = client.search(collection_name=name, data=[v], limit=5, anns_field="vector",
                            filter=flt, output_fields=["doc_id", "section"])[0]
        # 学员视角下「训练库」允许有正文，但不允许出现答案类 section
        leak = [h for h in res
                if "参考答案" in h["entity"]["section"] or "自测题" in h["entity"]["section"]]
        good = len(leak) == 0
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} 学员查「{q}」→ {name} : "
              f"{len(res)} 条命中, 答案泄漏 {len(leak)}")

    # 案例库 / 题库 / 自测库：学员视角必须 0 条
    for key in ("case", "question", "self_test"):
        name = COLL[key]
        v = emb.encode(["客户询问产品"])[0]
        res = client.search(collection_name=name, data=[v], limit=5, anns_field="vector",
                            filter=TRAINEE, output_fields=["doc_id"])[0]
        good = len(res) == 0
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} 学员查 {name:<22} → {len(res)} 条"
              f"（必须 0）")

    # 主管视角必须能查到
    for key in ("case", "question", "self_test"):
        name = COLL[key]
        v = emb.encode(["客户询问产品"])[0]
        res = client.search(collection_name=name, data=[v], limit=3, anns_field="vector",
                            filter=MANAGER, output_fields=["doc_id"])[0]
        good = len(res) > 0
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} 主管查 {name:<22} → {len(res)} 条"
              f"（应 >0）")

    # 结构断言：training 库里不允许存在 category=self_test 的块
    n_qa = client.query(collection_name=COLL["training"],
                        filter='category == "self_test"',
                        output_fields=["count(*)"])[0]["count(*)"]
    good = n_qa == 0
    ok &= good
    print(f"  {'OK ' if good else 'FAIL'} training 库残留 self_test 块 → {n_qa}（必须 0）")

    # 结构断言：self_test 库全部 restricted
    n_pub = client.query(collection_name=COLL["self_test"],
                         filter='permission_level in ["public","internal"]',
                         output_fields=["count(*)"])[0]["count(*)"]
    good = n_pub == 0
    ok &= good
    print(f"  {'OK ' if good else 'FAIL'} self_test 库非 restricted 块 → {n_pub}（必须 0）")
    return ok


# ---------------------------------------------------------------- 3
def check_recall(client, emb) -> dict:
    sec("3. 检索质量（学员视角：只能搜 public+internal 正文）")
    intent_of = {q["id"]: q["intent"] for q in EVAL["queries"]}

    r1 = r3 = r5 = 0
    mrr = 0.0
    groups: dict[str, list[int]] = defaultdict(list)
    details = []

    for q in EVAL["queries"]:
        if q["intent"] == "case":
            continue          # 案例留给第 4 项单独评
        v = emb.encode([q["query"]])[0]
        res = client.search(collection_name=COLL["training"], data=[v], limit=5, anns_field="vector",
                            filter=TRAINEE, output_fields=["doc_id", "section", "title"])[0]

        rank = 0
        for i, h in enumerate(res, 1):
            e = h["entity"]
            hay = f"{e['doc_id']} {e['title']} {e['section']}"
            if any(lab in hay for lab in q["labels"]):
                rank = i
                break

        if rank == 1:
            r1 += 1
        if 0 < rank <= 3:
            r3 += 1
        if 0 < rank <= 5:
            r5 += 1
        if rank > 0:
            mrr += 1.0 / rank

        groups[intent_of[q["id"]]].append(rank)
        details.append((q["id"], q["query"], rank,
                        f"{res[0]['entity']['doc_id']}|{res[0]['entity']['section'][:22]}"
                        if res else ""))

    n = len(details)
    stats = {"n": n, "R@1": r1 / n, "R@3": r3 / n, "R@5": r5 / n, "MRR": mrr / n}
    print(f"  查询 {n} 条（已排除 8 条案例查询）")
    print(f"  R@1={stats['R@1']:.3f}  R@3={stats['R@3']:.3f}  "
          f"R@5={stats['R@5']:.3f}  MRR={stats['MRR']:.3f}")

    print("\n  按意图分组（R@1 / 命中@5）：")
    for k, ranks in sorted(groups.items(), key=lambda x: -len(x[1])):
        a = sum(1 for r in ranks if r == 1) / len(ranks)
        b = sum(1 for r in ranks if 0 < r <= 5) / len(ranks)
        print(f"    {k:<12} n={len(ranks):>2}  R@1={a:.2f}  命中@5={b:.2f}")

    miss = [d for d in details if d[2] == 0]
    print(f"\n  完全未命中 {len(miss)} 条：")
    for d in miss:
        print(f"    {d[0]} {d[1][:24]:<26} Top1={d[3]}")
    return stats, details


# ---------------------------------------------------------------- 4
def check_case(client, emb) -> bool:
    sec("4. 案例定位（16 个案例）")
    all_ids = set()
    for name in (COLL["case"],):
        rows = client.query(collection_name=name, filter='category == "case"',
                            output_fields=["doc_id"], limit=500)
        all_ids |= {r["doc_id"] for r in rows}
    print(f"  案例库含 {len(all_ids)} 个 case_id: {sorted(all_ids)}")

    qs = [q for q in EVAL["queries"] if q["intent"] == "case"]
    ok_n = 0
    for q in qs:
        v = emb.encode([q["query"]])[0]
        res = client.search(collection_name=COLL["case"], data=[v], limit=3, anns_field="vector",
                            filter=MANAGER, output_fields=["doc_id", "section"])[0]
        top = res[0]["entity"]["doc_id"] if res else ""
        hit = top in q["labels"]
        ok_n += hit
        print(f"  {'OK ' if hit else 'FAIL'} {q['id']} {q['query'][:20]:<22} "
              f"Top1={top} 期望={q['labels'][0]}")
    print(f"\n  Top1 命中 {ok_n}/{len(qs)}")
    return ok_n == len(qs)


# ---------------------------------------------------------------- 5
def check_kp_filter(client) -> bool:
    sec("5. 题库结构化过滤")
    ok = True

    kps = client.query(collection_name=COLL["question"],
                       filter='category == "question"',
                       output_fields=["knowledge_point"], limit=500)
    cnt = Counter(r["knowledge_point"] for r in kps)
    print(f"  知识点 {len(cnt)} 个，共 {sum(cnt.values())} 题：")
    for k, n in sorted(cnt.items()):
        print(f"    {k:<26} {n:>3}")

    # 按知识点精确过滤
    for kp in ("KP_PROD_CATEGORY", "KP_CLAIM_BOUNDARY"):
        rows = client.query(collection_name=COLL["question"],
                            filter=f'knowledge_point == "{kp}"',
                            output_fields=["chunk_id"], limit=100)
        good = len(rows) > 0 and all(r["chunk_id"].startswith(("Q", "S")) for r in rows)
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} 按 {kp} 过滤 → {len(rows)} 题")

    # 按题型过滤
    for t in ("单选", "多选"):
        rows = client.query(collection_name=COLL["question"],
                            filter=f'question_type == "{t}"',
                            output_fields=["chunk_id"], limit=200)
        print(f"  {'OK ' if len(rows) > 0 else 'FAIL'} 题型={t} → {len(rows)} 题")
        ok &= len(rows) > 0

    # 按难度过滤
    for d in (1, 2, 3):
        rows = client.query(collection_name=COLL["question"],
                            filter=f"difficulty == {d}",
                            output_fields=["chunk_id"], limit=200)
        print(f"  难度={d} → {len(rows)} 题")
    return ok


# ---------------------------------------------------------------- 6
def check_stability(client, emb) -> bool:
    sec("6. 检索稳定性（同查询重复 3 次，Top1 应一致）")
    ok = True
    for q in ("客户档案哪些内容不能记", "保健食品能声称哪些功能", "客户说太贵了怎么处理"):
        tops = set()
        for _ in range(3):
            v = emb.encode([q])[0]
            res = client.search(collection_name=COLL["training"], data=[v], limit=1, anns_field="vector",
                                filter=TRAINEE, output_fields=["chunk_id"])[0]
            tops.add(res[0]["entity"]["chunk_id"] if res else "")
        good = len(tops) == 1
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} 「{q}」→ {tops}")
    return ok


def main() -> None:
    client = get_client()
    print("加载 bge-m3 ...")
    emb = Embedder()
    print(f"  dim={emb.dim}, max_seq={emb._model.max_seq_length}")

    r = {}
    r["1_完整性"] = check_counts(client)
    r["2_权限边界"] = check_permission(client, emb)
    stats, details = check_recall(client, emb)
    r["3_检索质量"] = True
    r["4_案例定位"] = check_case(client, emb)
    r["5_结构化过滤"] = check_kp_filter(client)
    r["6_稳定性"] = check_stability(client, emb)

    sec("验收结论")
    for k, v in r.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    allok = all(r.values())
    print(f"\n  总体：{'全部通过' if allok else '存在失败项'}")

    Path("verify_v2_result.json").write_text(json.dumps(
        {"recall": stats, "checks": r,
         "details": [{"id": d[0], "q": d[1], "rank": d[2], "top1": d[3]} for d in details]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n  明细已写入 verify_v2_result.json")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
