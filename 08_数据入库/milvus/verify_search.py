"""
Milvus 检索验证
验证三个 collection 能否被正确召回，含权限过滤
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, ".")
from common import Embedder, get_client


def show(title: str) -> None:
    print(f"\n{'=' * 74}")
    print(title)
    print("-" * 74)


def search(client, emb, collection: str, query: str, top_k: int = 3,
           filter_expr: str | None = None) -> list[dict]:
    vec = emb.encode([query])[0]
    kwargs = {
        "collection_name": collection,
        "data": [vec],
        "limit": top_k,
        "output_fields": [
            "chunk_id", "doc_id", "title", "section",
            "permission_level", "source_file", "text",
        ],
    }
    if filter_expr:
        kwargs["filter"] = filter_expr
    res = client.search(**kwargs)
    return res[0] if res else []


def main() -> None:
    client = get_client()
    print("加载 bge-m3 ...")
    emb = Embedder()
    print(f"  完成，维度 {emb.dim}")

    # ---------- 1. 培训文档检索 ----------
    show("① health_training_docs 检索验证")

    queries = [
        ("保健食品能说什么功能", "D005"),
        ("门店接待客户的流程", "D002"),
        ("客户说太贵了怎么处理", "D009"),
        ("体脂仪怎么操作", "D010"),
        ("客户吃了产品不舒服怎么办", "D013"),
    ]
    hit = 0
    for q, expect in queries:
        rows = search(client, emb, "health_training_docs", q, top_k=3)
        top = rows[0] if rows else None
        ok = top and top["entity"]["doc_id"] == expect
        hit += 1 if ok else 0
        mark = "✅" if ok else "⚠️"
        if top:
            e = top["entity"]
            print(f"{mark} 「{q}」")
            print(f"    Top1: {e['doc_id']} {e['title'][:28]} | 分数 {top['distance']:.4f}")
            if not ok:
                print(f"    期望: {expect}")

    print(f"\n  命中率: {hit}/{len(queries)}")

    # ---------- 2. 情景案例检索 ----------
    show("② health_case_library 检索验证")

    case_queries = [
        ("客户问能不能停降压药", "C04"),
        ("客户问孕妇能不能吃", "C06"),
        ("客户说别家更便宜", "C10"),
        ("客户问吃了没效果", "C09"),
    ]
    hit = 0
    for q, expect in case_queries:
        rows = search(client, emb, "health_case_library", q, top_k=3)
        top = rows[0] if rows else None
        ok = top and top["entity"]["doc_id"] == expect
        hit += 1 if ok else 0
        mark = "✅" if ok else "⚠️"
        if top:
            e = top["entity"]
            print(f"{mark} 「{q}」")
            print(f"    Top1: {e['doc_id']} {e['title'][:30]} | 分数 {top['distance']:.4f}")
            if not ok:
                print(f"    期望: {expect}")

    print(f"\n  命中率: {hit}/{len(case_queries)}")

    # ---------- 3. 题库检索 ----------
    show("③ health_question_bank 检索验证")

    q_queries = [
        ("产品类别辨识", None),
        ("功能声称边界", None),
        ("应急处理", None),
    ]
    for q, _ in q_queries:
        rows = search(client, emb, "health_question_bank", q, top_k=3)
        print(f"「{q}」")
        for r in rows:
            e = r["entity"]
            print(f"    {e['doc_id']} | {e['section']} | 分数 {r['distance']:.4f}")

    # ---------- 4. 权限过滤验证（关键） ----------
    show("④ 权限过滤验证（FR-050 核心）")

    q = "客户问能不能停降压药"

    # 学员视角：只能看 public + internal
    rows_student = search(
        client, emb, "health_case_library", q, top_k=5,
        filter_expr='permission_level in ["public", "internal"]',
    )
    print(f"学员视角（public+internal）: {len(rows_student)} 条")
    for r in rows_student:
        print(f"    {r['entity']['doc_id']} | {r['entity']['permission_level']}")

    # 主管视角：全部
    rows_admin = search(
        client, emb, "health_case_library", q, top_k=5,
        filter_expr='permission_level in ["public", "internal", "restricted"]',
    )
    print(f"主管视角（含 restricted）: {len(rows_admin)} 条")
    for r in rows_admin[:3]:
        print(f"    {r['entity']['doc_id']} | {r['entity']['permission_level']}")

    # 题库对学员应完全不可见
    rows_q = search(
        client, emb, "health_question_bank", "产品类别", top_k=5,
        filter_expr='permission_level in ["public", "internal"]',
    )
    print(f"\n学员检索题库: {len(rows_q)} 条 {'✅ 不可见' if len(rows_q) == 0 else '❌ 泄漏!'}")

    # ---------- 5. 权限分布统计 ----------
    show("⑤ 权限级别分布")

    for coll in ["health_training_docs", "health_case_library", "health_question_bank"]:
        print(f"\n{coll}:")
        for level in ["public", "internal", "restricted"]:
            rows = client.query(
                collection_name=coll,
                filter=f'permission_level == "{level}"',
                output_fields=["count(*)"],
            )
            cnt = rows[0]["count(*)"] if rows else 0
            if cnt:
                print(f"    {level:<12} {cnt:>4}")

    print("\n" + "=" * 74)
    print("验证完成")
    print("=" * 74)


if __name__ == "__main__":
    main()
