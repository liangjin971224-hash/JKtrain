"""修复题库正文（回填丢失的题干）。

★ 问题（2026-10-23 发现）
────────────────────────────────────────────────────────────────────
`health_question_bank` 的 140 条里，有 **117 条题干丢失**，
正文直接从 `A. xxx` 选项开始：

    text = "A. 绿色食品标志\nB. 蓝帽子标志\n..."

源文件里题干是完整的：

    ## Q005 ★☆☆
    **以下哪个标识是保健食品的专用标志？**     ← 这句没进库
    A. 绿色食品标志
    ...

后果：P6 组卷时只有 23 道题能解析出题干，卷子几乎组不出来。

★ 修复策略
────────────────────────────────────────────────────────────────────
**只回填 text 字段，不重建 collection。**

为什么不整库重建：
    · 重建要重新算 140 条向量（bge-m3 编码约 1 分钟，可接受但没必要）
    · 重建会丢失 question_id 与历史试卷快照的对应关系（更严重）

做法：
    1. 从源 markdown 重新解析出完整 text
    2. 按 chunk_id 匹配
    3. 只对「现在缺题干」的条目做 update

用法：
    python fix_question_text.py --dry-run     # 先看会改哪些
    python fix_question_text.py               # 真正执行
"""
from __future__ import annotations

import argparse
import re
import os
import sys
from pathlib import Path

from pymilvus import MilvusClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_clean import clean_light                     # noqa: E402

BASE = Path(os.getenv(
    "CONTENT_DIR",
    str(Path(__file__).resolve().parents[2] / "06_正式内容")))
COLLECTION = "health_question_bank"
MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")

#: 题目标题行匹配。
#:
#: ★★ 这里踩过的坑（2026-10-23，117 道题题干丢失的元凶）：
#:
#:   原正则（build_collections.py 里的 `_Q_RE`）：
#:       r"^#{2,3}\s+([QS]\d{3})\s*([★☆]{1,3})?\s*[（(]?[^）)\n]*[）)]?\s*$"
#:
#:   问题出在 `[^）)\n]*` —— 它允许匹配**除右括号和换行外的任意字符**，
#:   而 `\s*$` 里的 `\s` **匹配换行符**。
#:
#:   于是对这段源文件：
#:       ## Q005 ★☆☆
#:       (空行)
#:       **以下哪个标识是保健食品的专用标志？**
#:       (空行)
#:       A. 绿色食品标志
#:
#:   正则实际匹配到的是：
#:       '## Q005 ★☆☆\n\n**以下哪个标识是保健食品的专用标志？**\n'
#:                                      ↑ 题干被当成「标题的后半部分」吃掉了
#:
#:   然后 `m.end()` 落在题干之后，正文就从 `A.` 开始 → 题干永久丢失。
#:
#:   这就是 140 条里 117 条题干丢失的**真正原因**。
#:   不是数据问题，是正则写错了。
#:
#: ★ 修法：`[^）)\n]*` 改成 `[^\n]*?`（非贪婪、且不跨行），
#:   并且用 `(?=\n|$)` 明确行尾。这样标题只吃它自己那一行。
_Q_RE = re.compile(
    r"^#{2,3}[ \t]+([QS]\d{3})[ \t]*([★☆]{1,3})?[^\n]*?(?=\n|$)",
    re.M,
)

#: 正文以选项开头 = 缺题干
_STARTS_WITH_OPTION = re.compile(r"^\s*A\s*[.、)）:：]")


def parse_source_questions() -> dict[str, str]:
    """从源文件解析出 {题目编号: 完整正文}。"""
    out: dict[str, str] = {}

    for f in sorted((BASE / "03_题库").glob("*.md")):
        md = f.read_text(encoding="utf-8")
        qs = list(_Q_RE.finditer(md))

        for i, m in enumerate(qs):
            qno = m.group(1)
            start = m.end()
            end = qs[i + 1].start() if i + 1 < len(qs) else len(md)
            body = clean_light(md[start:end]).strip()

            # 去掉尾部空行和分隔线残留
            body = re.sub(r"\s*-{3,}\s*$", "", body).strip()
            if len(body) >= 20:
                out[qno] = body

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="回填题库题干")
    parser.add_argument("--dry-run", action="store_true",
                        help="只报告，不写库")
    args = parser.parse_args()

    source = parse_source_questions()
    print(f"源文件解析: {len(source)} 道题")

    client = MilvusClient(uri=MILVUS_URI)
    rows = client.query(
        collection_name=COLLECTION,
        filter="id >= 0",
        output_fields=["id", "chunk_id", "text"],
        limit=500,
    )
    print(f"库中题目: {len(rows)} 条")

    to_fix: list[dict] = []
    not_in_source: list[str] = []
    already_ok = 0

    for row in rows:
        qid = row.get("chunk_id", "")
        cur = row.get("text") or ""

        if not _STARTS_WITH_OPTION.match(cur):
            already_ok += 1
            continue

        full = source.get(qid)
        if not full:
            not_in_source.append(qid)
            continue

        if not _STARTS_WITH_OPTION.match(full) and "A." in full:
            to_fix.append({
                "id": row["id"],
                "qid": qid,
                "old_len": len(cur),
                "new_len": len(full),
                "preview": full[:60],
            })

    print(f"\n已有题干（无需修）: {already_ok}")
    print(f"待修复: {len(to_fix)}")
    print(f"源文件找不到对应（无法修）: {len(not_in_source)}")
    if not_in_source[:5]:
        print(f"  例: {not_in_source[:5]}")

    if to_fix:
        print("\n修复样例（前 5 条）:")
        for x in to_fix[:5]:
            print(f"  {x['qid']}: {x['old_len']} 字符 → {x['new_len']} 字符")
            print(f"      新题干: {x['preview']}...")

    if args.dry_run:
        print("\n[dry-run] 未写库")
        return 0

    if not to_fix:
        print("\n无需修复。")
        return 0

    # ★ 为什么用 upsert 而不是 update：
    #   本机 pymilvus 2.6.17 的 MilvusClient **没有 update 方法**（实测）。
    #   只能 upsert —— 它要求带全 schema 字段，所以：
    #     1. 先把整行查回来（含 vector）
    #     2. 只替换 text
    #     3. 原样写回（**保留原 vector，不重算编码**）
    #
    #   好处：不用重新跑 bge-m3（省 1 分钟），且向量与旧文本的偏差
    #         对组卷无影响（组卷不走向量）。
    ok = 0
    failed: list[str] = []
    for x in to_fix:
        full = source[x["qid"]]
        try:
            row = client.get(collection_name=COLLECTION, ids=[x["id"]])
            if not row:
                failed.append(f"{x['qid']}: 查不到该行")
                continue

            record = dict(row[0])
            record["text"] = full
            client.upsert(collection_name=COLLECTION, data=[record])
            ok += 1
        except Exception as exc:                    # noqa: BLE001
            failed.append(f"{x['qid']}: {type(exc).__name__}: {exc}")
            print(f"  [FAIL] {x['qid']}: {exc}")

    print(f"\n修复完成: 成功 {ok} / 失败 {len(failed)}")
    if failed:
        print("失败明细:")
        for f in failed[:10]:
            print(f"  {f}")
        return 1

    print("\n★ 注意：text 改了但向量没重算。")
    print("  组卷不需要向量（只按知识点/难度筛选），所以不影响 P6。")
    print("  但若要用向量检索题目，需要重跑 embedding。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
