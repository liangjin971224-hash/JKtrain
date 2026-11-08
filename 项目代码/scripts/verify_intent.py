"""端到端验证：意图分类是否真的改变了三路权重（2026-10-23）。

★ 为什么不能只靠单元测试：
  单元测试用的是假端口，只能证明「intent 传下去了」。
  这里接**真实** Neo4j / Milvus / BM25 / LLM，验证：
    1. 真实问题被判成什么意图
    2. 权重结构是否随之变化（图谱是否真的主导）
    3. 召回条数与答案是否正常（改造不能把链路搞坏）

用法：
    .venv/Scripts/python.exe scripts/verify_intent.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from application.use_cases.answer_question import AnswerQuestionUseCase  # noqa: E402
from config.assembly import build_container  # noqa: E402
from config.settings import get_settings  # noqa: E402

#: 覆盖 6 类意图 + 1 个乱码，故意包含「最该让图谱主导」的禁忌类。
QUESTIONS = (
    "在吃阿司匹林能吃鱼油吗",
    "吃了能停降压药吗",
    "孕妇能吃辅酶Q10吗",
    "客户进店第一步该做什么",
    "批准文号的格式有哪几种",
    "体脂仪测量前要做哪些准备",
    "鱼油有什么功能",
    "zzzzqqqqxxxx",
)

EXPECT_GRAPH_DOMINANT = {
    "禁忌咨询", "特殊人群", "要停药", "要诊断", "要处方", "超量服用",
}


def main() -> int:
    settings = get_settings()
    print("正在装配容器（加载 bge-m3 约 15s）…")
    t0 = time.perf_counter()
    container = build_container(settings)
    print(f"容器就绪 {len(container.ready_ports())}/7，用时 {time.perf_counter()-t0:.1f}s\n")

    uc = AnswerQuestionUseCase(
        graph=container.graph,
        vector=container.vector,
        keyword=container.keyword,
        llm=container.llm,
        repository=None,          # 不写未命中日志，避免污染运营数据
    )

    print(f"{'意图':<8}{'来源':<8}{'图/向/词权重':<20}{'融合':<6}结果")
    print("-" * 78)

    problems: list[str] = []
    for q in QUESTIONS:
        r = uc.execute(q, vector_collection=settings.coll_training)
        m = r.meta
        intent = m.get("intent", "?")
        w = m.get("weights", (0, 0, 0))
        fused = m.get("fused", 0)
        flag = "拒答" if r.refused else f"答 {len(r.answer)} 字"

        print(f"{intent:<8}{m.get('intent_source',''):<8}"
              f"{w[0]:.1f}/{w[1]:.1f}/{w[2]:.1f}{'':<10}{fused:<6}{flag}"
              f"   ← {q}")

        # 断言 1：安全类意图必须让图谱权重最高（结构不能反）
        if intent in EXPECT_GRAPH_DOMINANT and not w[0] > max(w[1], w[2]):
            problems.append(f"「{q}」判成 {intent}，但图谱权重 {w[0]} 不是最高")

        # 断言 2：除了纯乱码，分类来源必须是 rule（默认不调 LLM）
        if q != "zzzzqqqqxxxx" and m.get("intent_source") != "rule":
            problems.append(f"「{q}」意图来源是 {m.get('intent_source')}")

        # 断言 3：乱码必须退回开放问答（中性权重），不能被硬塞进安全类
        if q == "zzzzqqqqxxxx":
            if intent != "开放问答" or fused != 0 or not r.refused:
                problems.append(
                    f"乱码查询处理异常: intent={intent}, fused={fused}, "
                    f"refused={r.refused}（应为 开放问答 / 0 / 拒答）")

    # ---------------- 改造前后对比 ----------------
    #
    # ★ 这是最有说服力的证据：同一个问题，把 intent 强制写成「开放问答」
    #   （＝改造前的行为），与自动分类后的融合结果对比。
    print("\n改造前后融合结果对比（Top5 chunk_id）")
    print("-" * 78)
    for q in QUESTIONS[:4]:
        auto = uc.execute(q, vector_collection=settings.coll_training)
        old = uc.execute(q, vector_collection=settings.coll_training,
                         intent="开放问答")
        a = [c for c in auto.cited_chunk_ids[:5]]
        o = [c for c in old.cited_chunk_ids[:5]]
        same = "相同" if a == o else "**不同**"
        print(f"{q}\n  改造前(开放问答): {o}\n  改造后({auto.meta['intent']}): {a}  → {same}")

    print("-" * 78)
    if problems:
        print("发现问题：")
        for p in problems:
            print("  ✗", p)
        return 1
    print("结论：意图分类已生效，安全类意图的图谱权重均为最高")
    return 0


if __name__ == "__main__":
    sys.exit(main())
