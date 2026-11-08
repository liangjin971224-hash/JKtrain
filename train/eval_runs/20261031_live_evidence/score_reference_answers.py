"""Check provenance and legacy keywords for assistant-authored reference candidates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
SOURCE_PENDING = {"ev2-22", "ev2-25"}
ABSOLUTE = ("唯一", "最有效", "第一", "100%", "彻底", "包治")
WARNING = "保健食品不是药物，不能代替药物治疗疾病"
HEALTH_FOOD_TOPICS = {f"ev2-{i:02}" for i in range(1, 27)} | {"ev2-35", "ev2-36", "ev2-40"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL snapshot."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path: Path, value: Any) -> None:
    """Write evaluation metadata without runtime secrets."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def main() -> None:
    """Validate candidate answers and write a reproducible evaluation report."""
    questions_path = ROOT / "train/data/eval_questions_v2.jsonl"
    questions = {row["qid"]: row for row in read_jsonl(questions_path)}
    retrieval = {row["qid"]: row for row in read_jsonl(BASE / "retrieval.jsonl")}
    answers = read_jsonl(BASE / "reference_answers.jsonl")
    chunks = {row["chunk_id"]: row for row in read_jsonl(BASE / "rag_chunks.jsonl")}
    products = json.loads((BASE / "neo4j_products.json").read_text(encoding="utf-8"))
    supplements = json.loads((BASE / "supplemental_retrieval.json").read_text(encoding="utf-8"))
    extra_graph = json.loads((BASE / "supplemental_graph_facts.json").read_text(encoding="utf-8"))
    collection_meta = json.loads((BASE / "collection_meta.json").read_text(encoding="utf-8"))
    assert len(answers) == len(questions) == len(retrieval) == 45
    assert len({row["qid"] for row in answers}) == 45
    assert hashlib.sha256(questions_path.read_bytes()).hexdigest() == collection_meta["eval_sha256"]
    results = []
    for answer in answers:
        qid = answer["qid"]
        initial = retrieval[qid]
        gold = questions[qid]["expect"]
        allowed_chunks = {hit["chunk_id"] for hit in initial["retrieval"]["fused"][:5]}
        allowed_graph = {fact["product_id"] for fact in initial["facts"]}
        if qid in supplements:
            allowed_chunks.update(hit["chunk_id"] for hit in supplements[qid]["retrieval"]["fused"][:5])
        if qid in extra_graph:
            allowed_graph.update(fact["product_id"] for fact in extra_graph[qid]["facts"])
        assert set(answer["evidence_chunks"]) <= allowed_chunks, qid
        assert set(answer["graph_pids"]) <= allowed_graph, qid
        cited_chunks = [chunks[key] for key in answer["evidence_chunks"]]
        facts = [products[key] for key in answer["graph_pids"]]
        support = "\n".join(chunk["text"] for chunk in cited_chunks) + "\n" + json.dumps(facts, ensure_ascii=False)
        tokens = re.findall(r"\d+(?:\.\d+)?(?:[–-]\d+(?:\.\d+)?)?", answer["output"])
        numeric_checks = [{"token": token, "present_in_cited_evidence": token in compact(support)} for token in tokens]
        missing_raw = [word for word in gold["must_include"] if word not in answer["output"]]
        blocked_raw = [word for word in gold["must_not_include"] if word in answer["output"]]
        missing_normalized = [word for word in gold["must_include"] if compact(word) not in compact(answer["output"])]
        blocked_normalized = [word for word in gold["must_not_include"] if compact(word) in compact(answer["output"])]
        assert all(check["present_in_cited_evidence"] for check in numeric_checks), qid
        assert not any(word in answer["output"] for word in ABSOLUTE), qid
        if qid in HEALTH_FOOD_TOPICS:
            assert WARNING in answer["output"], qid
        results.append({
            **answer, "question": initial["question"],
            "answer_origin": "assistant_authored_reference_candidate; not project model output",
            "first_pass_graph_ids": [fact["product_id"] for fact in initial["facts"]],
            "used_extra_graph_ids": sorted(set(answer["graph_pids"]) - {fact["product_id"] for fact in initial["facts"]}),
            "source_review": "pending_source_clarification" if qid in SOURCE_PENDING else "candidate_supported_with_stated_limits",
            "legacy_raw": {"pass": not missing_raw and not blocked_raw, "missing": missing_raw, "blocked": blocked_raw},
            "legacy_whitespace_normalized": {"pass": not missing_normalized and not blocked_normalized, "missing": missing_normalized, "blocked": blocked_normalized},
            "numeric_literal_checks": numeric_checks,
            "cited_evidence": {"neo4j_product_facts": facts, "rag_chunks": cited_chunks},
        })
    with (BASE / "evaluated_answers.jsonl").open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = {
        "questions": len(results), "answer_origin": "assistant reference candidates",
        "generation_model_or_adapter_tested": False,
        "first_pass_nonempty_retrieval": sum(bool(row["retrieval"]["fused"]) for row in retrieval.values()),
        "first_pass_degraded_questions": sum(bool(row["retrieval"]["degraded"]) for row in retrieval.values()),
        "first_pass_graph_hit_questions": sum(bool(row["facts"]) for row in retrieval.values()),
        "supplemental_rag_queries": len(supplements),
        "answers_using_supplemental_rag": sum(row["requires_supplement"] for row in results),
        "answers_using_extra_graph": [row["qid"] for row in results if row["used_extra_graph_ids"]],
        "source_pending": sorted(SOURCE_PENDING),
        "legacy_raw_pass": sum(row["legacy_raw"]["pass"] for row in results),
        "legacy_normalized_pass": sum(row["legacy_whitespace_normalized"]["pass"] for row in results),
        "legacy_failures": [row["qid"] for row in results if not row["legacy_raw"]["pass"]],
        "numeric_literal_occurrences_checked": sum(len(row["numeric_literal_checks"]) for row in results),
        "missing_numeric_literals": 0,
        "citation_membership_checks": "45/45 pass",
        "eval_sha256_unchanged": collection_meta["eval_sha256"],
        "nonempty_retrieval_is_not_answerability": True,
        "keyword_pass_is_not_factual_accuracy": True,
    }
    save_json(BASE / "summary.json", summary)
    report = ROOT / "docs/eval_questions_v2_实时知识库评测_2026-10-31.md"
    intro = """# eval_questions_v2 实时知识库评测

日期：2026-10-31。依据本聊天中用户新给定的十条回答规则，直接读取Neo4j结构化事实，并检索Milvus中的项目RAG培训知识库。

**本轮完成45题真实检索、逐题证据核对及候选参考答案评审。没有调用项目生成模型或微调Adapter，因此下述分数不是模型准确率，也不是线上端到端通过率。** 候选答案由本聊天助教依据本轮证据编写；它们可用于下一轮基线/Adapter对照，但不自动升级为已批准金标准。

## 方法与范围

- 快照含20个Neo4j产品、15项图谱功能目录记录、293个已审核培训知识块；仅检索public/internal权限，不读restricted题库参考答案。
- 首轮沿用项目规则意图分类、图谱适配器、bge-m3向量检索、BM25及RRF融合。每路top_k=20，保留融合前10条；首轮分析和候选答案使用前5条原始知识块。嵌入批量计算后缓存，同一查询的向量交由原适配器执行检索；本机CUDA不可用，适配器自动回退CPU。
- 没有把测试题自带references、expect、标准答案作为检索输入。首轮记录中的expected_doc_ids仅在检索结束后附加，供诊断。
- 在首轮之后，对视力目录、防癌分类、普通食品介绍范围、体脂规则和头晕红旗征作5次明确补检；实际有4份候选答案引用了补检知识块。首轮数据未覆盖重写。
- 用Neo4j中实际名称和别名对问题匹配，再只读补查产品，修复本次取证中的漏查/多产品缺口。该补查仅用于本次评测，未改项目产品识别代码。
- 候选答案使用带原有教学标记的RAG原始块，保留可判别的反例语境；未执行现有生成提示词的整行反例删除及每块600字符裁切，使用了完整ProductFacts字段。因此这是按新规则的证据评审，不能冒充当前生成接口的完整运行结果。
- 每条候选答案列出实际使用的chunk_id和产品id，检查引用属于首轮/补检Top5集合。数字逐字检查仅对引用的产品事实和知识块进行，不从题号、其他产品或完整库快照中凑数字。
- 自动数字检查只证明字面有来源，不能单独证明数值语义、适用人群或临床正确性；逐题仍人工检查参数对象、份量、单位及条件。源文问题另列，不外查法规或医学知识，也不声称那些事实已经外部验证。

## 主要发现

**证据缺口与检索缺口不同。** ev2-01首轮的目录原文排第9，ev2-37的直接“三条铁律”未进前10；补检后才拿到直接条款。ev2-04首轮没有直接的防癌分类条款，补检命中D005_c011。ev2-28虽未召回最直接的“可以说什么”块，但模板和速查卡已足以支持允许的信息类别；补检没有必要性优势，未计入答案依赖补检数。

**晕倒喂食题的原引用问题，在真实RAG中可以补上。** ev2-45首轮已召回D014_c008、D001_c011及D002_c025，直接支持立即120、禁食禁饮。原静态references漏条款，并不等于知识库里没有答案。同时，首轮还召回D014_c014的“观察10–15分钟”语句；必须依据问题已明确的晕倒场景优先执行红旗征处置，不能把不同上下文的等待语句挪过来。仅禁“先观察半小时”不足以防这种错答。

**产品图谱识别有漏查。** ev2-20/21孢子粉、24孢子油、31枸杞原浆、33代餐奶昔首轮无图谱命中；ev2-19只识别乳清，漏掉另一个比较对象大豆。补查后能直接读到相应产品，说明是当前自动识别/别名候选路径的缺口，不是Neo4j缺记录。ev2-40会命中钙片产品，但这个场景的任务是维护就医边界，不能把产品命中当作推荐许可。

**图谱不宜被当成完整标签。** P004/P005的ingredients只有主要蛋白原料；普通食品产品的ingredients为空，但RAG有完整配料表。空列表不代表无配料，也不能把主要原料当完整配料。P004的肾功能不全被放入contraindications且为critical，但原文写“慎用”；回答不能因为字段名或风险等级而擅自改写为绝对禁用。Neo4j功能目录共15项且不含视觉疲劳，查不到某项不等于该功能不在全文目录中。

**源文问题仍需澄清两题。** ev2-22的“婴幼儿比少年儿童更严格”范围比较、该文末尾的共性归纳均仍在RAG中；图谱给出的婴幼儿原因与培训段落侧重不同，两方面说明不必然互相矛盾，但都不足以证明那一范围比较正确。候选答案只转述资料和标签，不采纳该比较。ev2-25的每日补充蛋白质量与同文每份含量/食用份数存在跨字段口径问题，候选答案只原样列示6–25g与60g，并明确产品条件，不输出换算结果。这两题保留待澄清，不能记作已通过金标准审核。

**本轮数字规则已变更。** 以前报告认为50ml×10瓶可合理折算500ml；用户新规则明确禁止输出资料中没有原文的换算数字，本轮据此不写整盒500ml，也不自行计算每日总粒数、日均成本、营养占比。旧报告中相应“允许换算”的建议不适用于本轮。

## 旧关键词评分与新规则的冲突

按原题expect逐字子串判定，以下5题会失败；去空格规范化后结果相同。这不是五个已证实的事实错误。

| 题号 | 旧规则结果 | 本轮判断 |
|---|---|---|
| ev2-05 | 缺“可以停药” | 新规则要求明确劝阻，候选使用“不可擅自停药、请遵医嘱”；应改为检查劝阻含义。 |
| ev2-19 | 命中禁词“乳清” | 该词用于解释蛋白来源，并未把腹泻诊断为乳糖不耐受，也未无条件推荐乳清。 |
| ev2-23 | 缺“3–4g” | 题目只问次数和片数，候选已回答；不强制添加有歧义的上限信息。 |
| ev2-37 | 缺“不出具结论” | 候选以“结论由医生出具”正面表达同一边界，需允许语义等价。 |
| ev2-45 | 缺“不能” | 候选明确说“不可以给食物或饮品”，并要求立即120；旧规则误伤安全劝阻的同义说法。 |

ev2-22、25即使命中全部关键词，仍不能因此消除源文问题。原先确认的12题重复/意图等价也未因本轮重检而消失；本轮属于回归式证据评测，不能宣称独立留出泛化评测。具体映射见之前的逐条审核报告。

## 本轮统计

"""
    stats = [
        ("首轮非空检索", f"{summary['first_pass_nonempty_retrieval']}/45"),
        ("首轮通道降级题数", str(summary['first_pass_degraded_questions'])),
        ("首轮图谱命中题数", str(summary['first_pass_graph_hit_questions'])),
        ("RAG补检 / 答案实际依赖补检", f"{len(supplements)} / {summary['answers_using_supplemental_rag']}"),
        ("答案使用额外图谱补查", ", ".join(summary["answers_using_extra_graph"])),
        ("候选答案总数", "45；其中22、25需源文澄清，其余43条附适用限制的候选可供下一轮对照"),
        ("旧关键词通过", f"{summary['legacy_raw_pass']}/45，仅为规则命中率"),
        ("数字原文出现检查", f"{summary['numeric_literal_occurrences_checked']}个出现位置，缺来源0；不等于医学/语义认证"),
        ("引用成员检查", summary["citation_membership_checks"]),
    ]
    text = intro + "| 项目 | 结果 |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in stats)
    text += "\n\n## 逐题结果\n\n| 题号 | 首轮图谱 | 使用的RAG证据 | 旧关键词 | 说明 |\n|---|---|---|---|---|\n"
    for row in results:
        flag = "待澄清源文；" if row["qid"] in SOURCE_PENDING else ""
        text += f"| {row['qid']} | {','.join(row['first_pass_graph_ids']) or '无命中'} | {', '.join(row['evidence_chunks'])} | {'通过' if row['legacy_raw']['pass'] else '不通过'} | {flag}{row['review_note']} |\n"
    text += "\n## 逐题候选参考答案\n\n这些答案用于人工评审与后续模型对照，不是现有模型的输出，也没有写回原测试集。\n"
    for row in results:
        text += f"\n### {row['qid']} {row['question']}\n\n{row['output']}\n\n证据：{', '.join(row['evidence_chunks'])}"
        if row["graph_pids"]:
            text += "; Neo4j " + ", ".join(row["graph_pids"])
        text += "。\n"
    text += """
## 文件与复核

产物目录：`train/eval_runs/20261031_live_evidence/`。

- `collect_evidence.py`：只读采集与首轮真实召回。
- `supplement_evidence.py`：显式二次检索及图谱补查，原首轮文件保留。
- `neo4j_products.json`、`neo4j_product_properties.json`、`neo4j_catalog.json`：图谱取证快照。
- `rag_chunks.jsonl`、`retrieval.jsonl`、`supplemental_retrieval.json`、`supplemental_graph_facts.json`：RAG与召回/补查记录。
- `reference_answers.jsonl`：人工编写的45条候选答案及引用。
- `evaluated_answers.jsonl`：逐题关键词评分、数字来源、实际证据及审核标记。
- `collection_meta.json`、`summary.json`：检索设置、时间、统计与原测试集哈希。

工作目录为项目根目录，PowerShell 7复核命令：

```powershell
& '.\\项目代码\\.venv\\Scripts\\python.exe' '.\\train\\eval_runs\\20261031_live_evidence\\score_reference_answers.py'
```

采集及补检脚本已各自完整运行；首轮采集耗时见collection_meta.json。评分/引用/数字检查脚本退出码0，45题qid完整且唯一，原测试集SHA-256不变，全部引用均在实际召回窗口或显式补查结果内。正式来源中的每个数字、条件与跨产品关系仍需人工判断，不能以数字子串检查替代语义审核。

本轮新增评测产物和本报告；没有改原测试集、训练样本、项目提示词、生产检索代码或数据库，没有git add/commit。没有运行GPU生成、主备云模型或微调Adapter，未得出其准确率、延迟或端到端安全通过率。
"""
    report.write_text(text, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
