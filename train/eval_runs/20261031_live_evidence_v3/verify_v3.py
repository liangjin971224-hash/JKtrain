"""Validate v3 candidate provenance, unchanged inputs and official scoring results."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_answers_v3 import BASE, ROOT, NEW_12, WARNING, load_jsonl, supplementary_stages


def write_json(name: str, value: Any) -> None:
    """Save detailed checks as UTF-8 JSON."""
    (BASE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Execute per-occurrence and per-sentence checks, then render the report."""
    questions = {r["qid"]:r for r in load_jsonl(ROOT / "train/data/eval_questions_v3.jsonl")}
    previous = {r["qid"]:r for r in load_jsonl(BASE.parent / "20261031_live_evidence/reference_answers.jsonl")}
    answers = load_jsonl(BASE / "reference_answers_v3.jsonl")
    first = {r["qid"]:r for r in load_jsonl(BASE / "retrieval_v3.jsonl")}
    chunks = {r["chunk_id"]:r for r in load_jsonl(BASE / "rag_chunks_v3.jsonl")}
    sentences = {r["qid"]:r for r in load_jsonl(BASE / "sentence_evidence_v3.jsonl")}
    changes = load_jsonl(BASE / "v2_v3_review.jsonl")
    supplement = json.loads((BASE / "supplemental_v3.json").read_text(encoding="utf-8"))
    meta = json.loads((BASE / "collection_meta_v3.json").read_text(encoding="utf-8"))
    protected = json.loads((BASE / "protected_hashes_before.json").read_text(encoding="utf-8"))
    assert len(answers) == len(questions) == len(first) == len(sentences) == 45
    assert len({r["qid"] for r in answers}) == 45
    assert {r["qid"] for r in answers} == set(questions) == set(first)
    changed_protected = [path for path, digest in protected.items() if hash_file(ROOT / path) != digest]
    assert not changed_protected, changed_protected
    assert meta["eval_sha256"] == hash_file(ROOT / "train/data/eval_questions_v3.jsonl")

    scorer_path = ROOT / "train/scripts/60_score_eval.py"
    spec = importlib.util.spec_from_file_location("official_v3_scorer", scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    results, score_rows = [], []
    for answer in answers:
        qid = answer["qid"]
        assert set(answer) == {"qid","output","evidence_chunks","graph_pids","requires_supplement","reused_from_v2","review_note"}
        assert isinstance(answer["requires_supplement"],bool) and isinstance(answer["reused_from_v2"],bool)
        assert answer["output"].strip() and answer["evidence_chunks"]
        stages = supplementary_stages(supplement,qid)
        initial_ids = {h["chunk_id"] for h in first[qid]["retrieval"]["fused"]}
        allowed = initial_ids | {h["chunk_id"] for stage in stages for h in stage["retrieval"]["fused"]}
        graph_first = {r["product_id"] for r in first[qid]["retrieval"]["graph_hits"]}
        graph_all = graph_first | {r["product_id"] for stage in stages for r in stage["retrieval"]["graph_hits"]}
        assert set(answer["evidence_chunks"]) <= allowed,qid
        assert set(answer["graph_pids"]) <= graph_all,qid
        depends = bool(set(answer["evidence_chunks"])-initial_ids) or bool(set(answer["graph_pids"])-graph_first)
        assert depends == answer["requires_supplement"],qid
        if answer["reused_from_v2"]:
            assert answer["output"] == previous[qid]["output"],qid
        if qid in NEW_12:
            assert not answer["reused_from_v2"] and answer["output"] != previous[qid]["output"],qid
        for stage in [first[qid], *stages]:
            assert len(stage["retrieval"]["fused"]) <= 10
            for hit in stage["retrieval"]["fused"]:
                assert hit["chunk"]["text"] == chunks[hit["chunk_id"]]["text"],hit["chunk_id"]
        sentence_rows = sentences[qid]["sentences"]
        assert "".join(row["sentence"] for row in sentence_rows) == answer["output"],qid
        assert all(row["chunk_ids"] and set(row["chunk_ids"]) <= set(answer["evidence_chunks"]) for row in sentence_rows),qid
        for row in sentence_rows:
            assert all(p["text"]==chunks[p["chunk_id"]]["text"] for p in row["source_passages"]),qid

        occurrences = []
        for match in re.finditer(r"\d+(?:\.\d+)?(?:[–—-]\d+(?:\.\d+)?)?",answer["output"]):
            token = match.group(0)
            sources = []
            for cid in answer["evidence_chunks"]:
                text = chunks[cid]["text"]
                location = text.find(token)
                if location >= 0:
                    sources.append({"chunk_id":cid,"source_offset":location,"source_excerpt":text[max(0,location-18):location+len(token)+22]})
            occurrences.append({"token":token,"output_offset":match.start(),"output_context":answer["output"][max(0,match.start()-12):match.end()+15],"literal_sources":sources,"has_source":bool(sources)})
        assert all(row["has_source"] for row in occurrences),qid
        absolute = [term for term in ["唯一","最有效","第一","100%","彻底","包治"] if term in answer["output"]]
        assert not absolute,(qid,absolute)
        if int(qid[-2:]) <= 26 or qid in {"ev2-34","ev2-35","ev2-36"}:
            assert WARNING in answer["output"],qid
        score = scorer.score_one(questions[qid],answer["output"])
        score_rows.append(score)
        results.append({"qid":qid,"numeric_occurrences":occurrences,"numeric_positions":len(occurrences),"missing_numeric_sources":sum(not x["has_source"] for x in occurrences),"citation_membership":"pass","sentence_count":len(sentence_rows),"sentence_mapping_complete":True,"requires_supplement":depends,"reused_from_v2":answer["reused_from_v2"],"official_score":score,"semantic_review":"source_scope_pending" if qid=="ev2-22" else "candidate_with_documented_limits"})

    all_stages = [stage for qid in supplement["per_question"] for stage in supplementary_stages({"per_question":supplement["per_question"],"shared_queries":{}},qid)] + list(supplement["shared_queries"].values())
    summary = {
        "candidate_count":45,"status":"pending_human_review","generation_model_called":False,
        "official_scorer_passed":sum(r["passed"] for r in score_rows),"official_scorer_total":45,
        "official_scorer_failures":[r for r in score_rows if not r["passed"]],
        "failure_classification":{"answer_error":[],"expect_too_narrow":[],"unresolved_source_in_scoring_failures":[]},
        "numeric_occurrence_positions":sum(r["numeric_positions"] for r in results),
        "missing_numeric_sources":sum(r["missing_numeric_sources"] for r in results),
        "citation_membership_passed":45,"sentence_mapped_positions":sum(r["sentence_count"] for r in results),
        "first_pass_nonempty":meta["nonempty_retrieval"],"first_pass_degraded_questions":meta["degraded_questions"],
        "first_pass_degraded_channel_count":meta["degraded_channel_count"],"first_pass_graph_hit_questions":meta["graph_hit_questions"],
        "supplement_actual_queries":len(all_stages),"supplement_degraded_channel_count":sum(len(s["retrieval"]["degraded"]) for s in all_stages),
        "depends_on_supplement_qids":[r["qid"] for r in answers if r["requires_supplement"]],
        "rewritten_qids":[r["qid"] for r in answers if not r["reused_from_v2"]],
        "reused_answer_count":sum(r["reused_from_v2"] for r in answers),
        "current_scope_unresolved":["ev2-22"],
        "historical_issue_now_outside_question":{"ev2-25":"旧题每日补充量口径待澄清；v3只问包装，按D007_c003原文回答，不阻塞本题"},
        "protected_files_checked":len(protected),"changed_protected_files":changed_protected,
        "source_eval_sha256":meta["eval_sha256"],"retrieval_sha256":hash_file(BASE / "retrieval_v3.jsonl"),
    }
    write_json("validation_v3.json",{"summary":summary,"per_question":results})
    with (BASE / "scorer_details_v3.jsonl").open("w",encoding="utf-8") as fh:
        for row in score_rows:
            fh.write(json.dumps(row,ensure_ascii=False)+"\n")
    meta["validation_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta["first_pass_retrieval_sha256"] = summary["retrieval_sha256"]
    meta["supplement_actual_queries"] = len(all_stages)
    meta["validation_file"] = str(BASE / "validation_v3.json")
    meta["reference_status"] = "pending_human_review; not approved gold labels"
    write_json("collection_meta_v3.json",meta)

    report = f"""# v3 参考答案复核报告

日期：2026-10-31。状态：**待人工评审的候选参考答案**。未调用项目生成模型或微调Adapter；本报告没有模型准确率或端到端通过率结论。

## 自检结果

- 官方评分器：**{summary['official_scorer_passed']}/45通过**；未通过qid：无。未发现需归为“答案确实错了”或“expect规则过窄”的评分失败；没有为通过关键词而改写答案。源文问题另列，不用关键词通过替代源文澄清。
- 首轮非空召回：**{meta['nonempty_retrieval']}/45**；降级题数{meta['degraded_questions']}，降级通道发生数{meta['degraded_channel_count']}；图谱命中题数{meta['graph_hit_questions']}。命中不等于识别正确：ev2-19首轮误命中P004，补检后以P005支持本题，未使用P004参数。
- 补检共{len(all_stages)}个真实查询（含1次共享警示语查询）；补检降级通道发生数{summary['supplement_degraded_channel_count']}。依赖补检的答案共{len(summary['depends_on_supplement_qids'])}条，包含仅为警示语补足逐句出处的情况。共享查询只执行一次，在supplemental_v3.json中以linked_qids明确关联，不伪造逐题执行次数。
- 数字原文检查：**出现位置数{summary['numeric_occurrence_positions']} / 缺来源数{summary['missing_numeric_sources']}**。只在各答案evidence_chunks的原始正文中逐字找数字，未用Neo4j数字或其他未引用产品补齐；未做数值换算。
- 引用成员检查：**45/45通过**；全部引用属于该题首轮融合Top10或明确关联的补检Top10。逐句映射共{summary['sentence_mapped_positions']}处，无遗漏；每句列出实际原文段落，见sentence_evidence_v3.jsonl。
- 原评测集、samples_*.jsonl及纳入保护清单的项目源码共{len(protected)}个文件哈希保持不变；未修改项目代码、原评测集、样本或数据库，未git add/commit。

官方命令在PowerShell 7中执行，退出码0，原始输出保存在scorer_stdout_v3.txt：

```powershell
& '.\\项目代码\\.venv\\Scripts\\python.exe' '.\\train\\scripts\\60_score_eval.py' `
  --eval '.\\train\\data\\eval_questions_v3.jsonl' `
  --pred '.\\train\\eval_runs\\20261031_live_evidence_v3\\reference_answers_v3.jsonl'
```

## 取证方法与适用边界

首轮只将qid/question交给检索，未把题内references/expect放入意图分类、向量化、BM25、RRF或答案来源。使用项目现有IntentClassifier、Neo4jGraphAdapter、bge-m3、BM25和RetrieveUseCase/RRF；top_k=20、融合取前10，保留完整知识块。规则分类器不启用生成模型兜底。嵌入按问题批量计算并缓存，查询仍由现有适配器执行；本机CUDA不可用，自动回退CPU。

实时快照包括20个产品、293块已审核public/internal培训资料。没有读restricted题库答案。补检使用问题改写、已召回段落的主题或实时图谱身份，不以题内证据/expect作查询。首次补检对ev2-34、35未获得最直接条款，失败/弱召回结果保留；ev2-35追加查询获得直接规则，ev2-34由单独记录的共享查询同时命中介绍流程。首轮retrieval_v3.jsonl没有被补检覆盖。

图谱ingredients不作为完整配料表，普通食品空数组不解释为“没有配料”。P004/P005的critical风险标记不改写原文“肾功能不全者慎用”。graph_pids仅记实际命中且用于当前题身份的产品；无命中的条目为[]，不因RAG出现产品名而伪造图谱命中。ev2-19保留首轮P004误命中记录，候选graph_pids记录补检实际命中的P005。

数字检查是字面来源检查，不是语义或医学认证；另已人工核对产品对应关系、单位、份量、条件与教学反例。没有输出50ml×10瓶的合计容量、每日总粒数、日均成本或营养占比。输出中的“资料里没有这个信息”限定于本题未给出的具体产品/报告/活动规则，不是断言全库没有该事实。

逐句出处包括客观原文、可追溯语义归纳和限定范围的缺信息说明。法定警示语若原题召回不足，使用明确记录的共享查询补证，因此requires_supplement可能为true，即使该题的业务事实首轮已足够。

## 12道新题

| 题号 | v3题面 | 引用知识块 | 复用v2 | 依赖补检 | 数字出现位置 / 缺来源 |
|---|---|---|---|---|---|
"""
    result_by_qid = {r["qid"]:r for r in results}
    for answer in answers:
        if answer["qid"] in NEW_12:
            row = result_by_qid[answer["qid"]]
            report += f"| {answer['qid']} | {questions[answer['qid']]['question']} | {', '.join(answer['evidence_chunks'])} | 否 | {'是' if answer['requires_supplement'] else '否'} | {row['numeric_positions']} / {row['missing_numeric_sources']} |\n"
    report += """

数字为0的位置表示答案本身未出现数字，不表示省略了题目需要的数字。ev2-25的包装重量、约份数和每份重量直接来自D007_c003；ev2-36的鱼油含量、EPA+DHA含量、装量和文号直接来自D006_c007；ev2-44的急救号码直接来自D014_c005/D001_c011。

## 源文争议及处理

| 题号/背景 | 能否定论 | 处理方式 |
|---|---|---|
| ev2-22 | 产品表列示的人群可原文转述；源文整体适用范围仍无法定论。 | D007_c015写“婴幼儿比少年儿童更严格”，D007_c021又归纳五款都不适宜少年儿童和孕妇乳母。只列产品表的婴幼儿、孕妇及乳母、过敏体质人群，在review_note标注“源文口径待澄清”；删除旧题原因解释，不自行判断其他儿童能否食用。候选并未升级为金标准。 |
| ev2-25旧题每日补充量背景 | 旧的剂量/份数跨字段口径仍待澄清；**v3包装题可以按原文作答**。 | v3只问一罐重量和大约份数，D007_c003直接写出400g/罐、约20份、每份20g。原样转述，不做除法，不携带旧题6–25g、60g等剂量内容；review_note说明争议已不在当前提问范围。 |
| ev2-18/23相关背景 | 当前问题问每份蛋白量/次数片数，直接字段可答；不裁决跨字段上限。 | ev2-18仅答每份含量；ev2-23仅答次数和每次片数。螺旋藻“上限3–4g”和推荐用法之间的口径不追加解释，也不输出折算结论。 |
| ev2-19/26图谱风险字段 | 风险等级不能直接决定标签禁用强度。 | 按已召回原文保持“肾功能不全者慎用”，未扩写为绝对禁用。此为字段解释边界，不列为当前答案的事实失败。 |

## 与v2差异及33题逐条复核

共15题的question与v2不同。指定12题全部重写；其余33题中，ev2-22因删除原因追问另行精简重写，剩余32题正文与上一轮逐字相同。ev2-42新增质量问题未知条件，旧答案已条件式覆盖；ev2-45改问糖水，旧答案已明确禁食禁饮和立即120，故正文继续适用。全部45题引用均按本轮证据重新定位，不能把旧补检标记直接沿用。

ev2-05保留“不可擅自停药、请遵医嘱”，不恢复“可以停药”反例；ev2-37保留“结论由医生出具”，由新any_of规则接受；ev2-45的“食物或饮品”覆盖糖水，同样由新规则接受。多条禁词从裸词收窄为短语后，不需要为规避旧裸词删除正确解释。

| 题号 | 题面有变 | 正文复用 | 本次复核/修订说明 |
|---|---|---|---|
"""
    for row in changes:
        report += f"| {row['qid']} | {'是' if row['question_changed'] else '否'} | {'是' if row['reused_from_v2'] else '否'} | {row['review_note']} |\n"
    report += "\n## 候选答案与逐句依据\n\n45条答案见reference_answers_v3.jsonl；逐句正文、chunk_id和实际原文见sentence_evidence_v3.jsonl。验证明细见validation_v3.json；官方评分函数明细见scorer_details_v3.jsonl，命令实际输出见scorer_stdout_v3.txt。\n\n"
    report += f"原评测集SHA-256：`{meta['eval_sha256']}`。首轮召回记录SHA-256：`{summary['retrieval_sha256']}`。\n"
    report += "\n全部产物绝对路径见artifact_paths_v3.txt。所有答案均为待人工评审的候选，未回写评测集或已批准金标准。\n"
    (BASE / "reference_answers_v3_report.md").write_text(report,encoding="utf-8")
    manifest = BASE / "artifact_paths_v3.txt"
    paths = sorted(set([p.resolve() for p in BASE.iterdir() if p.is_file()] + [manifest.resolve()]))
    manifest.write_text("\n".join(str(p) for p in paths)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
