"""Assemble manually reviewed v3 reference candidates and sentence-level provenance."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
WARNING = "保健食品不是药物，不能代替药物治疗疾病"
NEW_12 = {f"ev2-{i:02}" for i in (14, 19, 21, 25, 27, 34, 35, 36, 38, 40, 41, 44)}

# These answers were authored after reviewing the live retrieved blocks. No
# evaluation references or expectations are loaded by this builder.
REWRITTEN: dict[int, tuple[str, str]] = {
    14: (f"先问清客户还在吃哪些保健品，可能存在叠加，建议咨询医生后再决定。{WARNING}。", "新题改为其他保健品叠加情形，重写；依据D006_c006第7条，不沿用旧题海鲜过敏答案。"),
    19: (f"少年儿童不适宜；无明显缺乏蛋白质的人群不推荐食用；植物蛋白过敏者不宜使用；肾功能不全者慎用；孕妇、乳母建议咨询医生。{WARNING}。", "新题改问大豆分离蛋白粉，按P005对应条款重写。首轮图谱误命中P004，补检别名后命中P005；不使用P004参数，不把慎用升级为绝对禁用。"),
    21: (f"说明书中没有特别指定空腹或饭后，按个人习惯即可。{WARNING}。", "新题改问食用时机，引用D007_c014原文；去掉旧题每日用量内容。"),
    22: (f"不适宜人群为婴幼儿、孕妇及乳母、过敏体质人群。{WARNING}。", "33题复核中发现题面另有变化：删除了原因追问，因此精简旧答案中的原因解释。源文口径待澄清：D007_c015的范围比较与D007_c021共性归纳有争议；只列产品表原文，不自行裁决其他儿童是否适宜。"),
    25: (f"规格为 400g/罐，约 20 份，每份 20g。{WARNING}。", "新题改问包装规格，三项数字均直接摘自补检D007_c003，不计算份数。旧题每日补充量存在源文口径待澄清，但该争议不影响本题直接转述包装原文，不把6–25g或60g带回来。"),
    27: ("普通食品不能使用与已批准保健食品相同的名称。", "新题只问名称限制，直接按D008_c002对应条款回答，不沿用整份三条铁律答案。"),
    34: (f"先确认具体产品并说明产品类别。若是保健食品，介绍本产品批准文号，并说明：{WARNING}；若是普通食品，按资料说明生产许可证号。当前没有指定具体产品，其类别和编号，资料里没有这个信息。", "新题改为产品介绍，给条件式介绍流程；没有指定产品，不填具体文号或功能。共享警示语补检同时召回D005_c018及D002_c014，支持分类介绍。"),
    35: (f"无依据的专家推荐属于权威暗示，应避免用它为产品效果背书。介绍时回到本产品批准功能和可核实的产品资料。{WARNING}。", "新题改为电视专家推荐；首次补检未取到直接条款，追加检索命中D005_c013。无依据是条件，不判断电视节目或专家实际真假，不扩写认证或出处。"),
    36: (f"我不方便比较别家，只介绍本产品的客观参数：每粒含鱼油 1000mg，其中 EPA+DHA 合计 300mg，规格为 60 粒装，批准文号是国食健注G20250012。{WARNING}。", "新题改为鱼油竞品比较，按D006_c007原话介绍本品客观参数；所有数字直接有原文，不评价竞品优劣。"),
    38: ("可输出体重、体脂率、骨骼肌率、基础代谢、内脏脂肪等级、身体水分、BMI。", "新题改问输出指标，完整列示D010_c002的指标，不沿用异常结果重测答案。"),
    40: ("SOS 是声速值，按检测报告客观报出，单位为 m/s；具体数值未提供，资料里没有这个信息。结果的具体意义交由医生判断。", "新题改问SOS，按D011_c007只解释指标名和客观读数；不填写XXXX占位，不套用T/Z界线，不作疾病关联或诊断。"),
    41: ("赠品的单独退换按促销规则处理。具体活动规则，资料里没有这个信息，应先核实规则并提交主管确认。", "新题改问赠品退换，依据D013_c005、c018；不作能单独退或不能退的无条件结论，不编具体活动政策。"),
    44: ("剧烈腹痛伴面色苍白属于资料列明的红旗征，应立即拨打 120，并通知店长。不要自行判断病因或病情严重程度。", "新题是明确红旗征场景，按D014_c005和D001_c011要求立即急救；不复述整份清单，也不以观察代替120。"),
}

# Chunk IDs were selected from fresh first-pass Top10 or documented supplement
# Top10, not from the evaluation's provided references.
CONTENT_CHUNKS: dict[int, list[str]] = {
    1:["D005_c005","D005_c018"], 2:["D004_c011","D005_c009"], 3:["D004_c011","D005_c009"],
    4:["D005_c011"], 5:["D005_c012","D009_c015"], 6:["D005_c013"], 7:["D005_c014","D005_c018"],
    8:["D005_c015","D005_c019"], 9:["D005_c019"], 10:["D004_c027","D009_c013"],
    11:["D006_c005","D006_c017"], 12:["D006_c005"], 13:["D006_c006","D004_c016","D006_c007"],
    14:["D006_c006"], 15:["D006_c013"], 16:["D006_c013"], 17:["D006_c011","D006_c012"],
    18:["D007_c004"], 19:["D007_c011"], 20:["D007_c013"], 21:["D007_c014"],
    22:["D007_c015"], 23:["D007_c015"], 24:["D007_c016"], 25:["D007_c003"],
    26:["D007_c005","D004_c018"], 27:["D008_c002"], 28:["D008_c004"], 29:["D008_c005"],
    30:["D008_c006"], 31:["D008_c007"], 32:["D008_c008"], 33:["D008_c009"],
    34:["D005_c018","D002_c014"], 35:["D005_c013"], 36:["D006_c007"],
    37:["D010_c008","D010_c015","D010_c010"], 38:["D010_c002"], 39:["D011_c006"],
    40:["D011_c007"], 41:["D013_c005","D013_c018"], 42:["D013_c005","D013_c017"],
    43:["D013_c009","D001_c011","D014_c008"], 44:["D014_c005","D001_c011"],
    45:["D014_c008","D001_c011","D002_c025"],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSONL input without modifying it."""
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def supplementary_stages(supplement: dict[str, Any], qid: str) -> list[dict[str, Any]]:
    """Return actual supplementary queries linked to this question."""
    entry = supplement["per_question"].get(qid, {})
    result = [entry] if "retrieval" in entry else []
    result.extend(entry.get("additional_results", []))
    result.extend(stage for stage in supplement["shared_queries"].values() if qid in stage["linked_qids"])
    return result


def main() -> None:
    """Build all 45 candidates plus per-sentence evidence and v2 review records."""
    previous = {r["qid"]:r for r in load_jsonl(BASE.parent / "20261031_live_evidence/reference_answers.jsonl")}
    questions = {r["qid"]:r["question"] for r in load_jsonl(ROOT / "train/data/eval_questions_v3.jsonl")}
    old_questions = {r["qid"]:r["question"] for r in load_jsonl(ROOT / "train/data/eval_questions_v2.jsonl")}
    initial = {r["qid"]:r for r in load_jsonl(BASE / "retrieval_v3.jsonl")}
    supplement = json.loads((BASE / "supplemental_v3.json").read_text(encoding="utf-8"))
    chunks = {r["chunk_id"]:r for r in load_jsonl(BASE / "rag_chunks_v3.jsonl")}
    candidates, provenance, reviews = [], [], []
    for n in range(1,46):
        qid = f"ev2-{n:02}"
        first_chunks = {hit["chunk_id"] for hit in initial[qid]["retrieval"]["fused"]}
        stages = supplementary_stages(supplement, qid)
        allowed = first_chunks | {hit["chunk_id"] for stage in stages for hit in stage["retrieval"]["fused"]}
        body_ids = CONTENT_CHUNKS[n].copy()
        assert set(body_ids) <= allowed, (qid, set(body_ids)-allowed)
        if n in REWRITTEN:
            output, note = REWRITTEN[n]
        else:
            output = previous[qid]["output"]
            note = "逐条复核v3题面及新规则后沿用v2正文；引用按本轮实际Top10重新核对，不复用旧召回记录。"
            if n == 42:
                note += "题面新增未说明是否质量问题，旧答案已先核实原因并保留非质量问题条件，仍适用。"
            elif n == 45:
                note += "题面改为糖水，旧答案明确禁止食物或饮品并要求立即120，完整覆盖新场景，仍适用。"
            elif n == 5:
                note += "保留停药劝阻和遵医嘱，不恢复旧版要求复述的危险反例。"
            elif n == 37:
                note += "保留结论由医生出具的同义表达，不为关键词改写为否认句。"
            elif n == 23:
                note += "仅答原文次数与片数；上限区间的背景口径不自行解释、不换算。"

        warning_id = None
        if WARNING in output:
            ordered = body_ids + [h["chunk_id"] for h in initial[qid]["retrieval"]["fused"]]
            for cid in ordered:
                text = chunks[cid]["text"]
                if "保健食品" in text and "不是药物" in text and "不能代替药物" in text:
                    warning_id = cid
                    break
            if warning_id is None:
                warning_id = "D004_c012"
            assert warning_id in allowed, qid
        cited = list(dict.fromkeys(body_ids + ([warning_id] if warning_id else [])))
        depends = bool(set(cited)-first_chunks)
        graph_ids = [row["product_id"] for row in initial[qid]["retrieval"]["graph_hits"]]
        if n == 19:
            actual = {row["product_id"] for stage in stages for row in stage["retrieval"]["graph_hits"]}
            assert "P005" in actual
            graph_ids = ["P005"]
            depends = True
        if depends:
            note += "依赖显式补检（包括引用所需的通用警示语依据）；首轮记录保留。"
        candidate = {"qid": qid, "output": output, "evidence_chunks": cited, "graph_pids": graph_ids, "requires_supplement": depends, "reused_from_v2": n not in REWRITTEN, "review_note": note}
        assert qid not in NEW_12 or not candidate["reused_from_v2"]
        if candidate["reused_from_v2"]:
            assert output == previous[qid]["output"]
        candidates.append(candidate)
        sentence_rows = []
        sentences = [s for s in re.split(r"(?<=[。！？])", output) if s.strip()]
        for sentence_index, sentence in enumerate(sentences,1):
            is_warning_only = sentence.strip("。 ") == WARNING
            sentence_ids = [warning_id] if is_warning_only else body_ids.copy()
            if WARNING in sentence and warning_id not in sentence_ids:
                sentence_ids.append(warning_id)
            sentence_rows.append({
                "sentence_index": sentence_index, "sentence": sentence,
                "chunk_ids": sentence_ids,
                "source_passages": [{"chunk_id":cid, "section":chunks[cid]["section"], "text":chunks[cid]["text"]} for cid in sentence_ids],
                "support_kind": "literal_warning" if is_warning_only else ("scope_limited_answer_and_source_paraphrase" if "资料里没有这个信息" in sentence else "reviewed_literal_or_semantic_paraphrase"),
                "scope_note": "资料不足指当前问题未指定的具体对象/字段；不推断整个数据库不存在该事实。" if "资料里没有这个信息" in sentence else "不把教学反例作为可用话术，不作数值换算。",
            })
        provenance.append({"qid":qid,"sentences":sentence_rows,"review_status":"pending_human_review","source_scope_pending":n==22})
        reviews.append({"qid":qid,"question_v3":questions[qid],"question_changed":questions[qid]!=old_questions[qid],"mandatory_new_12":qid in NEW_12,"output_changed":output!=previous[qid]["output"],"reused_from_v2":candidate["reused_from_v2"],"review_note":note})
    for name, rows in [("reference_answers_v3.jsonl",candidates),("sentence_evidence_v3.jsonl",provenance),("v2_v3_review.jsonl",reviews)]:
        with (BASE / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False)+"\n")
    print(json.dumps({"candidates":len(candidates),"rewritten":[r['qid'] for r in candidates if not r['reused_from_v2']],"requires_supplement":[r['qid'] for r in candidates if r['requires_supplement']]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
