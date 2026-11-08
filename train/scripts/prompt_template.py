"""线上提示模板的本机冻结副本，训练数据构造与推理脚本共用。

来源：`项目代码/application/use_cases/answer_question.py`
  - SYSTEM_PROMPT        ← :611-633
  - _BAD_EXAMPLE_MARK    ← :109-111
  - _strip_bad_examples  ← :114-142
  - _build_prompt        ← :693-784

线上这三个位置一旦改动，本文件必须同步重拷。否则训练时模型见到的输入
与线上运行时的输入形态不一致，微调收益无法迁移到业务链路。
"""

from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "你是连锁健康食品门店的培训助教，回答新员工的问题。\n"
    "必须遵守：\n"
    "1. 只用【结构化事实】和【参考资料】里的内容作答。资料里没有的，直接说「资料里没有这个信息」，"
    "不要推测、不要补内容。\n"
    "2. 回答里出现的每一个数字（价格、含量、规格、用量、参考范围、批准文号），都必须能在资料里找到原文。"
    "不能自行填写，不能换算后写一个资料没有的数字，也不能把别的产品的参数搬过来。\n"
    "3. 只讲本产品批准文号对应的保健功能，并且一字不差使用目录原文。"
    "目录里有某一项功能，不等于本产品有这一项。\n"
    "4. 保健食品的回答必须包含「保健食品不是药物，不能代替药物治疗疾病」。\n"
    "5. 普通食品不得提及任何保健功能，只介绍配料、营养成分、规格、食用方法、生产许可证号。\n"
    "6. 资料里出现的「❌ 不能说」「错误说法」是教学反例，不要把反例当成可以使用的表述。\n"
    "7. 不做诊断、不做病因判断、不给用药或停药建议、不承诺效果与时间。"
    "客户出现症状时，按红旗征判断是否立即拨打 120，并建议就医。\n"
    "8. 禁止绝对化用语：唯一、最有效、第一、100%、彻底、包治。\n"
    "9. 涉及停药或替代药物时明确劝阻，写「不可擅自停药」「请遵医嘱」。\n"
    "10. 直接给答案。不要复述资料里的分节标题（如【结构化事实】【参考资料】），不要输出思考过程。"
)

_BAD_EXAMPLE_MARK = re.compile(
    # 不能把 × 当错误标记：规格写法天然含乘号（0.5g/片 × 120 片），
    # 整行剔除会把规格、价格为整段的参考资料删空。
    r"(?:❌|✗|错误示范|错误说法|错误表述|反例|不建议这样|不要这样|别这么说)"
)

_REFERENCE_NOTE = (
    "⚠️ 以下材料已剔除错误示范。复述时只做正面陈述；"
    "安全劝阻（如「不可擅自停药」「请遵医嘱」）除外。"
)

MAX_REFERENCES = 5
MAX_REFERENCE_CHARS = 600


def strip_bad_examples(text: str) -> str:
    kept: list[str] = []
    for line in (text or "").split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _BAD_EXAMPLE_MARK.search(stripped):
            continue
        kept.append(line)
    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def build_evidence(
    *,
    fact_hints: list[str] | None = None,
    facts: list[dict] | None = None,
    references: list[dict] | None = None,
) -> str:
    """只渲染证据部分：已核实的事实 → 结构化事实 → 参考资料。

    线上 `_build_prompt` 把证据和问题拼成一条消息；训练侧按 LLaMA-Factory 的
    alpaca 习惯拆成 instruction（问题）+ input（证据），所以单独暴露这一段。
    """
    parts: list[str] = []

    if fact_hints:
        lines = [
            "【已核实的事实】（系统查询产品库与批准功能目录得出，"
            "权威，优先于参考资料，必须以此为准）",
        ]
        lines.extend(f"- {h}" for h in fact_hints)
        parts.append("\n".join(lines))

    if facts:
        lines = ["【结构化事实】（来自产品库，权威，可直接引用）"]
        for f in facts:
            lines.append(f"产品：{f['name']}（{f['product_id']}）")
            lines.append(f"类别：{f.get('category', '')}")
            if f.get("approval_no"):
                lines.append(f"批准文号：{f['approval_no']}")
            if f.get("claims"):
                claims = "；".join(f"「{c}」" for c in f["claims"])
                lines.append(f"批准功能：{claims}")
            if f.get("suitable_for"):
                lines.append(f"适宜人群：{'、'.join(f['suitable_for'])}")
            if f.get("contraindications"):
                lines.append(f"不适宜人群：{'、'.join(f['contraindications'])}")
            if f.get("dosage"):
                lines.append(f"用法用量：{f['dosage']}")
                if f.get("daily_limit"):
                    lines.append(f"每日上限：{f['daily_limit']}")
            lines.append("")
        parts.append("\n".join(lines).strip())

    if references:
        lines = [
            "【参考资料】（来自培训文档，仅供参考）",
            _REFERENCE_NOTE,
        ]
        for i, ref in enumerate(references[:MAX_REFERENCES], 1):
            body = strip_bad_examples(ref.get("body", ""))
            if not body:
                continue
            lines.append(f"[{i}] 《{ref.get('title', '')}》{ref.get('section', '')}")
            lines.append(body[:MAX_REFERENCE_CHARS])
            lines.append("")
        if len(lines) > 2:
            parts.append("\n".join(lines).strip())

    return "\n\n".join(parts)


def build_user_prompt(
    question: str,
    *,
    fact_hints: list[str] | None = None,
    facts: list[dict] | None = None,
    references: list[dict] | None = None,
) -> str:
    """线上口径：证据在前、问题在后，合成一条用户消息。"""
    evidence = build_evidence(fact_hints=fact_hints, facts=facts, references=references)
    tail = f"【员工的问题】\n{question}"
    return f"{evidence}\n\n{tail}" if evidence else tail


def build_messages(
    question: str,
    *,
    fact_hints: list[str] | None = None,
    facts: list[dict] | None = None,
    references: list[dict] | None = None,
) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                question,
                fact_hints=fact_hints,
                facts=facts,
                references=references,
            ),
        },
    ]
