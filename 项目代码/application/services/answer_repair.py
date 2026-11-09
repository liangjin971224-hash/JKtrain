"""条件式答案补全信号与修复 prompt。

只在初稿出现可判定的缺失时追加一次生成；正常回答不触发，避免把
P1-b 验证过的「多喂上下文反而掉分」问题再引回来。

三类信号都来自 2026-11-09 的 100 题归因：
- `spurious_no_info`：检索明明有关键词高分命中，初稿却说「资料里没有」；
- `medical_escalation`：停药/同服/不适/赔偿类问题，初稿没有就医或上报；
- `numeric_missing`：问题在问数字/用量/规格，初稿一个数字都没有。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_NO_INFO = (
    "资料里没有",
    "资料中没有",
    "没有这个信息",
    "未提及",
    "没有提及",
    "资料未给出",
    "无法回答",
)
#: 只认「用药冲突」或「已经出现身体不适」的问法；「副作用能不能说」
#: 属于合规话术题，不应触发就医补全。
_MEDICAL = re.compile(
    r"停药|同服|一起吃|抗凝|阿司匹林|华法林|头晕|不适|过敏|赔偿|就医|"
    r"拉肚子|睡不着|失眠"
)
#: 合规话术/案例评分题：即使出现「停药」字样，也不是客户正在发生的不良事件。
_CLAIM_Q = re.compile(r"说法|表述|违禁|评分|检测表|案例|话术|能不能说|能不能讲|该怎么答")
#: 医疗升级表述。「上报主管」不算——它解决的是流程，不解决就医建议。
_ESCALATION = ("医生", "医嘱", "就医", "医院", "120", "急救")
_NUMERIC_Q = re.compile(
    r"多少|几克|几份|几粒|几次|每日|每次|上限|含量|规格|得分|几分|多少分|比例"
)
_NUMBER = re.compile(
    r"[0-9０-９]"
    r"|[一二三四五六七八九十百千万亿]+(?:g|mg|克|份|袋|片|粒|次|天|年|岁|%|分|类|项|处)"
)

#: 只用于「初稿说没有、证据却有」的粗筛；去掉高频词，避免「客户/产品」这类
#: 泛词把一篇无关文档也算成高重合。
_STOP_BIGRAMS = {
    "什么",
    "怎么",
    "怎样",
    "如何",
    "可以",
    "不能",
    "我们",
    "你们",
    "他们",
    "这个",
    "那个",
    "哪个",
    "多少",
    "是否",
    "没有",
    "有没",
    "应该",
    "需要",
    "客户",
    "产品",
    "资料",
    "问题",
    "这种",
    "那种",
    "能不",
    "不是",
    "是不",
    "是什",
    "么样",
    "么办",
}


@dataclass(frozen=True, slots=True)
class RepairSignal:
    code: str
    instruction: str


def _has_any(text: str, words: Sequence[str]) -> bool:
    return any(word in text for word in words)


def _content_bigrams(text: str) -> set[str]:
    chars = re.findall(r"[\u4e00-\u9fff]", text or "")
    return {
        a + b
        for a, b in zip(chars, chars[1:])
        if a + b not in _STOP_BIGRAMS
    }


def _spurious_no_info(
    question: str,
    evidence: str,
    keyword_score_max: float,
    vector_score_max: float,
    min_keyword_score: float,
) -> bool:
    overlap = _content_bigrams(question) & _content_bigrams(evidence)
    if len(overlap) >= 2:
        return True
    # 关键词路本身已过绝对阈值时，一个具体重合词也足以说明
    # 不是「完全没资料」，而是模型过早放弃。
    if keyword_score_max >= min_keyword_score and len(overlap) >= 1:
        return True
    # 向量分很高（≥0.75）且至少一个具体重合词，同样说明有直接依据。
    return vector_score_max >= 0.75 and len(overlap) >= 1


def detect_gaps(
    question: str,
    answer: str,
    evidence_texts: Sequence[str],
    *,
    keyword_score_max: float = 0.0,
    vector_score_max: float = 0.0,
    min_keyword_score: float = 14.0,
) -> tuple[RepairSignal, ...]:
    """返回初稿需要补全的信号；空元组表示不触发补全。"""
    answer = answer or ""
    evidence = " ".join(evidence_texts or ())
    signals: list[RepairSignal] = []

    if _has_any(answer, _NO_INFO) and _spurious_no_info(
        question, evidence, keyword_score_max, vector_score_max, min_keyword_score
    ):
        signals.append(
            RepairSignal(
                "spurious_no_info",
                "初稿说「资料里没有」，但参考资料中有关键词直接命中的内容。"
                "请基于参考资料直接回答，先给结论，再说明依据；不要再用「资料里没有」。",
            )
        )

    if (
        _MEDICAL.search(question or "")
        and not _CLAIM_Q.search(question or "")
        and not _has_any(answer, _ESCALATION)
    ):
        signals.append(
            RepairSignal(
                "medical_escalation",
                "问题涉及用药、身体不适或赔偿，回答必须补上「建议就医」或「遵医嘱」，"
                "并写明「上报主管/详细记录」等处置；不要替客户判断原因。",
            )
        )

    if _NUMERIC_Q.search(question or "") and not _NUMBER.search(answer):
        signals.append(
            RepairSignal(
                "numeric_missing",
                "问题在问具体数字、用量或规格，回答必须给出资料中的数字和单位；"
                "资料确实没有时写「资料未给出」，不要用文字含糊带过。",
            )
        )

    return tuple(signals)


def build_repair_prompt(
    original_prompt: str,
    draft: str,
    signals: Sequence[RepairSignal],
) -> str:
    """在原始 prompt（含同一批参考资料）后追加初稿与补全要求。"""
    checklist = "\n".join(f"- {signal.instruction}" for signal in signals)
    return (
        f"{original_prompt}\n\n"
        "【初稿】（信息可能不完整）\n"
        f"{draft.strip()}\n\n"
        "【必须补全】\n"
        f"{checklist}\n\n"
        "请输出修订后的完整回答。要求：\n"
        "1. 保留初稿中已有且正确的结论，不要改变结论方向；\n"
        "2. 只补充【结构化事实】/【参考资料】中能直接核实的内容，不引入资料外信息；\n"
        "3. 不写自检过程，不解释你在补什么；\n"
        "4. 资料确实没有的项写「资料未给出」，不要编造。"
    )
