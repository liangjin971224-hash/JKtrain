"""口语同义词的检索查询扩展。

只扩展**检索查询**，不改写用户问题：prompt 里仍然是原话，避免把
「有没有适合三高的」改成模型不认识的书面语。

来源：2026-11-09 实测 h2-03「有没有适合三高的？」在原始查询下三路召回
全空（向量 0 / 关键词 0），直接拒答；补上「高血压 高血糖 高血脂」后
命中 10 条。培训资料写的是规范词，顾客说的是口语，这层映射是必要的。
"""

from __future__ import annotations

#: 口语 → 培训资料里的规范词。
#: 只放高频、歧义低的词；不确定的词不要加，避免污染其他查询。
_COLLOQUIAL = {
    "三高": "高血压 高血糖 高血脂 血压 血糖 血脂",
    "血压高": "高血压 血压",
    "血糖高": "高血糖 血糖",
    "血脂高": "高血脂 血脂",
    "胆固醇高": "胆固醇 血脂",
}


def expand_query(text: str) -> str:
    """在原问题后追加规范词；没有口语词时原样返回。"""
    if not text:
        return text
    extras: list[str] = []
    for colloquial, formal in _COLLOQUIAL.items():
        if colloquial in text and formal not in text:
            extras.append(formal)
    if not extras:
        return text
    # dict.fromkeys 去重保序：一句话里既有「三高」又有「血压高」时不重复。
    return f"{text} {' '.join(dict.fromkeys(extras))}"
