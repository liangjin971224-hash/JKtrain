"""题库读取（组卷的数据来源）。

★ 为什么单独一层
────────────────────────────────────────────────────────────────────
`interfaces` 不能 import `infrastructure`（门禁强制），
`application` 也不能（只能依赖 Port）。

所以：Milvus 查询放在 infrastructure，这里做「行 → 快照」的转换。
接口层和用例层只看到 `QuestionSnapshot`，不知道数据来自 Milvus 还是别处。

★ 题库里的实际字段形态（实测）
────────────────────────────────────────────────────────────────────
Milvus `health_question_bank` 的字段：

    chunk_id           题目编号（如 Q_S1_001）
    text               题干
    knowledge_point    知识点编码（如 KP_CLAIM_BOUNDARY）
    kp_name            知识点名称
    question_type      单选 / 多选
    difficulty         1 / 2 / 3
    extra              JSON 字符串：{"stage": "S1", "answer": "B", ...}
    doc_id             来源文档

★ 题目正文（题干 + 选项 + 解析）都在 `text` 里，需要用正则解析。
  这是当初入库时的格式决定的，解析器必须容错——某题格式不标准时
  不能让整个组卷挂掉，跳过并记日志。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from domain.value_objects.exam import QuestionSnapshot

logger = logging.getLogger(__name__)

#: 选项行：`A. xxx` / `B、xxx` / `C)xxx` / `D：xxx`
_OPTION_RE = re.compile(
    r"^\s*([A-H])\s*[.、)）:：]\s*(.+?)\s*$", re.MULTILINE)

#: 解析行：`解析：xxx` / `【解析】xxx` / `答案：B`
_EXPLAIN_RE = re.compile(
    r"(?:【解析】|解析[:：]|答案解析[:：])\s*(.+?)(?:\n\n|$)", re.DOTALL)

#: 题干与选项之间的分界：第一个选项行之前都是题干
_ANSWER_IN_TEXT_RE = re.compile(
    r"(?:【答案】|答案[:：])\s*([A-H]+)")


def to_snapshot(row: dict[str, Any]) -> QuestionSnapshot | None:
    """把 Milvus 的一行转成题目快照。

    Args:
        row: Milvus 查询结果中的一行。

    Returns:
        快照；格式不可解析时返回 None（跳过，不中断组卷）。

    ★ 为什么要容错：题库是人工+合成混合来的，个别题格式可能不标准。
      一道题解析失败不该让整次组卷失败——跳过它，
      还有 139 道可用。但要记日志，便于事后修正。
    """
    qid = str(row.get("chunk_id") or "").strip()
    text = str(row.get("text") or "")
    if not qid or not text:
        logger.warning("题目缺编号或正文，跳过: %r", row.get("chunk_id"))
        return None

    extra = _parse_extra(row.get("extra"))

    stem, options = _split_stem_options(text)
    if not stem or not options:
        logger.warning("题目格式无法解析（无选项），跳过: %s", qid)
        return None

    answer = str(extra.get("answer") or "").strip().upper()
    if not answer:
        m = _ANSWER_IN_TEXT_RE.search(text)
        answer = m.group(1) if m else ""
    if not answer:
        logger.warning("题目缺答案，跳过: %s", qid)
        return None

    # 答案字母必须落在选项范围内
    if any(ord(ch) - ord("A") >= len(options) for ch in answer if ch.isalpha()):
        logger.warning("题目答案 %s 超出选项范围（%d 项），跳过: %s",
                       answer, len(options), qid)
        return None

    explanation = ""
    em = _EXPLAIN_RE.search(text)
    if em:
        explanation = em.group(1).strip()

    try:
        difficulty = int(row.get("difficulty") or extra.get("difficulty") or 2)
    except (TypeError, ValueError):
        difficulty = 2
    difficulty = min(3, max(1, difficulty))     # 夹到 1–3，脏数据不至于炸

    return QuestionSnapshot(
        question_id=qid,
        knowledge_point=str(row.get("knowledge_point") or
                            extra.get("knowledge_point") or "").strip(),
        kp_name=str(row.get("kp_name") or "").strip(),
        question_type=str(row.get("question_type") or "单选").strip(),
        difficulty=difficulty,
        stem=stem,
        options=tuple(options),
        answer=answer,
        explanation=explanation,
        product_id=str(extra.get("product_id") or "").strip(),
    )


def build_bank(
    rows: list[dict[str, Any]],
    *,
    stage: str = "",
    require_approved: bool = True,
) -> list[QuestionSnapshot]:
    """把一批题库行转成快照列表。

    Args:
        rows: Milvus 查询结果。
        stage: 只要该阶段的题。空表示不限。
        require_approved: 只要审核通过的题。

    Returns:
        可用题目快照。解析失败的已跳过并记日志。
    """
    out: list[QuestionSnapshot] = []
    skipped_review = 0
    skipped_stage = 0

    for row in rows:
        if require_approved:
            status = str(row.get("review_status") or "").strip().lower()
            # 空状态按「已审核」处理（历史数据可能没这个字段）
            if status and status != "approved":
                skipped_review += 1
                continue

        if stage:
            extra = _parse_extra(row.get("extra"))
            row_stage = str(extra.get("stage") or "").strip()
            if row_stage and row_stage != stage:
                skipped_stage += 1
                continue

        snap = to_snapshot(row)
        if snap is not None:
            out.append(snap)

    logger.info(
        "题库构建: 输入 %d 行 → 可用 %d 题（跳过未审核 %d / 非目标阶段 %d）",
        len(rows), len(out), skipped_review, skipped_stage)
    return out


# ===================================================================
# 内部
# ===================================================================
def _parse_extra(raw: Any) -> dict[str, Any]:
    """解析 extra 字段（可能是 JSON 字符串，也可能是已解析的 dict）。"""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        v = json.loads(str(raw))
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, TypeError):
        # 有些题的 extra 格式坏了，从正文里兜底取（见 to_snapshot）
        logger.debug("extra 解析失败: %r", str(raw)[:80])
        return {}


def _split_stem_options(text: str) -> tuple[str, list[str]]:
    """从题目正文里拆出题干和选项。

    ★ 题干 = 第一个选项行之前的内容。
      「答案」「解析」之后的内容不算题干。

    Returns:
        (题干, 选项文本列表)。选项少于 2 个视为解析失败（返回空题干）。
    """
    # 先切掉「答案/解析」之后的部分，避免把解析里的字母当成选项
    body = text
    for pat in (r"【答案】", r"\n答案[:：]", r"【解析】", r"\n解析[:：]"):
        m = re.search(pat, body)
        if m:
            body = body[:m.start()]

    matches = list(_OPTION_RE.finditer(body))
    if len(matches) < 2:
        return "", []

    stem = body[:matches[0].start()].strip()
    # 去掉「题干：」这类前缀
    stem = re.sub(r"^(?:题干|题目)[:：]\s*", "", stem)
    # 去掉【单选】【多选】这类标记
    stem = re.sub(r"^【[^】]{1,6}】\s*", "", stem)

    options = [m.group(2).strip() for m in matches]
    return stem, options


__all__ = ["build_bank", "to_snapshot"]
