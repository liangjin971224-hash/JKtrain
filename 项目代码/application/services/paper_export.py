"""试卷导出（P6）。

★ 输出 Markdown，能直接贴进 Word / 飞书文档，也能转 PDF。

────────────────────────────────────────────────────────────────────
为什么就出 Markdown
────────────────────────────────────────────────────────────────────
主管的诉求是「打印出来发给学员」，所以：
    · 试卷页：题干 + 选项，**不给答案**
    · 答案页：答案 + 解析，单独一页（方便只印试卷页）

Markdown 是最低公分母：想转 Word/PDF 都能转，而直接生成 .docx
要引入 python-docx，对「打印」这个目的来说是过度设计。

────────────────────────────────────────────────────────────────────
★ 一个细节：题号从 1 开始连续编号
────────────────────────────────────────────────────────────────────
`QuestionSnapshot` 里存的是**题库原始题号**（如 Q005），
但试卷上应该显示 1、2、3…，否则学员看到「第 7 题是 Q019」会困惑。
原始题号只在答案页的脚注里保留，供主管核对。
"""
from __future__ import annotations

from domain.value_objects.exam import ExamPaper

#: 题型在卷面上的标注
_TYPE_HINT = {"单选": "（单选）", "多选": "（多选）"}

#: 难度星标（卷面不显示给学员，只在答案页给主管看）
_DIFF_STAR = {1: "★☆☆", 2: "★★☆", 3: "★★★"}


def render_exam(paper: ExamPaper, *, with_answer: bool = False) -> str:
    """渲染试卷为 Markdown。

    Args:
        paper: 试卷。
        with_answer: True 时在每题下方直接跟答案（用于「带答案版」）；
            False 时只出试卷页，答案另出（`render_answer_key`）。

    Returns:
        Markdown 文本。
    """
    lines: list[str] = []

    # ---- 卷头 ----
    lines.append(f"# {paper.title or '培训考核试卷'}")
    lines.append("")
    lines.append(f"- 卷号：{paper.paper_no}")
    lines.append(f"- 阶段：{paper.stage}")
    lines.append(f"- 题量：{paper.count} 题")
    lines.append(f"- 姓名：____________　　得分：__________")
    lines.append("")
    lines.append("> 说明：单选题每题只有一个正确答案；多选题每题有两个及以上正确答案。")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 题目 ----
    for idx, q in enumerate(paper.questions, start=1):
        hint = _TYPE_HINT.get(q.question_type, "")
        lines.append(f"**{idx}. {q.stem}**{hint}")
        lines.append("")

        for i, opt in enumerate(q.options):
            letter = chr(ord("A") + i)
            lines.append(f"{letter}. {opt}")
        lines.append("")

        if with_answer:
            ans_text = _answer_text(q)
            lines.append(f"> ✅ 答案：**{q.answer}**　{ans_text}")
            if q.explanation:
                lines.append(f"> 📖 解析：{q.explanation}")
            lines.append("")

    return "\n".join(lines)


def render_answer_key(paper: ExamPaper) -> str:
    """渲染答案页（答案 + 解析 + 原始题号）。"""
    lines: list[str] = []

    lines.append(f"# {paper.title or '培训考核试卷'} · 答案与解析")
    lines.append("")
    lines.append(f"- 卷号：{paper.paper_no}")
    lines.append(f"- 题量：{paper.count} 题")
    lines.append("")
    lines.append("---")
    lines.append("")

    for idx, q in enumerate(paper.questions, start=1):
        stars = _DIFF_STAR.get(q.difficulty, "")
        lines.append(f"**{idx}. 答案：{q.answer}**　{stars}")
        lines.append("")
        lines.append(f"- 题干：{q.stem}")
        lines.append(f"- 知识点：{q.kp_name or q.knowledge_point}")
        lines.append(f"- 原题号：{q.question_id}")
        if q.explanation:
            lines.append(f"- 解析：{q.explanation}")
        lines.append("")

    return "\n".join(lines)


def render_summary(paper: ExamPaper) -> str:
    """组卷报告（给主管看：这次考了哪些知识点、难度如何、有没有警告）。"""
    lines: list[str] = []

    lines.append(f"# 组卷报告 · {paper.paper_no}")
    lines.append("")
    lines.append(f"- 标题：{paper.title or '（未命名）'}")
    lines.append(f"- 阶段：{paper.stage}")
    lines.append(f"- 题量：{paper.count} 题")
    lines.append(f"- 创建人：{paper.created_by or '（未填）'}")
    lines.append(f"- 创建时间：{paper.created_at}")
    lines.append("")

    # 知识点覆盖
    lines.append("## 知识点覆盖")
    lines.append("")
    lines.append("| 知识点 | 题数 |")
    lines.append("|---|---|")
    dist = paper.knowledge_point_distribution
    # 反查知识点名称
    name_of = {q.knowledge_point: q.kp_name for q in paper.questions
               if q.knowledge_point}
    for kp, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        lines.append(f"| {name_of.get(kp) or kp} | {cnt} |")
    lines.append("")

    # 难度分布
    lines.append("## 难度分布")
    lines.append("")
    lines.append("| 难度 | 题数 |")
    lines.append("|---|---|")
    for d, cnt in paper.difficulty_distribution.items():
        lines.append(f"| {_DIFF_STAR.get(d, d)} | {cnt} |")
    lines.append("")

    # 警告
    if paper.warnings:
        lines.append("## ⚠️ 组卷警告")
        lines.append("")
        for w in paper.warnings:
            lines.append(f"- {w}")
        lines.append("")
    else:
        lines.append("## 组卷警告")
        lines.append("")
        lines.append("无。知识点全覆盖，难度配比已满足。")
        lines.append("")

    return "\n".join(lines)


def _answer_text(q) -> str:
    """把答案字母转成选项文本（多选拼起来）。"""
    parts: list[str] = []
    for ch in (q.answer or "").upper():
        if not ch.isalpha():
            continue
        i = ord(ch) - ord("A")
        if 0 <= i < len(q.options):
            parts.append(f"{ch}. {q.options[i]}")
    return "；".join(parts)


__all__ = ["render_answer_key", "render_exam", "render_summary"]
