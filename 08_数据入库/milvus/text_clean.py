"""
文本清洗模块
三档强度，用于 A/B 测试清洗对召回的影响

L0 none      不清洗
L1 light     去格式符（emoji/加粗/分隔线/引用元信息）
L2 deep      L1 + 表格转述 + 去元信息段（文档信息块/学习目标/自测题）
"""
from __future__ import annotations

import re


# ===================================================================
# L1 轻度清洗：只去格式符号，不动内容结构
# ===================================================================
# 常见的 Markdown 格式噪声
_EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001F9FF"      # 各类 emoji
    "\u2600-\u27BF"                # 杂项符号（含 ✅❌⚠）
    "\u2B00-\u2BFF"                # 箭头符号
    "\uFE0F"                       # 变体选择符
    "\u2605\u2606"                 # ★☆
    "]+"
)

_CJK = r"\u4e00-\u9fff"


def clean_light(text: str) -> str:
    """轻度清洗：去格式噪声，保留全部内容。

    做四件事：
      1. 去 emoji / ★☆ / ✅❌ 等符号
      2. 去 Markdown 加粗/斜体标记（**xxx** → xxx）
      3. 去行内代码反引号
      4. 去分隔线 --- / ***
    """
    if not text:
        return text

    t = text
    # 分隔线
    t = re.sub(r"^\s*[-*_]{3,}\s*$", "", t, flags=re.M)
    # 加粗、斜体
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", t)
    t = re.sub(r"__([^_]+)__", r"\1", t)
    # 行内代码
    t = re.sub(r"`([^`]+)`", r"\1", t)
    # emoji
    t = _EMOJI_PATTERN.sub("", t)
    # 多余空格/空行
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


# ===================================================================
# L2 深度清洗：L1 + 表格转述 + 元信息剔除
# ===================================================================
# 文档信息块（如 "> **文档编号**：D005 ｜ **类别**：培训资料"）
_FRONT_MATTER = re.compile(
    r"^>\s*\*\*文档编号\*\*.*?(?=\n(?!#|>|\s*$))",
    re.M | re.S,
)


def table_to_text(md_table: str) -> str:
    """把 Markdown 表格转成「表头: 值」的自然语言行。

    为什么这样做：表格的 `|` 和 `---` 对向量模型是纯噪声，
    而「序号: 1 | 功能声称原文: 有助于增强免疫力」这种形式更接近自然语言。
    """
    lines = [l.strip() for l in md_table.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return md_table

    # 解析表头
    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = cells(lines[0])
    # 跳过分隔行（|---|）
    data_lines = [l for l in lines[1:] if not re.fullmatch(r"[\s|:\-]+", l)]
    if not data_lines:
        return md_table

    out: list[str] = []
    for row in data_lines:
        vals = cells(row)
        pairs = []
        for i, v in enumerate(vals):
            if not v:
                continue
            h = header[i] if i < len(header) else f"列{i+1}"
            h = re.sub(r"[*`]", "", h)
            pairs.append(f"{h}：{v}" if h else v)
        if pairs:
            out.append("；".join(pairs))
    return "\n".join(out)


_TABLE_BLOCK = re.compile(
    r"(?:^\|.*\|\s*$\n?)+",
    re.M,
)


def convert_tables(text: str) -> str:
    """把文本中所有 Markdown 表格转成自然语言。"""
    def repl(m: re.Match) -> str:
        converted = table_to_text(m.group(0))
        # 前后各加一个空行，避免与相邻段落粘连
        return "\n\n" + converted + "\n\n"

    return _TABLE_BLOCK.sub(repl, text)


# 元信息段落（对检索无帮助，且会抢排名）
_META_SECTION_TITLES = [
    r"^学习目标$",
    r"^文档信息$",
    r"^审核重点$",
    r"^数据性质$",
    r"^适用范围$",
]

_META_BLOCK = re.compile(
    r"^>\s*\*\*(文档编号|类别|权限|目标读者|预计字数|版本|编制日期|审核重点|重要程度|数据性质|适用范围)\*\*.*$",
    re.M,
)


def clean_deep(text: str, drop_qa: bool = False) -> str:
    """深度清洗：L1 + 表格转述 + 去元信息。

    Args:
        drop_qa: 是否剔除「自测题 / 参考答案」段（实验用开关）
    """
    t = clean_light(text)

    # 去文档头部的元信息引用块
    t = _META_BLOCK.sub("", t)

    # 表格转自然语言
    t = convert_tables(t)

    # 去残留的表格分隔符与空行
    t = re.sub(r"^\s*\|[\s|:\-]+\|\s*$", "", t, flags=re.M)
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


# ===================================================================
# 额外的清洗项（可选）
# ===================================================================
def strip_self_test(text: str) -> str:
    """剔除自测题/参考答案段（按 ## 标题定位）。"""
    lines = text.splitlines()
    out: list[str] = []
    skipping = False

    for line in lines:
        m = re.match(r"^#{2,3}\s+(.*)$", line)
        if m:
            title = m.group(1).strip()
            skipping = bool(re.search(r"自测题|参考答案", title))
        if not skipping:
            out.append(line)

    return "\n".join(out)


def normalize_punct(text: str) -> str:
    """统一标点：全角括号、中文标点规范化。"""
    t = text.replace("（", "(").replace("）", ")")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()
