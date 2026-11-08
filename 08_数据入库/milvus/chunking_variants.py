"""
四种切片方案对比
  A. heading   —— 按 Markdown 标题（当前方案）
  B. recursive —— 递归字符切分（RecursiveCharacterTextSplitter）
  C. parentchild —— 父子块（子块检索，父块返回）
  D. casemulti —— 案例分段 + 文档按标题（混合方案）

每个方案都产出统一的 Chunk 列表，接口一致便于对比
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

BASE_DIR_STR = r"D:\项目\某健康公司培训系统\06_正式内容"

TRAINING_PERMISSION = {
    "D001": "public", "D002": "internal", "D003": "public", "D004": "public",
    "D005": "internal", "D006": "public", "D007": "public", "D008": "public",
    "D009": "internal", "D010": "internal", "D011": "internal", "D012": "internal",
    "D013": "internal", "D014": "internal", "D015": "internal",
}


@dataclass
class Unit:
    """统一的分块单元。parent_id 非空表示这是子块。"""

    chunk_id: str
    text: str
    doc_id: str
    title: str
    section: str
    source_file: str
    permission_level: str
    category: str
    chunk_index: int = 0
    parent_id: str = ""          # 父子块方案用
    parent_text: str = ""        # 父块全文（检索命中后返回这个）
    extra: str = ""


def clean(t: str) -> str:
    t = re.sub(r"^```[a-zA-Z]*\s*$", "", t, flags=re.M)
    t = re.sub(r"^-{3,}\s*$", "", t, flags=re.M)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ===================================================================
# A. 按标题切（当前方案）
# ===================================================================
def chunk_heading(md: str) -> list[tuple[str, str]]:
    """返回 [(section, text)]"""
    lines = md.splitlines()
    out: list[tuple[str, str]] = []
    cur_title = ""
    cur: list[str] = []

    def flush():
        if not cur:
            return
        body = "\n".join(cur).strip()
        if not body:
            return
        if len(body) > 1400:
            parts = re.split(r"\n\s*\n", body)
            buf: list[str] = []
            for p in parts:
                buf.append(p)
                if sum(len(x) for x in buf) >= 700:
                    out.append((cur_title, "\n\n".join(buf).strip()))
                    buf = []
            if buf:
                out.append((cur_title, "\n\n".join(buf).strip()))
        else:
            out.append((cur_title, body))

    for line in lines:
        if re.match(r"^#{1,3}\s", line):
            flush()
            cur = []
            cur_title = line.lstrip("#").strip()
        else:
            cur.append(line)
    flush()
    return [(t, b) for t, b in out if b]


# ===================================================================
# B. 递归字符切分
# ===================================================================
def chunk_recursive(md: str, chunk_size: int = 500, overlap: int = 80) -> list[tuple[str, str]]:
    """模拟 RecursiveCharacterTextSplitter 的行为。

    分隔符优先级：\n## → \n### → \n\n → \n → 。
    但不跨标题合并，保留 section 归属。
    """
    # 先按顶层标题分段以保留 section
    sections = chunk_heading(md)

    out: list[tuple[str, str]] = []
    for section, body in sections:
        if len(body) <= chunk_size:
            out.append((section, body))
            continue

        # 超长：按分隔符递归切
        pieces = _split_by_separators(body, chunk_size, overlap)
        for p in pieces:
            out.append((section, p))
    return [(s, b) for s, b in out if len(b) >= 30]


def _split_by_separators(text: str, size: int, overlap: int) -> list[str]:
    """递归按分隔符切分，带 overlap。"""
    separators = ["\n\n", "\n", "。", "；", "，"]

    def split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= size:
            return [t]
        if not seps:
            # 硬切
            chunks = []
            for i in range(0, len(t), size - overlap):
                chunks.append(t[i:i + size])
            return chunks

        sep = seps[0]
        parts = t.split(sep)
        if len(parts) == 1:
            return split(t, seps[1:])

        result: list[str] = []
        buf = ""
        for p in parts:
            candidate = (buf + sep + p) if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    result.append(buf)
                if len(p) > size:
                    result.extend(split(p, seps[1:]))
                    buf = ""
                else:
                    buf = p
        if buf:
            result.append(buf)

        # 加 overlap：每个块尾部拼接下一块头部
        if overlap > 0 and len(result) > 1:
            merged = []
            for i, r in enumerate(result):
                if i == 0:
                    merged.append(r)
                else:
                    tail = result[i - 1][-overlap:]
                    merged.append(tail + r)
            return merged
        return result

    return split(text, separators)


# ===================================================================
# C. 父子块
# ===================================================================
def chunk_parent_child(md: str, child_size: int = 300,
                       child_overlap: int = 60) -> list[tuple[str, str, str, str]]:
    """父子块：父块 = 整个 section，子块 = section 内按 300 字切。

    Returns:
        [(section, child_text, parent_id, parent_text)]
    """
    sections = chunk_heading(md)
    out: list[tuple[str, str, str, str]] = []

    for idx, (section, body) in enumerate(sections):
        parent_id = f"P{idx:03d}"
        if len(body) <= child_size:
            out.append((section, body, parent_id, body))
            continue

        children = _split_by_separators(body, child_size, child_overlap)
        for c in children:
            if len(c) >= 30:
                out.append((section, c, parent_id, body))
    return out


# ===================================================================
# 案例：整块 vs 分段
# ===================================================================
# 案例分段正则。
#
# 修正记录（2026-10-22）：原正则写死 `###`，但两个源文件的标题层级不同——
#   · 情景演练案例集_14个.md      → 案例用 `# C02`，段落用 `## 一、`
#   · 情景演练案例集_核心2个      → 案例用 `## 案例 C01`，段落用 `### 一、`
# 导致 14 个案例全部退化成「全文一块」，案例分段方案实际只在 2 个案例上生效。
# 改为 {2,4} 任意层级后，两种源文件都能正确命中。
CASE_SECTION_PATTERN = re.compile(
    r"^#{2,4}\s+([一二三四五六七八九十])、(.+)$", re.M
)


def chunk_case_whole(md: str) -> list[tuple[str, str, str]]:
    """整案例一块。Returns [(case_id, case_title, full_text)]"""
    pattern = re.compile(r"^(?:#\s+|\#{2}\s+案例\s*)(C\d{2})[：:\s]*(.*)$", re.M)
    matches = list(pattern.finditer(md))
    out = []
    for i, m in enumerate(matches):
        cid = m.group(1)
        title = m.group(2).strip().rstrip("：:")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        out.append((cid, title or cid, clean(md[start:end])))
    return out


def chunk_case_split(md: str) -> list[tuple[str, str, str, str]]:
    """案例按 一、二、三... 分段。

    Returns: [(case_id, case_title, section_title, text)]
    """
    cases = chunk_case_whole(md)
    out = []

    for cid, title, body in cases:
        # 在正文里找 "### 一、" 这类小标题
        subs = list(CASE_SECTION_PATTERN.finditer(body))
        if not subs:
            out.append((cid, title, "全文", body))
            continue

        # 开头部分（第一个小标题之前）作为「概述」
        head = body[:subs[0].start()].strip()
        if len(head) >= 30:
            out.append((cid, title, "概述", head))

        for i, sm in enumerate(subs):
            sec_title = f"{sm.group(1)}、{sm.group(2).strip()}"
            start = sm.end()
            end = subs[i + 1].start() if i + 1 < len(subs) else len(body)
            text = body[start:end].strip()
            if len(text) >= 30:
                out.append((cid, title, sec_title, text))
    return out


# ===================================================================
# 噪声章节判定（实验结果驱动）
# ===================================================================
# 实测发现：「自测题」「参考答案」「学习目标」等章节在检索时会抢占排名，
# 因为它们的措辞与查询高度相似（如"客户档案哪些内容不能记"→"自测题"），
# 但不携带答案正文。
#
# 修正记录（2026-10-22）：
#   原实现有两个问题——
#   ① `strip_self_test()` 靠 `^## 自测题` 定位，但 chunk_heading 已经剥掉了标题行，
#      body 里根本没有标题可匹配 → 对培训文档 100% 失效（47 个噪声块 0 个被剔）。
#   ② 该函数作用在案例块上时，删掉的是「六、参考答案」段 → 演练评阅失去判定依据。
#   现改为在 **section 层面**判定（section 由 chunk_heading 保留），
#   并区分两档：
#     NOISE_META  纯元信息，无答案，直接丢
#     NOISE_QA    「自测题 / 参考答案」，不丢内容，分流到 health_self_test

# 纯元信息：剔除无损
NOISE_META_PATTERNS = [
    r"^学习目标$",
]

# 问答区：从正文库剥离，但内容保留在 health_self_test
NOISE_QA_PATTERNS = [
    r"自测题",
    r"参考答案",
]

# 曾尝试剔除的「小结 / 总结 / 速查表 / 难点」**未纳入**：
# 实测它们携带正文信息，剔了反而降低 R@3/R@5。速查表尤其不能剔（见 D004/D008）。


def _match_any(section: str, patterns: list[str]) -> bool:
    s = section.strip()
    return any(re.search(p, s) for p in patterns)


def is_meta_section(section: str) -> bool:
    """纯元信息章节（学习目标），剔除无损。"""
    return _match_any(section, NOISE_META_PATTERNS)


def is_qa_section(section: str) -> bool:
    """问答区章节（自测题/参考答案），应分流到 health_self_test。"""
    return _match_any(section, NOISE_QA_PATTERNS)


# 兼容旧调用：噪声 = 元信息 + 问答区
NOISE_SECTION_PATTERNS = NOISE_META_PATTERNS + NOISE_QA_PATTERNS


def is_noise_section(section: str) -> bool:
    """判断章节是否为噪声（不携带答案正文）。"""
    return is_meta_section(section) or is_qa_section(section)
