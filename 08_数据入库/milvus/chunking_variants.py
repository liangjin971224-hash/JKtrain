"""
四种切片方案对比
  A. heading   —— 按 Markdown 标题（当前方案）
  B. recursive —— 递归字符切分（RecursiveCharacterTextSplitter）
  C. parentchild —— 父子块（子块检索，父块返回）
  D. casemulti —— 案例分段 + 文档按标题（混合方案）

每个方案都产出统一的 Chunk 列表，接口一致便于对比
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

#: 数据源目录。★ 改为相对本文件推导 + 环境变量覆盖（迁移时改），
#: 原先硬编码 `D:\项目\某健康公司培训系统\06_正式内容`。
BASE_DIR_STR = os.getenv(
    "CONTENT_DIR",
    str(Path(__file__).resolve().parents[2] / "06_正式内容"))

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
# 案例：7 节 → 3 块（P0-2 切片重构，2026-11-07）
# ===================================================================
# 归组按「小节序号前缀」判定，不按标题文本精确匹配。原因：两个源文件的
# 标题写法不同（「一、基本设定」/「一、案例基本设定」，「五、违禁检测」/
# 「五、违禁表述检测（全部 critical）」），按中文序号才稳定。
#
# 附录剥离（2026-11-07 补）：C16 是文件最后一个案例，文件末尾的附录
# （一、按难度分布 … 五、评分阈值建议）也被 CASE_SECTION_PATTERN 命中，
# 旧 chunk_case_split 会把它算进 C16（s98~s102）。若仍按序号归组，附录
# 会混进 C16_m1 / C16_m2 污染正文。规则：同一案例正文出现第二个「一、」
# 时，其后小节全部剥离为独立单元 {cid}_APX_m1，合并成 1 块（group=附录）。
# 只看「序号从头重启」这一结构特征，不硬编码任何附录标题；旧 117 块
# 仍 100% 有归宿（5 条附录 id 归入 C16_APX_m1）。
CASE_GROUP_BY_NUM = {
    "一": (1, "场景"), "二": (1, "场景"), "三": (1, "场景"),
    "四": (2, "要求"), "五": (2, "要求"),
    "六": (3, "示范"), "七": (3, "示范"),
}
CASE_GROUP_NAME = {1: "场景", 2: "要求", 3: "示范"}
CASE_SHORT_CASE_CHARS = 120   # 合并后总长 < 120 字 → 整案例一块（group=全文）
CASE_LONG_CHUNK_CHARS = 1200  # 单块 > 1200 字 → 块内 ### 子标题处二次切分
CASE_MIN_CHARS = 30           # 每块 ≥ 30 字才保留（沿用旧阈值）

_H3_SUBHEAD = re.compile(r"^###[ \t]+\S.*$", re.M)


def _case_sections(body: str) -> list[tuple[str, str]]:
    """按与 chunk_case_split 完全相同的解析口径拆出 (小节标题, 正文)。

    只有两点为合并所需而不同：
      · 保留「概述」/「全文」并归入场景，而不是丢掉；
      · 不在小节粒度做 30 字过滤，过滤推迟到合并之后的块粒度。
    """
    subs = list(CASE_SECTION_PATTERN.finditer(body))
    if not subs:
        return [("全文", body.strip())] if body.strip() else []
    out: list[tuple[str, str]] = []
    head = body[:subs[0].start()].strip()
    if head:
        out.append(("概述", head))
    for i, sm in enumerate(subs):
        sec_title = f"{sm.group(1)}、{sm.group(2).strip()}"
        start = sm.end()
        end = subs[i + 1].start() if i + 1 < len(subs) else len(body)
        text = body[start:end].strip()
        if text:
            out.append((sec_title, text))
    return out


def _case_group_of(sec_title: str) -> int:
    """按序号前缀归组；概述 / 全文 / 未知标题一律并入场景，不产生「其他」组。"""
    return CASE_GROUP_BY_NUM.get(sec_title[:1], (1, "场景"))[0]


#: 附录的通用结构特征：中文序号小节重新开始编号。不硬编码任何附录标题。
_APX_RESTART = "一、"


def _split_appendix(
    secs: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """通用附录剥离：同一案例正文出现第二个「一、」时，其后内容不属于该案例。

    ★ 只作用于 chunk_case_merge；chunk_case_split 行为完全不变，旧 `_sNN`
      id 空间逐字节不变（附录小节在旧口径下仍属于该案例）。
    ★ 判定只看「序号小节从头重启」，不看标题文字、不看案例编号、不看文件名；
      无第二个「一、」的案例一律原样返回（附录为空列表）。

    Returns: (案例本体小节, 附录小节)
    """
    restarts = [i for i, (s, _) in enumerate(secs)
                if s.startswith(_APX_RESTART)]
    if len(restarts) < 2:
        return secs, []
    cut = restarts[1]
    return secs[:cut], secs[cut:]


def _split_long_chunk(body: str) -> list[str]:
    """长块在内部 `###` 子标题处二次切分；无可切子标题时原样返回一块。"""
    marks = list(_H3_SUBHEAD.finditer(body))
    if not marks:
        return [body]
    pieces: list[str] = []
    head = body[:marks[0].start()].strip()
    if head:
        pieces.append(head)
    for i, m in enumerate(marks):
        start = m.start()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        piece = body[start:end].strip()
        if piece:
            pieces.append(piece)
    return pieces if len(pieces) > 1 else [body]


def chunk_case_merge(md: str) -> list[tuple[str, str, str, str, str, list[str]]]:
    """案例 7 节 → 3 块：场景(一二三) / 要求(四五) / 示范(六七)。

    合并时保留小节标题前缀 `【四、必答要素】…【五、违禁检测】…`，避免丢失可读性。
    同一案例正文出现第二个「一、」时，其后小节剥离为独立单元
    `{cid}_APX_m1`（所有小节合并成 1 块，group=`附录`），见 `_split_appendix()`。

    Returns:
        [(new_id, case_id, case_title, group, body, sections)]
        sections = 该块收录的来源小节标题（按文件顺序）。这是与旧 chunker
        对齐、反查 `_sNN` idx 的唯一依据：不按 id 数字推断，不按文本相似度反查。
        附录单元的 case_id 仍是原 cid（旧口径下附录就属于该案例），
        保证 build_case_chunk_map 能按 (cid, 小节标题) 反查到 s98~s102。
    """
    out: list[tuple[str, str, str, str, str, list[str]]] = []
    for cid, ctitle, full in chunk_case_whole(md):
        secs = _case_sections(full)
        if not secs:
            continue

        body_secs, apx_secs = _split_appendix(secs)

        # ① 案例本体：3 组（短案例兜底为 1 块）
        if body_secs:
            # 短块兜底：整案例作为一块
            if sum(len(t) for _, t in body_secs) < CASE_SHORT_CASE_CHARS:
                body = "\n\n".join(f"【{s}】\n{t}" for s, t in body_secs)
                if len(body) >= CASE_MIN_CHARS:
                    out.append((f"{cid}_m1", cid, ctitle, "全文", body,
                                [s for s, _ in body_secs]))
            else:
                grouped: dict[int, list[tuple[str, str]]] = {1: [], 2: [], 3: []}
                for sec_title, text in body_secs:
                    grouped[_case_group_of(sec_title)].append((sec_title, text))

                for g in (1, 2, 3):
                    items = grouped[g]
                    if not items:
                        continue
                    body = "\n\n".join(f"【{s}】\n{t}" for s, t in items)
                    if len(body) < CASE_MIN_CHARS:
                        continue
                    sections = [s for s, _ in items]

                    # 长块上限：仅在块内确有 ### 子标题时才二次切分
                    if len(body) > CASE_LONG_CHUNK_CHARS:
                        pieces = _split_long_chunk(body)
                        if len(pieces) > 1:
                            for k, piece in enumerate(pieces, 1):
                                if len(piece) >= CASE_MIN_CHARS:
                                    out.append((f"{cid}_m{g}_{k}", cid, ctitle,
                                                CASE_GROUP_NAME[g], piece, sections))
                            continue

                    out.append((f"{cid}_m{g}", cid, ctitle, CASE_GROUP_NAME[g],
                                body, sections))

        # ② 附录：所有小节恒合并成 1 块，不做长块二次切分
        if apx_secs:
            apx_body = "\n\n".join(f"【{s}】\n{t}" for s, t in apx_secs)
            if len(apx_body) >= CASE_MIN_CHARS:
                out.append((f"{cid}_APX_m1", cid, ctitle, "附录", apx_body,
                            [s for s, _ in apx_secs]))
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


# ===================================================================
# P0-2：old _sNN → new _mN 映射（gold 重映射的唯一依据）
# ===================================================================
def build_case_chunk_map(base: Path | None = None) -> tuple[dict, dict]:
    """新旧两个 chunker 对同一份文件并行解析，产出确定性映射。

    ⚠️ 旧 chunk_id 的 `_sNN` 是 chunk_case_split() 在**文件内**的位置序号
       （build_collections.py:170-177 的 enumerate），不是小节编号。
       这里直接取 enumerate 的 idx，禁止按 id 数字推断、禁止按文本相似度反查。

    ⚠️ 真实链路是「先切块 → 再 clean_light(body) → <30 字丢弃」
       （build_collections.py:170-173），两侧都按这条链路复现，
       绝不能先清洗整篇再切块。

    Returns:
        (doc, stats)
        doc  = {old_id: new_id, ..., new_id: {case_id, group, sections,
                chars, chars_raw, old_ids}, ...}
        stats = 对账统计（旧块数 / 新块数 / 字数分布 / 长块检查 / 缺漏）
    """
    import json  # noqa: F401  (保持本模块顶层依赖不变)
    import statistics
    import sys as _sys

    _here = str(Path(__file__).resolve().parent)
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from text_clean import clean_light

    src = Path(base) if base else (Path(BASE_DIR_STR) / "02_情景案例")
    mapping: dict[str, str] = {}
    meta: dict[str, dict] = {}
    all_old: list[str] = []
    long_checked: list[dict] = []

    for f in sorted(src.glob("*.md")):
        md = f.read_text(encoding="utf-8")

        # ① 旧 chunker：idx 直接来自 enumerate，落库口径复现真实链路
        old_by_sec: dict[tuple[str, str], list[str]] = {}
        for idx, (cid, _ctitle, sec, body) in enumerate(chunk_case_split(md)):
            if len(clean_light(body)) < CASE_MIN_CHARS:
                continue
            old_id = f"{cid}_s{idx:02d}"
            all_old.append(old_id)
            old_by_sec.setdefault((cid, sec), []).append(old_id)

        # ② 新 chunker：同一份 md 独立解析，自带来源小节标题
        for new_id, cid, _ctitle, group, body, sections in chunk_case_merge(md):
            text = clean_light(body)
            if len(text) < CASE_MIN_CHARS:
                continue

            old_ids: list[str] = []
            for sec in sections:
                for oid in old_by_sec.get((cid, sec), []):
                    if oid not in old_ids:
                        old_ids.append(oid)

            for oid in old_ids:
                if oid in mapping and mapping[oid] != new_id:
                    raise ValueError(f"旧块被映射到两个新块: {oid} → "
                                     f"{mapping[oid]} / {new_id}")
                mapping[oid] = new_id
            if new_id in meta:
                raise ValueError(f"新块 id 冲突: {new_id}")

            meta[new_id] = {
                "case_id": cid,
                "group": group,
                "sections": sections,
                "chars": len(text),          # 落库口径：clean_light 后
                "chars_raw": len(body),      # 切块原文（含【小节】前缀）
                "old_ids": old_ids,          # 反向表，供 expandedgold 使用
            }
            if len(body) > CASE_LONG_CHUNK_CHARS:
                long_checked.append({
                    "new_id": new_id, "chars_raw": len(body),
                    "h3_subheads": len(_H3_SUBHEAD.findall(body)),
                    "split": "_" in new_id.rsplit("_m", 1)[-1],
                })

    missing = [oid for oid in all_old if oid not in mapping]
    if missing:
        raise ValueError(f"以下旧块没有归宿（{len(missing)} 条）: {missing}")
    if len(mapping) != len(all_old):
        raise ValueError(f"映射条目数 {len(mapping)} != 落库旧块数 {len(all_old)}")

    # 反向表一致性：meta.old_ids 必须与正向映射完全吻合
    for new_id, m in meta.items():
        expect = sorted(oid for oid, nid in mapping.items() if nid == new_id)
        if sorted(m["old_ids"]) != expect:
            raise ValueError(f"新块 {new_id} 的 old_ids 与正向映射不一致")

    lens_clean = [m["chars"] for m in meta.values()]
    lens_raw = [m["chars_raw"] for m in meta.values()]
    by_group: dict[str, list[int]] = {}
    by_case: dict[str, int] = {}
    for m in meta.values():
        by_group.setdefault(m["group"], []).append(m["chars"])
        by_case[m["case_id"]] = by_case.get(m["case_id"], 0) + 1

    def _dist(xs: list[int]) -> dict:
        s = sorted(xs)
        return {
            "mean": round(sum(s) / len(s), 1),
            "median": statistics.median(s),
            "p90": s[min(len(s) - 1, int(len(s) * 0.9))],
            "min": s[0], "max": s[-1],
        }

    stats = {
        "source_files": sorted(p.name for p in src.glob("*.md")),
        "old_chunks_landed": len(all_old),
        "map_entries": len(mapping),
        "new_chunks": len(meta),
        "cases": len(by_case),
        "blocks_per_case": sorted(set(by_case.values())),
        "groups": {g: {"blocks": len(v), "mean_chars": round(sum(v) / len(v), 1)}
                   for g, v in sorted(by_group.items())},
        "chars_clean": _dist(lens_clean),
        "chars_raw": _dist(lens_raw),
        "long_chunk_checks": long_checked,
        "missing_old_ids": missing,
        "samples": [
            {"old": oid, "new": mapping[oid]}
            for oid in ["C11_s59", "C12_s68", "C16_s97"]
            if oid in mapping
        ],
    }
    doc: dict = dict(mapping)
    doc.update(meta)
    return doc, stats


if __name__ == "__main__":
    import json as _json

    _doc, _stats = build_case_chunk_map()
    _out = Path(__file__).with_name("case_chunk_map.json")
    _out.write_text(_json.dumps(_doc, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(_json.dumps({"output": str(_out), "stats": _stats},
                      ensure_ascii=False, indent=2))
