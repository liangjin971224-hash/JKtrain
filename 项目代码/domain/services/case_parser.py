"""案例文档解析器（P5）。

把 `06_正式内容/02_情景案例/` 的两个 Markdown 解析成 `CaseSpec`。

────────────────────────────────────────────────────────────────────
★ 两个源文件的标题层级不同（P0 数据层踩过的同一个坑）
────────────────────────────────────────────────────────────────────
    情景演练案例集_核心2个_C01与C04.md   用 ### 一、案例基本设定
    情景演练案例集_14个.md              用 ## 一、基本设定

    所以匹配章节标题必须用 `#{2,4}`，写死 `##` 或 `###` 都会漏。

────────────────────────────────────────────────────────────────────
解析策略：宁可漏解析一项，也不要解析出脏数据
────────────────────────────────────────────────────────────────────
案例文档是**人工维护**的，格式难免不齐。所以：
    · 表格解析失败时跳过该行，打 warning，不填默认值
    · 正则解析失败的不进 `ForbiddenRule`（因为非法正则会直接抛）
    · 必答要素的「权重」「类型」解析不到时，用保守默认值并记 warning

这样系统能跑起来，同时把问题暴露在日志里——比静默塞假数据好。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from domain.value_objects import (
    CaseSpec,
    ElementSpec,
    ForbiddenRule,
    RiskLevel,
    ScriptedFollowUp,
)

logger = logging.getLogger(__name__)

#: 章节标题。★ 用 {2,4} 覆盖两个源文件的层级差异。
#:
#: ★ 不能用 f-string / .format 拼接：`{2,4}` 会被当成格式占位符
#:   （实测报 `KeyError: '2,4'`）。所以用显式字符串加法。
_SEC_PREFIX = r"^#{2,4}\s*[一二三四五六七]、\s*"

#: 案例分隔：`## 案例 C01：xxx` 或 `# C02 xxx`
_CASE_HEAD = re.compile(
    r"^#{1,3}\s*(?:案例\s*)?(C\d{2})\s*[：:、\s]\s*(.+?)\s*$", re.M)


def _section(text: str, name_pattern: str, next_names: list[str]) -> str:
    """截取某个章节的正文。

    Args:
        text: 全文。
        name_pattern: 章节名正则片段，如 `基本设定`。
        next_names: 后续可能的章节名，用于确定结束位置。

    Returns:
        章节正文（不含标题行）。找不到返回空串。
    """
    start_m = re.search(_SEC_PREFIX + name_pattern, text, re.M)
    if start_m is None:
        return ""

    start = start_m.end()
    # ★ 保留标题行的剩余部分（如「（全部 critical）」）——
    #   8 个案例把级别写在标题里而不是表格里，丢了就读不到级别。
    #   用换行分隔，让 _parse_forbidden 仍能取到首行。
    tail = text[start:]
    end_m = re.search(r"^#{2,4}\s*[一二三四五六七]、", tail, re.M)
    if end_m is not None:
        return tail[:end_m.start()]
    # 没找到下一章，截到下一个案例或文件尾
    next_case = re.search(r"^#{1,3}\s*(?:案例\s*)?C\d{2}", tail, re.M)
    if next_case is not None:
        return tail[:next_case.start()]
    return tail


def _parse_kv_table(body: str) -> dict[str, str]:
    """解析 `| 字段 | 内容 |` 形式的两列表格。

    ★ 键要先去掉 markdown 强调符号（2026-10-31 修）：
      两批案例的表格写法不一致——
        14 个案例：`| 提问类型 | 产品信息询问（product_ingredient） |`
        C01/C04  ：`| **提问类型** | 产品信息询问（product_existence） |`  ← 带粗体
      原来直接把 `key` 当字典键，C01/C04 的键变成 `**提问类型**`，
      于是 `basic.get("提问类型")` 取不到 → question_type 是空串
      → 月度报告里这两个案例归不进任何能力项（其中 C04 是唯一的红线案例）。

      去符号后两批写法统一，取值恢复正常。
    """
    out: dict[str, str] = {}
    for line in body.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = _split_table_row(line)
        if len(cells) < 2:
            continue
        # ★ 键去粗体/行内代码标记，值也去（部分值与键一样带 `**`）
        key, val = _strip_md(cells[0]), _strip_md(cells[1])
        if key in ("字段", "项") or set(key) <= set("-: "):
            continue
        if not key or not val:
            continue
        out[key] = val
    return out


def _strip_md(s: str) -> str:
    """去掉 markdown 强调符号。"""
    return re.sub(r"[*`]", "", s).strip()


def _split_table_row(line: str) -> list[str]:
    """切分 markdown 表格行。

    ★ 不能直接 `line.strip("|").split("|")`（实测踩坑）：
      违禁规则的正则单元格本身就含 `|`，例如
          | 违规宣称 | `降血脂|降胆固醇|治疗` | **critical** | 说明 |
      直接 split 会把正则切成好几段，导致整行解析失败、
      C01/C04 的违禁规则全部丢失。

    ★ 处理方式：优先识别被反引号包裹的单元格，反引号内的 `|` 不当分隔符。

    Args:
        line: 一行表格文本（已 strip）。

    Returns:
        单元格列表。
    """
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]

    cells: list[str] = []
    buf: list[str] = []
    in_code = False

    for ch in s:
        if ch == "`":
            in_code = not in_code
            buf.append(ch)
        elif ch == "|" and not in_code:
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    cells.append("".join(buf).strip())

    return cells


def _parse_elements(body: str, case_id: str) -> tuple[ElementSpec, ...]:
    """解析「必答要素」表格。

    表格列：| ID | 要素 | 权重 | 类型 | 判定说明 |
    """
    elements: list[ElementSpec] = []
    for line in body.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = _split_table_row(line)
        if len(cells) < 3:
            continue

        eid = _strip_md(cells[0])
        if not re.fullmatch(r"E\d+", eid):
            continue

        name = _strip_md(cells[1])
        try:
            weight = int(re.sub(r"[^\d]", "", cells[2]) or 0)
        except ValueError:
            weight = 0
            logger.warning("%s 要素 %s 权重解析失败: %r",
                           case_id, eid, cells[2])

        required = True
        note = ""
        if len(cells) >= 4:
            type_cell = _strip_md(cells[3])
            required = "建议" not in type_cell
            if len(cells) >= 5:
                note = _strip_md(cells[4])

        elements.append(ElementSpec(
            element_id=eid, name=name, weight=weight,
            required=required, judgement_note=note,
        ))
    return tuple(elements)


def _parse_forbidden(body: str, case_id: str) -> tuple[ForbiddenRule, ...]:
    """解析「违禁检测」表格。

    表格有**两种形态**（实测）：
        4 列：| 类型 | 正则 | 级别 | 说明 |
        3 列：| 类型 | 正则 |                      ← 级别写在章节标题里
              如 `## 五、违禁检测（全部 critical）`

    ★ 3 列形态不能当成「解析失败」跳过——那会丢掉 8 个案例的全部规则。
      处理：正则列之后的列若像级别就用它，否则从章节首行找「全部 critical」。
    """
    # 章节首行可能是「（全部 critical）」——先确定默认级别
    default_level = RiskLevel.MEDIUM
    head_line = body.strip().split("\n", 1)[0] if body.strip() else ""
    if "全部" in head_line:
        low = head_line.lower()
        if "critical" in low:
            default_level = RiskLevel.CRITICAL
        elif "high" in low:
            default_level = RiskLevel.HIGH

    rules: list[ForbiddenRule] = []
    for line in body.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = _split_table_row(line)
        if len(cells) < 2:
            continue

        rule_type = _strip_md(cells[0])
        if rule_type in ("类型", "项目") or set(rule_type) <= set("-: "):
            continue

        # 正则可能被 markdown 反引号包住
        pattern = cells[1].strip().strip("`").strip()
        if not pattern or pattern in ("正则", "正则模式", "模式"):
            continue

        # 第 3 列：像级别就是级别，否则当说明
        level = default_level
        note = ""
        if len(cells) >= 3:
            third = _strip_md(cells[2]).lower()
            if "critical" in third:
                level = RiskLevel.CRITICAL
            elif "high" in third:
                level = RiskLevel.HIGH
            elif "medium" in third:
                level = RiskLevel.MEDIUM
            elif third:
                note = _strip_md(cells[2])
        if len(cells) >= 4:
            note = _strip_md(cells[3])

        try:
            rules.append(ForbiddenRule(
                rule_type=rule_type, pattern=pattern, level=level, note=note))
        except ValueError as exc:
            # 非法正则：跳过并告警（案例文档手写，难免有错）
            logger.warning(
                "%s 违禁规则 %r 正则非法，已跳过: %s", case_id, rule_type, exc)

    return tuple(rules)


def _parse_follow_ups(body: str) -> tuple[ScriptedFollowUp, ...]:
    """解析「对话脚本」的追问表格。

    表格列：| # | 追问 | 触发条件 | 考察点 |
    """
    out: list[ScriptedFollowUp] = []
    for line in body.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = _split_table_row(line)
        if len(cells) < 2:
            continue

        fid = _strip_md(cells[0])
        if not re.fullmatch(r"F\d+", fid):
            continue

        out.append(ScriptedFollowUp(
            follow_up_id=fid,
            question=_strip_md(cells[1]),
            trigger_note=_strip_md(cells[2]) if len(cells) >= 3 else "",
            focus=_strip_md(cells[3]) if len(cells) >= 4 else "",
        ))
    return tuple(out)


def _parse_one(case_id: str, title: str, block: str) -> CaseSpec:
    """解析单个案例块。"""
    basic = _parse_kv_table(_section(block, r"(?:案例)?基本设定", []))
    customer = _section(block, r"客户(?:人物)?设定", [])
    script = _section(block, "对话脚本", [])
    elements_body = _section(block, "必答要素", [])
    forbidden_body = _section(block, r"违禁(?:表述)?检测", [])
    reference_body = _section(block, "参考答案", [])

    # 开场白：「系统扮演的客户开场」后面的引用块
    opening = ""
    m = re.search(r"客户开场[^\n]*\n+\s*>\s*(.+)", script)
    if m:
        opening = _strip_md(m.group(1))
    else:
        # 兜底：脚本里第一个引用行
        m2 = re.search(r"^\s*>\s*(.+)$", script, re.M)
        if m2:
            opening = _strip_md(m2.group(1))

    # 难度：★☆☆ → 1
    diff_text = basic.get("难度", "")
    difficulty = diff_text.count("★") or 0

    is_red = "红线" in basic.get("教学定位", "") or "一票否决" in title
    if not is_red:
        is_red = "critical" in forbidden_body and "全部" in forbidden_body

    return CaseSpec(
        case_id=case_id,
        title=_strip_md(title),
        opening_line=opening,
        question_type=basic.get("提问类型", ""),
        difficulty=difficulty,
        stage=_strip_md(basic.get("阶段", basic.get("培训阶段", ""))),
        teaching_point=_strip_md(basic.get("教学定位", "")),
        elements=_parse_elements(elements_body, case_id),
        forbidden_rules=_parse_forbidden(forbidden_body, case_id),
        follow_ups=_parse_follow_ups(script),
        reference_answer=_strip_md(reference_body)[:800],
        is_red_line=is_red,
    )


def parse_case_text(text: str, source_name: str = "") -> list[CaseSpec]:
    """解析案例集**文本**（无 IO）。

    ★ 为什么拆出这一层：
      domain 层不得做任何 IO（门禁 `test_domain_has_no_io` 强制）。
      所以「读文件」放在 infrastructure，本函数只处理字符串。

    Args:
        text: markdown 全文。
        source_name: 来源文件名（仅用于日志）。

    Returns:
        解析出的案例列表。**解析失败不抛异常**，返回已成功解析的部分。
    """
    heads = list(_CASE_HEAD.finditer(text))
    if not heads:
        logger.warning("文本中未找到任何案例标题: %s", source_name or "(未命名)")
        return []

    cases: list[CaseSpec] = []
    for i, m in enumerate(heads):
        case_id = m.group(1)
        title = m.group(2)
        start = m.end()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        block = text[start:end]

        try:
            spec = _parse_one(case_id, title, block)
        except Exception as exc:                  # noqa: BLE001
            logger.error("案例 %s 解析失败: %s: %s",
                         case_id, type(exc).__name__, exc)
            continue

        if not spec.elements:
            logger.warning("案例 %s 未解析出必答要素", case_id)
        if not spec.forbidden_rules:
            logger.warning("案例 %s 未解析出违禁规则", case_id)
        cases.append(spec)

    return cases


def dedupe_cases(cases: list[CaseSpec]) -> list[CaseSpec]:
    """按 case_id 去重，保留要素更全的那份。

    ★ 两个文件有重叠（C01/C04 各出现一次），
      「核心 2 个」那份写得比「14 个」那份更细，应优先。

    Args:
        cases: 可能含重复 case_id 的列表。

    Returns:
        去重后按 case_id 排序的列表。
    """
    by_id: dict[str, CaseSpec] = {}
    for spec in cases:
        existing = by_id.get(spec.case_id)
        if existing is None or len(spec.elements) > len(existing.elements):
            by_id[spec.case_id] = spec
    return sorted(by_id.values(), key=lambda s: s.case_id)


__all__ = [
    "dedupe_cases",
    "parse_case_text",
]
