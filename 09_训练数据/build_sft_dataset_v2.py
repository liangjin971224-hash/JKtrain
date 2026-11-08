"""按「推理侧真实格式」构建 SFT 训练集。

════════════════════════════════════════════════════════════════════
为什么重写：与推理侧对齐（最重要的一条设计决策）
════════════════════════════════════════════════════════════════════
初版数据集是「裸问题 → 答案」，但真实推理（`answer_question.py:_build_prompt`）
每次都会拼一大段上下文：

    【已核实的事实】  ← 法规层结论（可选）
    【结构化事实】    ← 产品库事实（品类/批号/批准功能/不适宜人群/用量）
    【前几轮对话】    ← 仅用于理解指代（可选）
    【参考资料】      ← 检索到的文档片段
    【员工的问题】    ← 用户问题

如果把「裸问题」训进去，模型学到的是**凭记忆答题**，
而项目的核心设计恰恰是「只用资料作答、不凭常识补充」。
两者直接冲突 → 训出来的模型会编造，且拒绝能力退化。

因此本脚本训的是「**带着资料回答**」：

    instruction = 与推理完全同构的上下文 + 问题
    input       = ""（占位，alpaca 要求）
    output      = 引用资料的回答
    system      = 项目 `_system_prompt()` 原文，逐字复用

════════════════════════════════════════════════════════════════════
★ 三条铁律
════════════════════════════════════════════════════════════════════
1. system 必须与 `answer_question.py:_system_prompt()` **逐字一致**。
   训练跟推理不一样，等于白训。这里直接复制，不做任何改写。
2. 上下文里的产品事实、法规结论全部来自项目真实数据文件，不编造。
3. 答案必须「只用给出的资料」组织，不能引入资料外的知识——
   否则模型学会「资料不够也能答」，破坏拒答机制。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(r"D:\项目\某健康公司培训系统")
BANK_DIR = ROOT / "06_正式内容" / "03_题库"
DOC_DIR = ROOT / "06_正式内容" / "01_培训文档"
KNOW_DIR = ROOT / "06_正式内容" / "04_知识条目"
OUT_DIR = ROOT / "09_训练数据"

# ══════════════════════════════════════════════════════════════
# system prompt 直接**导入**推理侧实现，不手工复制。
#
# ★ 为什么不用手抄：初版手抄了 8 条约束，结果是 403 vs 409 字符 ——
#   多行字符串的缩进空格混进了正文，逐字校验不过。
#   手抄必然漂移（那里改一个字、这里忘了改，训出来就是错配）。
#   直接 import 是唯一可靠的做法。
# ══════════════════════════════════════════════════════════════
import sys as _sys

_CODE_ROOT = str(ROOT / "项目代码")
if _CODE_ROOT not in _sys.path:
    _sys.path.insert(0, _CODE_ROOT)

try:
    from application.use_cases.answer_question import (
        AnswerQuestionUseCase as _AQUC,
    )
    SYSTEM_PROMPT = _AQUC._system_prompt()
except Exception as _exc:  # pragma: no cover
    raise SystemExit(f"无法从推理侧导入 system prompt，训练与推理会错配: {_exc}")

#: 推理侧拒答模板，用于训练「资料不足」类样本，保证措辞一致
try:
    from application.use_cases.answer_question import (
        REFUSAL_TEMPLATE as _REFUSAL,
    )
except Exception:  # pragma: no cover
    _REFUSAL = ""

EXAM_PAT = re.compile(r"以下哪(一)?个|下列哪(一)?项|以下说法(正确|错误)|属于[^，。]*的是[？?]")


# ════════════════════════════════════════ 上下文构造（与推理同构）

def build_context(
    *,
    fact_hints: list[str] | None = None,
    facts: list[dict] | None = None,
    docs: list[dict] | None = None,
    question: str,
) -> str:
    """按 `answer_question.py:_build_prompt` 的分区顺序拼上下文。"""
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
            lines.append(f"类别：{f['category']}")
            if f.get("approval_no"):
                lines.append(f"批准文号：{f['approval_no']}")
            if f.get("claims"):
                lines.append(
                    "批准功能：" + "；".join(f"「{c}」" for c in f["claims"])
                )
            if f.get("suitable_for"):
                lines.append(f"适宜人群：{'、'.join(f['suitable_for'])}")
            if f.get("contraindications"):
                lines.append(f"不适宜人群：{'、'.join(f['contraindications'])}")
            if f.get("usage"):
                lines.append(f"用法用量：{f['usage']}")
            if f.get("daily_limit"):
                lines.append(f"每日上限：{f['daily_limit']}")
            lines.append("")
        parts.append("\n".join(lines).strip())

    if docs:
        lines = [
            "【参考资料】（来自培训文档，仅供参考）",
            "⚠️ 以下材料已剔除错误示范。写作时不要使用否认句（如「不能说X」），"
            "只做正面陈述。",
        ]
        for i, d in enumerate(docs, 1):
            lines.append(f"[{i}] 《{d['title']}》{d['section']}")
            lines.append(d["body"][:600])
            lines.append("")
        parts.append("\n".join(lines).strip())

    parts.append(f"【员工的问题】\n{question}")
    return "\n\n".join(parts)


# ════════════════════════════════════════ 数据源

def parse_bank(path: Path) -> tuple[list[dict], list[dict]]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n## ([QS]\d+)", text)
    good, dropped = [], []
    for i in range(1, len(blocks) - 1, 2):
        qid, body = blocks[i].strip(), blocks[i + 1]
        m_q = re.search(r"\*\*(.+?)\*\*", body, re.S)
        if not m_q:
            continue
        question = re.sub(r"\s+", "", m_q.group(1).strip())
        if EXAM_PAT.search(question):
            dropped.append({"qid": qid, "question": question, "why": "纯考题"})
            continue
        m_a = re.search(r"\*\*答案[：:]\s*([A-D]+)\*\*", body)
        m_e = re.search(
            r"\*\*解析\*\*[：:]\s*(.+?)(?=\n\n|\*\*难点\*\*|\Z)", body, re.S
        )
        if not (m_a and m_e):
            dropped.append({"qid": qid, "question": question, "why": "缺答案或解析"})
            continue
        options = dict(re.findall(r"^([A-D])[.、]\s*(.+)$", body, re.M))
        picked = [options.get(c, "").strip() for c in m_a.group(1)]
        picked = [p for p in picked if p]
        if not picked:
            dropped.append({"qid": qid, "question": question, "why": "选项缺失"})
            continue
        good.append({
            "qid": qid,
            "question": question,
            "answer_text": "；".join(picked),
            "explanation": re.sub(r"\s+", "", m_e.group(1).strip()),
        })
    return good, dropped


def _sections(text: str) -> dict[str, str]:
    parts = re.split(r"\n## (.+?)\n", text)
    return {parts[i].strip(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def parse_compliance(path: Path) -> tuple[list[dict], list[str]]:
    text = path.read_text(encoding="utf-8")
    secs = _sections(text)
    pairs, skipped = [], []

    ch3 = next((v for k, v in secs.items() if k.startswith("第三章")), "")
    last_why = ""
    for row in re.findall(
        r"^\|\s*\d+\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|$", ch3, re.M
    ):
        wrong, right, why = (re.sub(r"\*+", "", c).strip() for c in row)
        wrong, right = wrong.strip("「」\"' "), right.strip("「」\"' ")
        if not (wrong and right):
            continue
        if why in ("同上", "同前", "同上。", ""):
            why = last_why or "表述不合规"
        else:
            last_why = why
        if wrong.replace("有助于", "") == right.replace("有助于", ""):
            skipped.append(wrong)          # 24 项目录项，非错→对
            continue
        pairs.append({"wrong": wrong, "right": right, "why": why, "kind": "term"})

    lines = text.splitlines()
    for i, ln in enumerate(lines[:-1]):
        cur, nxt = ln.strip(), lines[i + 1].strip()
        if cur.startswith("> ❌") and nxt.startswith("> ✅"):
            w = cur.replace("> ❌", "").strip().strip("「」\"' ")
            r = nxt.replace("> ✅", "").strip().strip("「」\"' ")
            if w and r:
                pairs.append({"wrong": w, "right": r, "why": "", "kind": "script"})

    ch4 = next((v for k, v in secs.items() if k.startswith("第四章")), "")
    for row in re.findall(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$", ch4, re.M):
        term, reason = (c.strip() for c in row)
        if term in ("禁用词", "禁用表述") or len(term) > 30 or set(term) <= set("-: "):
            continue
        pairs.append({"wrong": term, "right": "", "why": reason, "kind": "banned"})

    best: dict[str, dict] = {}
    for p in pairs:
        k = p["wrong"]
        if k not in best or (p["right"] and not best[k]["right"]):
            best[k] = p
    return list(best.values()), skipped


def normalize_answer(text: str, max_len: int = 200) -> str:
    """把 output 规范成「单段、限长、句号收尾」。

    ★ 为什么必须做（v4 训练后实测的两个退化症状）：
      1. **尾部回吐参考资料**：微调后模型答完正确答案后会继续抄资料原文
         （实测出现 `| 序号 | 最容易说错的表述 |`）。根因是训练集里
         106/705 条 output 不以句末标点结束（15%），模型学不到
         「说完就停」的信号。
      2. **越写越长**：362/705 条 output 含 2 个以上换行（51%），
         模型学成多段落输出，容易收不住。

    做法：合并成一段、截到最后一个完整句子、强制句号收尾、限长 200 字。
    """
    t = strip_markdown(text)
    # 多段合并为一段（段内用句号分隔，不保留空行）
    t = re.sub(r"\s*\n+\s*", "", t)
    t = re.sub(r"[ \t]+", " ", t).strip()

    # 超长时截到最后一个句末标点，避免半句结尾
    if len(t) > max_len:
        cut = max(t.rfind(c, 0, max_len) for c in "。！？")
        t = t[: cut + 1] if cut > max_len // 2 else t[:max_len]

    # 强制句号收尾：模型需要明确的「结束」信号
    if t and t[-1] not in "。！？」":
        t = t.rstrip("，,；;、 ") + "。"
    return t


# ════════════════════════════════════════ 文档 QA 派生

#: 只从这些"知识型"文档派生问答。
#: 排除 D005（已单独处理为合规改写）、D010/D011（设备操作，非话术）。
QA_DOCS = [
    "D001", "D002", "D003", "D004", "D006", "D007", "D008", "D009", "D012", "D013", "D014",
]

#: 章节 → 客户/新员工问题 的映射规则。
#:
#: ★ 覆盖率的教训：初版只有 13 条规则，99 个章节标题映射不上，
#:   文档 QA 只出了 8 条（目标 200）。根因不是正文短，是问句表太窄。
#:   这里按「客户会怎么问」扩到 40+ 条，并补一档兜底模板。
#:   顺序即优先级：越具体的越靠前，泛化规则放后面。
SECTION_Q = [
    (r"区别|概念|是什么|什么是", "保健食品和普通食品到底怎么区分？"),
    (r"身份证|批准文号|蓝帽子|核实|真伪", "怎么判断一个产品是不是正规的保健食品？"),
    (r"允许声称|24 ?项|功能目录|功能范围", "保健食品可以声称哪些功能？"),
    (r"禁止|禁用|红线|不能说|违规|越界", "哪些话是绝对不能说的？"),
    (r"警示|免责", "回答里必须带哪句话？"),
    (r"适宜|不适宜|人群|谁能吃", "这个产品哪些人能吃、哪些人不能吃？"),
    (r"用量|用法|剂量|一天吃", "这个产品怎么吃？一天吃多少？"),
    (r"异议|拒绝|太贵|没用|骗人|质疑", "客户说太贵了／吃了没用，该怎么回应？"),
    (r"接待|迎宾|流程|步骤|SOP|环节", "接待客户的标准流程是什么？"),
    (r"需求|提问|沟通|话术|表达", "怎么问才能知道客户真正想要什么？"),
    (r"记录|档案|跟进", "客户档案要记哪些内容？"),
    (r"退换|售后|退款", "客户要退货怎么处理？"),
    (r"应急|突发|晕倒|不适|胸闷", "客户在店里突然不舒服，该怎么处理？"),
    (r"成分|配料|规格|含量", "这个产品的成分和规格是什么？"),
    (r"配伍|禁忌|同吃|相互作用|一起吃", "这个产品和其他东西能一起吃吗？"),
    (r"职责|岗位|工作标准|做什么", "健康顾问这个岗位具体要做哪些事？"),
    (r"边界|不做什么|越权", "哪些事不在健康顾问的职责范围内？"),
    (r"仪器|设备|体脂|骨密度|测量", "门店的检测仪器怎么用、结果怎么解读？"),
    (r"自测|练习|复习", ""),            # 自测题章节 → 跳过
    (r"学习目标|本章小结|小结|附录", ""),  # 元信息 → 跳过
]

#: 兜底问句。命中不了任何规则时，按文档主题拼一句自然提问，
#: 总比丢掉整节好（但跳过自测/学习目标这类元信息章节）。
FALLBACK_Q = "{topic}这块要注意什么？"

#: 元信息章节——直接跳过，不生成样本
SKIP_PAT = re.compile(r"自测|练习|学习目标|本章小结|文档小结|附录|修订记录")


def section_question(title: str, topic: str = "") -> str:
    """按章节标题选一个自然的提问。

    Returns:
        问题字符串；空字符串表示该章节应跳过（元信息类）。
    """
    if SKIP_PAT.search(title):
        return ""
    for pat, q in SECTION_Q:
        if re.search(pat, title):
            return q
    return FALLBACK_Q.format(topic=topic or title)


def section_topic(title: str) -> str:
    """把章节标题洗成可嵌入问句的主题词。"""
    t = re.sub(r"^第[一二三四五六七八九十]+[章步节][:：\s]*", "", title)
    t = re.sub(r"[（(].*?[)）]", "", t).strip()
    return t or title


def clean_doc_body(body: str, limit: int = 500) -> str:
    """清理正文：去表格竖线装饰、去空行、去掉❌反例行。"""
    kept = []
    for ln in body.splitlines():
        t = ln.strip()
        if not t or set(t) <= set("-|: "):
            continue
        if t.startswith("❌") or "❌" in t:
            continue                    # 反例行不能进参考资料
        if t.startswith("**") and t.endswith("**") and len(t) < 20:
            continue
        kept.append(t)
    return "\n".join(kept)[:limit]


def gen_doc_qa(limit: int = 200) -> list[dict]:
    """从培训文档的章节派生问答。

    ★ 只取有实质内容的章节（>=120 字），且必须能映射到一个自然提问；
      否则丢弃——宁可少，不要造出「这一章讲了什么」这种废样本。
    """
    out: list[dict] = []
    for code in QA_DOCS:
        files = list(DOC_DIR.glob(f"{code}_*.md"))
        if not files:
            continue
        path = files[0]
        text = path.read_text(encoding="utf-8")
        # 按 ## 章节切
        parts = re.split(r"\n## (.+?)\n", text)
        doc_title = path.stem.split("_", 1)[1].replace("_", " ")
        for i in range(1, len(parts) - 1, 2):
            sec_title = parts[i].strip()
            body = clean_doc_body(parts[i + 1])
            if len(body) < 120:
                continue
            q = section_question(sec_title, section_topic(sec_title))
            if not q:
                continue
            ctx = build_context(
                docs=[{"title": doc_title, "section": sec_title, "body": body}],
                question=q,
            )
            out.append({
                "instruction": ctx,
                "input": "",
                "output": "",          # 稍后由 answer_from_body 生成
                "system": SYSTEM_PROMPT,
                "meta": {"source": "qa_from_doc", "ref": f"{code}:{sec_title[:20]}"},
                "_body": body,
            })
    return out[:limit]


def table_to_sentences(body: str) -> list[str]:
    """把 Markdown 表格转成「字段：内容」式的自然语句。

    ★ 为什么不能直接丢表格：
      本项目大量章节（各服务步骤、红线清单）主体就是表格，
      丢掉后这些章的答案不足 60 字被门禁拦掉 —— 实测 33/63 条被拦。
      而表格里恰恰是**最有操作价值的信息**（该说什么、不该说什么）。

    做法：按 `| a | b | c |` 拆格，跳过表头与分隔行，
    输出「a——b；c」形式。不做语义改写，只做格式转换，保证零编造。
    """
    out: list[str] = []
    for raw in body.splitlines():
        t = raw.strip()
        if not t.startswith("|"):
            continue
        if set(t) <= set("|-: "):
            continue
        cells = [c.strip().replace("**", "") for c in t.strip("|").split("|")]
        cells = [c for c in cells if c and not set(c) <= set("-: ")]
        if len(cells) < 2:
            continue
        # 表头行：首格是「项目/字段/内容/类型」这类词
        if cells[0] in ("项目", "字段", "内容", "类型", "序号", "#", "场景", "要素"):
            continue
        # 纯序号列（1 / 2 / 一 / ①）没有语义，去掉后用下一格做字段名
        while cells and (cells[0].isdigit() or set(cells[0]) <= set("①②③④⑤⑥⑦⑧⑨⑩一二三四五六七八九十.")):
            cells = cells[1:]
        if len(cells) < 2:
            continue
        head, rest = cells[0], cells[1:]
        # ★ 表头误判为数据：两格都很短、且没有任何标点 → 大概率是「首列名 | 次列名」
        #   例：| 首次到店 | 老客户复访 | 是分组表头，不是内容。
        if (
            len(cells) == 2
            and len(cells[0]) <= 6 and len(cells[1]) <= 6
            and not re.search(r"[，。；：、]", cells[0] + cells[1])
            and any(k in cells[0] for k in ("首次", "类型", "场景", "情况", "对象"))
        ):
            continue
        if len(cells) == 2:
            out.append(f"{head}：{rest[0]}")
        else:
            out.append(f"{head}——{'；'.join(rest)}")
    return out


def answer_from_body(body: str) -> str:
    """把章节正文压成一段可读的回答。

    ★ 不能简单「每行加句号」——实测会把流程图和代码块毁掉：
          ```。   ① 迎宾…。   ↓。
      训练集里出现这种输出，模型会学会「往任何符号后加句号」。
      规则：
        1. 跳过 ``` 代码块整段（流程图留在参考资料里即可，不必进答案）
        2. 跳过纯符号行（↓ → | --- 等）
        3. 去掉 Markdown 强调标记 ** 与列表前缀 - ，但保留句子
        4. 已经是句末标点的行不再追加
    """
    lines: list[str] = []
    in_code = False
    for raw in body.splitlines():
        t = raw.strip()
        if t.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not t:
            continue
        # 纯符号行：箭头、竖线、分隔线
        if set(t) <= set("↓→|—-_= ."):
            continue
        # 表格行（| … |）不是句子，切碎后毫无意义 → 整行丢弃
        if t.startswith("|"):
            continue
        # 残留的 Markdown 标题（### 4.1 …）不是回答内容 → 丢弃
        if re.match(r"^#{1,6}\s", t):
            continue
        t = t.replace("**", "").lstrip("-*+ ").strip()
        if not t or set(t) <= set("↓→|—-_= ."):
            continue
        # 去掉行内的 <br> 等 HTML
        t = re.sub(r"<br\s*/?>", " ", t).strip()
        if t:
            lines.append(t)

    # 正文段落优先；不足时补表格转述（表格往往是这些章的主体内容）
    lines.extend(table_to_sentences(body))

    picked: list[str] = []
    total = 0
    for ln in lines:
        if total + len(ln) > 340:
            break
        if ln[-1] not in "。！？；：":
            ln += "。"
        picked.append(ln)
        total += len(ln)
    return "\n".join(picked) if picked else "资料里没有这个信息。"


# ════════════════════════════════════════ 情景案例对话

def parse_cases() -> list[dict]:
    """解析情景案例：客户开场 + 追问 + 参考答案。"""
    path = CASE_FILE if (CASE_FILE := DOC_DIR.parent / "02_情景案例" / "情景演练案例集_14个.md").exists() else None
    if path is None:
        return []
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n# (C\d+) (.+?)\n", text)
    cases = []
    for i in range(1, len(blocks) - 1, 3):
        cid, title, body = blocks[i], blocks[i + 1].strip(), blocks[i + 2]
        m_open = re.search(r"\*\*客户开场\*\*[：:]\s*\n+>\s*(.+?)\n", body)
        m_ref = re.search(r"## 六、参考答案\s*\n+(.+?)(?=\n## |\Z)", body, re.S)
        if not (m_open and m_ref):
            continue
        opening = m_open.group(1).strip()
        answer = re.sub(r"\*\*", "", m_ref.group(1)).strip()
        answer = "\n".join(l.strip("> ").rstrip() for l in answer.splitlines() if l.strip())
        # 追问列表
        follows = re.findall(r"^\| F\d+ \| (.+?) \|", body, re.M)
        if opening and answer:
            cases.append({
                "cid": cid, "title": title,
                "opening": opening, "answer": answer[:600],
                "follows": [f.strip() for f in follows],
            })
    return cases


def gen_case_dialogues(limit: int = 150) -> list[dict]:
    """把每个案例展开成多条「单轮问答」（alpaca 不擅多轮，故拆成单轮）。"""
    out: list[dict] = []
    for c in parse_cases():
        doc = [{
            "title": "门店合规话术要点",
            "section": c["title"],
            "body": (
                "只讲批准的功能，不承诺效果；不涉及疾病治疗与用药建议；"
                "涉及人群适配时不替客户判断，引导其自行核对或咨询医生；"
                "保健食品不是药物，不能代替药物治疗疾病。"
            ),
        }]
        # 主问
        out.append({
            "instruction": build_context(docs=doc, question=c["opening"]),
            "input": "", "output": clean_punct(c["answer"]),
            "system": SYSTEM_PROMPT,
            "meta": {"source": "dialogue", "ref": c["cid"]},
        })
        # 追问
        for f in c["follows"][:2]:
            out.append({
                "instruction": build_context(docs=doc, question=f),
                "input": "",
                "output": (
                    "这个问题我按资料里的口径回答：只讲产品批准的功能与客观信息，"
                    "不做效果承诺、不替您判断是否适合。"
                    "如果涉及您的身体状况或正在用药，建议先咨询医生。"
                    "保健食品不是药物，不能代替药物治疗疾病。"
                ),
                "system": SYSTEM_PROMPT,
                "meta": {"source": "dialogue", "ref": f"{c['cid']}-追问"},
            })
    return out[:limit]


# ════════════════════════════════════════ 负样本（拒答）

#: 用推理侧的拒答模板，保证训练与线上措辞一致
try:
    from application.use_cases.answer_question import REFUSAL_TEMPLATE as _REFUSAL_TEXT
except Exception:
    _REFUSAL_TEXT = "抱歉，知识库里暂时没有找到与这个问题直接相关的资料，我不能凭空给您答复。"


#: 知识库里没有的内容——按「客户会怎么问」的真实口吻构造。
#: ★ 必须造这一类样本的原因（这是 v1 训练暴露的最大缺陷）：
#:   v1 的 459 条全部来自有答案的题库与文档，**没有任何一条「查不到」**。
#:   结果模型学到「任何问题都要给出答案」，推理时面对资料外的产品
#:   直接编出了「一款专门针对孕妇设计的钙片」。拒答能力不是调参能补的，
#:   只能靠负样本显式教。
_UNANSWERABLE = [
    # ── A. 产品库没有的产品（客户最常问，也最容易编）──────────
    ("你们有适合孕妇吃的钙片吗？", "产品库中没有孕妇钙片"),
    ("有没有适合糖尿病人吃的无糖饼干？", "产品库中没有无糖饼干"),
    ("你们卖不卖蛋白棒？", "产品库中没有蛋白棒"),
    ("有给小孩吃的 DHA 软糖吗？", "产品库中没有儿童 DHA 软糖"),
    ("你们这个牌子的胶原蛋白口服液多少钱？", "产品库中没有胶原蛋白口服液"),
    ("你们有减肥茶吗？", "产品库中没有减肥茶"),
    ("有没有贴的膏药？", "产品库中没有外用贴剂"),
    ("你们有进口的鱼油吗？", "产品库未收录进口鱼油"),
    ("有羊奶粉吗？", "产品库中没有羊奶粉"),
    ("你们卖不卖蜂蜜？", "产品库中没有蜂蜜"),
    ("有没有益生菌粉？", "产品库中没有益生菌粉"),
    ("你们有软胶囊的维生素 E 吗？", "产品库中没有维生素 E"),
    ("有儿童钙铁锌口服液吗？", "产品库中没有儿童口服液剂型"),
    ("你们这个牌子有没有男士专用款？", "产品库无男士专用产品"),
    ("有没有不含糖的蛋白粉？", "产品库未标注无糖款"),
    ("有大桶装的吗？", "产品库无大规格装"),
    ("你们有试吃装吗？", "产品库无试吃装规格"),
    ("有没有礼盒装？", "产品库无礼盒装"),
    ("你们卖不卖中药材？", "门店不经营中药饮片"),
    ("有花胶吗？", "产品库中没有花胶"),
    ("有没有护肝片？", "产品库中没有护肝片"),
    ("你们有褪黑素吗？", "目录与产品库均无褪黑素"),
    ("有没有胶原蛋白肽粉？", "产品库中没有胶原蛋白肽"),
    ("有灵芝孢子粉吗？", "产品库中没有灵芝孢子粉"),
    ("你们卖不卖膏方？", "产品库中没有膏方"),
    # ── B. 目录外的功能声称（法规层面无此功能）────────────
    ("这个产品能治失眠吗？", "目录无治疗类功能"),
    ("吃这个能长高吗？", "目录无增高类功能"),
    ("有没有防脱发的？", "目录无防脱发功能"),
    ("能排毒养颜吗？", "目录无排毒功能"),
    ("这个吃了能抗癌吗？", "疾病治疗宣称，明确禁止"),
    ("能提高记忆力吗？", "目录表述是辅助改善记忆，不能用「提高」"),
    ("吃这个能治胃病吗？", "疾病治疗宣称，明确禁止"),
    ("能降尿酸吗？", "目录无降尿酸功能"),
    ("能不能治便秘？", "目录无治疗便秘功能"),
    ("吃这个能长肌肉吗？", "目录无增肌功能"),
    ("能改善视力吗？", "目录表述是缓解视觉疲劳，不能用「改善视力」"),
    ("可以治贫血吗？", "疾病治疗宣称，明确禁止"),
    ("能调理月经吗？", "目录无此功能"),
    ("吃这个能提高性功能吗？", "目录无此功能"),
    ("能不能治湿疹？", "疾病治疗宣称，明确禁止"),
    ("吃了能戒烟吗？", "目录无此功能"),
    ("能解酒护肝吗？", "目录无解酒护肝功能"),
    ("吃这个能防止脱发吗？", "目录无防脱发功能"),
    ("能补脑吗？", "目录无补脑功能"),
    ("吃这个能让皮肤变白吗？", "目录无美白功能"),
    ("可以治颈椎病吗？", "疾病治疗宣称，明确禁止"),
    ("能缓解痛经吗？", "目录无此功能"),
    ("吃这个能治感冒吗？", "疾病治疗宣称，明确禁止"),
    ("能提高精子质量吗？", "目录无此功能"),
    ("吃了能治便秘和口臭吗？", "目录无治疗便秘功能"),
    ("能治脂肪肝吗？", "疾病治疗宣称，明确禁止"),
    ("这个能缓解更年期症状吗？", "目录无此功能"),
    ("吃这个能治痛风吗？", "疾病治疗宣称，明确禁止"),
    ("能不能治鼻炎？", "疾病治疗宣称，明确禁止"),
    ("吃了能长高五厘米吗？", "目录无增高类功能"),
    # ── C. 诊断 / 用药 / 医疗判断（一律不做）──────────────
    ("帮我看看我这份体检报告？", "不提供诊断服务"),
    ("我这个症状该吃什么药？", "不提供用药建议"),
    ("我血压 160/100，吃这个能降下来吗？", "不提供诊断与用药建议"),
    ("我血糖 8.5，该吃多少？", "不提供用量与诊断建议"),
    ("我这个结节要紧吗？", "不提供诊断服务"),
    ("我化验单上这项偏高是什么意思？", "不提供诊断服务"),
    ("我该吃哪个药见效快？", "不提供用药建议"),
    ("我能不能把二甲双胍停了换成这个？", "不提供停药建议"),
    ("我吃这个会不会和化疗药冲突？", "不提供用药相互作用判断"),
    ("我这个情况要不要做手术？", "不提供医疗决策建议"),
    ("我怀孕了能吃吗？", "不判断个体是否适宜，引导咨询医生"),
    ("我哺乳期能不能吃？", "不判断个体是否适宜，引导咨询医生"),
    ("我刚做完手术能吃吗？", "不判断个体是否适宜，引导咨询医生"),
    ("我有肾病能吃蛋白粉吗？", "不判断个体是否适宜，引导咨询医生"),
    ("我在吃华法林，能吃鱼油吗？", "不提供用药相互作用判断"),
    ("我孩子三岁能吃吗？", "不判断个体是否适宜，引导咨询医生"),
    ("我有甲亢能吃吗？", "不判断个体是否适宜，引导咨询医生"),
    ("我做过心脏支架能吃吗？", "不判断个体是否适宜，引导咨询医生"),
    ("我这个病吃多久能好？", "不承诺效果与时间"),
    ("吃几个疗程能见效？", "不承诺效果与时间"),
    # ── D. 门店不提供的服务 ────────────────────────────
    ("能帮我预约医院专家号吗？", "门店不提供挂号服务"),
    ("帮我联系一下主治医生？", "门店不提供医疗对接"),
    ("能上门给我做体检吗？", "门店不提供上门医疗服务"),
    ("你们能代购国外产品吗？", "门店不提供代购"),
    ("帮我代收快递行吗？", "门店不提供代收服务"),
    ("能帮我测血糖吗？", "门店不提供血糖检测"),
    ("你们能输液吗？", "门店不提供医疗操作"),
    ("能帮我打针吗？", "门店不提供医疗操作"),
    ("可以帮我开处方吗？", "门店无处方权"),
    ("能帮我办理医保报销吗？", "门店不办理医保业务"),
    # ── E. 内部经营信息 ──────────────────────────────
    ("你们公司去年营收多少？", "内部经营数据，不在知识库"),
    ("这款的出厂价是多少？", "进货价属内部信息，不在知识库"),
    ("你们进货价多少？", "进货价属内部信息，不在知识库"),
    ("你们利润率多高？", "内部经营数据，不在知识库"),
    ("门店一个月卖多少？", "内部经营数据，不在知识库"),
    ("你们老板是谁？", "知识库无人员信息"),
    ("这家店什么时候开的？", "知识库无门店沿革"),
    ("你们下个月有什么促销？", "知识库无促销计划"),
    ("你们今年开店计划是什么？", "内部经营数据，不在知识库"),
    ("能给我看看你们的进货单吗？", "进货单属内部资料"),
    # ── F. 知识库未收录的流程与政策 ──────────────────────
    ("员工请假流程是什么？", "知识库无人事流程"),
    ("门店装修预算怎么报？", "知识库无装修预算规范"),
    ("会员卡积分怎么兑换？", "知识库无积分兑换规则"),
    ("你们支持花呗分期吗？", "知识库无支付方式说明"),
    ("我要开发票怎么开？", "知识库无开票流程"),
    ("你们的加盟政策是什么？", "知识库无加盟政策"),
    ("退货要扣手续费吗？", "知识库无手续费规则"),
    ("能开专票吗？", "知识库无开票流程"),
    ("你们有会员等级制度吗？", "知识库无会员等级规则"),
    ("员工提成怎么算？", "知识库无薪酬制度"),
    ("门店考勤怎么规定？", "知识库无人事流程"),
    ("你们招人吗？", "知识库无招聘信息"),
    ("我要投诉找谁？", "知识库无投诉渠道说明"),
    ("能加你微信私下咨询吗？", "知识库无个人联系方式规范"),
    ("你们有没有线上小程序？", "知识库无线上渠道信息"),
    ("能微信支付吗？", "知识库无支付方式说明"),
    ("你们有送货上门吗？", "知识库无配送规则"),
    ("可以预定吗？", "知识库无预定规则"),
    ("你们营业时间到几点？", "知识库无营业时间信息"),
    ("你们周末上班吗？", "知识库无营业时间信息"),
    ("门店地址具体在哪？", "知识库无门店地址信息"),
    ("你们有分店吗？", "知识库无门店布局信息"),
    ("停车方便吗？", "知识库无停车信息"),
    # ── G. 与业务无关的闲聊 / 越界 ──────────────────────
    ("你今年多大了？", "闲聊，与业务无关"),
    ("你是真人还是机器人？", "闲聊，与业务无关"),
    ("今天天气怎么样？", "闲聊，与业务无关"),
    ("给我讲个笑话吧。", "闲聊，与业务无关"),
    ("帮我写一份工作总结。", "非知识库职责范围"),
    ("帮我改一下这份简历。", "非知识库职责范围"),
    ("你能帮我翻译一段英文吗？", "非知识库职责范围"),
    ("帮我算一下这个数学题。", "非知识库职责范围"),
    ("推荐几部电影看看。", "闲聊，与业务无关"),
    ("你觉得哪个牌子的车好？", "闲聊，与业务无关"),
    # ── H. 竞品与比较（无资料可比）──────────────────────
    ("你们和汤臣倍健比哪个好？", "知识库无竞品对比资料"),
    ("和安利的产品比怎么样？", "知识库无竞品对比资料"),
    ("网上那个牌子更便宜，你们为什么贵？", "知识库无竞品价格资料"),
    ("你们这个和药店卖的有区别吗？", "知识库无渠道对比资料"),
    ("为什么别的店卖得比你们便宜？", "知识库无竞品价格资料"),
    ("你们产品通过了 FDA 认证吗？", "知识库无该认证信息"),
    ("有蓝帽子认证吗？", "需按具体产品核实，知识库无此项"),
    ("你们是上市公司吗？", "知识库无公司信息"),
    ("有 ISO 认证吗？", "知识库无认证信息"),
    ("有没有检测报告可以看？", "知识库无检测报告"),
    # ── I. 资料不足的模糊问法 ───────────────────────────
    ("我这种情况该吃什么？", "缺信息，无法定位产品"),
    ("有什么推荐的吗？", "缺信息，无法定位产品"),
    ("哪个最好？", "缺信息，无法定位产品"),
    ("你们有什么好东西？", "缺信息，无法定位产品"),
    ("我上次买的那个还有吗？", "缺信息，无法定位产品"),
    ("老客户有优惠吗？", "知识库无优惠规则"),
    ("我妈让我来买点东西，买什么好？", "缺信息，无法定位产品"),
    ("随便给我拿一个。", "缺信息，无法定位产品"),
    ("你们这儿什么卖得最好？", "知识库无销量数据"),
    ("这个和那个有啥不一样？", "缺信息，无法定位产品"),
    ("都行，你看着推荐。", "缺信息，无法定位产品"),
    ("我这情况吃什么能好？", "缺信息，无法定位产品"),
    ("有什么适合我的？", "缺信息，无法定位产品"),
    ("你帮我选一个吧。", "缺信息，无法定位产品"),
    ("还有别的牌子吗？", "缺信息，无法定位品牌"),
    # ── J. 超出培训范围的专业问题 ────────────────────────
    ("保健食品的注册流程要走几步？", "知识库无注册流程细节"),
    ("GMP 认证具体审什么？", "知识库无 GMP 细则"),
    ("功能性成分的检测方法是什么？", "知识库无检测方法"),
    ("这个成分的分子式是什么？", "知识库无分子结构信息"),
    ("原料是从哪个基地采购的？", "知识库无供应链信息"),
    ("产品的稳定性试验怎么做的？", "知识库无研发细节"),
    ("保健食品和药品的审评差异在哪？", "知识库无审评细节"),
    ("营养素补充剂的备案要求是什么？", "知识库无备案细则"),
    ("跨境电商的保健食品怎么监管？", "知识库无跨境政策"),
    ("保健食品广告审批要多久？", "知识库无广告审批时限"),
    # ── K. 不存在或虚构的产品名 ────────────────────────
    ("康源牌灵芝胶囊有吗？", "产品库无此产品"),
    ("康源牌蜂胶软胶囊多少钱？", "产品库无此产品"),
    ("康源牌氨糖软骨素在哪？", "产品库无此产品"),
    ("康源牌铁皮石斛有货吗？", "产品库无此产品"),
    ("康源牌辅酶Q10有吗？", "产品库无此产品"),
    ("康源牌叶黄素酯片有吗？", "产品库无此产品"),
    ("康源牌胶原蛋白粉怎么卖？", "产品库无此产品"),
    ("康源牌益生元冲剂有吗？", "产品库无此产品"),
    ("康源牌钙镁锌片有吗？", "产品库无此产品"),
    ("康源牌大豆异黄酮有吗？", "产品库无此产品"),
    ("康源牌番茄红素软胶囊有吗？", "产品库无此产品"),
    ("康源牌硒片多少钱？", "产品库无此产品"),
    ("康源牌维生素B族有吗？", "产品库无此产品"),
    ("康源牌葡萄籽提取物有吗？", "产品库无此产品"),
    ("康源牌螺旋藻片有吗？", "产品库无此产品"),
    # ── L. 安全边缘问题（必须拒答且提示风险）─────────────
    ("这个可以天天吃吗？吃多了会怎样？", "超量风险需医生评估，知识库无答案"),
    ("一次吃十片会怎么样？", "超量风险，不提供建议"),
    ("我多吃几片能不能好快点？", "超量风险，不提供建议"),
    ("这个和酒一起喝行吗？", "知识库无饮酒相互作用资料"),
    ("吃完这个能开车吗？", "需按具体产品核实，知识库无此项"),
    ("能不能空腹吃？", "需按具体产品核实，知识库无此项"),
    ("可以给宠物吃吗？", "知识库无宠物适用信息"),
    ("过期的还能吃吗？", "知识库无过期处理规范"),
    ("这个能当饭吃吗？", "知识库无代餐方案"),
    ("能长期当保健品一直吃吗？", "需按具体产品核实，知识库无此项"),
    ("这个对肝肾功能有影响吗？", "不提供安全性判断"),
    ("吃了会过敏吗？", "不判断个体过敏风险"),
    ("这个含激素吗？", "知识库无成分安全性结论"),
    ("吃了会发胖吗？", "知识库无热量数据"),
    # ── M. 补充：客户口吻的连续追问与情绪化表达 ─────────────
    ("你们这儿能刷医保卡吗？", "知识库无医保结算信息"),
    ("能不能便宜点？给个底价。", "知识库无议价规则"),
    ("我要买十盒，能打几折？", "知识库无批量折扣规则"),
    ("有赠品吗？", "知识库无赠品政策"),
    ("能不能先拿回去试用？", "知识库无试用政策"),
    ("这个吃完能退吗？", "知识库无退换细则"),
    ("我昨天买贵了，能补差价吗？", "知识库无补差价规则"),
    ("能寄到我家吗？", "知识库无配送规则"),
    ("能开成办公用品吗？", "知识库无开票品目规则"),
    ("你们有没有内部员工价？", "知识库无员工价政策"),
    ("你们这个是不是传销？", "知识库无经营模式说明"),
    ("我朋友说这个是骗人的。", "知识库无品牌质疑应答资料"),
    ("网上说保健食品都是智商税，你怎么说？", "知识库无行业争议应答资料"),
    ("你们是不是快倒闭了？", "知识库无经营状况信息"),
    ("我要找你们领导。", "知识库无投诉升级流程"),
    ("你们领导电话多少？", "知识库无人员联系方式"),
    ("给我一张名片。", "知识库无个人联系方式规范"),
    ("你们公司全称是什么？", "知识库无主体登记信息"),
    ("营业执照能看看吗？", "知识库无证照信息"),
    ("你们有实体店吗？", "知识库无门店布局信息"),
    ("能上网查到这个产品吗？", "需按具体产品核实，知识库无此项"),
    ("这个批号在药监局能查到吗？", "需按具体产品核实，知识库无此项"),
    ("你们产品出口吗？", "知识库无出口信息"),
    ("这个 FDA 注册号是多少？", "知识库无海外注册信息"),
    ("有没有临床试验数据？", "知识库无临床数据"),
    ("做过人体试验吗？", "知识库无临床数据"),
    ("有多少人吃过这个？", "知识库无销量数据"),
    ("有没有用户评价可以看？", "知识库无用户评价"),
    ("你们哪个产品卖得最好？", "知识库无销量数据"),
    ("有什么新品吗？", "知识库无新品信息"),
    ("这个下架了吗？", "知识库无上下架状态"),
    ("你们什么时候补货？", "知识库无库存计划"),
    ("这个现在有货吗？", "知识库无实时库存"),
    ("能预留两天吗？", "知识库无预留规则"),
    ("我家老人吃这个行不行？", "缺信息，无法定位产品与人群"),
    ("我老公应酬多，吃什么好？", "缺信息，无法定位产品"),
    ("孩子挑食，补点什么？", "缺信息，无法定位产品"),
    ("我最近总失眠，有办法吗？", "不提供诊断，需就医"),
    ("我掉头发厉害，吃什么？", "不提供诊断，需就医"),
    ("我老是乏力，是不是缺什么？", "不提供诊断，需就医"),
    ("我手脚冰凉，吃什么调理？", "不提供诊断，需就医"),
    ("我肠胃不好，能吃这个吗？", "不判断个体是否适宜，引导咨询医生"),
    ("我血压有点高，吃什么好？", "不提供诊断与用药建议"),
    ("我血脂高，你们有什么？", "需具体产品与医生评估，知识库无答案"),
    ("我妈骨质疏松，吃哪个？", "缺信息，无法定位产品"),
    ("我经常熬夜，吃什么补？", "缺信息，无法定位产品"),
    ("我想减肥，吃什么？", "目录无减肥功能，不能推荐"),
    ("我想长点肌肉，吃什么？", "目录无增肌功能，不能推荐"),
    ("我要备孕，吃什么？", "不提供备孕营养方案"),
    ("我术后恢复吃什么？", "不提供术后营养方案"),
    ("我化疗期间能吃什么？", "不提供治疗期营养方案"),
    ("我想提高免疫力吃什么？", "缺信息，无法定位产品"),
    ("你们有没有增强抵抗力的？", "缺信息，无法定位产品"),
    ("我想补钙，哪个好？", "缺信息，无法定位产品"),
]


def gen_refusal_samples() -> list[dict]:
    """生成「资料不足 → 拒答」负样本。

    关键设计：
    · 上下文里**放一段与问题无关的资料**，模拟真实检索（召回总有结果，
      但都不相关）——只给空上下文反而学不到「有资料但不相关也要拒答」。
    · 输出用推理侧的 REFUSAL_TEMPLATE，一字不改，保证线上行为一致。
    """
    unrelated = [{
        "title": "门店接待服务流程SOP",
        "section": "第一步 迎宾与初步沟通",
        "body": "主动问候「您好，欢迎光临」并微笑；判断是否首次到店；观察客户状态；引导入座；递水。",
    }]
    rows = []
    for q, why in _UNANSWERABLE:
        rows.append({
            "instruction": build_context(docs=unrelated, question=q),
            "input": "",
            "output": _REFUSAL_TEXT,
            "system": SYSTEM_PROMPT,
            "meta": {"source": "refusal", "ref": why},
        })
    return rows


# ════════════════════════════════════════ 样本组装

def fmt_products_ref(pairs: list[dict]) -> list[dict]:
    """把 D005 的对照转成「参考资料」形态——训练时模型看到的原始材料。"""
    rows = ["对照表（第三章 7 组最容易说错的表述）"]
    for p in pairs:
        if p["kind"] == "term":
            rows.append(f"| {p['wrong']} | {p['right']} | {p['why']} |")
    return [{
        "title": "保健食品功能声称规范与禁用表述",
        "section": "第三章 7 组最容易说错的表述",
        "body": "\n".join(rows),
    }]


#: 训练数据里绝不能出现的排版符号。
#: ★ 实测教训：v1 训练后模型会输出 `### 标题`、`| 字段 | 内容 |`、```` ``` ````、
#:   `---`、`⚠️ 以下为错误示范`——它学到的是「文档长什么样」而不是「该怎么回答」。
#:   这些符号必须在训练前剥离，否则模型会把排版当内容复读。
_MD_NOISE = [
    (re.compile(r"^#{1,6}\s*", re.M), ""),           # ### 标题
    (re.compile(r"^\s*\|.*\|\s*$", re.M), ""),      # | 表格 | 行
    (re.compile(r"^\s*[-:]{2,}\s*$", re.M), ""),      # | :--: | 分隔
    (re.compile(r"^```.*$", re.M), ""),               # 代码块围栏
    (re.compile(r"^---+\s*$", re.M), ""),             # 水平线
    (re.compile(r"⚠️[^\n]*"), ""),               # ⚠️ 提示行
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),         # 列表符号
    (re.compile(r"\*\*|__"), ""),                    # 加粗标记
    (re.compile(r"<br\s*/?>"), " "),                 # HTML 换行
]


def strip_markdown(text: str) -> str:
    """剥离排版符号，只留可读句子。

    这是训练数据进库前的最后一道清洗——模型不该学任何排版。
    """
    out = text
    for pat, repl in _MD_NOISE:
        out = pat.sub(repl, out)
    # 收拾空行与多余空格
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{2,}", "\n", out)
    return out.strip()


def clean_punct(s: str) -> str:
    """清掉拼接造成的重复标点，并剥离 Markdown 符号。"""
    s = strip_markdown(s)
    s = s.replace("」。", "」。").replace("。。", "。")
    s = re.sub(r"」{2,}", "」", s)
    s = re.sub(r"。{2,}", "。", s)
    s = re.sub(r"^[：:；;，,。]+", "", s)
    return s.strip()


def build(sample_only: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_q, all_dropped = [], []
    for f in sorted(BANK_DIR.glob("题库_*.md")):
        good, dropped = parse_bank(f)
        print(f"  {f.name}: 可用 {len(good)} / 丢弃 {len(dropped)}")
        all_q.extend(good)
        all_dropped.extend(dropped)

    comp, skipped = parse_compliance(DOC_DIR / "D005_保健食品功能声称规范与禁用表述.md")
    print(f"  D005 对照: {len(comp)} 条 | 排除目录项 {len(skipped)} 条")

    if sample_only:
        all_q, comp = all_q[:8], comp[:8]

    rows: list[dict] = []

    # ── A. 题库问答：带【结构化事实】或【参考资料】，答案只讲结论+依据
    for it in all_q:
        # 只给真实存在的资料：类别判定依据段落。
        # ★ 不要造「（见题干）」这种占位事实——真推理时不会有占位符，
        #   训进去模型会学到「可以给模糊值」，而项目要求只用资料作答。
        ctx = build_context(
            docs=[{
                "title": "保健食品与普通食品辨识",
                "section": "产品类别判定依据",
                "body": (
                    "国食健注 / 食健备 开头为保健食品批准文号；"
                    "SC 开头为食品生产许可证号，对应普通食品。\n"
                    "没有蓝帽子、没有保健食品批准文号的一律不是保健食品。\n"
                    "普通食品不得作任何保健功能声称。"
                ),
            }],
            question=it["question"],
        )
        ans = it["answer_text"] + "。" + it["explanation"]
        if it["explanation"].startswith(it["answer_text"][:6]):
            ans = it["explanation"]
        rows.append({
            "instruction": ctx,
            "input": "",
            "output": clean_punct(ans)[:260],
            "system": SYSTEM_PROMPT,
            "meta": {"source": "qa_from_bank", "ref": it["qid"]},
        })

    # ── B. 合规改写：参考资料给对照表，答案做纠正
    ref = fmt_products_ref(comp)
    for c in comp:
        if c["kind"] == "term":
            q = f"新员工介绍产品时说「{c['wrong']}」，这样对吗？应该怎么说？"
            a = (
                f"不对。按参考资料，正确表述是「{c['right']}」，"
                f"错在{c['why']}。\n"
                f"保健食品的功能声称必须与批准目录一字不差，"
                f"不能换成医疗术语，也不能省略「有助于」这类限定词。\n"
                f"保健食品不是药物，不能代替药物治疗疾病。"
            )
        elif c["kind"] == "script":
            q = f"新员工这样回答客户：「{c['wrong']}」。这样说有问题吗？"
            a = (
                f"有问题，这么说会被判定违规。应改为：\n"
                f"「{c['right']}」\n"
                f"核心是只讲批准的功能，不承诺效果，不涉及疾病治疗。\n"
                f"保健食品不是药物，不能代替药物治疗疾病。"
            )
        else:
            q = f"新员工说了「{c['wrong']}」这句话，可以这样说吗？"
            a = (
                f"不可以。这是明确禁止的表述（{c['why']}）。\n"
                f"保健食品不能涉及疾病治疗或预防，也不能替代药物。\n"
                f"应改用批准目录内的功能声称原文。"
            )
        rows.append({
            "instruction": build_context(docs=ref, question=q),
            "input": "",
            "output": clean_punct(a),
            "system": SYSTEM_PROMPT,
            "meta": {"source": "compliance", "ref": "D005", "kind": c["kind"]},
        })

    # ── C. 题库换问法：同一知识点 × 3 种客户口吻 ────────────────
    #  ★ 换的是**问法**，不是答案——答案仍来自同一题解析，零编造。
    #    目的：让模型适应客户五花八门的说法，而不是只会答标准题干。
    REPHRASE = [
        ("客户直接问", "{}"),
        ("客户口语化地问", "我问问啊，{}"),
        ("新员工请教", "新人问我：{} 我该怎么答？"),
    ]
    bank = [r for r in rows if r["meta"]["source"] == "qa_from_bank"]
    if sample_only:
        bank = bank[:3]
    for r in bank:
        q = r["instruction"].split("【员工的问题】\n")[-1].strip()
        ans = r["output"]
        ctx_prefix = r["instruction"].split("【员工的问题】")[0]
        for style, tpl in REPHRASE[1:]:          # 第 1 式已在原样本里
            new_q = tpl.format(q)
            rows.append({
                "instruction": ctx_prefix + "【员工的问题】\n" + new_q,
                "input": "",
                "output": ans,
                "system": r["system"],
                "meta": {"source": "bank_rephrase", "ref": r["meta"]["ref"], "style": style},
            })

    # ── D. 文档 QA 派生 ─────────────────────────────────────
    doc_rows = gen_doc_qa(limit=400 if not sample_only else 4)
    kept: list[dict] = []
    for r in doc_rows:
        body = r.pop("_body")
        ans = answer_from_body(body)
        # ★ 质量门禁：答案不足 60 字说明这节基本是表格/流程图，问不出东西
        if len(ans) < 60 or ans == "资料里没有这个信息。":
            continue
        r["output"] = ans
        kept.append(r)
    rows.extend(kept)
    print(f"  文档 QA: {len(kept)} 条（原始候选 {len(doc_rows)}）")

    # ── E. 情景案例对话 ─────────────────────────────────────
    case_rows = gen_case_dialogues(limit=150 if not sample_only else 4)
    rows.extend(case_rows)
    print(f"  情景案例: {len(case_rows)} 条")

    # ── F. 负样本（拒答）────────────────────────────────────
    ref_rows = gen_refusal_samples()
    if sample_only:
        ref_rows = ref_rows[:4]
    rows.extend(ref_rows)
    print(f"  负样本（拒答）: {len(ref_rows)} 条")

    # ── 最终统一清洗 ──────────────────────────────────────
    # ★ instruction 也要洗：实测 output 里 ** 已清零，但 instruction 的
    #   【参考资料】区还有 62 条带 ** 的文档原文，模型会从这里学会加粗。
    #   参考资料是给模型「读」的，不该带排版符号。
    # ★ 恢复 v4 做法：output 只剥 Markdown，不压成单段/不限长。
    #   v5 试过 normalize_answer（单段 + 200字 + 强制句号），
    #   结果格式干净了但**答案的正确表述丢了**（「有助于维持血压健康水平」
    #   变成「有助于改善血管弹性，降低血压」——后者是违规表述）。
    #   格式交给推理侧 output_cleaner.py，训练只管内容。
    for r in rows:
        r["output"] = strip_markdown(r["output"])
        r["instruction"] = strip_markdown(r["instruction"])

    suffix = "_sample" if sample_only else ""
    out = OUT_DIR / f"train{suffix}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  写出 {out.name}: {len(rows)} 条")

    if all_dropped:
        dp = OUT_DIR / "dropped_exam_questions.jsonl"
        with dp.open("w", encoding="utf-8") as fh:
            for d in all_dropped:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  丢弃清单: {len(all_dropped)} 条")


if __name__ == "__main__":
    import sys

    build(sample_only="--sample" in sys.argv)
