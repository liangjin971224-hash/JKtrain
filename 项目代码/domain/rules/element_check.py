"""必答要素覆盖判定。

★ 纯规则、纯函数、零依赖。
  同一份答卷今天判和明天判必须一致——培训场景里「判定不确定」比「判错」更伤信任。

────────────────────────────────────────────────────────────────────
判定规则从哪来
────────────────────────────────────────────────────────────────────
`06_正式内容/02_情景案例/` 每个案例的「四、必答要素」表格里，
「判定说明」列已经写成了可执行的描述，例如：

    E1  出现「有」+ 产品名，或「没有」。不能是"好像有""我们产品挺多"
    E5  出现"不是药物，不能代替药物治疗疾病"或等价完整表述

本模块就是把这类描述翻译成规则。
每个内置规则都标注了它对应哪个案例的哪条要素，便于追溯。

────────────────────────────────────────────────────────────────────
两类要素
────────────────────────────────────────────────────────────────────
内置规则（`_BUILTIN`）：覆盖 C01/C04 等核心案例的高频要素，判定较严。
兜底规则：名称关键词命中即算覆盖，标记为「需人工确认」。
    14+2 个案例共 60+ 条要素，不可能每条都写专门规则；
    兜底让系统能跑通，准确性交给人工。
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable

from domain.value_objects import ElementCheck, ElementSpec

logger = logging.getLogger(__name__)


# ===================================================================
# 判定器实现
# ===================================================================
def _has_any(text: str, *patterns: str) -> str:
    """返回第一个命中的片段，没命中返回空串。"""
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(0)
    return ""


#: 产品名特征（康源牌 + 品类词）
_PRODUCT_NAME = (
    r"康源牌[\u4e00-\u9fff]{2,12}",
    r"(鱼油|辅酶Q10|蛋白粉|乳清蛋白|钙维生素D|褪黑素|益生菌|代餐奶昔|"
    r"固体饮料|膳食纤维|胶原蛋白|维生素C|叶黄素|氨糖|蜂胶|螺旋藻)"
    r"(软胶囊|片|粉|胶囊)?",
)

#: 模糊话术（不能算「明确回答」）。
#:
#: ★ 必须带边界，否则会误匹配（实测踩坑）：
#:   `我看看` 会命中「我们这款」里的「我看」，
#:   导致「有的！我们这款鱼油特别好」被判为「没明确回答」。
#:   这个误判很严重——学员开头明明说了「有的」。
_VAGUE = (
    r"好像", r"应该有", r"大概有", r"似乎", r"不太清楚",
    r"您问.{0,4}同事", r"我(?:再)?看看(?:啊|吧|哦|哈)?(?![们])",
    r"产品挺多",
)

#: 合规功能声称（带「有助于」限定词，来自目录原文）
_CLAIM_WITH_HEDGE = re.compile(
    r"有助于(?:维持|增强|改善|控制|缓解|促进|补充)[^。！？\n]{2,24}"
)

#: 警示语等价片段（与 warning_text.py 保持一致）
_WARNING = (
    "不能代替药物治疗疾病",
    "不能代替药物",
    "不能替代药物治疗疾病",
    "不能替代药物",
    "不是药物，不能代替",
    "保健食品不是药物",
    "本品不能代替药物",
)


class _Judge:
    """一条判定规则。

    用「必须全部满足」的语义，而不是「任一满足」——
    因为案例的判定说明基本都是「A 且 B」的形式
    （如 E4：说清类别 **且** 给出功能原文）。
    """

    def __init__(
        self,
        element_id: str,
        name_keywords: tuple[str, ...],
        check: Callable[[str], tuple[bool, str]],
    ) -> None:
        self.element_id = element_id
        self.name_keywords = name_keywords
        self.check = check


def _explicit_answer(text: str) -> tuple[bool, str]:
    """E1：明确回答有 / 没有。

    判定说明要求：出现「有」+ 产品名，或「没有」。
    不能是「好像有」「产品挺多」这类模糊话术。

    ★ 判断顺序（实测修正）：
      先看有没有「明确的肯定/否定」，有就直接判定覆盖。
      只有在**没看到明确回答**时才去看模糊话术。

      原实现是先查模糊话术，导致「有的！我们这款鱼油特别好」
      因为后面的「我们这款」被误认为「我看看」而判成未覆盖。
    """
    # ---- ① 先找明确回答 ----
    if re.search(r"没有(这|该|此)?(款|个)?(产品|货)?", text):
        return True, "没有"

    # ★ 「有的」独立成句就算明确回答（实测修正）。
    #
    #   原实现要求「有的」紧跟产品名，但真实口语是：
    #       「有的！我们这款鱼油特别好……」
    #   「有的」后面接的是「我们」，不是产品名 → 匹配失败 → 判成未覆盖。
    #   而判定说明写的是「出现『有』+ 产品名，**或**『没有』」，
    #   「有的」本身已经是明确肯定，产品名由 E2 单独判定，不该在 E1 重复要求。
    m0 = re.search(r"(?:^|[。！？!?\n])\s*(?:有的|有啊|有货|有这个)", text)
    if m0 is None:
        m0 = re.search(r"^(?:有的|有啊|有货|有这个)", text.strip())
    if m0:
        return True, m0.group(0).strip("。！？!?\n ")

    # 「有的 / 有货 / 有这个 + 产品名」（保留，覆盖「有的，康源牌…」）
    m = _has_any(
        text,
        *[rf"(?:有的|有货|有这个)[，,、！!。\s]*{p}" for p in _PRODUCT_NAME])
    if m:
        return True, m

    # 「有，<产品名>」
    m2 = _has_any(text, r"有[，,、\s]*" + _PRODUCT_NAME[0])
    if m2:
        return True, m2

    # 「<产品名>…有货」——肯定语在句尾的语序
    m3 = _has_any(
        text,
        *[rf"{p}[^。！？\n]{{0,12}}(?:有货|有的|有卖)" for p in _PRODUCT_NAME])
    if m3:
        return True, m3

    # ---- ② 没找到明确回答，再看有没有模糊话术（有则明确判未覆盖）----
    for vague in _VAGUE:
        if re.search(vague, text):
            return False, ""

    # 既没说清楚也不算模糊 → 未覆盖
    return False, ""


def _product_name(text: str) -> tuple[bool, str]:
    """E2：给出产品名称。"""
    m = _has_any(text, *_PRODUCT_NAME)
    return (True, m) if m else (False, "")


def _objective_params(text: str) -> tuple[bool, str]:
    """E3：给出客观参数（规格 / 含量 / 装量）。"""
    m = _has_any(
        text,
        r"每(?:粒|片|袋|份|罐|支)[^。，,\n]{0,12}\d+(?:\.\d+)?\s*(?:mg|g|毫克|克|亿)",
        r"\d+(?:\.\d+)?\s*(?:mg|g|毫克|克)\s*(?:的)?(?:鱼油|EPA|DHA|蛋白质)",
        r"\d+\s*粒装", r"\d+\s*片装", r"\d+\s*袋装", r"\d+\s*克装",
        r"EPA\s*\+\s*DHA",
        r"净含量", r"规格",
    )
    return (True, m) if m else (False, "")


def _category_and_claim(text: str) -> tuple[bool, str]:
    """E4：说明产品类别 **且** 给出批准功能原文。

    ★ 两个都要有。只说「这是保健食品」不算，只说功能也不算。
    ★ 旧声称「降血脂」不算——这是最关键的一条。
    """
    has_category = bool(re.search(r"保健食品", text))
    claim_match = _CLAIM_WITH_HEDGE.search(text)

    if has_category and claim_match:
        return True, f"保健食品 + {claim_match.group(0)}"
    return False, ""


def _warning_present(text: str) -> tuple[bool, str]:
    """E5：含强制警示语（等价表述均可）。"""
    for frag in _WARNING:
        if frag in text:
            return True, frag
    return False, ""


def _doctor_advice(text: str) -> tuple[bool, str]:
    """E6：提及服药时建议咨询医生。"""
    m = _has_any(
        text,
        r"(?:建议|请|最好|需要)?[^。！？\n]{0,8}咨询(?:一下)?(?:医生|医师|药师)",
        r"遵医嘱",
        r"听医生的",
        r"问(?:一下)?医生",
    )
    # 必须与「在服药」语境相关，否则「咨询医生」可能是别的事
    if m and re.search(r"药|服|吃|降压|降糖|抗凝", text):
        return True, m
    return False, ""


def _contraindicated(text: str) -> tuple[bool, str]:
    """E7：提及不适宜人群。"""
    m = _has_any(
        text,
        r"不适宜人群",
        r"出血倾向",
        r"海产品过敏",
        r"孕妇", r"乳母", r"哺乳期",
        r"少年儿童", r"肝功能不全",
        r"不建议.{0,6}(?:人群|者|服用)",
    )
    return (True, m) if m else (False, "")


#: 内置规则表。
#: key 是「要素 ID + 名称关键词」的组合匹配，
#: 因为不同案例的 E5 含义可能不同（有的要求「若提及功能才加」）。
_BUILTIN: tuple[_Judge, ...] = (
    _Judge("E1", ("明确回答", "回答有", "回答有没有"), _explicit_answer),
    _Judge("E2", ("产品名称", "给出产品名", "报出产品"), _product_name),
    _Judge("E3", ("客观参数", "规格", "含量"), _objective_params),
    _Judge("E4", ("类别与批准功能", "批准功能原文", "类别和功能",
                  "说明产品类别", "引用批准原文"), _category_and_claim),
    _Judge("E5", ("警示语",), _warning_present),
    _Judge("E6", ("咨询医生", "服药", "就医"), _doctor_advice),
    _Judge("E7", ("不适宜人群", "禁忌人群"), _contraindicated),
)


def _match_builtin(spec: ElementSpec) -> _Judge | None:
    """为一条要素找匹配的内置判定器。

    匹配条件：ID 相同 **且** 名称含该规则的关键词之一。
    只按 ID 匹配是不行的——不同案例的 E5 语义可能差别很大。
    """
    for judge in _BUILTIN:
        if judge.element_id != spec.element_id:
            continue
        if any(k in spec.name for k in judge.name_keywords):
            return judge
    return None


def _fallback_judge(spec: ElementSpec, text: str) -> tuple[bool, str]:
    """兜底判定：要素名称里的关键词出现在文本中即算覆盖。

    ★ 这个策略是**宽松**的，会有误判。
      所以调用方（评阅用例）应把兜底判定的结果标记为「需人工确认」。
      但兜底本身是必要的：60+ 条要素不可能每条都写规则，
      没有兜底系统就跑不通。
    """
    # 从名称里提取 2 字以上的中文片段作为关键词
    keywords = [w for w in re.split(r"[^\u4e00-\u9fffA-Za-z0-9]+", spec.name)
                if len(w) >= 2]
    if not keywords:
        return False, ""

    for kw in keywords:
        if kw in text:
            return True, kw
    return False, ""


# ===================================================================
# 对外入口
# ===================================================================
def judge_element(spec: ElementSpec, text: str) -> ElementCheck:
    """判定一条要素是否被覆盖。

    Args:
        spec: 要素定义。
        text: 学员的全部发言（多轮拼在一起）。

    Returns:
        判定结果。`covered=False` + `required=True` 即「缺失的必答要素」。

    Note:
        空文本一律判定未覆盖——不能因为「没提到」就当作「没问题」。
    """
    if not text or not text.strip():
        return ElementCheck(
            element_id=spec.element_id,
            element_name=spec.name,
            weight=spec.weight,
            required=spec.required,
            covered=False,
            evidence="",
        )

    judge = _match_builtin(spec)
    if judge is not None:
        covered, evidence = judge.check(text)
    else:
        covered, evidence = _fallback_judge(spec, text)
        logger.debug(
            "要素 %s「%s」无内置规则，使用兜底判定（结果需人工确认）: %s",
            spec.element_id, spec.name, covered,
        )

    return ElementCheck(
        element_id=spec.element_id,
        element_name=spec.name,
        weight=spec.weight,
        required=spec.required,
        covered=covered,
        evidence=evidence,
    )


def check_elements(
    specs: tuple[ElementSpec, ...], text: str
) -> list[ElementCheck]:
    """批量判定。

    Args:
        specs: 要素定义列表。
        text: 学员发言（已按多轮拼接）。

    Returns:
        与 specs **同序**的判定结果。
    """
    return [judge_element(s, text) for s in specs]


def is_fallback_judged(spec: ElementSpec) -> bool:
    """该要素是否只能靠兜底判定（提示人工复核用）。"""
    return _match_builtin(spec) is None
