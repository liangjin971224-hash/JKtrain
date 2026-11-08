"""意图分类（纯领域服务，零框架零 IO）。

────────────────────────────────────────────────────────────────────
为什么需要它
────────────────────────────────────────────────────────────────────
P3 就把 13 套意图权重表（`rrf_fusion.INTENT_WEIGHTS`）建好了，但接口层
从不传 `intent`，于是 `AnswerQuestionUseCase` 一直走「开放问答」
(图谱 0.1 / 向量 0.8 / 关键词 0.1)。后果是**权重结构是反的**：

    「在吃阿司匹林能吃鱼油吗」
        应该是「禁忌咨询」(0.6, 0.2, 0.2) —— 答案在图谱的禁忌边上
        实际走「开放问答」(0.1, 0.8, 0.1) —— 图谱这个最准的一路被压到 0.1

本模块就是补上这块：把问题映射到权重表里的标签。

────────────────────────────────────────────────────────────────────
为什么先做规则、不做 LLM
────────────────────────────────────────────────────────────────────
1. **不破坏红线**：`AnswerQuestionUseCase` 的硬约束是「召回为空不调 LLM」。
   若分类器先调一次 LLM，这条红线在链路上就被绕过了 ——
   乱码输入也会打一次模型，白白花钱还变慢。
2. **可复现**：合规培训场景下，同一句话每次判成不同意图会让
   「为什么这条排第一」无法解释。
3. **够用**：培训场景的问法是**收敛**的（禁忌/停药/文号/设备/流程…），
   不像开放域对话那样长尾。规则覆盖不到时还有 LLM 兜底开关（默认关）。

────────────────────────────────────────────────────────────────────
规则表的顺序就是优先级（这一点最容易改错）
────────────────────────────────────────────────────────────────────
安全类必须排在前面。反例：
    「鱼油哪些人不能吃」
        · 先进「特殊人群」（含「人」）→ 判成特殊人群
        · 应该进「禁忌咨询」（含「不能吃」）
    两者权重恰好相同，所以这次没出问题；但
    「吃了能停降压药吗」若被「产品功效」（含「降压」）抢走，
    权重就从 (0.5,0.3,0.2) 变成 (0.5,0.3,0.2)…
    —— 真正会出事的是「客户在店里突然胸闷怎么办」被判成「业务流程」，
    那种情况下检索范围完全偏了。

────────────────────────────────────────────────────────────────────
判定与「不匹配」的边界
────────────────────────────────────────────────────────────────────
不匹配就返回「开放问答」，**不要硬塞**。硬塞的代价是给一个不相干的
问题套上安全类权重，让召回偏向一个错误的范围。宁可 0.1/0.8/0.1 这种
中性权重，也不要错位的强权重。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from domain.services.rrf_fusion import INTENT_WEIGHTS

logger = logging.getLogger(__name__)

#: 兜底意图。未命中任何规则时用它。
OPEN_INTENT = "开放问答"

#: 安全类意图。这些意图的图谱权重是 0.5–0.6，判错代价最高。
SAFETY_INTENTS = frozenset({
    "禁忌咨询", "特殊人群", "要停药", "要诊断", "要处方", "超量服用",
})

#: 分类来源，写进 meta 便于回放「这次为什么这么判」。
SOURCE_RULE = "rule"
SOURCE_LLM = "llm"
SOURCE_DEFAULT = "default"

#: LLM 兜底时允许返回的「无法判断」标记。
UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class IntentResult:
    """意图分类结果。

    Attributes:
        label: 意图标签，**一定**是 `INTENT_WEIGHTS` 里的键（或 `OPEN_INTENT`）。
        confidence: 置信度 0–1。规则命中词越多越高。
        matched: 命中的词，用于解释「凭什么这么判」。
        source: `rule` / `llm` / `default`。
    """

    label: str
    confidence: float = 0.0
    matched: tuple[str, ...] = ()
    source: str = SOURCE_DEFAULT

    @property
    def is_safety_sensitive(self) -> bool:
        return self.label in SAFETY_INTENTS


#: 规则表：**(意图, 正则)**，按优先级从上到下。
#:
#: ★ 写规则的三条纪律：
#:   1. 只写**业务上确有区分力**的说法，不追求覆盖所有同义句
#:      （覆盖不了还有 LLM 兜底开关，但错的规则比漏的规则更难查）
#:   2. 不用 `禁止` 这种会跨类命中的裸词：「接待过程中有哪些禁止行为」
#:      是业务流程，不是禁忌咨询
#:   3. 安全类规则里要带上**具体的药名/人群词**，不要靠「能不能」这类
#:      疑问形式判断 —— 「能不能宣传功效」显然不是禁忌咨询
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # ---------------- 第一梯队：安全类 ----------------
    (
        "应急处理",
        re.compile(
            r"胸闷|胸痛|心慌|晕倒|昏倒|休克|抽搐|呼吸困难|喘不上气|大出血|"
            r"烫伤|摔倒|摔伤|抽搐|过敏反应|急救|打120|叫救护车|突发事件|"
            r"意外|身体不适|不舒服得厉害"
        ),
    ),
    (
        "超量服用",
        re.compile(
            r"吃多|多吃|误食|超量|过量|超过剂量|一天最多|最多吃|吃几粒|"
            r"几粒|剂量上限|重复吃|忘了已经吃过|加量|加倍"
        ),
    ),
    (
        "要停药",
        re.compile(
            # ★ 「能停」要单独列出：「吃了能停降压药吗」里没有「停药」这个词，
            #   只有「能停」——而它恰好被「降压药」抢进了禁忌咨询。
            r"停药|停掉|停用|停了|能停|能不能停|可以停|可不可以停|"
            r"代替药|替代药|不吃药|不用吃药|取代药|换掉药|断药"
        ),
    ),
    (
        "要处方",
        re.compile(
            r"开药|开点药|处方|给我配|配点药|推荐吃什么药|用什么药|吃什么药好"
        ),
    ),
    (
        "要诊断",
        re.compile(
            r"是不是得|是不是患|算不算病|是什么病|什么病|诊断|确诊|"
            r"帮我看看.{0,6}(指标|报告|化验)|我这个.{0,4}是不是|病情"
        ),
    ),
    (
        "禁忌咨询",
        re.compile(
            # 明确的否定/禁忌说法
            r"不能吃|不可以吃|不宜|忌口|禁忌|禁用|忌食|"
            r"哪些人不能|什么人不适合|谁不能吃|"
            # 同服与相互作用
            r"同服|同时吃|一起吃|一起服用|相互作用|冲突|相克|不良反应|"
            r"能一起|会不会有影响|有没有影响|"
            # 过敏
            r"过敏|不耐受|"
            # 具体药名（客户最常问的合并用药场景）
            r"阿司匹林|华法林|氯吡格雷|他汀|二甲双胍|降压药|降糖药|降脂药|"
            r"胰岛素|抗凝|抗血小板|优甲乐|甲状腺素|布洛芬|头孢|抗生素|激素"
        ),
    ),
    (
        "特殊人群",
        re.compile(
            r"孕妇|孕期|怀孕|备孕|哺乳|喂奶|产妇|月子|"
            r"儿童|小孩|宝宝|婴儿|幼儿|青少年|学生|"
            r"老人|老年|年纪大|八十岁|七十岁|"
            r"糖尿病|血糖高|高血压|血压高|高血脂|血脂高|痛风|尿酸高|"
            r"肾病|肾功能|肾不好|肝功能|肝不好|脂肪肝|甲亢|甲减|甲状腺|"
            r"心脏病|冠心病|做过手术|术后|化疗|经期|更年期|慢性病|三高|"
            r"体质|敏感体质"
        ),
    ),
    # ---------------- 第二梯队：查询类 ----------------
    (
        "文号查询",
        re.compile(
            r"批准文号|文号|国食健注|国食健字|食健备|卫食健字|"
            r"蓝帽子|注册号|备案号|怎么查真伪|查真伪|是不是正规"
        ),
    ),
    (
        "设备使用",
        re.compile(
            r"体脂仪|体脂秤|骨密度仪|骨密度|血压计|血糖仪|仪器|设备|"
            r"T值|Z值|体脂率|基础代谢|内脏脂肪|骨量|测量|怎么测|校准"
        ),
    ),
    (
        "业务流程",
        re.compile(
            r"流程|SOP|第一步|第二步|步骤|接待|迎宾|开场|送客|"
            r"档案|记录|登记|回访|跟进|交接班|排班|"
            r"退货|换货|退款|售后|投诉|发票|开单|收银|"
            r"禁止行为|必须说|必须做|注意事项|规范要求|考核|职责|"
            r"怎么应对|怎么回答|怎么答|话术|异议"
        ),
    ),
    # ---------------- 第三梯队：产品与兜底 ----------------
    (
        "产品名查询",
        re.compile(
            # ★ 只写「明确在问某个货」的说法。
            #   早期写过 `有.{1,6}么`，结果「鱼油有什么功能」里的「有什么」
            #   被当成产品名查询，把功效询问整类抢走（红灯抓出来的）。
            r"多少钱|价格|几盒|一盒|一瓶|一袋|"
            r"有卖|有货|现货|有没有卖|有没有得卖|规格|包装|"
            # 「你们有鱼油吗」这类问货句式。
            # ★ 负向前瞻排除「有什么/有啥/哪些」——否则「鱼油有什么功能」
            #   会被当成产品名查询（红灯抓出来的真实误判）。
            r"有(?!什么|啥|哪些|哪几|没有|没)[^什啥哪吗]{1,8}吗"
        ),
    ),
    (
        "产品功效",
        re.compile(
            r"功效|功能|作用|效果|有什么用|有什么用|管用|有效|好不好|"
            r"适合|推荐|保健品|保健食品|普通食品|成分|配料|含量|"
            r"增强免疫力|免疫|补钙|降血脂|降血糖|降血压|助眠|睡眠|"
            r"抗氧化|护肝|养胃|通便|减肥|增肌|补气血|抗疲劳|"
            r"维生素|蛋白|鱼油|辅酶|褪黑素|益生菌|胶原|叶黄素|氨糖|"
            r"灵芝|螺旋藻|蜂胶|芝麻丸|代餐|膳食纤维|声称"
        ),
    ),
)

#: 未命中时的默认置信度。不设 0，因为「判成开放问答」本身是有依据的判断。
_DEFAULT_CONFIDENCE = 0.35

#: 单个命中词的基准置信度，以及每多命中一个词的增量与上限。
_BASE_CONFIDENCE = 0.60
_CONFIDENCE_STEP = 0.10
_MAX_CONFIDENCE = 0.95


class IntentClassifier:
    """把问题映射到意图标签。

    ★ 构造函数接受 Port，不碰具体适配器 —— 与项目其他用例一致，
      这样 LLM 兜底可以用假实现测。
    """

    def __init__(
        self,
        *,
        llm=None,
        use_llm_fallback: bool = False,
        rules: tuple[tuple[str, re.Pattern[str]], ...] | None = None,
    ) -> None:
        """
        Args:
            llm: `LanguageModelPort`。仅在 `use_llm_fallback=True` 时使用。
            use_llm_fallback: 是否在规则全部未命中时调 LLM 兜底。

                ★ 默认 False，且**不建议随手打开**：
                  分类器位于召回链路最前面，一旦在这里调 LLM，
                  「召回为空不调 LLM」这条红线的成本假设就不成立了
                  （乱码输入也会打一次模型）。而且每次问答多 0.3s。
            rules: 自定义规则表（测试用，便于验证优先级）。
        """
        self._llm = llm
        self._use_llm_fallback = use_llm_fallback
        self._rules = rules if rules is not None else _RULES

    # ---------------- 主入口 ----------------
    def classify(self, question: str) -> IntentResult:
        """判断问题意图。

        Args:
            question: 用户问题。

        Returns:
            分类结果。**永不抛异常** —— 它在链路最前面，
            抛出去等于整个问答不可用。
        """
        text = (question or "").strip()
        if not text:
            return IntentResult(
                label=OPEN_INTENT, confidence=0.0, source=SOURCE_DEFAULT)

        hit = self._match_rule(text)
        if hit is not None:
            return hit

        # 规则没命中 → 可选 LLM 兜底
        if self._use_llm_fallback and self._llm is not None:
            return self._classify_by_llm(text)

        return IntentResult(
            label=OPEN_INTENT,
            confidence=_DEFAULT_CONFIDENCE,
            source=SOURCE_DEFAULT,
        )

    # ---------------- 规则 ----------------
    def _match_rule(self, text: str) -> IntentResult | None:
        """按优先级找第一条命中的规则。

        返回 `None` 表示全部未命中（由调用方决定兜底策略）。
        """
        for label, pattern in self._rules:
            matched = tuple(dict.fromkeys(m.group(0) for m in pattern.finditer(text)))
            if not matched:
                continue

            # 兜底：规则表里有、但权重表里没有的标签不能返回，
            # 否则 weights_for_intent 会静默走默认权重（看起来像「生效了」）。
            if label not in INTENT_WEIGHTS:
                logger.warning(
                    "意图规则命中「%s」，但 INTENT_WEIGHTS 里没有该标签，忽略", label)
                continue

            confidence = min(
                _BASE_CONFIDENCE + _CONFIDENCE_STEP * (len(matched) - 1),
                _MAX_CONFIDENCE,
            )
            return IntentResult(
                label=label, confidence=confidence,
                matched=matched, source=SOURCE_RULE,
            )
        return None

    # ---------------- LLM 兜底 ----------------
    def _classify_by_llm(self, text: str) -> IntentResult:
        """规则没覆盖时问模型。

        ★ 给模型的候选集就是**权重表的键**：
          分类的唯一目的就是选权重，给一个查不到权重的标签没有意义。
        ★ 模型返回任何意外值都退回「开放问答」，不猜。
        """
        labels = [*INTENT_WEIGHTS.keys(), UNKNOWN]
        try:
            raw = self._llm.classify(text, labels, allow_unknown=True)
        except Exception as exc:                    # noqa: BLE001 兜底不能拖垮主链路
            logger.warning("意图 LLM 兜底失败，退回开放问答: %s: %s",
                           type(exc).__name__, exc)
            return IntentResult(
                label=OPEN_INTENT, confidence=0.0, source=SOURCE_DEFAULT)

        label = (raw or "").strip()
        if label not in INTENT_WEIGHTS:
            return IntentResult(
                label=OPEN_INTENT, confidence=0.0, source=SOURCE_DEFAULT)

        return IntentResult(
            label=label, confidence=0.70, matched=(f"llm:{label}",),
            source=SOURCE_LLM,
        )


__all__ = [
    "OPEN_INTENT",
    "SAFETY_INTENTS",
    "SOURCE_DEFAULT",
    "SOURCE_LLM",
    "SOURCE_RULE",
    "IntentClassifier",
    "IntentResult",
]
