"""产品类别判定与功能声称准入。

★ 约束 #2 的实现点：普通食品不得有任何功能声称。

────────────────────────────────────────────────────────────────────
规则来源
────────────────────────────────────────────────────────────────────
`06_正式内容/D008` 第一章「普通食品的三条铁律」：

| # | 铁律 | 违反后果 |
|---|---|---|
| 1 | 不能作任何保健功能声称 | 《食品安全法》虚假宣传 |
| 2 | 不能暗示疾病预防、治疗功能 | 《广告法》 |
| 3 | 不能使用与已批准保健食品相同的名称 | 市场监管总局规定 |

D008 1.2 明确「什么是任何功能声称」的范围比直觉宽得多：
    增强免疫力 / 补充蛋白质效果好 / 适合免疫力低的人 / 改善皮肤弹性 /
    抗衰老 / 养肝明目 / 补肾 / 乌发 / 补钙 / 减肥 / 控卡 / 营养均衡
    —— 全部不能说。

────────────────────────────────────────────────────────────────────
一个容易混淆的点
────────────────────────────────────────────────────────────────────
「有助于增强免疫力」：
    · 对**保健食品**是合规的（它是目录里的批准功能原文）
    · 对**普通食品**是违规的（普通食品没有批准功能）

所以判定必须结合类别，不能只看文本。这正是本模块存在的理由。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from domain.value_objects import ProductCategory


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    """功能声称准入判定结果。

    ★ 不含 score / verdict 分数字段（约束 #3）。
      `blocked` 是布尔判断。
    """

    blocked: bool
    reason: str = ""
    matched: str = ""

    @property
    def allowed(self) -> bool:
        return not self.blocked


# ===================================================================
# 普通食品禁用的功能声称词表
# ===================================================================
# 来源：D008 1.2 的完整清单，逐条对应
_ORDINARY_FOOD_FORBIDDEN = (
    # 保健功能声称（24 项目录里的常见表述）
    "增强免疫力", "提高免疫力", "调节免疫",
    "改善睡眠", "缓解疲劳", "抗疲劳",
    "改善骨密度", "增加骨密度",
    "控制体内脂肪", "减肥", "减脂", "控卡",
    # ★ 体脂 / 体重相关（2026-10-23 补）。
    #
    #   依据：D005 第 62 行「第 9 项：有助于控制体内脂肪」、
    #        第 110 行「旧称『减肥』，2023 版改称『有助于控制体内脂肪』」、
    #        第 269 行「不是减肥药，不能承诺减重效果」。
    #
    #   为什么必须补：拆多跳时实测发现「能改善体脂」被**放行**。
    #   客户口语说的是「体脂」，不是法规原文的「体内脂肪」——
    #   规则表按法规原文写，就会漏掉最常用的口语说法。
    #   这是「按客户实际说法写规则」这条教训在禁用词表上的又一次复现。
    #
    #   ★ 注意：这里**不能**直接放裸词「体脂」。
    #     店里有台「体脂仪」（D010），「请用体脂仪测量」是客观描述，
    #     放裸词会把整个设备使用类问题一起拦掉（实测 4/4 误报）。
    #     所以体脂类改走 `_EFFICACY_CONTEXT_RE`（见下），要求带功效动词。
    "减重", "控制体重", "降低体重",
    "改善皮肤", "皮肤弹性", "美容",
    "抗衰老", "延缓衰老",
    "维持血脂", "血脂健康", "降血脂", "降血糖", "降血压",
    "维持血糖", "血糖健康",
    "缓解视疲劳", "改善视力",
    "促进消化", "改善肠道", "润肠通便",
    "对化学性肝损伤有辅助保护", "养肝", "护肝",
    "清咽", "改善咽部",
    "促进排铅", "改善睡眠质量",
    # 营养强化 / 中医功效声称
    "补钙", "补铁", "补锌", "补充维生素",
    "补肾", "壮阳", "乌发", "黑发", "生发",
    "明目", "益气", "补血", "调经",
    # 功能暗示
    "效果好", "有功效", "有效果", "管用",
    "适合免疫力低", "免疫力低的人",
    "营养均衡", "补充营养",
    # 疾病相关（对普通食品尤其严重）
    "治疗", "预防疾病", "辅助治疗", "康复",
)

# 预编译：一次性匹配全部禁用词
_FORBIDDEN_RE = re.compile("|".join(
    re.escape(w) for w in sorted(_ORDINARY_FOOD_FORBIDDEN, key=len, reverse=True)
))

#: 「体脂/脂肪率」这类**带歧义**的词：只有出现在功效语境里才算违规声称。
#:
#: ★ 为什么不能进上面的裸词表（2026-10-23 实测）：
#:     门店有一台「体脂仪」（D010 专门讲它的操作与指标解读）。
#:     「请用体脂仪测量」「体脂率从体脂仪读取」是**客观描述设备**，完全合规。
#:     把「体脂」放进裸词表 → 设备使用类问题 4 条全被误拦
#:     （这个类别的权重是向量 0.7，是条正经业务线，堵掉就是业务事故）。
#:
#: ★ 判据：功效动词 + 体脂类词，或体脂类词 + 功效方向。
#:     「能改善体脂」「体脂降下来了」「减体脂」→ 拦
#:     「用体脂仪测」「体脂率是 28%」        → 放行（客观数据）
_EFFICACY_CONTEXT_RE = re.compile(
    r"(?:改善|降低|减少|减|降|控制|调节|消除|燃烧|分解|攻克|祛除|抑制)\s*"
    r"(?:体脂|体脂肪|脂肪率|内脏脂肪)"
    r"|(?:体脂|体脂肪|脂肪率|内脏脂肪)\s*"
    r"(?:降下来|降了|会降|变低|减少了|减少|下降|高是.{0,4}问题)"
)


def check_claim_allowed(
    category: ProductCategory | str | None,
    text: str,
) -> ClaimVerdict:
    """判定某段文本对该类别产品是否允许出现。

    Args:
        category: 产品类别。None 或未知值按「普通食品」处理（fail-safe）。
        text: 待判定文本。

    Returns:
        判定结果。`blocked=True` 表示不得输出。

    Note:
        **fail-safe 语义**：类别未知时按最严格处理（不允许声称）。
        理由：脏数据不能成为放行理由，宁可误拦也不可漏放。
    """
    if not text or not text.strip():
        return ClaimVerdict(blocked=False)

    # 归一化类别：未知一律按普通食品处理（fail-safe）
    cat: ProductCategory
    if isinstance(category, ProductCategory):
        cat = category
    else:
        try:
            cat = ProductCategory(str(category)) if category else \
                ProductCategory.ORDINARY_FOOD
        except ValueError:
            # 未知类别 → 最严格：按普通食品
            cat = ProductCategory.ORDINARY_FOOD

    # 保健食品允许功能声称（但禁用词由 forbidden_claims 负责）
    if cat is ProductCategory.HEALTH_FOOD:
        return ClaimVerdict(blocked=False)

    # 普通食品：任何功能声称都拦
    #
    # ★ 必须先挖掉合规表述（实测踩坑）：
    #   「普通食品不能宣称任何保健功能或疾病预防、治疗作用。」
    #   这是**禁止性陈述**，完全合规，但裸正则会命中「治疗」→ 误拦。
    #
    #   挖除逻辑与 `check_forbidden_claims` **共用同一个函数**
    #   （`neutralize_compliant`），避免两处规则漂移。
    #
    # ★ 但必须传 `allow_youzhuyu=False`：
    #   普通食品**没有**批准功能，「有助于增强免疫力」对它是违规声称。
    #   若沿用保健食品的放行名单，就会漏放（实测踩过）。
    from domain.rules.forbidden_claims import neutralize_compliant

    working = neutralize_compliant(text, allow_youzhuyu=False)

    m = _FORBIDDEN_RE.search(working)
    if m is None:
        # 裸词没命中，再查「功效语境 + 体脂类词」。
        # 顺序不能反：裸词精确，先查便宜的；语境正则慢且宽，后查。
        m = _EFFICACY_CONTEXT_RE.search(working)
    if m is None:
        return ClaimVerdict(blocked=False)

    return ClaimVerdict(
        blocked=True,
        reason=(
            f"普通食品不得作任何保健功能声称（违反《食品安全法》）。"
            f"命中「{m.group(0)}」。可改为客观描述配料、规格、食用方法。"
        ),
        matched=m.group(0),
    )


def is_ordinary_food(category: ProductCategory | str | None) -> bool:
    """便捷判定：是否按普通食品处理（含未知类别 fail-safe）。"""
    if isinstance(category, ProductCategory):
        return category is ProductCategory.ORDINARY_FOOD
    if not category:
        return True
    try:
        return ProductCategory(str(category)) is ProductCategory.ORDINARY_FOOD
    except ValueError:
        return True
