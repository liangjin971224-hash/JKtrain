"""LLM 多跳规划：50 题评测集。

设计原则：
  1. **六成是「不该触发」** —— 只测该触发的情况会高估收益，
     测不出误触发率（第一版就是因此把 44% 当成正常）。
  2. **问法取自真实语料** —— 参考 `08_数据入库/milvus/eval_set.json`
     与 16 个情景案例，而不是自己编抽象句子。
  3. **每题标注「期望」** —— 便于发现回归：
     规则应当判对 / 应当触发复核 / 两者皆可。

分类（`expect` 字段）：
  · `rule_ok`   规则应判对，不该触发 LLM
  · `trigger`   规则大概率判不了，应触发 LLM 复核
  · `either`    两种都可以接受（判据模糊）
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlanCase:
    question: str
    expect: str          # rule_ok / trigger / either
    note: str = ""


# ===================================================================
# 一、单步问题（rule_ok）—— 占比最高，是误触发的主要考验
# ===================================================================
SINGLE_STEP: list[PlanCase] = [
    PlanCase("保健食品能声称哪些功能", "rule_ok", "单步-法规"),
    PlanCase("降血糖这个说法对不对", "rule_ok", "单步-合规判定"),
    PlanCase("绝对不能说的话有哪些", "rule_ok", "单步-禁用表述"),
    PlanCase("保健食品的强制警示语是什么", "rule_ok", "单步-警示语"),
    PlanCase("怎么区分保健食品和普通食品", "rule_ok", "单步-分类"),
    PlanCase("批准文号的格式有哪几种", "rule_ok", "单步-文号"),
    PlanCase("客户进店第一步该做什么", "rule_ok", "单步-流程"),
    PlanCase("介绍产品时必须说哪五件事", "rule_ok", "单步-流程"),
    PlanCase("客户说太贵了怎么应对", "rule_ok", "单步-异议"),
    PlanCase("客户怀疑我们是骗子怎么处理", "rule_ok", "单步-异议"),
    PlanCase("体脂仪测量前要做哪些准备", "rule_ok", "单步-设备"),
    PlanCase("骨密度的T值是什么意思", "rule_ok", "单步-设备"),
    PlanCase("客户档案里的自述要怎么标", "rule_ok", "单步-档案"),
    PlanCase("什么情况下可以退货", "rule_ok", "单步-售后"),
    PlanCase("客户在店里突然胸闷怎么办", "rule_ok", "单步-应急"),
    PlanCase("健康顾问不能做什么", "rule_ok", "单步-边界"),
    PlanCase("鱼油哪些人不能吃", "rule_ok", "单步-产品"),
    PlanCase("辅酶Q10对心脏好不好", "rule_ok", "单步-产品"),
    PlanCase("乳清蛋白和大豆蛋白怎么选", "rule_ok", "单步-横向比较(不是多跳)"),
    PlanCase("普通食品能不能宣传功效", "rule_ok", "单步-合规"),
    PlanCase("复合蛋白固体饮料能增强免疫力吗", "rule_ok", "单步-产品"),
    PlanCase("客户犹豫不决时怎么跟进", "rule_ok", "单步-经验"),
    PlanCase("老员工有什么处理经验", "rule_ok", "单步-经验"),
    PlanCase("客户问你们有鱼油吗", "rule_ok", "单步-案例"),
    PlanCase("客户说孕妇能不能吃", "rule_ok", "单步-案例"),
    PlanCase("客户拿我们和别的品牌比较", "rule_ok", "单步-案例"),
    PlanCase("客户说太贵了别家便宜一半", "rule_ok", "单步-案例"),
    PlanCase("辅助降血脂是什么意思", "rule_ok", "单步-术语"),
    PlanCase("蓝帽子标志代表什么", "rule_ok", "单步-术语"),
    PlanCase("客户问吃了能不能停药怎么答", "rule_ok", "单步-红线"),
]

# ===================================================================
# 二、规则能判对的并列（rule_ok）—— 不该触发
# ===================================================================
PARALLEL_RULE_OK: list[PlanCase] = [
    PlanCase("补钙的有什么推荐，有什么禁忌", "rule_ok", "并列-无回指"),
    PlanCase("鱼油有什么功能，孕妇能吃吗", "rule_ok", "并列-两话题"),
    PlanCase("蛋白粉有什么功效，肾不好能吃吗", "rule_ok", "并列-产品+禁忌"),
    PlanCase("鱼油几点吃好，能空腹吃吗", "trigger", "★规则缺陷-「几点吃好」被整段丢掉"),
    PlanCase("辅酶Q10有什么作用，什么时候吃最好", "rule_ok", "并列-功效+用法"),
    PlanCase("客户说太贵了怎么答，能不能打折", "rule_ok", "并列-异议+价格"),
]

# ===================================================================
# 三、规则能判对的依赖（rule_ok）—— 回指词在表内
# ===================================================================
DEP_RULE_OK: list[PlanCase] = [
    PlanCase("改善体脂都可以吃什么保健品，其中那个最便宜", "rule_ok", "依赖-其中"),
    PlanCase("降血脂的推荐几款，里面哪款划算", "rule_ok", "依赖-里面哪款"),
    PlanCase("补钙的产品有哪些，这些里面最便宜的", "rule_ok", "依赖-这些"),
    PlanCase("适合老人的蛋白粉有几款，上述产品哪个销量好", "trigger", "★规则缺陷-前半句被丢掉"),
    PlanCase("免疫力的产品推荐一下，那几款里最便宜的", "rule_ok", "依赖-那几款"),
]

# ===================================================================
# 四、规则大概率判不了（trigger）—— 这是 LLM 的价值所在
# ===================================================================
SHOULD_TRIGGER: list[PlanCase] = [
    # 4.1 片段被丢弃（含回指/比价，但词表没覆盖组合）
    PlanCase("改善体脂吃什么，这几样给我挑个划算的", "trigger", "丢弃-这几样+划算"),
    PlanCase("补钙的推荐一下，刚才说的那批里挑一个", "trigger", "丢弃-刚才说的"),
    PlanCase("降血脂的有什么，上面提到的几款哪个好", "rule_ok", "规则判对-「上面提到的」+「哪个」"),
    PlanCase("缓解疲劳的产品有哪些，前面提到的那款怎么样", "trigger", "丢弃-前面提到的"),

    # 4.2 条件依赖（规则词表没有"不行的话/否则"）
    PlanCase("鱼油和阿司匹林能一起吃吗，不行的话有什么替代", "trigger", "条件-不行的话"),
    PlanCase("孕妇能吃蛋白粉吗，如果不能那吃什么好", "trigger", "条件-如果不能"),
    PlanCase("这个和降压药冲突吗，冲突的话换哪个", "rule_ok", "规则判对-「哪个」在词表"),
    PlanCase("辅酶Q10能长期吃吗，否则有什么副作用", "trigger", "条件-否则"),

    # 4.3 无标点但语义两步
    PlanCase("那种吃了对血脂好的最便宜的多少钱", "either", "无标点-隐含两步"),
    PlanCase("适合糖尿病人又能降血脂的有哪些", "either", "条件筛选-单步或双步"),
    PlanCase("先看体脂再推荐产品的流程是什么", "either", "流程内含顺序"),

    # 4.4 三个问句（可能拆漏）
    PlanCase("鱼油有什么功能？孕妇能吃吗？多少钱一盒？", "rule_ok", "规则判对-问号切分正常"),
    PlanCase("保健食品能治高血压吗？能吃多久？有副作用吗？", "trigger", "多问号"),
]

# ===================================================================
# 五、边界与陷阱（either）—— 检验是否被过度拆解
# ===================================================================
TRAPS: list[PlanCase] = [
    PlanCase("我和我老婆都能吃鱼油吗", "rule_ok", "陷阱-两个主语但一个问题"),
    PlanCase("鱼油和卵磷脂有什么区别和联系", "rule_ok", "陷阱-两个对象一个问题"),
    PlanCase("吃了没效果能不能退，退了还能再买吗", "either", "售后连问"),
    PlanCase("客户问买了能不能停降压药，停了会怎样", "either", "红线追问"),
    PlanCase("这个产品既能降血脂又能增强免疫力吗", "rule_ok", "陷阱-两个功效一个问题"),
]


def all_cases() -> list[PlanCase]:
    return (SINGLE_STEP + PARALLEL_RULE_OK + DEP_RULE_OK
            + SHOULD_TRIGGER + TRAPS)


def stats() -> dict[str, int]:
    from collections import Counter

    return dict(Counter(c.expect for c in all_cases()))


__all__ = [
    "DEP_RULE_OK",
    "PARALLEL_RULE_OK",
    "PlanCase",
    "SHOULD_TRIGGER",
    "SINGLE_STEP",
    "TRAPS",
    "all_cases",
    "stats",
]
