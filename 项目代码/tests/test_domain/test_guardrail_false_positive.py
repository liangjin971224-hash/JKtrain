"""护栏误报回归测试。

★ 这个文件的每一条都来自**真实误报**，不是假想场景。

背景：护栏的第一版单元测试全绿（68 passed），但拿真实培训内容一跑，
1166 个段落里 127 个被误判（10.9%）。逐条看下来，绝大多数是规则写得太宽：
    · 「第一条铁律」「## 第一步」被当成绝对化用语
    · 「您最好先咨询医生」（合规建议）被当成绝对化
    · 「第一选择」漏检

**教训：单元测试只覆盖了自己想到的情况。**
误报必须用真实语料扫，扫出来的每一条都要固化成测试。

本文件的作用：防止后续调规则时又把误报带回来。
"""
from __future__ import annotations

import pytest

from domain.rules.forbidden_claims import check_forbidden_claims
from domain.rules.product_category import check_claim_allowed
from domain.rules.warning_text import check_warning_present
from domain.value_objects import ProductCategory


# ===================================================================
# 序号类「第一」不得误报
# ===================================================================
class TestOrdinalDiYi:
    """「第一条 / 第一步 / 第一次」是序号，不是绝对化宣称。"""

    @pytest.mark.parametrize("text", [
        "第一条铁律：客户主动问了产品，必须如实回答",
        "## 第一步 迎宾与初步沟通（1–3 分钟）",
        "第一次来吗？我先了解一下您的情况",
        "第一个环节是需求了解",
        "第一组是最容易说错的",
        "第一阶段的重点是产品知识",
        "第一步的选择很重要",
        "客户第一次到店",
        "优先级第一的是安全",
    ])
    def test_ordinal_not_flagged(self, text: str) -> None:
        result = check_forbidden_claims(text)
        assert not result.has_critical, (
            f"序号用法被误判为绝对化: {text}\n"
            f"命中: {[v.matched for v in result.violations]}"
        )

    def test_category_heading_flags_on_content_word_not_ordinal(self) -> None:
        """「第一类：疾病治疗类」会因「治疗」命中，不因「第一类」命中。

        ★ 这是**已知且可接受**的边界：
          「疾病治疗类」是分类名词，不是医疗宣称，
          但护栏是纯文本规则，无法区分「引用分类名」与「做宣称」。
          处理方式是：护栏只对**模型输出**生效，
          而模型输出不会出现「第一类：疾病治疗类」这种文档标题结构。
          若真出现，误拦的代价（人工复核一条）远小于漏放的代价（违法宣传）。
        """
        result = check_forbidden_claims("第一类：疾病治疗类（严禁）")
        matched = [v.matched for v in result.violations]
        assert "第一类" not in matched, "「第一类」不应作为绝对化命中"
        assert "治疗" in matched, "命中的是「治疗」这个内容词"

    @pytest.mark.parametrize("text", [
        "这是行业第一品牌",
        "行业第一",
        "全国第一",
        "排名第一",
        "第一品牌",
        "第一选择",
        "我们的销量第一",
    ])
    def test_ranking_still_flagged(self, text: str) -> None:
        """真正的排名宣称仍必须检出——修误报不能把真违规也放掉。"""
        result = check_forbidden_claims(text)
        assert result.has_critical, f"排名宣称漏检: {text}"


# ===================================================================
# 「最」的两种用法
# ===================================================================
class TestZuiUsage:
    """「最好」既可能是绝对化，也可能是副词「建议最好……」"""

    @pytest.mark.parametrize("text", [
        "您最好先咨询医生",
        "建议最好遵医嘱",
        "最好还是去医院检查一下",
        "最好问一下医生",
        "最好别自己停药",
    ])
    def test_adverbial_zui_not_flagged(self, text: str) -> None:
        """「最好 + 建议动作」是合规话术，拦掉会把正常业务堵死。"""
        result = check_forbidden_claims(text)
        assert not result.has_critical, (
            f"合规建议被误判为绝对化: {text}\n"
            f"命中: {[v.matched for v in result.violations]}"
        )

    @pytest.mark.parametrize("text", [
        "这是最有效的产品",
        "最有效的产品",
        "最好的选择",
        "效果最强",
        "见效最快",
    ])
    def test_superlative_product_flagged(self, text: str) -> None:
        result = check_forbidden_claims(text)
        assert result.has_critical, f"绝对化用语漏检: {text}"


# ===================================================================
# 合规话术不得误报（来自真实培训文档的表述）
# ===================================================================
class TestCompliantPhrases:
    """这些是 D001/D002/D004 的标准话术，必须放行。"""

    @pytest.mark.parametrize("text", [
        # 法定警示语及其变体
        "保健食品不是药物，不能代替药物治疗疾病",
        "本品不能代替药物",
        "不是药物，不能代替药物治疗疾病",
        # 目录原文的合规功能声称
        "有助于维持血糖健康水平",
        "有助于维持血脂（胆固醇/甘油三酯）健康水平",
        "有助于增强免疫力",
        "有助于改善睡眠",
        "有助于改善骨密度",
        "有助于控制体内脂肪",
        # 边界说明
        "个体差异较大，不能承诺具体效果",
        "保健食品是长期调理，没有固定的见效时间",
        "便秘人群不建议长期服用",
        "请咨询医生或药师",
        "建议遵医嘱，不要自行停药",
        # 客观描述
        "每份含蛋白质 15g，一罐 400g 大概 20 份",
        "适宜人群是血脂偏高者",
        "不适宜人群包括少年儿童、孕妇、乳母",
        "产品类别是保健食品，批准文号国食健注G20250012",
        # 教学/警示语境里的类别名与否定式（2026-11-09 实测误拒）
        "「防癌」不能讲，属于疾病治疗类禁用词",
        "保健食品不能声称治疗作用",
        "本品不能替代药物，服用治疗药物请咨询医生",
        "不建议先停一段时间，应先咨询医生",
    ])
    def test_standard_scripts_pass(self, text: str) -> None:
        result = check_forbidden_claims(text)
        assert not result.has_critical, (
            f"标准话术被误判: {text}\n"
            f"命中: {[v.matched for v in result.violations]}"
        )


# ===================================================================
# 时间/疗程承诺 + 停药：真违规要拦，教学提及要放行
# ===================================================================
class TestTimePromiseStopMedication:
    @pytest.mark.parametrize("text", [
        "吃几个月就能停",
        "吃几个月就能停药",
        "一年左右就能停",
        "三个疗程就能停药",
        "先停一段时间观察观察",
    ])
    def test_bare_time_promise_is_blocked(self, text: str) -> None:
        result = check_forbidden_claims(text)
        assert result.has_critical, f"诱导停药漏检: {text}"

    @pytest.mark.parametrize("text", [
        "不能吃几个月就停药，必须遵医嘱",
        "不建议先停一段时间，应先咨询医生",
        "资料里的错误说法是「吃几个月就能停药」，不能照用",
    ])
    def test_teaching_or_negated_context_passes(self, text: str) -> None:
        result = check_forbidden_claims(text)
        assert not result.has_critical, (
            f"教学/否定语境被误判: {text}\n"
            f"命中: {[v.matched for v in result.violations]}"
        )


# ===================================================================
# 普通食品：真违规仍要拦
# ===================================================================
class TestOrdinaryFoodNoRegression:
    @pytest.mark.parametrize("text", [
        "有助于增强免疫力",
        "补充蛋白质效果好",
        "适合免疫力低的人",
        "改善皮肤弹性",
        "抗衰老",
        "养肝明目",
        "补肾",
        "补钙",
        "减肥",
        "控卡",
        "营养均衡",
    ])
    def test_still_blocked(self, text: str) -> None:
        verdict = check_claim_allowed(ProductCategory.ORDINARY_FOOD, text)
        assert verdict.blocked, f"普通食品功能声称漏检: {text}"

    @pytest.mark.parametrize("text", [
        "每份含蛋白质 12g",
        "配料是乳清蛋白粉和麦芽糊精",
        "一罐 400g，建议每日 1 次",
        "这是普通食品，不是保健食品",
        "保质期 18 个月",
    ])
    def test_plain_description_passes(self, text: str) -> None:
        """客观描述配料/规格/保质期，不是功能声称。"""
        verdict = check_claim_allowed(ProductCategory.ORDINARY_FOOD, text)
        assert not verdict.blocked, (
            f"客观描述被误判为功能声称: {text}（命中「{verdict.matched}」）"
        )


# ===================================================================
# 体脂类：漏词缺口 + 体脂仪误报（2026-10-23 补，两者必须同时守）
# ===================================================================
class TestBodyFatClaims:
    """★ 这一组是「漏报」和「误报」成对出现的典型。

    背景：接多跳时实测「能改善体脂」被**放行**——
    禁词表里只有法规原文「控制体内脂肪」和旧称「减肥」，
    没有客户口语最常用的「体脂」。

    但补词时立刻踩到反向问题：门店有一台**体脂仪**（D010 专门讲它），
    「请用体脂仪测量」是客观描述，裸词「体脂」会把设备类问题全部误拦。

    所以规则改成了「功效动词 + 体脂类词」的组合式，
    下面两组断言分别守住漏报和误报。**删任何一组都会让另一半悄悄坏掉。**
    """

    @pytest.mark.parametrize("text", [
        "这个能改善体脂",
        "有助于改善体脂率",
        "可以减体脂",
        "能降低内脏脂肪",
        "体脂会降下来",
        "能减重",
        "可以控制体重",
    ])
    def test_body_fat_claims_blocked(self, text: str) -> None:
        """普通食品说这些就是违规声称（D005 第 9 项对应的口语说法）。"""
        verdict = check_claim_allowed(ProductCategory.ORDINARY_FOOD, text)
        assert verdict.blocked, (
            f"体脂类功能声称漏检: {text}\n"
            f"  依据：D005 第 62 行「第 9 项 有助于控制体内脂肪」"
        )

    @pytest.mark.parametrize("text", [
        "请用体脂仪测量",
        "体脂仪的测量结果是参考值",
        "体脂率从体脂仪读取后要跟客户解释",
        "基础代谢和内脏脂肪看体脂仪数据",
        "体脂率 28% 属于偏高区间（仪器读数为参考）",
    ])
    def test_body_fat_device_not_flagged(self, text: str) -> None:
        """★ 设备与客观数据必须放行。

        「体脂仪」是门店真实存在的设备（D010 整篇讲它）。
        若把「体脂」当裸禁词，设备使用这类问题会被整类堵死 ——
        误报的代价是业务事故，与漏报的违法风险同等严重。
        """
        verdict = check_claim_allowed(ProductCategory.ORDINARY_FOOD, text)
        assert not verdict.blocked, (
            f"体脂仪/客观数据被误判为功能声称: {text}"
            f"（命中「{verdict.matched}」）"
        )

    def test_health_food_still_allowed(self) -> None:
        """同一句话对保健食品是合规的（它可能是批准功能原文）。

        判定必须结合类别，不能只看文本 —— 这是本模块存在的理由。
        """
        verdict = check_claim_allowed(ProductCategory.HEALTH_FOOD,
                                      "有助于控制体内脂肪")
        assert not verdict.blocked


# ===================================================================
# 警示语：等价表述不得误判为缺失
# ===================================================================
class TestWarningNoRegression:
    @pytest.mark.parametrize("answer", [
        "产品说明。保健食品不是药物，不能代替药物治疗疾病。",
        "产品说明。本品不能代替药物。",
        "产品说明。不能代替药物治疗疾病。",
        "产品说明。不是药物，不能代替药物治疗疾病。",
        "产品说明。不能替代药物。",
    ])
    def test_equivalents_recognized(self, answer: str) -> None:
        result = check_warning_present(answer, ProductCategory.HEALTH_FOOD)
        assert result.present, f"等价警示语未识别: {answer}"

    def test_missing_still_blocked(self) -> None:
        result = check_warning_present(
            "这款鱼油有助于维持血脂健康水平，适宜血脂偏高者。",
            ProductCategory.HEALTH_FOOD)
        assert result.blocked
