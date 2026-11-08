"""输出护栏测试（P4 红灯先行）。

★ 这是全项目**最重要的测试文件**。
  三条不能碰的约束里，前两条都靠这里守：
    1. 输出护栏是必经节点（否则构成违法宣传）
    2. 普通食品不得有任何功能声称

★ 本文件完全不依赖 LLM、不依赖数据库、不依赖网络。
  规则是纯函数，必须能在任何机器上秒级跑绿。

  这是刻意的设计：合规判定的正确性不能依赖模型的不确定性。
  LLM 只负责「组织表达」，判断「这句话违规不违规」由规则做。

规则来源（逐条可追溯）：
    06_正式内容/D005 第四章「四类禁用表述」
    06_正式内容/D005 第三章「7 组最容易说错的表述」
    06_正式内容/D004 4.1「必须说的那句话」
    06_正式内容/D008 第一章「普通食品的三条铁律」
"""
from __future__ import annotations

import pytest

from domain.value_objects import ProductCategory, Severity


# ===================================================================
# 1. 禁用词检测
# ===================================================================
class TestForbiddenClaims:
    """六类禁用表述必须全部检出，合规表述不得误报。"""

    # ---------------- 第一类：疾病治疗 ----------------
    @pytest.mark.parametrize("text", [
        "这个能治疗糖尿病",
        "可以治愈高血压",
        "彻底根治失眠",
        "有很好的疗效",
        "对症下药就行",
        "能抗炎消炎",
        "可以降血糖",
        "降血脂效果好",
        "降血压有奇效",
        "能防癌抗癌",
    ])
    def test_disease_treatment_detected(self, text: str) -> None:
        """疾病治疗类必须检出 critical。"""
        from domain.rules.forbidden_claims import check_forbidden_claims

        result = check_forbidden_claims(text)
        assert result.has_critical, f"未检出疾病治疗类禁用词: {text}"
        assert result.max_severity is Severity.CRITICAL

    # ---------------- 第二类：药物替代 ----------------
    @pytest.mark.parametrize("text", [
        "这个可以替代药物",
        "吃了就不用吃药了",
        "可以停降压药了",
        "比药还管用",
        "药食同源没有副作用",
        "天然无害可以长期吃",
    ])
    def test_drug_replacement_detected(self, text: str) -> None:
        """药物替代类必须检出 critical（「可以停药」是最危险的一条）。"""
        from domain.rules.forbidden_claims import check_forbidden_claims

        result = check_forbidden_claims(text)
        assert result.has_critical, f"未检出药物替代类禁用词: {text}"

    # ---------------- 第三类：绝对化 ----------------
    @pytest.mark.parametrize("text", [
        "这是最有效的产品",
        "行业第一品牌",
        "唯一的解决方案",
        "彻底解决问题",
        "100% 有效",
        "包治百病",
        "三天见效",
        "一个疗程就见效",
    ])
    def test_absolute_claims_detected(self, text: str) -> None:
        """绝对化表述必须检出。"""
        from domain.rules.forbidden_claims import check_forbidden_claims

        result = check_forbidden_claims(text)
        assert result.has_critical, f"未检出绝对化表述: {text}"

    # ---------------- 第四类：效果承诺 ----------------
    @pytest.mark.parametrize("text", [
        "保证有效",
        "吃了没效果可以退款",
        "吃好了很多人",
    ])
    def test_effect_promise_detected(self, text: str) -> None:
        """效果承诺必须检出。"""
        from domain.rules.forbidden_claims import check_forbidden_claims

        result = check_forbidden_claims(text)
        assert result.has_critical, f"未检出效果承诺: {text}"

    # ---------------- 第五类：他人经验背书 ----------------
    @pytest.mark.parametrize("text", [
        "很多人吃了都说好",
        "我朋友吃了就有效",
        "客户反馈都说管用",
        "老客户都在回购",
    ])
    def test_testimonial_detected(self, text: str) -> None:
        """他人经验背书必须检出。

        这一类最容易被误认为「正常话术」，但同样违规。
        """
        from domain.rules.forbidden_claims import check_forbidden_claims

        result = check_forbidden_claims(text)
        assert result.has_critical, f"未检出经验背书: {text}"

    # ---------------- 第六类：缺「有助于」限定词 ----------------
    @pytest.mark.parametrize("text", [
        "增强免疫力",
        "改善睡眠",
        "增加骨密度",
    ])
    def test_missing_hedge_word_detected(self, text: str) -> None:
        """省略「有助于」就从「辅助」变成「保证」，必须检出。"""
        from domain.rules.forbidden_claims import check_forbidden_claims

        result = check_forbidden_claims(text)
        assert result.has_critical, f"未检出缺限定词表述: {text}"

    # ---------------- 合规表述不得误报 ----------------
    @pytest.mark.parametrize("text", [
        "有助于维持血糖健康水平",
        "有助于维持血脂（胆固醇/甘油三酯）健康水平",
        "有助于增强免疫力",
        "有助于改善睡眠",
        "有助于改善骨密度",
        "本品不能代替药物治疗疾病",
        "保健食品不是药物",
        "个体差异较大，不能承诺具体效果",
        "便秘人群不建议长期服用",
        "每份含蛋白质 15g",
    ])
    def test_compliant_text_not_flagged(self, text: str) -> None:
        """合规表述**绝不能**误报，否则护栏会过度拦截把正常业务堵死。"""
        from domain.rules.forbidden_claims import check_forbidden_claims

        result = check_forbidden_claims(text)
        assert not result.has_critical, (
            f"误报！这是合规表述却被判 critical: {text}\n"
            f"命中: {[v.matched for v in result.violations]}"
        )


# ===================================================================
# 2. 普通食品功能声称拦截（约束 #2）
# ===================================================================
class TestOrdinaryFoodClaim:
    """普通食品不得有任何功能声称。"""

    @pytest.mark.parametrize("text", [
        "有助于增强免疫力",
        "补充蛋白质效果好",
        "适合免疫力低的人",
        "改善皮肤弹性",
        "抗衰老",
        "养肝明目",
        "补肾",
        "乌发",
        "补钙",
        "减肥",
        "控卡",
    ])
    def test_ordinary_food_efficacy_blocked(self, text: str) -> None:
        """普通食品提到任何保健功能都必须拦截。

        注意：「有助于增强免疫力」对保健食品是合规的，
        对普通食品是违规的——所以判定必须结合产品类别。
        """
        from domain.rules.product_category import check_claim_allowed

        verdict = check_claim_allowed(ProductCategory.ORDINARY_FOOD, text)
        assert verdict.blocked, (
            f"普通食品的功能声称未拦截: {text}\n"
            f"这是约束 #2 的直接违反（《食品安全法》虚假宣传）"
        )

    @pytest.mark.parametrize("text", [
        "有助于增强免疫力",
        "有助于维持血脂健康水平",
    ])
    def test_health_food_efficacy_allowed(self, text: str) -> None:
        """同样的表述，保健食品是合规的（用目录原文）。"""
        from domain.rules.product_category import check_claim_allowed

        verdict = check_claim_allowed(ProductCategory.HEALTH_FOOD, text)
        assert not verdict.blocked

    def test_unknown_category_fails_safe(self) -> None:
        """未知类别必须按「不允许声称」处理。

        脏数据不能让普通食品蒙混过关。
        """
        from domain.rules.product_category import check_claim_allowed

        verdict = check_claim_allowed(None, "有助于增强免疫力")
        assert verdict.blocked, "未知类别必须 fail-safe 拦截"


# ===================================================================
# 3. 强制警示语（保健食品必须带）
# ===================================================================
class TestWarningText:
    """保健食品的回答必须包含法定警示语。"""

    WARNING = "保健食品不是药物，不能代替药物治疗疾病"

    def test_health_food_missing_warning_blocked(self) -> None:
        """保健食品回答缺警示语必须拦截。"""
        from domain.rules.warning_text import check_warning_present

        answer = "这款鱼油的批准功能是有助于维持血脂健康水平，适宜血脂偏高者。"
        result = check_warning_present(
            answer, ProductCategory.HEALTH_FOOD)

        assert not result.present
        assert result.blocked, "保健食品回答必须有法定警示语"

    def test_health_food_with_warning_passes(self) -> None:
        from domain.rules.warning_text import check_warning_present

        answer = (
            "这款鱼油的批准功能是有助于维持血脂健康水平。"
            f"需要说明的是，{self.WARNING}。"
        )
        result = check_warning_present(
            answer, ProductCategory.HEALTH_FOOD)
        assert result.present
        assert not result.blocked

    def test_ordinary_food_no_warning_needed(self) -> None:
        """普通食品不需要保健食品警示语（它本来就不该声称功效）。"""
        from domain.rules.warning_text import check_warning_present

        result = check_warning_present(
            "这是一款固体饮料，每份含蛋白质 12g。",
            ProductCategory.ORDINARY_FOOD)
        assert not result.blocked

    def test_warning_variants_accepted(self) -> None:
        """警示语的等价表述应被接受，不能只认逐字一致。"""
        from domain.rules.warning_text import check_warning_present

        variants = [
            "保健食品不是药物，不能代替药物治疗疾病",
            "本品不能代替药物",
            "不能代替药物治疗疾病",
            "不是药物，不能代替药物治疗疾病",
        ]
        for v in variants:
            result = check_warning_present(
                f"产品说明……{v}。", ProductCategory.HEALTH_FOOD)
            assert result.present, f"等价警示语未被识别: {v}"


# ===================================================================
# 4. 输出护栏（集成三项检测）
# ===================================================================
class TestOutputGuardrail:
    """护栏是输出路径的必经节点。"""

    def test_blocks_forbidden_claim(self) -> None:
        from domain.services.output_guardrail import OutputGuardrail

        verdict = OutputGuardrail().check(
            "这个能降血糖，很多人吃了都说好",
            category=ProductCategory.HEALTH_FOOD,
        )
        assert verdict.blocked, "含禁用词的回答必须拦截"
        assert verdict.reason

    def test_blocks_ordinary_food_claim(self) -> None:
        from domain.services.output_guardrail import OutputGuardrail

        verdict = OutputGuardrail().check(
            "这款固体饮料有助于增强免疫力",
            category=ProductCategory.ORDINARY_FOOD,
        )
        assert verdict.blocked

    def test_blocks_missing_warning(self) -> None:
        from domain.services.output_guardrail import OutputGuardrail

        verdict = OutputGuardrail().check(
            "这款鱼油有助于维持血脂健康水平。",
            category=ProductCategory.HEALTH_FOOD,
        )
        assert verdict.blocked
        assert "警示语" in verdict.reason or "警示" in verdict.reason

    def test_passes_compliant_answer(self) -> None:
        """完全合规的回答必须放行——不能过度拦截。"""
        from domain.services.output_guardrail import OutputGuardrail

        answer = (
            "康源牌鱼油软胶囊是保健食品，批准的功能是"
            "「有助于维持血脂（胆固醇/甘油三酯）健康水平」。"
            "适宜人群是血脂偏高者；不适宜人群包括少年儿童、孕妇、乳母、"
            "出血倾向者、肝功能不全者、海产品过敏者。"
            "保健食品不是药物，不能代替药物治疗疾病。"
        )
        verdict = OutputGuardrail().check(
            answer, category=ProductCategory.HEALTH_FOOD)
        assert not verdict.blocked, (
            f"合规回答被误拦: {verdict.reason}\n"
            f"命中: {verdict.violations}"
        )

    def test_ordinary_food_plain_answer_passes(self) -> None:
        """普通食品的纯描述性回答应放行。"""
        from domain.services.output_guardrail import OutputGuardrail

        answer = (
            "康源牌复合蛋白固体饮料是普通食品，"
            "每份含蛋白质 12g，配料是乳清蛋白粉和麦芽糊精。"
        )
        verdict = OutputGuardrail().check(
            answer, category=ProductCategory.ORDINARY_FOOD)
        assert not verdict.blocked

    def test_verdict_has_no_score(self) -> None:
        """护栏判定不得含 score/verdict 分数字段（约束 #3）。"""
        from domain.services.output_guardrail import OutputGuardrail

        verdict = OutputGuardrail().check(
            "有助于维持血脂健康水平。保健食品不是药物，不能代替药物治疗疾病。",
            category=ProductCategory.HEALTH_FOOD,
        )
        # `blocked` 是布尔判断，不是分数
        assert isinstance(verdict.blocked, bool)
        assert not hasattr(verdict, "score")
