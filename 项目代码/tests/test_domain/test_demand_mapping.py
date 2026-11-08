"""口语需求 → 目录功能映射的测试（2026-10-24）。

────────────────────────────────────────────────────────────────────
本文件守两个真实缺陷
────────────────────────────────────────────────────────────────────
**缺陷一：有货却说没货**（最伤信任）
    问「改善精力」，规则映射为空 → 系统答「资料里没有」。
    而本店 P008 西洋参含片、P009 人参胶囊恰恰声称
    「缓解体力疲劳」（目录第 7 项）。

**缺陷二：把"资料里没有"当成"法规不允许"**
    问「提升关节舒适度」，系统答「资料里没有这个信息」。
    真相是**27 项批准功能目录里根本没有关节这一项**。
    对学员来说这两句话意义完全不同：
        「资料里没有」    → 我再翻翻别的资料
        「法规没有这一项」 → 这句不能说，说了就违规
    合规培训系统必须说后者。

────────────────────────────────────────────────────────────────────
为什么不再用「3 次投票」校验 LLM（实测否掉的方案）
────────────────────────────────────────────────────────────────────
真实千问实测（2026-10-24）：

    需求            temp=0.0        temp=0.8        一致性
    改善精力        缓解体力疲劳×3   缓解体力疲劳×3    3/3
    提升关节舒适度   NONE×3          NONE×3          3/3

`temperature=0.0` 是确定性的，跑 3 次拿到的「一致」不是高置信，
只是同一个计算重复了 3 次。真正可能产生分歧的做法（调高温度）
等于故意注入噪声来检验稳定性，逻辑自相矛盾。

所以改成三层校验，本文件重点守 L1（词表封闭）与 L2（反向验证）。
"""
from __future__ import annotations

import pytest

from application.services.semantic_efficacy import (
    NONE_TOKEN,
    SemanticEfficacyMatcher,
    boolean_prompt,
    match_prompt,
    normalize_to_catalog,
)
from domain.services.demand_matcher import (
    absent_reason,
    demand_terms,
    guess_efficacy_ids,
)

#: 真实目录快照（从 `jiankangshujuku` 取，15 项）
CATALOG: list[tuple[str, str]] = [
    ("EFF_01", "有助于增强免疫力"),
    ("EFF_02", "有助于抗氧化"),
    ("EFF_03", "辅助改善记忆"),
    ("EFF_06", "有助于改善睡眠"),
    ("EFF_07", "缓解体力疲劳"),
    ("EFF_09", "有助于控制体内脂肪"),
    ("EFF_10", "有助于改善骨密度"),
    ("EFF_15", "有助于调节肠道菌群"),
    ("EFF_17", "有助于润肠通便"),
    ("EFF_19", "有助于维持血脂（胆固醇/甘油三酯）健康水平"),
    ("EFF_20", "有助于维持血糖健康水平"),
    ("EFF_21", "有助于维持血压健康水平"),
    ("EFF_NUT_CA", "补充钙、维生素D"),
    ("EFF_NUT_MV", "补充多种维生素和矿物质"),
    ("EFF_NUT_VC", "补充维生素C"),
]


# ===================================================================
# 1. 精力类：规则必须命中（缺陷一）
# ===================================================================
class TestEnergyMapping:
    """客户在门店说的是「精力/没精神/容易累」，不是「体力疲劳」。"""

    @pytest.mark.parametrize("text", [
        "改善精力",
        "想改善精力",
        "提升精力",
        "没精神",
        "容易累",
        "容易疲劳",
        "无精打采",
        "总是乏力",
        "想提神",
    ])
    def test_energy_words_map_to_eff_07(self, text: str) -> None:
        ids = guess_efficacy_ids(text, CATALOG)
        assert "EFF_07" in ids, (
            f"「{text}」没映射到「缓解体力疲劳」（EFF_07）。\n"
            f"  后果：本店 P008 西洋参、P009 人参声称这项功能，"
            f"系统却答「资料里没有」——有货说没货。"
        )

    def test_standard_word_still_works(self) -> None:
        """标准说法不能因为补口语而失效（回归）。"""
        assert "EFF_07" in guess_efficacy_ids("缓解疲劳", CATALOG)


# ===================================================================
# 2. 关节类：必须明确「目录里没有」，而不是含糊（缺陷二）
# ===================================================================
class TestAbsentDemands:
    @pytest.mark.parametrize("text", [
        "提升关节舒适度",
        "关节疼",
        "关节不舒服",
        "软骨磨损",
        "韧带",
    ])
    def test_joint_is_marked_absent(self, text: str) -> None:
        reason = absent_reason(text)
        assert reason, (
            f"「{text}」应命中断言「目录里没有关节类功能」。\n"
            f"  否则系统只能说「资料里没有」，学员会以为知识库没收录，"
            f"而不是「法规不允许这么宣称」。"
        )
        assert "27 项" in reason, "原因里必须点明这是法规层面的空缺"

    def test_absent_word_not_mapped_to_catalog(self) -> None:
        """关节类不能映射到任何目录项（包括骨密度）。"""
        ids = guess_efficacy_ids("提升关节舒适度", CATALOG)
        assert ids == [], (
            f"关节不该映射到任何批准功能，实际: {ids}。"
            f"骨密度是骨头，不是关节 —— 混为一谈就是推荐错产品"
        )

    @pytest.mark.parametrize("text", [
        "改善骨密度", "补钙", "关节酸痛想补骨密度",
    ])
    def test_legit_demands_not_marked_absent(self, text: str) -> None:
        """★ 双向：不能因为加了关节规则，把真正的骨密度需求也判成"没有"。"""
        if "骨密度" in text:
            assert "EFF_10" in guess_efficacy_ids(text, CATALOG)

    def test_absent_and_present_can_coexist(self) -> None:
        """一句话里既有能答的也有不能答的 —— 两边都要处理。"""
        q = "关节酸痛，想补骨密度"
        assert absent_reason(q), "关节部分应被标为目录无此项"
        assert "EFF_10" in guess_efficacy_ids(q, CATALOG), (
            "骨密度部分仍应正常映射 —— 不能因为一句里有关节就整句作废"
        )


# ===================================================================
# 3. L1 词表封闭：模型输出必须能归一化
# ===================================================================
class TestL1Normalization:
    @pytest.mark.parametrize("raw", [
        "缓解体力疲劳",
        "5. 缓解体力疲劳",
        "5、缓解体力疲劳",
        "（5）缓解体力疲劳",
        "- 缓解体力疲劳",
        "**缓解体力疲劳**",
        "  缓解体力疲劳  ",
        "缓解体力疲劳。",
    ])
    def test_noise_is_stripped(self, raw: str) -> None:
        """实测模型真实输出是 `5. 缓解体力疲劳|NONE`（带序号还自加后缀）。"""
        got = normalize_to_catalog(raw, CATALOG)
        assert got == ("EFF_07", "缓解体力疲劳"), f"{raw!r} 归一化失败: {got}"

    def test_model_self_appended_suffix(self) -> None:
        """★ 实测原样：模型自己拼了 `|NONE` """
        got = normalize_to_catalog("5. 缓解体力疲劳|NONE", CATALOG)
        assert got == ("EFF_07", "缓解体力疲劳")

    @pytest.mark.parametrize("raw", ["NONE", "none", "None", "无"])
    def test_none_recognized(self, raw: str) -> None:
        assert normalize_to_catalog(raw, CATALOG) == NONE_TOKEN

    def test_paraphrase_is_rejected(self) -> None:
        """★ 改写过的表述必须拒绝，不能"差不多就采纳"。

        批准功能只能用目录原文。若模型输出「改善精力」这种改写法，
        采纳了就会变成对外宣称一个不存在的批准功能。

        ★ 红灯记录（2026-10-24）：这组用例最初只有「改善精力」「养肝护肝」，
          而把归一化放宽成「包含就算」后它们**照样全部通过**——
          因为这两个词既不是任何目录名的子串、也不包含目录名，
          包含匹配同样命中不到，测不出放宽。
          补了下面两条「目录名的真子串」才让自证 FAIL：
              · 「体力疲劳」  是「缓解体力疲劳」的子串 → 包含匹配会误采纳
              · 「维生素」    是「补充维生素C」的子串
          教训：**用来验证「不放宽」的用例，必须是放宽后真的会变的那条**。
        """
        assert normalize_to_catalog("改善精力", CATALOG) is None
        assert normalize_to_catalog("养肝护肝", CATALOG) is None

    @pytest.mark.parametrize("raw", [
        "体力疲劳",        # 只是「缓解体力疲劳」的一部分，不是原文
        "维生素",          # 只是「补充维生素C」的一部分
        "免疫力",          # 只是「有助于增强免疫力」的一部分
        "改善骨密度",      # 少了「有助于」前缀 → 不是批准功能原文
    ])
    def test_partial_name_rejected(self, raw: str) -> None:
        """★ 只给了目录名的一部分，必须拒绝。

        这是「包含匹配」与「精确匹配」的分水岭：
        包含匹配认了这些，就等于允许对外宣称一个不存在的批准功能表述。
        """
        assert normalize_to_catalog(raw, CATALOG) is None, (
            f"{raw!r} 只是目录名的一部分，不是批准功能原文，不能采纳"
        )

    def test_paren_variant_accepted(self) -> None:
        """血脂那条目录原文带括号补充，客户/模型常省略。"""
        got = normalize_to_catalog("有助于维持血脂健康水平", CATALOG)
        assert got is not None and got[0] == "EFF_19"

    def test_empty_rejected(self) -> None:
        assert normalize_to_catalog("", CATALOG) is None
        assert normalize_to_catalog("   ", CATALOG) is None


# ===================================================================
# 4. 预算：红线（最多 1 次）
# ===================================================================
class _FakeLLM:
    def __init__(self, reply: str = "缓解体力疲劳") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)

        class R:
            text = self.reply
        return R()


class TestBudget:
    def test_only_one_call_by_default(self) -> None:
        """★ 红线：每个请求最多 1 次。

        第 2 次调用会返回预算不足，**不再碰模型**。
        L2 反向验证因此跳过（见 matcher 里"降级采纳"的注释）。
        """
        llm = _FakeLLM()
        m = SemanticEfficacyMatcher(llm, max_calls=1)
        r = m.match("改善精力", CATALOG)

        assert len(llm.prompts) == 1, (
            f"只允许 1 次调用，实际 {len(llm.prompts)} 次。"
            f"每多一次就多 637ms 延迟与一份成本"
        )
        assert r.efficacy_id == "EFF_07"
        assert r.verified is True
        assert m.calls_used == 1
        assert m.budget_left == 0

    def test_budget_zero_never_calls(self) -> None:
        """预算 0 = 完全不调（用于纯规则模式/测试环境）。"""
        llm = _FakeLLM()
        m = SemanticEfficacyMatcher(llm, max_calls=0)
        r = m.match("改善精力", CATALOG)

        assert llm.prompts == []
        assert not r.usable
        assert "预算" in r.reason

    def test_second_match_blocked(self) -> None:
        """同一实例第 2 次 match 不再调模型（预算不重复给）。"""
        llm = _FakeLLM()
        m = SemanticEfficacyMatcher(llm, max_calls=1)
        m.match("改善精力", CATALOG)
        r2 = m.match("改善睡眠", CATALOG)

        assert len(llm.prompts) == 1
        assert not r2.usable

    def test_reset_gives_new_budget(self) -> None:
        llm = _FakeLLM()
        m = SemanticEfficacyMatcher(llm, max_calls=1)
        m.match("改善精力", CATALOG)
        m.reset_budget()
        assert m.budget_left == 1

    def test_no_llm_returns_unusable(self) -> None:
        m = SemanticEfficacyMatcher(None, max_calls=1)
        assert not m.match("改善精力", CATALOG).usable

    def test_empty_catalog_never_calls(self) -> None:
        """没有目录就没有约束，模型的输出无从校验 —— 不调。"""
        llm = _FakeLLM()
        m = SemanticEfficacyMatcher(llm, max_calls=1)
        assert not m.match("改善精力", []).usable
        assert llm.prompts == []


# ===================================================================
# 5. L2 反向验证（预算允许时）
# ===================================================================
class _SeqLLM:
    """按顺序返回预设答案，模拟「先映射再验证」。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        text = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]

        class R:
            pass
        r = R()
        r.text = text
        return r


class TestL2ReverseCheck:
    def test_agreement_accepted(self) -> None:
        llm = _SeqLLM(["5. 缓解体力疲劳", "YES"])
        m = SemanticEfficacyMatcher(llm, max_calls=2)
        r = m.match("改善精力", CATALOG)

        assert r.usable and r.efficacy_id == "EFF_07"
        assert r.checks.get("L2", {}).get("agreed") is True

    def test_disagreement_rejected(self) -> None:
        """★ 反向验证说 NO 就必须放弃 —— 这是防"映射错了"的关键一层。

        投票测的是「模型稳不稳」，这一层测的是「这个映射对不对」。
        后者才是本项目要防的东西。
        """
        llm = _SeqLLM(["有助于改善骨密度", "NO"])
        m = SemanticEfficacyMatcher(llm, max_calls=2)
        r = m.match("关节不舒服", CATALOG)

        assert not r.usable, "反向验证不通过却采用了映射"
        assert "反向验证未通过" in r.reason

    def test_reverse_prompt_asks_for_sameness(self) -> None:
        """★ 问法必须是「是否同一件事」，不能是「是否相关」。

        「相关」太宽：骨密度和关节也算"相关"，那就没有区分力了。
        """
        p = boolean_prompt("改善精力", "缓解体力疲劳")
        assert "同一件" in p
        assert "YES" in p and "NO" in p

    def test_ambiguous_answer_treated_as_no(self) -> None:
        """答非 YES 一律不采用 —— 这一层的失败方向必须保守。"""
        llm = _SeqLLM(["缓解体力疲劳", "可能是相关的吧"])
        m = SemanticEfficacyMatcher(llm, max_calls=2)
        assert not m.match("改善精力", CATALOG).usable


# ===================================================================
# 6. 提示词约束
# ===================================================================
class TestPrompt:
    def test_catalog_is_closed_set(self) -> None:
        p = match_prompt("改善精力", CATALOG)
        for _, name in CATALOG:
            assert name in p, f"目录项「{name}」没进提示词，模型无法从封闭列表选"
        assert NONE_TOKEN in p, "不允许答 NONE 的话，模型会硬凑一项出来"

    def test_no_number_leak_requirement(self) -> None:
        """提示词要求"原样输出、不要加序号"（虽然模型常不遵守，L1 兜住）。"""
        p = match_prompt("改善精力", CATALOG)
        assert "原样输出" in p


# ===================================================================
# 7. demand_terms 回归
# ===================================================================
class TestDemandTerms:
    @pytest.mark.parametrize("q,expect", [
        ("改善体脂都可以吃什么保健品", "改善体脂"),
        ("补钙的有什么推荐", "补钙"),
    ])
    def test_terms_extracted(self, q: str, expect: str) -> None:
        assert expect in demand_terms(q)
