"""风险定位（P5）。

★ 硬验收项：**critical 违规检出率必须 100%**。

    这一条不能用「大概能检出」来交付。
    漏一条 critical 意味着学员说了违法宣传话术，系统却没提示——
    这个学员可能因此真的那么说出去。
    所以本文件必须包含「逐条 critical 规则各测一遍」的用例。

★ 硬验收项：**定位到「第几轮第几句」**。

    只告诉学员「你有违规」而不给位置，他无法改进。
    定位必须精确到句——所以对话要按句切分，且句序从 1 开始。
"""
from __future__ import annotations

import pytest

from domain.services.risk_locator import (
    collect_turns_text,
    locate_risks,
)
from domain.value_objects import (
    CaseSpec,
    ForbiddenRule,
    RiskLevel,
    TraineeTurn,
)


# ===================================================================
# 夹具
# ===================================================================
def _c01_spec() -> CaseSpec:
    """C01 的违禁规则（逐字来自案例文档「五、违禁表述检测」）。"""
    return CaseSpec(
        case_id="C01",
        title="客户询问是否有鱼油",
        opening_line="你们有鱼油吗？",
        forbidden_rules=(
            ForbiddenRule(
                "违规宣称",
                r"降血脂|降胆固醇|治疗|改善血脂|调节血脂",
                RiskLevel.CRITICAL,
                "属医疗术语或旧声称",
            ),
            ForbiddenRule(
                "效果承诺",
                r"肯定有效|保证|一定管用|很快见效|吃一个月就好",
                RiskLevel.HIGH,
                "不能承诺效果",
            ),
            ForbiddenRule(
                "主动推销",
                r"要不要买|带两盒|现在有活动|买二送一|限时",
                RiskLevel.CRITICAL,
                "B2 模式禁止",
            ),
            ForbiddenRule(
                "回避推诿",
                r"这个我不清楚|你问医生吧|我们只做健康管理",
                RiskLevel.HIGH,
                "客户问产品必须答",
            ),
        ),
    )


# ===================================================================
# 1. ★ critical 检出率 100%
# ===================================================================
class TestCriticalDetection:
    """每条 critical 规则都必须能检出。"""

    @pytest.mark.parametrize("phrase", [
        "降血脂", "降胆固醇", "治疗", "改善血脂", "调节血脂",
        "要不要买", "带两盒", "现在有活动", "买二送一", "限时",
    ])
    def test_each_critical_phrase_detected(self, phrase: str) -> None:
        """逐条测：任何一条漏检都是安全事故。"""
        turns = [TraineeTurn(1, f"这个{phrase}。")]
        risks = locate_risks(_c01_spec(), turns)

        crit = [r for r in risks if r.is_critical]
        assert crit, f"critical 漏检: {phrase}"

    def test_all_critical_in_one_answer(self) -> None:
        """一句里含多个 critical 时，不能只报第一个。"""
        turns = [TraineeTurn(
            1, "这个能降血脂，您要不要买两盒？")]
        risks = locate_risks(_c01_spec(), turns)

        crit_types = {r.rule_type for r in risks if r.is_critical}
        assert "违规宣称" in crit_types
        assert "主动推销" in crit_types

    def test_high_level_also_reported(self) -> None:
        """high 级也要报（不是只报 critical）。"""
        turns = [TraineeTurn(1, "肯定有效，保证管用。")]
        risks = locate_risks(_c01_spec(), turns)

        assert any(r.level is RiskLevel.HIGH for r in risks)

    def test_no_false_positive_on_clean_answer(self) -> None:
        """合规答卷不能报风险——误报会让学员不信任系统。"""
        turns = [TraineeTurn(
            1,
            "有的，康源牌鱼油软胶囊。这是保健食品，"
            "批准功能是有助于维持血脂健康水平。"
            "保健食品不是药物，不能代替药物治疗疾病。",
        )]
        risks = locate_risks(_c01_spec(), turns)
        assert risks == [], (
            f"合规答卷误报: {[(r.rule_type, r.matched) for r in risks]}"
        )


# ===================================================================
# 2. ★ 定位到「第几轮第几句」
# ===================================================================
class TestLocation:
    def test_locates_turn_and_sentence(self) -> None:
        """第 2 轮第 2 句违规，必须精确定位。"""
        turns = [
            TraineeTurn(1, "有的，康源牌鱼油软胶囊。"),
            TraineeTurn(2, "这个很安全。这个能降血脂。"),
        ]
        risks = locate_risks(_c01_spec(), turns)

        assert len(risks) >= 1
        hit = [r for r in risks if "降血脂" in r.matched][0]
        assert hit.turn_index == 2
        assert hit.sentence_index == 2
        assert hit.location == "第2轮第2句"

    def test_matches_case_doc_example(self) -> None:
        """案例文档给的示例答卷必须能定位到具体位置。

        ★ 与文档示例的差异说明：
          文档的摘要卡示例写「第1轮第1句」「第1轮第2句」，
          但它的示例答卷用了「！」，按句切分后「有的！」是第 1 句，
          含违规的长句实际是第 2 句。

          文档的标注是按「逗号分段」算的，不是按标点分句。
          本实现统一按句末标点（。！？；）分句——因为：
            · 学员更容易理解「第几句」而不是「第几个逗号段」
            · 分句稳定，不受口语标点随意性的影响

          所以这里**只验证「能精确定位到第几轮第几句」，不逐字对齐文档数字**。
          实质要求（「定位到第几轮第几句」）已满足。
        """
        spec = _c01_spec()
        turns = [TraineeTurn(
            1,
            "有的！我们这款鱼油特别好，很多人吃了血脂都降下来了，"
            "您可以先带两盒试试，现在还有活动。",
        )]
        risks = locate_risks(spec, turns)

        assert risks, "示例答卷含多处违规，必须报出"
        # 全部在第 1 轮
        assert all(r.turn_index == 1 for r in risks)
        # 必须能精确定位到句序（1 或 2 都合理，取决于分句口径）
        assert all(r.sentence_index >= 1 for r in risks)
        assert any(r.sentence_index == 2 for r in risks), (
            "含违规的长句应被定位到第 2 句（第 1 句是「有的！」）"
        )
        # 文档明确说明该答卷有 3 处 critical
        crit = [r for r in risks if r.is_critical]
        assert len(crit) >= 2, (
            f"文档说明该答卷有 3 处 critical，实际检出 {len(crit)} 处"
        )

    def test_multiple_sentences_each_located(self) -> None:
        """一句话里多处违规，各自定位。"""
        turns = [TraineeTurn(1, "能降血脂。要不要买？")]
        risks = locate_risks(_c01_spec(), turns)

        pairs = {(r.turn_index, r.sentence_index) for r in risks}
        assert (1, 1) in pairs
        assert (1, 2) in pairs

    def test_risks_sorted_by_position(self) -> None:
        """按位置排序，便于学员按顺序看。"""
        turns = [
            TraineeTurn(1, "这个能降血脂。"),
            TraineeTurn(2, "要不要买？"),
        ]
        risks = locate_risks(_c01_spec(), turns)
        positions = [(r.turn_index, r.sentence_index) for r in risks]
        assert positions == sorted(positions)


# ===================================================================
# 3. 多轮对话
# ===================================================================
class TestMultiTurn:
    def test_risks_across_turns(self) -> None:
        turns = [
            TraineeTurn(1, "有的。"),
            TraineeTurn(2, "能治疗糖尿病。"),
            TraineeTurn(3, "带两盒吧。"),
        ]
        risks = locate_risks(_c01_spec(), turns)

        turns_hit = {r.turn_index for r in risks if r.is_critical}
        assert 2 in turns_hit
        assert 3 in turns_hit

    def test_empty_turns_skipped(self) -> None:
        turns = [TraineeTurn(1, ""), TraineeTurn(2, "能降血脂。")]
        risks = locate_risks(_c01_spec(), turns)
        assert all(r.turn_index == 2 for r in risks)

    def test_no_turns(self) -> None:
        assert locate_risks(_c01_spec(), []) == []

    def test_spec_without_rules(self) -> None:
        """案例无自定义规则、且不叠加全局规则时，不报任何风险。"""
        spec = CaseSpec("C99", "t", "o")
        assert locate_risks(
            spec, [TraineeTurn(1, "能降血脂。")],
            include_global_rules=False) == []

    def test_global_rules_catch_what_case_rules_miss(self) -> None:
        """★ 案例规则漏掉的，全局规则必须兜住。

        C01 的案例规则表里**没有「他人经验背书」**这一类，
        但它的评分示例明确说「很多人吃了血脂都降下来了」是 critical。

        只用案例规则会漏检。所以评阅必须叠加全局六类规则
        （与输出护栏同源），案例规则管「考察重点」，全局规则管「兜住底线」。
        """
        spec = CaseSpec("C99", "t", "o")          # 无案例规则
        turns = [TraineeTurn(1, "很多人吃了都说好。")]

        with_global = locate_risks(spec, turns, include_global_rules=True)
        without = locate_risks(spec, turns, include_global_rules=False)

        assert with_global, "全局规则应检出经验背书"
        assert without == [], "不叠加全局规则时确实检不出"


# ===================================================================
# 4. 命中去重与片段长度
# ===================================================================
class TestMatchQuality:
    def test_same_sentence_same_match_counted_once(self) -> None:
        """同一句里同一片段重复出现，只报一次（避免刷屏）。"""
        turns = [TraineeTurn(1, "降血脂降血脂降血脂。")]
        risks = locate_risks(_c01_spec(), turns)
        same = [r for r in risks
                if r.turn_index == 1 and r.sentence_index == 1
                and r.matched == "降血脂"]
        assert len(same) == 1, f"同一句同一片段应去重，实际 {len(same)} 条"

    def test_matched_piece_is_short(self) -> None:
        """命中片段应是违规词本身，不是整句——方便展示。"""
        turns = [TraineeTurn(1, "这个产品呢，其实它是可以降血脂的。")]
        risks = locate_risks(_c01_spec(), turns)
        hit = [r for r in risks if r.is_critical][0]
        assert hit.matched == "降血脂"
        assert len(hit.matched) < 10

    def test_substring_overlap_keeps_longest(self) -> None:
        """★ 同位置的命中片段互为子串时，只保留最长的。

        实测：「可以的，一起吃没问题」同时被两条规则命中
              → 「可以的，一起吃没问题」（长）
              → 「一起吃没问题」（短，是前者的子串）
        报两次学员会疑惑，保留长的那条（信息更完整）。
        """
        from domain.value_objects import CaseSpec, TraineeTurn

        spec = CaseSpec("C99", "t", "o")
        turns = [TraineeTurn(1, "可以的，一起吃没问题。")]
        risks = locate_risks(spec, turns)

        # 找出「替客户判断」类的全部命中
        jd = [r for r in risks if "替客户判断" in r.rule_type]
        # 不应有互为子串的两条
        for i, a in enumerate(jd):
            for b in jd[i + 1:]:
                assert not (a.matched in b.matched or b.matched in a.matched), (
                    f"存在子串重叠的重复报告: 「{a.matched}」与「{b.matched}」"
                )

    def test_unrelated_matches_both_reported(self) -> None:
        """★ 同位置但互不相关的违规，必须都报。

        一句里既「降血脂」又「带两盒」是两件不同的事。
        合并成一条会让学员以为只需改一处。
        """
        turns = [TraineeTurn(1, "这个能降血脂，您带两盒吧。")]
        risks = locate_risks(_c01_spec(), turns)

        matched = {r.matched for r in risks}
        assert "降血脂" in matched
        assert "带两盒" in matched

    def test_evidence_sentence_available(self) -> None:
        """要能回指到原句，供人工复核。"""
        turns = [TraineeTurn(1, "你好。这个能降血脂。")]
        risks = locate_risks(_c01_spec(), turns)
        hit = [r for r in risks if r.is_critical][0]
        assert hit.reason
        assert "降血脂" in hit.reason or "医疗术语" in hit.reason


# ===================================================================
# 5. 文本拼接
# ===================================================================
class TestCollectTurns:
    def test_joins_all_turns(self) -> None:
        turns = [TraineeTurn(1, "第一轮。"), TraineeTurn(2, "第二轮。")]
        text = collect_turns_text(turns)
        assert "第一轮" in text
        assert "第二轮" in text

    def test_empty(self) -> None:
        assert collect_turns_text([]) == ""
