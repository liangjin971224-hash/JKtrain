"""图谱召回的「产品名抽取」测试。

★ 全部来自用户实测暴露的真实 bug。

背景：
    用户在前端**不选产品**直接问「康源牌鱼油软胶囊有什么功能」，结果保健食品被拒答。

    链路排查：
        1. 图谱路把**整句**当别名 → `resolve_alias("康源牌鱼油软胶囊有什么功能")` → None
           （而 `resolve_alias("鱼油")` → P001 是能命中的）
        2. 图谱没命中 → 产品事实没注入 prompt
        3. 参考资料里只有「基础信息」，**没有批准功能原文**
        4. LLM 凭参数知识编出旧版功能名「辅助降血脂」
        5. 护栏拦下（护栏是对的，但根因是事实没喂进去）

    所以修的是**召回**，不是护栏。

★ 这个文件的重点是「不指定产品 ID」这条路径——
  P3/P4 的测试都显式传了 product_ids，所以从没覆盖到。
"""
from __future__ import annotations

import pytest

from application.use_cases.retrieve import RetrieveUseCase
from domain.value_objects import ProductFacts
from tests.fakes import FakeGraphQueryPort


# ===================================================================
# 1. 产品名抽取
# ===================================================================
class TestExtractCandidates:
    @pytest.mark.parametrize(("query", "expected_first"), [
        ("康源牌鱼油软胶囊有什么功能", "康源牌鱼油软胶囊"),
        ("康源牌褪黑素片多少钱", "康源牌褪黑素片"),
        ("鱼油有什么功能", "鱼油"),
        ("辅酶Q10对心脏好不好", "辅酶Q10"),
        ("我们这款蛋白粉怎么吃", "蛋白粉"),
        ("钙片适合什么人", "钙片"),
        ("黑芝麻丸有什么功能", "黑芝麻丸"),
        ("益生菌能长期吃吗", "益生菌"),
    ])
    def test_extracts_product_name(self, query: str,
                                   expected_first: str) -> None:
        candidates = RetrieveUseCase._extract_product_candidates(query)
        assert candidates, f"未抽到产品名: {query}"
        assert candidates[0] == expected_first, (
            f"首选候选应为 {expected_first}，实际 {candidates}"
        )

    @pytest.mark.parametrize("query", [
        "吃了能停降压药吗",
        "你们有没有量子能量水",
        "客户说太贵了怎么应对",
        "孕妇能吃吗",
    ])
    def test_no_product_returns_empty(self, query: str) -> None:
        """不含产品名的问句返回空，不能瞎猜。"""
        assert RetrieveUseCase._extract_product_candidates(query) == []

    def test_longest_first(self) -> None:
        """★ 长匹配优先，避免短名截胡。

        「康源牌鱼油软胶囊」应整体命中，而不是先被「鱼油」截胡。
        当前两者指向同一产品，但若将来有多个鱼油产品就会错。
        """
        candidates = RetrieveUseCase._extract_product_candidates(
            "康源牌鱼油软胶囊有什么功能")
        assert candidates[0] == "康源牌鱼油软胶囊"
        # 后面才轮到短名
        assert "鱼油" in candidates
        assert candidates.index("鱼油") > 0

    def test_dedup(self) -> None:
        candidates = RetrieveUseCase._extract_product_candidates(
            "鱼油和康源牌鱼油软胶囊是一回事吗")
        assert len(candidates) == len(set(candidates))

    def test_empty_query(self) -> None:
        assert RetrieveUseCase._extract_product_candidates("") == []


# ===================================================================
# 2. ★ 不指定产品 ID 也能命中图谱
# ===================================================================
class TestGraphHitWithoutProductIds:
    def _fish_oil(self) -> ProductFacts:
        from domain.value_objects import ApprovalClaim

        return ProductFacts(
            product_id="P001", name="康源牌鱼油软胶囊",
            category="保健食品", approval_no="国食健注G20250012",
            claims=(ApprovalClaim(
                efficacy_id="EFF_19",
                text="有助于维持血脂（胆固醇/甘油三酯）健康水平",
                catalog_no=19),),
            aliases=("鱼油",),
        )

    def test_whole_sentence_resolves_via_extraction(self) -> None:
        """★ 核心回归：整句问句也能命中图谱。

        这是用户踩到的那条路径——前端不选产品，问句里带产品全名。
        """
        graph = FakeGraphQueryPort({"P001": self._fish_oil()})
        uc = RetrieveUseCase(graph=graph)

        result = uc.execute(
            "康源牌鱼油软胶囊有什么功能",
            vector_collection="(none)",     # 无向量，只测图谱路
        )

        assert result.graph_hits, (
            "整句问句未能命中图谱——这会导致产品事实无法注入 prompt，"
            "LLM 会凭参数知识编造批准功能"
        )
        assert result.graph_hits[0].product_id == "P001"

    def test_alias_only_query(self) -> None:
        """只说「鱼油」也要命中。"""
        graph = FakeGraphQueryPort({"P001": self._fish_oil()})
        uc = RetrieveUseCase(graph=graph)
        result = uc.execute("鱼油有什么功能", vector_collection="(none)")
        assert result.graph_hits
        assert result.graph_hits[0].product_id == "P001"

    def test_explicit_product_id_still_works(self) -> None:
        """显式传 ID 的老路径不能坏。"""
        graph = FakeGraphQueryPort({"P001": self._fish_oil()})
        uc = RetrieveUseCase(graph=graph)
        result = uc.execute("有什么功能", product_ids=("P001",),
                            vector_collection="(none)")
        assert result.graph_hits
        assert result.graph_hits[0].product_id == "P001"

    def test_no_product_still_no_graph_hit(self) -> None:
        """不含产品名的问句不能硬凑图谱命中。"""
        graph = FakeGraphQueryPort({"P001": self._fish_oil()})
        uc = RetrieveUseCase(graph=graph)
        result = uc.execute("吃了能停降压药吗", vector_collection="(none)")
        assert result.graph_hits == ()

    def test_unknown_product_no_false_hit(self) -> None:
        """问一个库里没有的产品，不能返回别的产品。"""
        graph = FakeGraphQueryPort({"P001": self._fish_oil()})
        uc = RetrieveUseCase(graph=graph)
        result = uc.execute("量子能量水有什么功能", vector_collection="(none)")
        assert result.graph_hits == ()
