"""月度报告用例的测试（P1 同步版）。

重点测三件事：
  1. 无薄弱项时**不调 LLM**（避免硬凑建议）
  2. 建议被护栏拦截时能重写，且拦不住时如实标记
  3. LLM 挂掉时**数据部分依然可用**（报告不能因 LLM 而不可用）
"""
from __future__ import annotations

import pytest

from application.use_cases.monthly_report import (
    MonthlyReportUseCase,
    build_suggestion_prompt,
)
from tests.fakes import FakeLanguageModelPort, FakeRepositoryPort
from domain.ports.repository_port import DrillReviewRow

_SAFE_REPLY = ("建议重练 C07 第1轮第1句的应答，那一句出现了功效声称。"
               "本周请复习第三章人群适配部分。")
_BAD_REPLY = "你可以告诉客户这个产品能降血脂，效果很好。"


def _review(drill_id: str, case_id: str, case_type: str, *,
            hit=4, total=4, high=0, medium=0, critical=0,
            created_at="2026-10-10") -> DrillReviewRow:
    return DrillReviewRow(
        drill_id=drill_id, trainee_id="E001", case_id=case_id,
        case_type=case_type, duration_sec=120,
        element_hit=hit, element_total=total,
        element_hit_all=hit, element_total_all=total,
        risk_count=high + medium + critical,
        high_count=high, medium_count=medium, critical_count=critical,
        created_at=created_at,
    )


def _repo(*reviews) -> FakeRepositoryPort:
    r = FakeRepositoryPort()
    for x in reviews:
        r.save_drill_review(x, ())
    return r


class TestNoWeak:
    """没有薄弱项时不该调 LLM。"""

    def test_no_llm_call_when_all_good(self) -> None:
        llm = FakeLanguageModelPort(reply=_SAFE_REPLY)
        repo = _repo(
            _review("d1", "C01", "产品信息询问（product_existence）"),
            _review("d2", "C02", "产品信息询问（product_ingredient）"),
        )
        uc = MonthlyReportUseCase(repository=repo, llm=llm)
        res = uc.execute("E001", period="2026-10")

        assert res.suggestion.skipped is True
        assert "无明显问题" in res.suggestion.text

    def test_empty_month(self) -> None:
        llm = FakeLanguageModelPort(reply=_SAFE_REPLY)
        uc = MonthlyReportUseCase(repository=_repo(), llm=llm)
        res = uc.execute("E001", period="2026-10")

        assert res.report.total_drills == 0
        assert res.suggestion.skipped is True


class TestSuggestionGeneration:
    def test_generates_when_weak(self) -> None:
        llm = FakeLanguageModelPort(reply=_SAFE_REPLY)
        repo = _repo(
            _review("d1", "C10", "价格异议（price_objection）",
                    hit=2, total=4, high=1),
            _review("d2", "C09", "疗效质疑（efficacy_challenge）",
                    hit=1, total=4, high=1),
        )
        uc = MonthlyReportUseCase(repository=repo, llm=llm)
        res = uc.execute("E001", period="2026-10")

        assert res.report.has_weak is True
        assert res.suggestion.text
        assert res.suggestion.passed is True
        assert res.suggestion.skipped is False

    def test_prompt_binds_cases(self) -> None:
        """提示词必须包含具体案例与违规位置，否则模型会写空话。"""
        repo = _repo(
            _review("d1", "C07", "禁忌询问（contraindication）",
                    hit=1, total=4, critical=1),
            _review("d2", "C07", "禁忌询问（contraindication）",
                    hit=1, total=4, critical=1),
        )
        uc = MonthlyReportUseCase(repository=repo, llm=None)
        res = uc.execute("E001", period="2026-10")
        prompt = build_suggestion_prompt(res.report)

        assert "C07" in prompt
        assert "禁忌与配伍" in prompt
        assert "必须引用具体案例编号" in prompt
        assert "不许出现任何产品功效表述" in prompt

    def test_blocked_suggestion_rewritten(self) -> None:
        """第一次输出违规 → 触发重写。"""

        class _SeqLLM:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt, **kw):
                self.calls += 1

                class R:
                    text = _BAD_REPLY if self.calls == 1 else _SAFE_REPLY

                return R()

        llm = _SeqLLM()
        repo = _repo(
            _review("d1", "C03", "功效导向询问（efficacy_inquiry）",
                    hit=1, total=4, high=1),
            _review("d2", "C03", "功效导向询问（efficacy_inquiry）",
                    hit=1, total=4, high=1),
        )
        uc = MonthlyReportUseCase(repository=repo, llm=llm, max_rewrites=1)
        res = uc.execute("E001", period="2026-10")

        assert llm.calls == 2, "应重写一次"
        assert res.suggestion.retries == 1
        assert res.suggestion.passed is True

    def test_still_blocked_after_rewrites(self) -> None:
        """重写后仍违规 → 如实标记未通过，但**报告数据仍在**。"""
        llm = FakeLanguageModelPort(reply=_BAD_REPLY)
        repo = _repo(
            _review("d1", "C03", "功效导向询问（efficacy_inquiry）",
                    hit=1, total=4, high=1),
            _review("d2", "C03", "功效导向询问（efficacy_inquiry）",
                    hit=1, total=4, high=1),
        )
        uc = MonthlyReportUseCase(repository=repo, llm=llm, max_rewrites=1)
        res = uc.execute("E001", period="2026-10")

        assert res.suggestion.passed is False
        assert res.suggestion.reason
        # ★ 数据部分必须完好
        assert res.report.total_drills == 2
        assert res.summary["capabilities"]

    def test_llm_down_data_still_available(self) -> None:
        """LLM 挂掉 → 报告数据必须仍然可用。"""

        class _Boom:
            def generate(self, *a, **kw):
                raise RuntimeError("llm down")

        repo = _repo(
            _review("d1", "C10", "价格异议（price_objection）",
                    hit=1, total=4, high=1),
            _review("d2", "C10", "价格异议（price_objection）",
                    hit=1, total=4, high=1),
        )
        uc = MonthlyReportUseCase(repository=repo, llm=_Boom())
        res = uc.execute("E001", period="2026-10")

        assert res.report.total_drills == 2
        assert res.suggestion.text == ""
        assert res.summary["overview"]["total_drills"] == 2


class TestSummaryShape:
    def test_three_layers_present(self) -> None:
        repo = _repo(
            _review("d1", "C07", "禁忌询问（contraindication）",
                    hit=2, total=4, critical=1),
            _review("d2", "C07", "禁忌询问（contraindication）",
                    hit=2, total=4, critical=1),
        )
        uc = MonthlyReportUseCase(repository=repo, llm=None)
        res = uc.execute("E001", period="2026-10", )
        s = res.summary

        # 第 1 层
        assert "capabilities" in s and s["capabilities"]
        assert s["capabilities"][0]["key"] == "C_contraindication"
        assert s["capabilities"][0]["level"] == "red_line"
        # 第 2 层
        assert "cases" in s and len(s["cases"]) == 2
        # 第 3 层（违规明细，这里没有 risk_items 所以是空列表）
        assert "risks" in s["cases"][0]
        # 薄弱项
        assert "C_contraindication" in s["weak_capabilities"]

    def test_case_titles_used(self) -> None:
        repo = _repo(_review("d1", "C07", "禁忌询问（contraindication）"))
        uc = MonthlyReportUseCase(
            repository=repo, llm=None,
            case_titles={"C07": "客户在服抗凝药询问能否吃鱼油"})
        res = uc.execute("E001")
        assert res.summary["cases"][0]["case_title"] == "客户在服抗凝药询问能否吃鱼油"

    def test_period_from_since(self) -> None:
        uc = MonthlyReportUseCase(repository=_repo(), llm=None)
        res = uc.execute("E001", since="2026-10-01", until="2026-11-01")
        assert res.report.period == "2026-10"

    def test_time_filter_passed_to_repo(self) -> None:
        class _SpyRepo(FakeRepositoryPort):
            def __init__(self):
                super().__init__()
                self.args = None

            def list_drill_reviews(self, **kw):
                self.args = kw
                return []

        spy = _SpyRepo()
        uc = MonthlyReportUseCase(repository=spy, llm=None)
        uc.execute("E001", since="2026-10-01", until="2026-11-01")

        assert spy.args["trainee_id"] == "E001"
        assert spy.args["since"] == "2026-10-01"
        assert spy.args["until"] == "2026-11-01"
        assert spy.args["with_items"] is True


class TestDocCatalog:
    """★ 真实 bug（2026-10-31）：不给资料清单时模型会编造资料名。

    实测它写了「《禁忌话术禁用词表（V3.2）》第4章」
    「《配伍禁忌应答四步法》第2章」「《健康顾问合规守则》第1.3条」——
    三份都不存在，学员按建议根本找不到东西。
    """

    def _weak_repo(self):
        return _repo(
            _review("d1", "C07", "禁忌询问（contraindication）",
                    hit=1, total=4, critical=1),
            _review("d2", "C07", "禁忌询问（contraindication）",
                    hit=1, total=4, critical=1),
        )

    def test_catalog_listed_in_prompt(self) -> None:
        uc = MonthlyReportUseCase(repository=self._weak_repo(), llm=None)
        res = uc.execute("E001")
        catalog = {
            "D005": "保健食品功能声称规范与禁用表述",
            "D009": "客户常见异议处理指引",
        }
        prompt = build_suggestion_prompt(res.report, catalog)

        assert "可引用的资料清单" in prompt
        assert "D005" in prompt
        assert "D009" in prompt
        assert "不得编造其他资料名" in prompt
        assert "不要写章号/条款号" in prompt

    def test_no_catalog_forbids_names(self) -> None:
        """没有目录时，必须明确禁止提资料名与章号。"""
        uc = MonthlyReportUseCase(repository=self._weak_repo(), llm=None)
        res = uc.execute("E001")
        prompt = build_suggestion_prompt(res.report, None)

        assert "不要提及任何具体资料名称" in prompt
        assert "章号或条款号" in prompt

    def test_usecase_passes_catalog(self) -> None:
        """用例要把 catalog 传进 prompt。"""
        captured = {}

        class _SpyLLM:
            def generate(self, prompt, **kw):
                captured["prompt"] = prompt

                class R:
                    text = _SAFE_REPLY

                return R()

        uc = MonthlyReportUseCase(
            repository=self._weak_repo(), llm=_SpyLLM(),
            doc_catalog={"D014": "门店突发情况应急处理"})
        uc.execute("E001")

        assert "D014" in captured["prompt"]
        assert "门店突发情况应急处理" in captured["prompt"]
