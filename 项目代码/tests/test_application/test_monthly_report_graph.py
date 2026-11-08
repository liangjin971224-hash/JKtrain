"""月度报告图的测试（P2）。

★ 全部用**真实 LangGraph 图**跑，不 mock 图本身。
  mock 图等于只测「我调了哪个方法」，证明不了「图真的能挂起能恢复」——
  而后者恰恰是引入 LangGraph 的唯一理由。

覆盖六个必须成立的场景：
  1. 挂起时**不写库**（草稿不进业务库）
  2. 审批后写库，且状态是 approved
  3. 驳回 → 回 ⑥ 重生成建议（LLM 再调一次），仍不写库
  4. 驳回后审批 → 用新建议定稿
  5. 无薄弱项 → 跳过下钻、跳过 LLM
  6. 跨实例恢复（模拟进程重启）
"""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from application.graph.monthly_report_graph import (
    DECISION_APPROVE,
    DECISION_REJECT,
    build_report_graph,
    new_thread_id,
)
from domain.ports.repository_port import DrillReviewRow
from tests.fakes import FakeLanguageModelPort, FakeRepositoryPort

_SAFE = "建议重练 C10，第1轮第1句「降血糖」属于疾病治疗，需复习 D005。"


def _seed(repo: FakeRepositoryPort, *, weak: bool = True) -> None:
    """塞两条同一能力项的记录。weak=True 时分数低，构成薄弱项。"""
    hit = 1 if weak else 4
    high = 1 if weak else 0
    for did, cid, ct in [
        ("d1", "C10", "价格异议（price_objection）"),
        ("d2", "C09", "疗效质疑（efficacy_challenge）"),
    ]:
        repo.save_drill_review(
            DrillReviewRow(
                drill_id=did, trainee_id="E001", case_id=cid,
                case_type=ct, duration_sec=100,
                element_hit=hit, element_total=4,
                element_hit_all=hit, element_total_all=4,
                risk_count=high, high_count=high, critical_count=0,
                created_at="2026-10-10",
            ), ())


class _SeqLLM:
    """每次调用返回不同内容，用于验证「重生成产生了新建议」。"""

    def __init__(self) -> None:
        self.n = 0
        self.prompts: list[str] = []

    def generate(self, prompt, **kw):
        self.n += 1
        self.prompts.append(prompt)

        class R:
            text = f"第{_SeqLLM._n_of(self)}版建议：重练 C10，「降血糖」属于疾病治疗。"

        return R()

    @staticmethod
    def _n_of(obj) -> int:
        return obj.n


def _graph(repo, *, llm=None, saver=None):
    return build_report_graph(
        repository=repo,
        llm=llm,
        checkpointer=saver or InMemorySaver(),
        case_titles={"C10": "价格异议", "C09": "质疑效果"},
        doc_catalog={"D005": "功能声称规范与禁用表述"},
    )


def _start(g, tid="t1"):
    cfg = {"configurable": {"thread_id": tid}}
    g.invoke({"trainee_id": "E001", "period": "2026-10", "thread_id": tid}, cfg)
    return cfg


# ===================================================================
# 一、挂起不写库（最关键的一条）
# ===================================================================
class TestInterruptNoPersist:
    def test_pending_does_not_write_db(self) -> None:
        """挂起 = 草稿，**库里不能有定稿**。

        ★ 这是「Redis 存中间态 + MySQL 存定稿」方案的核心承诺。
          一旦这里写库，方案就退化成「MySQL 存全部」，
          失去分层意义，且草稿会被当成正式报告。
        """
        repo = FakeRepositoryPort()
        _seed(repo)
        g = _graph(repo, llm=FakeLanguageModelPort(reply=_SAFE))
        cfg = _start(g)

        st = g.get_state(cfg)
        assert st.next == ("await_review",)
        assert repo.get_monthly_report("E001", "2026-10") is None

    def test_interrupt_payload_has_key_fields(self) -> None:
        """挂起载荷要给主管看的信息：多少人、哪些薄弱、建议内容。"""
        repo = FakeRepositoryPort()
        _seed(repo)
        g = _graph(repo, llm=FakeLanguageModelPort(reply=_SAFE))
        cfg = {"configurable": {"thread_id": "t2"}}
        r = g.invoke({"trainee_id": "E001", "period": "2026-10",
                      "thread_id": "t2"}, cfg)

        itf = r.get("__interrupt__") or []
        assert itf, "应该有中断"
        payload = itf[0].value
        assert payload["trainee_id"] == "E001"
        assert payload["total_drills"] == 2
        assert payload["weak"], "薄弱项不能为空"
        assert payload["suggestion"]


# ===================================================================
# 二、审批 → 定稿
# ===================================================================
class TestApprove:
    def test_approve_persists(self) -> None:
        repo = FakeRepositoryPort()
        _seed(repo)
        g = _graph(repo, llm=FakeLanguageModelPort(reply=_SAFE))
        cfg = _start(g)

        g.invoke(Command(resume={"decision": DECISION_APPROVE,
                                 "reviewed_by": "M001",
                                 "note": "同意"}), cfg)

        row = repo.get_monthly_report("E001", "2026-10")
        assert row is not None
        assert row.status == "approved"
        assert row.reviewed_by == "M001"
        assert row.review_note == "同意"
        assert row.total_drills == 2
        assert row.weak_capabilities

    def test_persist_stores_three_layers(self) -> None:
        """定稿快照必须含三层结构（能力项 / 案例 / 违规明细）。"""
        repo = FakeRepositoryPort()
        _seed(repo)
        g = _graph(repo, llm=FakeLanguageModelPort(reply=_SAFE))
        cfg = _start(g)
        g.invoke(Command(resume={"decision": DECISION_APPROVE}), cfg)

        row = repo.get_monthly_report("E001", "2026-10")
        snap = row.report_json
        assert snap["capabilities"], "缺第 1 层能力项"
        assert snap["cases"], "缺第 2 层案例"
        assert "risks" in snap["cases"][0], "缺第 3 层违规明细"
        assert snap["suggestion"]["text"] == _SAFE

    def test_persist_keeps_thread_id(self) -> None:
        """thread_id 要落库，便于回溯 Redis 里的中间态。"""
        repo = FakeRepositoryPort()
        _seed(repo)
        g = _graph(repo, llm=FakeLanguageModelPort(reply=_SAFE))
        cfg = _start(g, tid="trace-1")
        g.invoke(Command(resume={"decision": DECISION_APPROVE}), cfg)

        row = repo.get_monthly_report("E001", "2026-10")
        assert row.thread_id == "trace-1"


# ===================================================================
# 三、驳回 → 回 ⑥ 重生成
# ===================================================================
class TestReject:
    def test_reject_regenerates_suggestion(self) -> None:
        """驳回后 LLM 应被再调一次，且建议内容变了。"""
        repo = FakeRepositoryPort()
        _seed(repo)
        llm = _SeqLLM()
        g = _graph(repo, llm=llm)
        cfg = _start(g)

        first = g.get_state(cfg).values["suggestion_text"]
        assert llm.n == 1

        g.invoke(Command(resume={"decision": DECISION_REJECT,
                                 "note": "太笼统"}), cfg)

        st = g.get_state(cfg)
        assert llm.n == 2, "驳回应触发重新生成"
        assert st.values["suggestion_text"] != first
        assert st.next == ("await_review",), "重生成后应回到复核"
        # ★ 驳回不写库
        assert repo.get_monthly_report("E001", "2026-10") is None

    def test_reject_then_approve_uses_new_suggestion(self) -> None:
        """驳回后再同意，定稿应用**新版**建议。"""
        repo = FakeRepositoryPort()
        _seed(repo)
        llm = _SeqLLM()
        g = _graph(repo, llm=llm)
        cfg = _start(g)

        g.invoke(Command(resume={"decision": DECISION_REJECT}), cfg)
        g.invoke(Command(resume={"decision": DECISION_APPROVE,
                                 "reviewed_by": "M001"}), cfg)

        row = repo.get_monthly_report("E001", "2026-10")
        assert row is not None
        assert "第2版" in row.suggestion, f"应用第2版，实际: {row.suggestion}"

    def test_review_feedback_separate_from_guardrail_note(self) -> None:
        """★ 主管驳回意见与校验失败原因是两个字段，不能混用。

        踩过的坑：一开始复用 `suggestion_note` 传驳回意见，
        但它在校验通过时会被置空 → 驳回后读不到意见，提示词里没有它。
        """
        repo = FakeRepositoryPort()
        _seed(repo)
        g = _graph(repo, llm=_SeqLLM())
        cfg = _start(g)

        # 首次校验通过，suggestion_note 应为空
        assert g.get_state(cfg).values.get("suggestion_note") == ""

        g.invoke(Command(resume={"decision": DECISION_REJECT,
                                 "note": "太笼统了"}), cfg)
        st = g.get_state(cfg)
        assert st.values["review_feedback"] == "太笼统了"
        assert st.values["review_note"] == "太笼统了"

    def test_reject_does_not_recompute_data(self) -> None:
        """★ 驳回不回上游重算数据（用户 2026-10-31 决策）。

        验证方式：驳回前后 `summary` 里的能力项分数不变。
        如果回上游重算，`operator.add` 型字段会翻倍（实测踩过）。
        """
        repo = FakeRepositoryPort()
        _seed(repo)
        g = _graph(repo, llm=_SeqLLM())
        cfg = _start(g)

        before = g.get_state(cfg).values["summary"]["capabilities"]
        g.invoke(Command(resume={"decision": DECISION_REJECT}), cfg)
        after = g.get_state(cfg).values["summary"]["capabilities"]

        assert before == after, "驳回不该重算数据"

    def test_reject_note_passed_to_prompt(self) -> None:
        """驳回意见要进提示词，模型才知道要改什么。"""
        repo = FakeRepositoryPort()
        _seed(repo)
        llm = _SeqLLM()
        g = _graph(repo, llm=llm)
        cfg = _start(g)
        g.invoke(Command(resume={"decision": DECISION_REJECT,
                                 "note": "别只说要重练，要说怎么练"}), cfg)

        assert "主管的修改要求" in llm.prompts[-1]
        assert "别只说要重练" in llm.prompts[-1]


# ===================================================================
# 四、无薄弱项
# ===================================================================
class TestNoWeak:
    def test_no_weak_skips_llm(self) -> None:
        """无薄弱项不调 LLM（硬凑建议就是让模型编）。"""
        repo = FakeRepositoryPort()
        _seed(repo, weak=False)
        llm = _SeqLLM()
        g = _graph(repo, llm=llm)
        cfg = _start(g)

        assert llm.n == 0
        st = g.get_state(cfg)
        assert "无明显问题" in st.values["suggestion_text"]

    def test_no_weak_skips_drilldown(self) -> None:
        """无薄弱项时**下钻节点根本不被执行**。

        ★ 注意断言方式：`drilldown` 键在状态里**不存在**（是 None），
          不是空列表——因为条件边直接跳过了 `drill_down` 节点，
          节点没跑就不会写入这个字段。

          这恰恰证明了「条件边真的生效了」：
          如果下钻跑了，键存在且是空列表；
          如果没跑，键不存在。两种状态是可区分的。
        """
        repo = FakeRepositoryPort()
        _seed(repo, weak=False)
        g = _graph(repo, llm=FakeLanguageModelPort(reply=_SAFE))
        cfg = _start(g)

        st = g.get_state(cfg)
        assert st.values.get("drilldown") is None, "下钻节点不该被执行"

    def test_empty_month(self) -> None:
        """没有演练记录也要能出报告（不崩）。"""
        repo = FakeRepositoryPort()
        g = _graph(repo, llm=FakeLanguageModelPort(reply=_SAFE))
        cfg = _start(g)

        st = g.get_state(cfg)
        assert st.next == ("await_review",)
        assert st.values["summary"]["overview"]["total_drills"] == 0

    def test_empty_month_approve_still_persists(self) -> None:
        """空月份的定稿也要能落库（0 次演练也是一份合法报告）。"""
        repo = FakeRepositoryPort()
        g = _graph(repo, llm=NilLLM())
        cfg = _start(g)
        g.invoke(Command(resume={"decision": DECISION_APPROVE}), cfg)

        row = repo.get_monthly_report("E001", "2026-10")
        assert row is not None
        assert row.total_drills == 0


class NilLLM:
    def generate(self, *a, **kw):
        class R:
            text = ""

        return R()


# ===================================================================
# 五、跨实例恢复（模拟进程重启）
# ===================================================================
class TestCrossInstance:
    def test_resume_with_new_graph_instance(self) -> None:
        """★ 用**新图实例**恢复（共享同一个 checkpointer）。

        这模拟了「主管隔天审批」：进程重启，内存态全丢，
        只剩存储里的 checkpoint。
        """
        repo = FakeRepositoryPort()
        _seed(repo)
        saver = InMemorySaver()

        g1 = _graph(repo, llm=FakeLanguageModelPort(reply=_SAFE), saver=saver)
        cfg = _start(g1, tid="restart-1")

        # 新实例
        g2 = _graph(repo, llm=FakeLanguageModelPort(reply=_SAFE), saver=saver)
        st = g2.get_state(cfg)
        assert st.next == ("await_review",), "新实例应能读到挂起状态"

        g2.invoke(Command(resume={"decision": DECISION_APPROVE,
                                  "reviewed_by": "M002"}), cfg)
        row = repo.get_monthly_report("E001", "2026-10")
        assert row is not None
        assert row.reviewed_by == "M002"

    def test_rebuild_report_from_state(self) -> None:
        """dataclass 不进 State，但节点要能从 summary 重建出对象。"""
        from application.graph.monthly_report_graph import _rebuild_report

        summary = {
            "overview": {"total_drills": 3, "total_critical": 1},
            "capabilities": [{
                "key": "E_objection", "name": "异议处理", "samples": 2,
                "score": 52.5, "level": "weak", "level_label": "薄弱",
                "critical_count": 0, "note": "",
            }],
            "cases": [{
                "drill_id": "d1", "case_id": "C10", "case_title": "价格异议",
                "capability": "E_objection", "case_type": "价格异议",
                "score": 45.0, "score_note": "", "element_hit": 1,
                "element_total": 4, "risk_count": 1, "critical_count": 0,
                "created_at": "2026-10-10",
                "risks": [{"location": "第1轮第1句", "level": "high"}],
            }],
            "weak_capabilities": ["E_objection"],
        }
        r = _rebuild_report("E001", "2026-10", summary)

        assert r.total_drills == 3
        assert r.capabilities[0].capability == "E_objection"
        assert r.capabilities[0].score == 52.5
        assert len(r.outcomes) == 1
        assert r.outcomes[0].risks[0]["location"] == "第1轮第1句"
        assert r.weak_capabilities == ("E_objection",)


# ===================================================================
# 六、thread_id
# ===================================================================
class TestThreadId:
    def test_thread_id_contains_context(self) -> None:
        """thread_id 前两段要能看出「谁的哪个月」，便于排查。"""
        tid = new_thread_id("E001", "2026-10")
        assert tid.startswith("E001:2026-10:")
        assert len(tid.split(":")[2]) == 8

    def test_thread_id_unique(self) -> None:
        a = new_thread_id("E001", "2026-10")
        b = new_thread_id("E001", "2026-10")
        assert a != b


class TestRewriteCounting:
    """★ 真实 bug（2026-10-31）：驳回后 rewrites 仍是 0。

    最初只在校验失败分支累加 `rewrites`，主管驳回走的是另一条边，
    没有经过 `bump_rewrites`。结果「驳回 3 次」在状态里看不出来，
    前端也没法显示「已重写 N 次」。

    `rewrites` 的语义由此明确：**建议被重写了多少次，不区分谁要求的**。
    """

    def test_reject_increments_rewrites(self) -> None:
        repo = FakeRepositoryPort()
        _seed(repo)
        g = _graph(repo, llm=_SeqLLM())
        cfg = _start(g)

        assert g.get_state(cfg).values.get("rewrites", 0) == 0

        g.invoke(Command(resume={"decision": DECISION_REJECT}), cfg)
        assert g.get_state(cfg).values.get("rewrites") == 1

        g.invoke(Command(resume={"decision": DECISION_REJECT}), cfg)
        assert g.get_state(cfg).values.get("rewrites") == 2

    def test_approve_does_not_increment(self) -> None:
        repo = FakeRepositoryPort()
        _seed(repo)
        g = _graph(repo, llm=_SeqLLM())
        cfg = _start(g)
        g.invoke(Command(resume={"decision": DECISION_APPROVE}), cfg)

        assert g.get_state(cfg).values.get("rewrites", 0) == 0
