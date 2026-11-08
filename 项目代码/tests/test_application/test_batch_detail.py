"""批量任务明细 + 薄弱项可读性的测试。

★ 两个真实问题（2026-10-31 用户截图发现）：

  ① **批量进度只给数字，没有明细**
     进度条显示「20/20 完成」，但 `failed=1` 时无法定位到人，
     主管只能整批重跑。页面里 `rpBatchList` 是个空容器 ——
     我建了占位却没渲染，等于留了个 TODO 忘了做。

  ② **薄弱能力项显示英文 key**
     中断载荷里 `weak` 只存 `C_contraindication` 这种 key，
     前端直接 join 显示，主管看到的是一串标识符，
     等于没告诉他"哪里弱"。
"""
from __future__ import annotations

import pytest

from application.graph.monthly_report_graph import build_report_graph
from domain.ports.repository_port import DrillReviewRow
from domain.services.monthly_report import CAPABILITY_META
from tests.fakes import FakeLanguageModelPort, FakeRepositoryPort


def _seed(repo: FakeRepositoryPort) -> None:
    for did, cid, ct, high in [
        ("d1", "C07", "禁忌询问（contraindication）", 1),
        ("d2", "C13", "配伍询问（incompatibility）", 1),
    ]:
        repo.save_drill_review(
            DrillReviewRow(
                drill_id=did, trainee_id="E001", case_id=cid,
                case_type=ct, duration_sec=100,
                element_hit=1, element_total=3,
                element_hit_all=1, element_total_all=3,
                risk_count=high, high_count=high, critical_count=0,
                created_at="2026-10-10",
            ), ())


def _graph(repo):
    from langgraph.checkpoint.memory import InMemorySaver

    return build_report_graph(
        repository=repo,
        llm=FakeLanguageModelPort(reply="建议重练 C07，「降血糖」属于疾病治疗。"),
        checkpointer=InMemorySaver(),
        case_titles={"C07": "抗凝药", "C13": "配伍"},
    )


class TestWeakCapsReadable:
    """薄弱项必须给人能看懂的名字。"""

    def test_interrupt_payload_has_names(self) -> None:
        """★ 中断载荷里 `weak_caps` 要带中文名。"""
        repo = FakeRepositoryPort()
        _seed(repo)
        g = _graph(repo)
        cfg = {"configurable": {"thread_id": "w1"}}
        r = g.invoke({"trainee_id": "E001", "period": "2026-10",
                      "thread_id": "w1"}, cfg)

        payload = (r.get("__interrupt__") or [None])[0].value
        assert "weak_caps" in payload, "缺少 weak_caps，前端只能显示英文 key"
        caps = payload["weak_caps"]
        assert caps, "应有薄弱项"

        for c in caps:
            assert c["key"], "key 不能空"
            assert c["name"] != c["key"], (
                f"name 不该等于 key（实际 {c['name']}）—— "
                "说明没查到中文名，前端会显示标识符")

    def test_name_matches_domain_meta(self) -> None:
        """中文名必须来自 `CAPABILITY_META`（唯一来源），不能前端另抄一份。"""
        repo = FakeRepositoryPort()
        _seed(repo)
        g = _graph(repo)
        cfg = {"configurable": {"thread_id": "w2"}}
        r = g.invoke({"trainee_id": "E001", "period": "2026-10",
                      "thread_id": "w2"}, cfg)

        for c in (r.get("__interrupt__") or [None])[0].value["weak_caps"]:
            expected = CAPABILITY_META.get(c["key"], {}).get("name")
            assert c["name"] == expected

    def test_weak_keys_still_present(self) -> None:
        """`weak`（key 列表）要保留 —— 脚本和历史调用依赖它。"""
        repo = FakeRepositoryPort()
        _seed(repo)
        g = _graph(repo)
        cfg = {"configurable": {"thread_id": "w3"}}
        r = g.invoke({"trainee_id": "E001", "period": "2026-10",
                      "thread_id": "w3"}, cfg)
        payload = (r.get("__interrupt__") or [None])[0].value
        assert isinstance(payload["weak"], list)
        assert payload["weak"], "weak 不该为空"


class TestBatchTaskDetail:
    """批量任务明细：能定位到人。"""

    def test_list_report_tasks_returns_rows(self) -> None:
        repo = FakeRepositoryPort()
        for no in ("E001", "E002", "E003"):
            repo.upsert_report_task(
                task_id=f"{no}:2026-10", trainee_id=no,
                period="2026-10", status="pending_review")
        rows = repo.list_report_tasks(period="2026-10")
        assert len(rows) == 3
        assert all(r["trainee_id"] for r in rows)

    def test_filter_by_status(self) -> None:
        """失败的任务要能单独筛出来（主管只看有问题的）。"""
        repo = FakeRepositoryPort()
        repo.upsert_report_task(task_id="E001:2026-10", trainee_id="E001",
                                period="2026-10", status="pending_review")
        repo.upsert_report_task(task_id="E002:2026-10", trainee_id="E002",
                                period="2026-10", status="failed",
                                error="LLM timeout")
        failed = repo.list_report_tasks(period="2026-10", status="failed")
        assert len(failed) == 1
        assert failed[0]["trainee_id"] == "E002"
        assert "timeout" in failed[0]["last_error"]
