"""试卷接口测试（P6）。

★ 用假仓储 + 假题库，不连真实 MySQL / Milvus。

★ 但「假仓储能跑通」不代表真仓储能跑通，
  真实的读写另有 `test_mysql_repository.py` 覆盖（标 integration）。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from config.assembly import Container
from config.settings import Settings
from domain.exceptions import ExamBuildError
from domain.value_objects import ExamSpec, QuestionSnapshot
from domain.value_objects.exam import ExamStatus
from interfaces.http import deps
from interfaces.http.app import create_app


# ===================================================================
# 假题库 / 假仓储
# ===================================================================
def _snap(qid: str, kp: str, diff: int, product_id: str = "") -> QuestionSnapshot:
    return QuestionSnapshot(
        question_id=qid, knowledge_point=kp, kp_name=f"知识点{kp}",
        question_type="单选", difficulty=diff,
        stem=f"{qid} 的题干？",
        options=("选项A内容", "选项B内容", "选项C内容", "选项D内容"),
        answer="B", explanation=f"{qid} 的解析", product_id=product_id,
    )


def _bank() -> list[QuestionSnapshot]:
    out = []
    for k in range(3):
        kp = f"KP_{k}"
        for i in range(6):
            out.append(_snap(f"Q{k}{i}", kp, (i % 3) + 1))
    return out


class FakeRepo:
    """内存仓储（模拟 RepositoryPort 的试卷部分）。"""

    def __init__(self) -> None:
        self.papers: dict[str, dict] = {}
        self.usage: dict[str, set[str]] = {}

    # ---- 试卷 ----
    def save_exam_paper(self, paper: dict) -> str:
        no = paper["paper_no"]
        if no in self.papers:
            raise ValueError(f"卷号重复: {no}")
        stored = dict(paper)
        # 与真实仓储一致：created_at 缺省时补当前时间
        stored.setdefault("created_at", "2026-10-23 11:00:00")
        self.papers[no] = stored
        return no

    def get_exam_paper(self, paper_id: str) -> dict | None:
        p = self.papers.get(paper_id)
        if not p:
            return None
        out = dict(p)
        # 仓储的契约：返回的字典带 total_count（主表字段）
        out["total_count"] = len(p.get("questions", []))
        return out

    def list_exam_papers(self, *, stage: str = "", status: str = "",
                         limit: int = 50) -> list[dict]:
        out = []
        for p in self.papers.values():
            if stage and p.get("stage") != stage:
                continue
            if status and p.get("status") != status:
                continue
            out.append({
                "paper_no": p["paper_no"],
                "stage": p.get("stage", ""),
                "title": p.get("title", ""),
                "status": p.get("status", "draft"),
                "created_by": p.get("created_by", ""),
                "created_at": p.get("created_at", ""),
                "total_count": len(p.get("questions", [])),
            })
        return out[:limit]

    def update_exam_status(self, paper_no: str, status: str) -> None:
        if paper_no not in self.papers:
            raise ValueError(f"试卷不存在: {paper_no}")
        self.papers[paper_no]["status"] = status

    # ---- 使用记录 ----
    def record_question_usage(self, paper_no: str, question_ids: list[str],
                              *, trainee_id: str = "") -> None:
        self.usage.setdefault(trainee_id, set()).update(question_ids)

    def recent_used_question_ids(self, *, trainee_id: str = "",
                                 days: int = 30) -> set[str]:
        return set(self.usage.get(trainee_id, set()))


class FailingRepo(FakeRepo):
    """组卷必失败的仓储（验证错误处理）。"""

    def save_exam_paper(self, paper: dict) -> str:
        raise ExamBuildError("模拟：题库缺少 KP_X")


def _container(repo) -> Container:
    return Container(
        settings=Settings(_env_file=None, auth_required=False),  # type: ignore[call-arg]
        repository=repo,
    )


@asynccontextmanager
async def _noop_lifespan(app):
    yield


@pytest.fixture()
def make_client(monkeypatch):
    """建客户端，并注入假题库（绕开 Milvus）。"""
    from interfaces.http.routers import exam as exam_router

    def _make(repo=None):
        # 题库走假数据
        monkeypatch.setattr(
            exam_router, "_load_bank", lambda container, stage: _bank())
        app = create_app()
        app.router.lifespan_context = _noop_lifespan
        deps.set_container(_container(repo or FakeRepo()))
        # ★ 开发模式下按主管身份调用：组卷/试卷管理都需要 manager
        return TestClient(
            app, raise_server_exceptions=False,
            headers={"X-Debug-Role": "manager"})

    yield _make
    deps.set_container(None)


# ===================================================================
# 1. 组卷
# ===================================================================
class TestBuildExam:
    def test_build_ok(self, make_client) -> None:
        c = make_client()
        r = c.post("/api/exam/build", json={
            "total": 6, "stage": "S1", "title": "测试卷",
            "knowledge_points": ["KP_0", "KP_1"],
            "difficulty_quota": {"1": 2, "2": 2, "3": 2},
            "seed": 1,
        })
        assert r.status_code == 200
        p = r.json()
        assert p["total_count"] == 6
        assert p["status"] == "draft"
        assert set(p["covered_knowledge_points"]) == {"KP_0", "KP_1"}
        assert len(p["questions"]) == 6
        assert p["paper_no"].startswith("S1-")

    def test_difficulty_quota_respected(self, make_client) -> None:
        c = make_client()
        r = c.post("/api/exam/build", json={
            "total": 6, "stage": "S1",
            "knowledge_points": ["KP_0", "KP_1"],
            "difficulty_quota": {"1": 2, "2": 2, "3": 2},
            "seed": 1,
        })
        dist = r.json()["difficulty_distribution"]
        assert dist.get("1") == 2
        assert dist.get("2") == 2
        assert dist.get("3") == 2

    def test_missing_knowledge_point_400(self, make_client) -> None:
        """★ 缺知识点 → 400，且报错要说清缺哪个。"""
        c = make_client()
        r = c.post("/api/exam/build", json={
            "total": 4, "stage": "S1",
            "knowledge_points": ["KP_0", "KP_NOT_EXIST"],
        })
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "KP_NOT_EXIST" in detail, f"报错没指出缺哪个知识点: {detail}"

    def test_no_duplicate_across_two_papers(self, make_client) -> None:
        """★ 防重复：同一学员连组两卷，题目不重复。"""
        repo = FakeRepo()
        c = make_client(repo)

        body = {
            "total": 6, "stage": "S1",
            "knowledge_points": ["KP_0", "KP_1", "KP_2"],
            "trainee_id": "T1", "seed": 1,
        }
        p1 = c.post("/api/exam/build", json=body).json()
        ids1 = {q["question_id"] for q in p1["questions"]}

        body2 = dict(body, seed=2)
        p2 = c.post("/api/exam/build", json=body2).json()
        ids2 = {q["question_id"] for q in p2["questions"]}

        assert not (ids1 & ids2), f"两卷重复题: {ids1 & ids2}"

    def test_persist_false_not_saved(self, make_client) -> None:
        repo = FakeRepo()
        c = make_client(repo)
        r = c.post("/api/exam/build", json={
            "total": 3, "stage": "S1", "persist": False, "seed": 1,
        })
        assert r.status_code == 200
        assert not repo.papers, "persist=False 不应落库"

    def test_invalid_quota_key_400(self, make_client) -> None:
        c = make_client()
        r = c.post("/api/exam/build", json={
            "total": 3, "stage": "S1",
            "difficulty_quota": {"9": 1},      # 9 不是合法难度
        })
        assert r.status_code == 400


# ===================================================================
# 2. 列表与详情
# ===================================================================
class TestPaperList:
    def test_list_empty(self, make_client) -> None:
        c = make_client()
        r = c.get("/api/exam/papers")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_after_build(self, make_client) -> None:
        c = make_client()
        c.post("/api/exam/build", json={"total": 3, "stage": "S1", "seed": 1})
        r = c.get("/api/exam/papers")
        assert len(r.json()) == 1
        assert r.json()[0]["total_count"] == 3

    def test_detail_not_found(self, make_client) -> None:
        c = make_client()
        r = c.get("/api/exam/papers/NOT-EXIST")
        assert r.status_code == 404

    def test_detail_ok(self, make_client) -> None:
        c = make_client()
        no = c.post("/api/exam/build",
                    json={"total": 3, "stage": "S1", "seed": 1}).json()["paper_no"]
        r = c.get(f"/api/exam/papers/{no}")
        assert r.status_code == 200
        assert len(r.json()["questions"]) == 3


# ===================================================================
# 3. ★ 状态流转
# ===================================================================
class TestPaperStatus:
    def _build(self, c) -> str:
        return c.post("/api/exam/build",
                      json={"total": 3, "stage": "S1", "seed": 1}).json()["paper_no"]

    def test_draft_to_active(self, make_client) -> None:
        c = make_client()
        no = self._build(c)
        r = c.post(f"/api/exam/papers/{no}/status",
                   json={"paper_no": no, "target": "active"})
        assert r.status_code == 200
        assert r.json()["status"] == "active"

    def test_cannot_go_back(self, make_client) -> None:
        """★ 生效后不能退回草稿（FR-040 单向）。"""
        c = make_client()
        no = self._build(c)
        c.post(f"/api/exam/papers/{no}/status",
               json={"paper_no": no, "target": "active"})

        r = c.post(f"/api/exam/papers/{no}/status",
                   json={"paper_no": no, "target": "draft"})
        assert r.status_code == 409
        assert "回退" in r.json()["detail"] or "不能" in r.json()["detail"]

    def test_active_to_archived(self, make_client) -> None:
        c = make_client()
        no = self._build(c)
        for st in ("active", "archived"):
            r = c.post(f"/api/exam/papers/{no}/status",
                       json={"paper_no": no, "target": st})
            assert r.status_code == 200
        assert r.json()["status"] == "archived"

    def test_invalid_target_400(self, make_client) -> None:
        c = make_client()
        no = self._build(c)
        r = c.post(f"/api/exam/papers/{no}/status",
                   json={"paper_no": no, "target": "whatever"})
        assert r.status_code == 400


# ===================================================================
# 4. ★ 导出
# ===================================================================
class TestExport:
    def _paper(self, c, activate: bool = True) -> str:
        no = c.post("/api/exam/build",
                    json={"total": 3, "stage": "S1", "title": "测试卷",
                          "seed": 1}).json()["paper_no"]
        if activate:
            c.post(f"/api/exam/papers/{no}/status",
                   json={"paper_no": no, "target": "active"})
        return no

    def test_draft_cannot_export(self, make_client) -> None:
        """★ 草稿不能导出（还在变，导出没意义）。"""
        c = make_client()
        no = self._paper(c, activate=False)
        r = c.post(f"/api/exam/papers/{no}/export", json={})
        assert r.status_code == 409

    def test_active_can_export(self, make_client) -> None:
        c = make_client()
        no = self._paper(c)
        r = c.post(f"/api/exam/papers/{no}/export", json={})
        assert r.status_code == 200
        body = r.json()
        assert body["exam_markdown"]
        assert body["answer_markdown"]
        assert body["summary_markdown"]

    def test_exam_page_has_no_answer(self, make_client) -> None:
        """★★ 试卷页不能泄露答案。

        这是导出最关键的约束——试卷页是发给学员的。
        """
        c = make_client()
        no = self._paper(c)
        exam = c.post(f"/api/exam/papers/{no}/export", json={}).json()["exam_markdown"]

        # 试卷页不应出现答案标记
        assert "答案：" not in exam, "试卷页泄露了答案"
        assert "解析：" not in exam, "试卷页泄露了解析"
        # 但应有题干与选项
        assert "题干" in exam
        assert "A. " in exam

    def test_answer_key_has_answer(self, make_client) -> None:
        c = make_client()
        no = self._paper(c)
        key = c.post(f"/api/exam/papers/{no}/export", json={}).json()["answer_markdown"]
        assert "答案：" in key
        assert "解析：" in key

    def test_export_not_found_404(self, make_client) -> None:
        c = make_client()
        r = c.post("/api/exam/papers/NOT-EXIST/export", json={})
        assert r.status_code == 404

    def test_summary_has_kp_table(self, make_client) -> None:
        c = make_client()
        no = self._paper(c)
        s = c.post(f"/api/exam/papers/{no}/export", json={}).json()["summary_markdown"]
        assert "知识点覆盖" in s
        assert "难度分布" in s

    def test_export_after_reload(self, make_client) -> None:
        """★ 导出走的是「从库里读回 → 转 domain」，不是内存里的对象。

        这里验证这条路径不会因为类型转换出错（实测踩过：
        难度配比的键从 JSON 读回是字符串，导致 ValueError 500）。
        """
        repo = FakeRepo()
        c = make_client(repo)
        no = self._paper(c)

        # 模拟：换个客户端（等价于重启后从库读）
        c2 = make_client(repo)
        r = c2.post(f"/api/exam/papers/{no}/export", json={})
        assert r.status_code == 200, (
            f"从库读回后导出失败: {r.json()}"
        )
