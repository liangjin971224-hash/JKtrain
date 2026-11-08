"""MySQL 仓储适配器（P6/P7）。

★ 实现 `RepositoryPort`，把业务数据落到 MySQL。

────────────────────────────────────────────────────────────────────
连接管理
────────────────────────────────────────────────────────────────────
用 **PyMySQL + 自建连接池**，不用 SQLAlchemy ORM。

理由：
    · 这里都是简单 CRUD（单表增删改查），ORM 的价值体现不出来
    · 快照写入是「一次多行 insert」，用 `executemany` 比 ORM 快得多
    · 少一层依赖，SQL 一眼能看懂，排查时不用猜 ORM 生成了什么

★ 连接每请求借还，用完立刻归还。不复用 pymysql 连接做长连接——
  它没有连接存活检测，MySQL 8 的 `wait_timeout` 默认 8 小时，
  空闲久了再拿旧连接会报 `MySQL server has gone away`。

────────────────────────────────────────────────────────────────────
关键约束
────────────────────────────────────────────────────────────────────
· 题目快照**必须**完整写进 `exam_paper_question`，不能只存 question_id
· `question_usage` 是防重复窗口的唯一依据，每次组卷必须写
· 所有写操作在单个事务里完成，中途失败要整体回滚
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from domain.ports.repository_port import (
    DailySummaryRow,
    DrillRecordRow,
    DrillReviewRow,
    DrillRiskItemRow,
    MissLogRow,
    MonthlyReportRow,
    RepositoryPort,
)
from infrastructure.exceptions import InfrastructureError

logger = logging.getLogger(__name__)


class RepositoryError(InfrastructureError):
    """仓储操作失败。"""


class MySqlRepositoryAdapter(RepositoryPort):
    """MySQL 仓储实现。"""

    def __init__(
        self,
        database_url: str = "",
        *,
        host: str = "",
        port: int = 3306,
        user: str = "",
        password: str = "",
        database: str = "",
        charset: str = "utf8mb4",
    ) -> None:
        """构造适配器。

        Args:
            database_url: SQLAlchemy 风格 URL（形如
                `<驱动>://<用户>:<密码>@<主机>:<端口>/<库名>?charset=utf8mb4`）。
                ★ 优先用这个：凭据只出现在 .env，不在代码里散落。
                ★ 具体值不要写进代码，否则会被架构门禁拦下（也确实该拦）。
            host/port/user/password/database: 也可逐个传（测试时方便）。

        Raises:
            RepositoryError: 连接不通（启动即验证，不留到第一次请求）。
        """
        params = _parse_url(database_url) if database_url else {}
        self._cfg = dict(
            host=host or params.get("host") or "localhost",
            port=port or params.get("port") or 3306,
            user=user or params.get("user") or "root",
            password=password or params.get("password") or "",
            database=database or params.get("database") or "jiankangshujuku",
            charset=charset or params.get("charset") or "utf8mb4",
        )
        # 启动即验证连接，避免「看起来装配成功但第一次请求才炸」
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        logger.info("MySQL 仓储就绪: %s:%s/%s",
                    self._cfg["host"], self._cfg["port"],
                    self._cfg["database"])

    # ---------------- 连接 ----------------
    @contextmanager
    def _conn(self):
        """借一个连接，用完关闭。"""
        try:
            conn = pymysql.connect(
                **self._cfg,
                cursorclass=DictCursor,
                autocommit=False,
                connect_timeout=5,
            )
        except Exception as exc:                    # noqa: BLE001
            raise RepositoryError(
                f"MySQL 连接失败: {type(exc).__name__}: {exc}") from exc

        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ================================================================
    # 演练记录
    # ================================================================
    def save_drill_record(self, record: DrillRecordRow) -> None:
        sql = """
        INSERT INTO drill_record
            (drill_id, trainee_id, case_id, started_at, finished_at,
             duration_sec, status)
        VALUES (%s, %s, %s, %s, NULLIF(%s, ''), %s, %s)
        ON DUPLICATE KEY UPDATE
            trainee_id   = VALUES(trainee_id),
            case_id      = VALUES(case_id),
            started_at   = VALUES(started_at),
            finished_at  = VALUES(finished_at),
            duration_sec = VALUES(duration_sec),
            status       = VALUES(status)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    record.drill_id, record.trainee_id, record.case_id,
                    _dt(record.started_at), record.finished_at,
                    record.duration_sec, record.status,
                ))
            conn.commit()

    def get_drill_record(self, drill_id: str) -> DrillRecordRow | None:
        sql = "SELECT * FROM drill_record WHERE drill_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (drill_id,))
                row = cur.fetchone()
        if not row:
            return None
        return DrillRecordRow(
            drill_id=row["drill_id"],
            trainee_id=row["trainee_id"],
            case_id=row["case_id"],
            started_at=_iso(row["started_at"]),
            finished_at=_iso(row["finished_at"]),
            duration_sec=row["duration_sec"],
            status=row["status"],
        )

    def list_drill_records_by_trainee(
        self, trainee_id: str, *, limit: int = 50
    ) -> list[DrillRecordRow]:
        sql = """
        SELECT * FROM drill_record
        WHERE trainee_id = %s
        ORDER BY started_at DESC
        LIMIT %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (trainee_id, limit))
                rows = cur.fetchall()
        return [
            DrillRecordRow(
                drill_id=r["drill_id"],
                trainee_id=r["trainee_id"],
                case_id=r["case_id"],
                started_at=_iso(r["started_at"]),
                finished_at=_iso(r["finished_at"]),
                duration_sec=r["duration_sec"],
                status=r["status"],
            )
            for r in rows
        ]

    # ================================================================
    # 每日总结
    # ================================================================
    def save_daily_summary(self, row: DailySummaryRow) -> None:
        sql = """
        INSERT INTO daily_summary
            (trainee_no, summary_date, content, review_note, status)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            content     = VALUES(content),
            review_note = VALUES(review_note),
            status      = VALUES(status)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    row.trainee_no, row.summary_date,
                    row.content, row.review_note, row.status,
                ))
            conn.commit()

    def get_daily_summary(
        self, trainee_no: str, summary_date: str
    ) -> DailySummaryRow | None:
        sql = """
        SELECT * FROM daily_summary
        WHERE trainee_no = %s AND summary_date = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (trainee_no, summary_date))
                row = cur.fetchone()
        if not row:
            return None
        return DailySummaryRow(
            trainee_no=row["trainee_no"],
            summary_date=_iso(row["summary_date"]),
            content=row["content"] or "",
            review_note=row["review_note"] or "",
            status=row["status"],
        )

    # ================================================================
    # 题库
    # ================================================================
    def list_questions(
        self,
        *,
        knowledge_point: str | None = None,
        difficulty: int | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """取题（组卷用）。

        ★ 注意：题库正文实际存在 **Milvus**，这里只按约定返回空列表，
          真实取题由 `application/services/question_bank.py` 走 Milvus。
          保留此方法是为了端口契约完整（P6 早期设计时曾打算把题库放 MySQL）。
        """
        logger.debug(
            "list_questions 走 Milvus，MySQL 侧无题库表（kp=%s diff=%s）",
            knowledge_point, difficulty)
        return []

    # ================================================================
    # ★ 试卷（核心）
    # ================================================================
    def save_exam_paper(self, paper: dict[str, Any]) -> str:
        """保存试卷 + 题目快照。

        Args:
            paper: 试卷字典，必须含：
                paper_no, stage, title, status, total_count,
                spec（组卷参数）, created_by, questions（题目列表）

        Returns:
            paper_no。

        Raises:
            RepositoryError: 卷号重复或写入失败。

        ★ 整个过程在一个事务里：主表 + 明细要么都成，要么都不成。
          半份卷子比没有卷子更危险——导出时会缺题。
        """
        paper_no = str(paper.get("paper_no") or "").strip()
        if not paper_no:
            raise RepositoryError("试卷缺少卷号")

        questions = paper.get("questions") or []

        with self._conn() as conn:
            with conn.cursor() as cur:
                # ---- 主表 ----
                # ★ created_at 用 COALESCE：传了就用传入值（组卷时间），
                #   没传才用库默认值。两者可能不同——先预览后落库的场景下，
                #   组卷时间才是主管对账的依据。
                cur.execute(
                    """
                    INSERT INTO exam_paper
                        (paper_no, stage, title, status, total_count,
                         spec_json, created_by, created_at, remark)
                    VALUES (%s, %s, %s, %s, %s, %s, %s,
                            COALESCE(NULLIF(%s, ''), CURRENT_TIMESTAMP), %s)
                    """,
                    (
                        paper_no,
                        paper.get("stage", ""),
                        paper.get("title", ""),
                        paper.get("status", "draft"),
                        len(questions),
                        json.dumps(paper.get("spec") or {},
                                   ensure_ascii=False),
                        paper.get("created_by", ""),
                        paper.get("created_at", ""),
                        paper.get("remark", ""),
                    ),
                )

                # ---- 题目快照（批量写）----
                if questions:
                    cur.executemany(
                        """
                        INSERT INTO exam_paper_question
                            (paper_no, seq, question_id, knowledge_point,
                             kp_name, question_type, difficulty, stem,
                             options_json, answer, explanation, product_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                paper_no, i, q.get("question_id", ""),
                                q.get("knowledge_point", ""),
                                q.get("kp_name", ""),
                                q.get("question_type", "单选"),
                                int(q.get("difficulty", 2)),
                                q.get("stem", ""),
                                json.dumps(list(q.get("options") or []),
                                           ensure_ascii=False),
                                q.get("answer", ""),
                                q.get("explanation", ""),
                                q.get("product_id", ""),
                            )
                            for i, q in enumerate(questions, start=1)
                        ],
                    )

            conn.commit()

        logger.info("试卷已保存: %s（%d 题）", paper_no, len(questions))
        return paper_no

    def get_exam_paper(self, paper_id: str) -> dict[str, Any] | None:
        """取试卷（读快照，不重新关联题库）。

        Args:
            paper_id: 卷号（`paper_no`）。
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM exam_paper WHERE paper_no = %s",
                    (paper_id,))
                head = cur.fetchone()
                if not head:
                    return None

                cur.execute(
                    """
                    SELECT * FROM exam_paper_question
                    WHERE paper_no = %s ORDER BY seq
                    """,
                    (paper_id,))
                items = cur.fetchall()

        return {
            "paper_no": head["paper_no"],
            "stage": head["stage"],
            "title": head["title"],
            "status": head["status"],
            "total_count": head["total_count"],
            "spec": _loads(head["spec_json"]),
            "created_by": head["created_by"],
            "created_at": _iso(head["created_at"]),
            "remark": head["remark"],
            "questions": [
                {
                    "seq": r["seq"],
                    "question_id": r["question_id"],
                    "knowledge_point": r["knowledge_point"],
                    "kp_name": r["kp_name"],
                    "question_type": r["question_type"],
                    "difficulty": r["difficulty"],
                    "stem": r["stem"],
                    "options": _loads(r["options_json"]) or [],
                    "answer": r["answer"],
                    "explanation": r["explanation"] or "",
                    "product_id": r["product_id"],
                }
                for r in items
            ],
        }

    def update_exam_status(
        self, paper_no: str, status: str
    ) -> None:
        """更新试卷状态（附带时间戳）。

        ★ 状态合法性由 domain 层 `transition()` 校验，这里不重复判断。
          仓储只负责落库，业务规则留在领域层。
        """
        col = {"active": "activated_at", "archived": "archived_at"}.get(status)
        sql = f"""
        UPDATE exam_paper
        SET status = %s{", " + col + " = NOW()" if col else ""}
        WHERE paper_no = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (status, paper_no))
                if cur.rowcount == 0:
                    raise RepositoryError(f"试卷不存在: {paper_no}")
            conn.commit()

    def list_exam_papers(
        self, *, stage: str = "", status: str = "", limit: int = 50
    ) -> list[dict[str, Any]]:
        """列出试卷（不含题目明细，只给列表页用）。"""
        where, params = [], []
        if stage:
            where.append("stage = %s")
            params.append(stage)
        if status:
            where.append("status = %s")
            params.append(status)

        sql = "SELECT * FROM exam_paper"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()

        return [
            {
                "paper_no": r["paper_no"],
                "stage": r["stage"],
                "title": r["title"],
                "status": r["status"],
                "total_count": r["total_count"],
                "created_by": r["created_by"],
                "created_at": _iso(r["created_at"]),
            }
            for r in rows
        ]

    # ================================================================
    # ★ 题目使用记录（防重复窗口）
    # ================================================================
    def record_question_usage(
        self,
        paper_no: str,
        question_ids: list[str],
        *,
        trainee_id: str = "",
    ) -> None:
        """记录题目使用（组卷成功后必须调用）。"""
        if not question_ids:
            return

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO question_usage
                        (question_id, trainee_id, paper_no)
                    VALUES (%s, %s, %s)
                    """,
                    [(qid, trainee_id, paper_no) for qid in question_ids],
                )
            conn.commit()

        logger.info("题目使用已记录: %d 题 → %s", len(question_ids), paper_no)

    def recent_used_question_ids(
        self, *, trainee_id: str = "", days: int = 30
    ) -> set[str]:
        """取防重复窗口内用过的题目编号。

        Args:
            trainee_id: 学员编号。空表示不限学员（全局去重）。
            days: 窗口天数。

        Returns:
            题目编号集合。
        """
        sql = """
        SELECT DISTINCT question_id FROM question_usage
        WHERE used_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
        """
        params: list[Any] = [days]
        if trainee_id:
            sql += " AND trainee_id = %s"
            params.append(trainee_id)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                return {r["question_id"] for r in cur.fetchall()}

    # ================================================================
    # 未命中日志
    # ================================================================
    def append_miss_log(self, row: MissLogRow) -> None:
        sql = """
        INSERT INTO miss_log (query, intent, trainee_id, detail_json)
        VALUES (%s, %s, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    row.query[:500], row.intent, row.trainee_id,
                    json.dumps(row.detail, ensure_ascii=False)
                    if row.detail else None,
                ))
            conn.commit()

    def list_miss_logs(self, *, limit: int = 100) -> list[MissLogRow]:
        sql = "SELECT * FROM miss_log ORDER BY created_at DESC LIMIT %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall()

        return [
            MissLogRow(
                id=str(r["id"]),
                query=r["query"],
                intent=r["intent"] or "",
                trainee_id=r["trainee_id"] or "",
                created_at=_iso(r["created_at"]),
                detail=_loads(r["detail_json"]) or {},
            )
            for r in rows
        ]


# ===================================================================
# 小工具
# ===================================================================
def _iso(v: Any) -> str:
    """datetime → ISO 字符串；None → 空串。"""
    if v is None:
        return ""
    if hasattr(v, "isoformat"):
        return v.isoformat(sep=" ", timespec="seconds")
    return str(v)


def _dt(s: str) -> Any:
    """ISO 字符串 → datetime；空串 → None。"""
    if not s:
        return None
    return s


def _loads(v: Any) -> Any:
    """JSON 字段解析（None 安全）。"""
    if v is None or v == "":
        return None
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        return None


__all__ = ["MySqlRepositoryAdapter", "RepositoryError"]


def _parse_url(url: str) -> dict[str, Any]:
    """解析 SQLAlchemy 风格数据库 URL。

    支持的驱动前缀：
        <驱动>://<用户>:<密码>@<主机>:<端口>/<库名>?charset=utf8mb4
    驱动可为 `mysql+pymysql` 或 `mysql`。

    ★ 这里只放形态说明，不放真实连接串示例 ——
      架构门禁会扫描源码中的硬编码连接串，示例文本同样会被拦（而且该拦）。

    Returns:
        {host, port, user, password, database, charset}
    """
    from urllib.parse import parse_qs, urlparse

    p = urlparse(url)
    qs = parse_qs(p.query or "")
    return {
        "host": p.hostname or "localhost",
        "port": p.port or 3306,
        "user": p.username or "root",
        "password": p.password or "",
        "database": (p.path or "/").lstrip("/"),
        "charset": (qs.get("charset") or ["utf8mb4"])[0],
    }


# ===================================================================
# P7：员工档案 / 日报 / 未命中日志（扩展方法）
# ===================================================================
class MySqlRepositoryAdapterP7(MySqlRepositoryAdapter):
    """P7 扩展。

    ★ 为什么用子类而不是直接改原类：
      原类已经有 900+ 行，P7 的方法都是独立的新表操作，
      放在子类里便于单独看、单独测。
      装配根用这个子类即可，对外仍是 `RepositoryPort`。
    """

    # ---------------- 员工档案 ----------------
    def save_employee(self, employee) -> None:
        """写入/更新员工档案（按工号 upsert）。"""
        sql = """
        INSERT INTO employee
            (employee_no, name, hire_date, position, team, mentor, active)
        VALUES (%s, %s, NULLIF(%s, ''), %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            name      = VALUES(name),
            hire_date = VALUES(hire_date),
            position  = VALUES(position),
            team      = VALUES(team),
            mentor    = VALUES(mentor),
            active    = VALUES(active)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    employee.employee_no,
                    employee.name,
                    employee.hire_date or "",
                    employee.position,
                    employee.team,
                    employee.mentor,
                    1 if getattr(employee, "active", True) else 0,
                ))
            conn.commit()

    def get_employee(self, employee_no: str):
        from domain.value_objects.employee import EmployeeProfile

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM employee WHERE employee_no = %s",
                    (employee_no,))
                row = cur.fetchone()
        if not row:
            return None
        return _to_employee(row)

    def list_employees(self, *, team: str = "", active_only: bool = True,
                       limit: int = 500) -> list:
        where, params = [], []
        if team:
            where.append("team = %s")
            params.append(team)
        if active_only:
            where.append("active = 1")

        sql = "SELECT * FROM employee"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY employee_no LIMIT %s"
        params.append(limit)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        return [_to_employee(r) for r in rows]

    def count_employees(self, *, active_only: bool = True) -> int:
        sql = "SELECT COUNT(*) AS c FROM employee"
        if active_only:
            sql += " WHERE active = 1"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return int(cur.fetchone()["c"])

    # ---------------- 每日总结 ----------------
    def save_daily_summary(self, row) -> None:
        """写入/更新日报。

        ★ 用 `uk_trainee_date` 保证「每人每天一条」（FR-045）。
          重复写会覆盖草稿内容 —— 这正是「草稿可反复修改」的实现方式。

        ★ 但**已提交的不能被覆盖**：SQL 里加了 `status = 'draft'` 条件，
          命中 0 行说明已提交，抛异常交给上层。
        """
        from domain.exceptions import DailySummaryFrozenError

        check_sql = """
        SELECT status FROM daily_summary
        WHERE trainee_no = %s AND summary_date = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(check_sql, (row.trainee_no, row.summary_date))
                existing = cur.fetchone()

                if existing and existing["status"] == "submitted":
                    # 已提交：只允许更新评语，不允许改正文
                    cur.execute(
                        """
                        UPDATE daily_summary
                        SET review_note = %s
                        WHERE trainee_no = %s AND summary_date = %s
                        """,
                        (row.review_note, row.trainee_no, row.summary_date),
                    )
                    conn.commit()
                    raise DailySummaryFrozenError(
                        "该总结已提交，正文不可修改（已保存评语）")

                cur.execute(
                    """
                    INSERT INTO daily_summary
                        (trainee_no, summary_date, content,
                         review_note, status, submitted_at)
                    VALUES (%s, %s, %s, %s, %s,
                            IF(%s = 'submitted', NOW(), NULL))
                    ON DUPLICATE KEY UPDATE
                        content      = VALUES(content),
                        review_note  = VALUES(review_note),
                        status       = VALUES(status),
                        submitted_at = COALESCE(
                            daily_summary.submitted_at,
                            VALUES(submitted_at))
                    """,
                    (
                        row.trainee_no, row.summary_date,
                        row.content, row.review_note,
                        str(row.status) if row.status else "draft",
                        str(row.status) if row.status else "draft",
                    ),
                )
            conn.commit()

    def get_daily_summary(self, trainee_no: str, summary_date: str):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM daily_summary
                    WHERE trainee_no = %s AND summary_date = %s
                    """,
                    (trainee_no, summary_date))
                row = cur.fetchone()
        return _to_summary(row) if row else None

    def list_daily_summaries(
        self,
        *,
        trainee_no: str = "",
        start_date: str = "",
        end_date: str = "",
        status: str = "",
        limit: int = 100,
    ) -> list:
        where, params = [], []
        if trainee_no:
            where.append("trainee_no = %s")
            params.append(trainee_no)
        if start_date:
            where.append("summary_date >= %s")
            params.append(start_date)
        if end_date:
            where.append("summary_date <= %s")
            params.append(end_date)
        if status:
            where.append("status = %s")
            params.append(status)

        sql = "SELECT * FROM daily_summary"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY summary_date DESC, trainee_no LIMIT %s"
        params.append(limit)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        return [_to_summary(r) for r in rows]

    def save_review(self, trainee_no: str, summary_date: str,
                    note: str, *, by: str = "") -> None:
        """只更新评语（不改正文、不改状态）。"""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE daily_summary
                    SET review_note = %s
                    WHERE trainee_no = %s AND summary_date = %s
                    """,
                    (note, trainee_no, summary_date))
                if cur.rowcount == 0:
                    raise ValueError(
                        f"日报不存在: {trainee_no} / {summary_date}")
            conn.commit()

    # ---------------- 未命中日志 ----------------
    def list_miss_logs(self, *, limit: int = 100) -> list:
        """★ 只按时间倒序返回，**不按学员聚合、不排序**。

        FR-047 明确「该日志不得用于对学员的绩效评价」。
        如果这里按学员分组排序，就相当于给了「谁问得多」的排序视图，
        很容易被拿去做成「谁学得差」。
        """
        sql = "SELECT * FROM miss_log ORDER BY created_at DESC LIMIT %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall()

        from domain.ports.repository_port import MissLogRow

        return [
            MissLogRow(
                id=str(r["id"]),
                query=r["query"],
                intent=r["intent"] or "",
                trainee_id=r["trainee_id"] or "",
                created_at=_iso(r["created_at"]),
                detail=_loads(r["detail_json"]) or {},
            )
            for r in rows
        ]

    def top_miss_queries(self, *, limit: int = 20) -> list[dict]:
        """按**问题内容**聚合（不是按学员），给主管看知识库缺口。"""
        sql = """
        SELECT query, COUNT(*) AS times, MAX(created_at) AS last_at
        FROM miss_log
        GROUP BY query
        ORDER BY times DESC
        LIMIT %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                return [
                    {"query": r["query"], "times": int(r["times"]),
                     "last_at": _iso(r["last_at"])}
                    for r in cur.fetchall()
                ]


    # ---------------- 演练评阅结果 ----------------
    def save_drill_review(
        self,
        review: DrillReviewRow,
        risk_items: tuple[DrillRiskItemRow, ...] = (),
    ) -> None:
        """保存评阅结果。幂等：同 drill_id 覆盖。

        ★ 事务内先删明细再插入：
          前端重试评阅请求时会重复调用本方法，
          只 UPSERT 主表而不清明细，违规条目会累积成 2 倍、3 倍，
          月度报告的风险数就全错了。
        """
        sql_main = """
        INSERT INTO drill_review (
            drill_id, trainee_id, case_id, case_type, duration_sec,
            element_hit, element_total, element_hit_all, element_total_all,
            risk_count, high_count, medium_count, critical_count
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            trainee_id=VALUES(trainee_id), case_id=VALUES(case_id),
            case_type=VALUES(case_type), duration_sec=VALUES(duration_sec),
            element_hit=VALUES(element_hit), element_total=VALUES(element_total),
            element_hit_all=VALUES(element_hit_all),
            element_total_all=VALUES(element_total_all),
            risk_count=VALUES(risk_count), high_count=VALUES(high_count),
            medium_count=VALUES(medium_count),
            critical_count=VALUES(critical_count)
        """
        sql_del = "DELETE FROM drill_risk_item WHERE drill_id = %s"
        sql_item = """
        INSERT INTO drill_risk_item (
            drill_id, turn_index, sentence_index,
            rule_type, matched, level, reason
        ) VALUES (%s,%s,%s,%s,%s,%s,%s)
        """

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_main, (
                    review.drill_id, review.trainee_id, review.case_id,
                    review.case_type, review.duration_sec,
                    review.element_hit, review.element_total,
                    review.element_hit_all, review.element_total_all,
                    review.risk_count, review.high_count,
                    review.medium_count, review.critical_count,
                ))
                cur.execute(sql_del, (review.drill_id,))
                if risk_items:
                    cur.executemany(sql_item, [
                        (review.drill_id, it.turn_index, it.sentence_index,
                         it.rule_type, it.matched[:255], it.level,
                         it.reason[:512])
                        for it in risk_items
                    ])
            conn.commit()

    def list_drill_reviews(
        self,
        *,
        trainee_id: str,
        since: str = "",
        until: str = "",
        with_items: bool = False,
    ) -> list[DrillReviewRow]:
        where = ["trainee_id = %s"]
        params: list = [trainee_id]
        if since:
            where.append("created_at >= %s")
            params.append(f"{since} 00:00:00")
        if until:
            where.append("created_at < %s")
            params.append(f"{until} 00:00:00")

        sql = (
            "SELECT * FROM drill_review WHERE " + " AND ".join(where)
            + " ORDER BY created_at ASC"
        )
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()

        reviews = [_to_drill_review(r) for r in rows]
        if not with_items or not reviews:
            return reviews

        # 一次性把明细全查回来，避免 N+1
        ids = [r.drill_id for r in reviews]
        placeholders = ",".join(["%s"] * len(ids))
        sql_items = (
            "SELECT * FROM drill_risk_item WHERE drill_id IN ("
            + placeholders + ") ORDER BY id"
        )
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_items, tuple(ids))
                item_rows = cur.fetchall()

        grouped: dict[str, list[DrillRiskItemRow]] = {}
        for r in item_rows:
            grouped.setdefault(r["drill_id"], []).append(_to_risk_item(r))

        return [
            replace(r, risk_items=tuple(grouped.get(r.drill_id, ())))
            for r in reviews
        ]

    def get_drill_review(self, drill_id: str) -> DrillReviewRow | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM drill_review WHERE drill_id = %s",
                    (drill_id,))
                row = cur.fetchone()
                if not row:
                    return None
                cur.execute(
                    "SELECT * FROM drill_risk_item WHERE drill_id = %s "
                    "ORDER BY id", (drill_id,))
                items = tuple(_to_risk_item(r) for r in cur.fetchall())
        return replace(_to_drill_review(row), risk_items=items)


    # ---------------- 批量任务（P3） ----------------
    def upsert_report_task(
        self,
        *,
        task_id: str,
        trainee_id: str,
        period: str,
        status: str,
        error: str = "",
        thread_id: str = "",
    ) -> None:
        """写任务状态。冲突键是 `(trainee_id, period)`。

        ★ 注意 ON DUPLICATE KEY 里**不改 task_id**：
          task_id 是首次投递时定的，后续状态更新沿用它。
          如果改了，前端拿着的旧 task_id 就查不到进度了。
        """
        sql = """
        INSERT INTO report_task (
            task_id, trainee_id, period, thread_id, status,
            attempts, last_error, started_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,
                  CASE WHEN %s IN ('running') THEN NOW() ELSE NULL END)
        ON DUPLICATE KEY UPDATE
            thread_id = IF(VALUES(thread_id) = '', thread_id, VALUES(thread_id)),
            status = VALUES(status),
            attempts = attempts + IF(VALUES(status) = 'failed', 1, 0),
            last_error = VALUES(last_error),
            started_at = IF(started_at IS NULL AND VALUES(status) = 'running',
                            NOW(), started_at),
            finished_at = IF(VALUES(status) IN ('pending_review','done','dead'),
                             NOW(), finished_at)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    task_id, trainee_id, period, (thread_id or "")[:128],
                    status, 0, (error or "")[:512], status,
                ))
            conn.commit()

    def get_report_task(self, trainee_id: str, period: str) -> dict | None:
        sql = ("SELECT * FROM report_task "
               "WHERE trainee_id = %s AND period = %s")
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (trainee_id, period))
                return cur.fetchone()

    def list_report_tasks(
        self, *, period: str = "", status: str = "", limit: int = 500,
    ) -> list[dict]:
        where, params = [], []
        if period:
            where.append("period = %s")
            params.append(period)
        if status:
            where.append("status = %s")
            params.append(status)
        sql = "SELECT * FROM report_task"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY enqueued_at DESC LIMIT %s"
        params.append(limit)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                return list(cur.fetchall())

    # ---------------- 月度报告定稿 ----------------
    def save_monthly_report(self, row: MonthlyReportRow) -> None:
        """保存定稿。幂等：同 (trainee_id, period) 覆盖。

        ★ `status='approved'` 时才应该调这里（草稿不进这张表），
          但实现不做强制检查——状态判断放用例/图里，
          基础设施只负责「写进去」。
        """
        sql = """
        INSERT INTO monthly_report (
            trainee_id, period, thread_id, status, report_json,
            suggestion, suggestion_passed, suggestion_note,
            total_drills, total_critical, weak_json, graph_steps,
            reviewed_by, reviewed_at, review_note
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            thread_id=VALUES(thread_id), status=VALUES(status),
            report_json=VALUES(report_json), suggestion=VALUES(suggestion),
            suggestion_passed=VALUES(suggestion_passed),
            suggestion_note=VALUES(suggestion_note),
            total_drills=VALUES(total_drills),
            total_critical=VALUES(total_critical),
            weak_json=VALUES(weak_json), graph_steps=VALUES(graph_steps),
            reviewed_by=VALUES(reviewed_by), reviewed_at=VALUES(reviewed_at),
            review_note=VALUES(review_note)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    row.trainee_id, row.period, row.thread_id[:128],
                    row.status,
                    json.dumps(row.report_json, ensure_ascii=False)
                    if row.report_json else None,
                    row.suggestion or None,
                    1 if row.suggestion_passed else 0,
                    (row.suggestion_note or "")[:512],
                    row.total_drills, row.total_critical,
                    json.dumps(list(row.weak_capabilities), ensure_ascii=False),
                    row.graph_steps,
                    (row.reviewed_by or "")[:64],
                    row.reviewed_at or None,
                    (row.review_note or "")[:512],
                ))
            conn.commit()

    def get_monthly_report(
        self, trainee_id: str, period: str,
    ) -> MonthlyReportRow | None:
        sql = ("SELECT * FROM monthly_report "
               "WHERE trainee_id = %s AND period = %s")
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (trainee_id, period))
                row = cur.fetchone()
        return _to_monthly_report(row) if row else None

    def list_monthly_reports(
        self,
        *,
        period: str = "",
        status: str = "",
        trainee_id: str = "",
        limit: int = 200,
    ) -> list[MonthlyReportRow]:
        where, params = [], []
        if period:
            where.append("period = %s")
            params.append(period)
        if status:
            where.append("status = %s")
            params.append(status)
        if trainee_id:
            where.append("trainee_id = %s")
            params.append(trainee_id)
        sql = "SELECT * FROM monthly_report"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY updated_at DESC LIMIT %s"
        params.append(limit)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        return [_to_monthly_report(r) for r in rows]


# ---------------- 行转换 ----------------
def _to_drill_review(row: dict) -> DrillReviewRow:
    return DrillReviewRow(
        drill_id=row["drill_id"],
        trainee_id=row.get("trainee_id", "") or "",
        case_id=row.get("case_id", "") or "",
        case_type=row.get("case_type", "") or "",
        duration_sec=int(row.get("duration_sec") or 0),
        element_hit=int(row.get("element_hit") or 0),
        element_total=int(row.get("element_total") or 0),
        element_hit_all=int(row.get("element_hit_all") or 0),
        element_total_all=int(row.get("element_total_all") or 0),
        risk_count=int(row.get("risk_count") or 0),
        high_count=int(row.get("high_count") or 0),
        medium_count=int(row.get("medium_count") or 0),
        critical_count=int(row.get("critical_count") or 0),
        created_at=_iso(row.get("created_at")),
    )


def _to_risk_item(row: dict) -> DrillRiskItemRow:
    return DrillRiskItemRow(
        id=int(row.get("id") or 0),
        drill_id=row["drill_id"],
        turn_index=int(row.get("turn_index") or 0),
        sentence_index=int(row.get("sentence_index") or 0),
        rule_type=row.get("rule_type", "") or "",
        matched=row.get("matched", "") or "",
        level=row.get("level", "") or "",
        reason=row.get("reason", "") or "",
    )


def _to_monthly_report(row: dict) -> MonthlyReportRow:
    report_json = row.get("report_json")
    if isinstance(report_json, str):
        try:
            report_json = json.loads(report_json)
        except (ValueError, TypeError):
            report_json = {}
    weak = row.get("weak_json")
    if isinstance(weak, str):
        try:
            weak = json.loads(weak)
        except (ValueError, TypeError):
            weak = []

    return MonthlyReportRow(
        id=int(row.get("id") or 0),
        trainee_id=row.get("trainee_id", "") or "",
        period=row.get("period", "") or "",
        thread_id=row.get("thread_id", "") or "",
        status=row.get("status", "draft") or "draft",
        report_json=report_json or {},
        suggestion=row.get("suggestion") or "",
        suggestion_passed=bool(row.get("suggestion_passed", 1)),
        suggestion_note=row.get("suggestion_note", "") or "",
        total_drills=int(row.get("total_drills") or 0),
        total_critical=int(row.get("total_critical") or 0),
        weak_capabilities=tuple(weak or ()),
        graph_steps=int(row.get("graph_steps") or 0),
        reviewed_by=row.get("reviewed_by", "") or "",
        reviewed_at=_iso(row.get("reviewed_at")),
        review_note=row.get("review_note", "") or "",
        created_at=_iso(row.get("created_at")),
        updated_at=_iso(row.get("updated_at")),
    )


def _to_employee(row: dict):
    from domain.value_objects.employee import EmployeeProfile

    return EmployeeProfile(
        employee_no=row["employee_no"],
        name=row["name"],
        hire_date=_iso_date(row.get("hire_date")),
        position=row.get("position") or "",
        team=row.get("team") or "",
        mentor=row.get("mentor") or "",
        active=bool(row.get("active", 1)),
    )


def _to_summary(row: dict):
    from domain.value_objects.employee import DailySummary, SummaryStatus

    return DailySummary(
        trainee_no=row["trainee_no"],
        summary_date=_iso_date(row.get("summary_date")),
        content=row.get("content") or "",
        status=SummaryStatus(row.get("status") or "draft"),
        submitted_at=_iso(row.get("submitted_at")),
        review_note=row.get("review_note") or "",
    )


def _iso_date(v) -> str:
    """date 对象 → YYYY-MM-DD。"""
    if v is None:
        return ""
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    return str(v)[:10]


# ===================================================================
# P8：账号与审计日志（扩展）
# ===================================================================
class MySqlRepositoryAdapterFull(MySqlRepositoryAdapterP7):
    """P8 扩展：账号 + 审计日志。

    ★ 逐层继承（Base → P7 → P8）而不是一次写全：
      每层对应一个阶段的表，改动边界清楚。
      装配根只用最外层，对外仍是 `RepositoryPort`。
    """

    # ---------------- 账号 ----------------
    def get_account(self, employee_no: str):
        """取账号（含密码哈希，仅供认证链路使用）。"""
        from domain.value_objects.account import EmployeeAccount

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM employee_account WHERE employee_no = %s",
                    (employee_no,))
                row = cur.fetchone()
        if not row:
            return None

        return EmployeeAccount(
            employee_no=row["employee_no"],
            role=row.get("role") or "trainee",
            password_hash=row.get("password_hash") or "",
            active=bool(row.get("active", 1)),
        )

    def save_account(self, account) -> None:
        """写入/更新账号（按工号 upsert）。"""
        sql = """
        INSERT INTO employee_account
            (employee_no, role, password_hash, active)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            role          = VALUES(role),
            password_hash = VALUES(password_hash),
            active        = VALUES(active)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    account.employee_no,
                    account.role,
                    account.password_hash,
                    1 if account.active else 0,
                ))
            conn.commit()

    def set_account_role(self, employee_no: str, role: str) -> None:
        """改角色。

        ★ FR-048「权限变更应即时生效」的落点：
          这里只改库；因为 `ResolveIdentityUseCase` 每次请求都回库取，
          所以改完**下一个请求立即生效**，不用等 token 过期。
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE employee_account SET role = %s "
                    "WHERE employee_no = %s",
                    (role, employee_no))
                if cur.rowcount == 0:
                    raise ValueError(f"账号不存在: {employee_no}")
            conn.commit()

    def set_account_active(self, employee_no: str, active: bool) -> None:
        """启用/停用账号。"""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE employee_account SET active = %s "
                    "WHERE employee_no = %s",
                    (1 if active else 0, employee_no))
                if cur.rowcount == 0:
                    raise ValueError(f"账号不存在: {employee_no}")
            conn.commit()

    def touch_last_login(self, employee_no: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE employee_account SET last_login_at = NOW() "
                    "WHERE employee_no = %s", (employee_no,))
            conn.commit()

    def list_accounts(self, *, limit: int = 500) -> list:
        """列账号（**不含密码哈希**，供管理页用）。"""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT employee_no, role, active, last_login_at "
                    "FROM employee_account ORDER BY employee_no LIMIT %s",
                    (limit,))
                return [
                    {
                        "employee_no": r["employee_no"],
                        "role": r["role"],
                        "active": bool(r["active"]),
                        "last_login_at": _iso(r.get("last_login_at")),
                    }
                    for r in cur.fetchall()
                ]

    # ---------------- 审计日志 ----------------
    def append_audit_log(self, entry) -> None:
        """写审计日志。

        ★ detail 里塞不进任何凭据 —— 由 `AuditLogEntry` 的字段定义保证，
          这里不做二次过滤（字段都没有，无从塞起）。
        """
        sql = """
        INSERT INTO audit_log
            (actor, role, action, target, result, detail_json, ip)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    entry.actor,
                    entry.role,
                    entry.action,
                    entry.target,
                    entry.result,
                    json.dumps(entry.detail, ensure_ascii=False)
                    if entry.detail else None,
                    entry.ip,
                ))
            conn.commit()

    def list_audit_logs(
        self, *, actor: str = "", action: str = "",
        limit: int = 100,
    ) -> list[dict]:
        """查审计日志（按时间倒序）。"""
        where, params = [], []
        if actor:
            where.append("actor = %s")
            params.append(actor)
        if action:
            where.append("action = %s")
            params.append(action)

        sql = "SELECT * FROM audit_log"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC LIMIT %s"
        params.append(limit)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()

        return [
            {
                "actor": r["actor"],
                "role": r["role"],
                "action": r["action"],
                "target": r["target"],
                "result": r["result"],
                "detail": _loads(r.get("detail_json")) or {},
                "ip": r["ip"],
                "created_at": _iso(r["created_at"]),
            }
            for r in rows
        ]
