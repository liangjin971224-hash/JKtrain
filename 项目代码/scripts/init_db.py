"""建库建表（P6/P7 业务数据）。

★ 凭据从 .env 的 DATABASE_URL 读，不在命令行暴露。

用法：
    .venv\\Scripts\\python.exe scripts\\init_db.py --dry-run   # 只打印 SQL
    .venv\\Scripts\\python.exe scripts\\init_db.py             # 执行

────────────────────────────────────────────────────────────────────
六张表的职责
────────────────────────────────────────────────────────────────────
    exam_paper            试卷主表（存组卷参数 + 状态）
    exam_paper_question   试卷题目快照（**存完整副本，不存引用**）
    question_usage        题目使用记录（防重复窗口的依据）
    drill_record          演练记录
    drill_review          演练评阅结果（月度报告算分依据）
    drill_risk_item       演练违规明细（逐句定位）
    monthly_report        月度报告定稿（★ 只存定稿，草稿在 Redis）
    report_task           报告生成任务（P3 队列，唯一键保证幂等）
    daily_summary         每日总结（每人每天一条）
    miss_log              未命中日志（召回为空时记录）

★ 为什么题目快照要单独一张表而不是 JSON 存主表：
  FR-039 要求「即使题库中的题目被修改或删除，历史试卷内容不得变化」。
  用独立表存快照，每道题一行，便于：
    · 按题目反查「这道题出现在哪些卷子里」
    · 分批读取（一份卷子 50 题，一次查回来快）
  而 JSON 塞主表虽然也能存，但查不出去、改不了、也没法做统计。

★ 引擎与字符集
  InnoDB + utf8mb4。题目里含中文与特殊符号（★、「」），
  utf8mb4 才能完整存储（utf8 三字节版有坑）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymysql                                    # noqa: E402

from config.settings import Settings               # noqa: E402

DB_NAME = "jiankangshujuku"

#: 建库语句
CREATE_DATABASE = f"""
CREATE DATABASE IF NOT EXISTS `{DB_NAME}`
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci
"""

#: 建表语句（顺序有依赖：exam_paper 先于 exam_paper_question）
TABLES: list[tuple[str, str]] = [
    ("employee_account", """
CREATE TABLE IF NOT EXISTS `employee_account` (
  `employee_no`   VARCHAR(64)  NOT NULL COMMENT '工号（与 employee 对应）',
  `role`          VARCHAR(16)  NOT NULL DEFAULT 'trainee'
                               COMMENT 'trainee / manager',
  `password_hash` VARCHAR(128) NOT NULL COMMENT 'bcrypt 哈希（绝不存明文）',
  `active`        TINYINT(1)   NOT NULL DEFAULT 1 COMMENT '启用状态',
  `last_login_at` DATETIME     NULL COMMENT '最后登录时间',
  `created_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                               ON UPDATE CURRENT_TIMESTAMP,

  PRIMARY KEY (`employee_no`),
  KEY `idx_role` (`role`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='员工账号（P8，凭据与档案分离）'
"""),

    ("audit_log", """
CREATE TABLE IF NOT EXISTS `audit_log` (
  `id`         BIGINT       NOT NULL AUTO_INCREMENT,
  `actor`      VARCHAR(64)  NOT NULL COMMENT '操作人工号',
  `role`       VARCHAR(16)  NOT NULL DEFAULT '' COMMENT '操作时角色',
  `action`     VARCHAR(32)  NOT NULL COMMENT '动作（login/exam_build…）',
  `target`     VARCHAR(128) NOT NULL DEFAULT '' COMMENT '操作对象',
  `result`     VARCHAR(16)  NOT NULL DEFAULT 'ok' COMMENT 'ok/failed/denied',
  `detail_json` JSON        NULL COMMENT '补充信息（不含凭据）',
  `ip`         VARCHAR(64)  NOT NULL DEFAULT '',
  `created_at` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (`id`),
  KEY `idx_actor_time` (`actor`, `created_at`),
  KEY `idx_action` (`action`),
  KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='审计日志（P8，不含任何凭据字段）'
"""),

    ("employee", """
CREATE TABLE IF NOT EXISTS `employee` (
  `employee_no` VARCHAR(64)  NOT NULL COMMENT '工号（业务主键）',
  `name`        VARCHAR(64)  NOT NULL COMMENT '真实姓名（FR-044 要求）',
  `hire_date`   DATE         NULL COMMENT '入职日期',
  `position`    VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '岗位',
  `team`        VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '所属团队',
  `mentor`      VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '带教人',
  `active`      TINYINT(1)   NOT NULL DEFAULT 1 COMMENT '是否在职',
  `created_at`  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                             ON UPDATE CURRENT_TIMESTAMP,

  PRIMARY KEY (`employee_no`),
  KEY `idx_team` (`team`),
  KEY `idx_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='员工档案（P7）'
"""),

    ("exam_paper", """
CREATE TABLE IF NOT EXISTS `exam_paper` (
  `paper_id`       BIGINT       NOT NULL AUTO_INCREMENT COMMENT '内部主键',
  `paper_no`       VARCHAR(32)  NOT NULL COMMENT '可读卷号 S1-20261105-001',
  `stage`          VARCHAR(16)  NOT NULL DEFAULT '' COMMENT '所属阶段',
  `title`          VARCHAR(128) NOT NULL DEFAULT '' COMMENT '试卷标题',
  `status`         VARCHAR(16)  NOT NULL DEFAULT 'draft'
                                COMMENT '草稿 draft / 生效 active / 归档 archived',
  `total_count`    INT          NOT NULL DEFAULT 0 COMMENT '题目数',

  -- 组卷参数：原样存下来，便于「按同样参数再组一份」
  `spec_json`      JSON         NULL COMMENT '组卷参数快照',

  `created_by`     VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '创建人',
  `created_at`     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `activated_at`   DATETIME     NULL COMMENT '定稿时间',
  `archived_at`    DATETIME     NULL COMMENT '归档时间',
  `remark`         VARCHAR(512) NOT NULL DEFAULT '' COMMENT '备注',

  PRIMARY KEY (`paper_id`),
  UNIQUE KEY `uk_paper_no` (`paper_no`),
  KEY `idx_stage_status` (`stage`, `status`),
  KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='试卷主表（P6）'
"""),

    ("exam_paper_question", """
CREATE TABLE IF NOT EXISTS `exam_paper_question` (
  `id`             BIGINT       NOT NULL AUTO_INCREMENT,
  `paper_no`       VARCHAR(32)  NOT NULL COMMENT '所属卷号',
  `seq`            INT          NOT NULL COMMENT '卷内题序（从 1 开始）',
  `question_id`    VARCHAR(64)  NOT NULL COMMENT '原题编号（仅作追溯线索）',

  -- ★★ 以下是**快照副本**：题目改了也不影响这里
  `knowledge_point` VARCHAR(64) NOT NULL DEFAULT '' COMMENT '知识点编码',
  `kp_name`        VARCHAR(128) NOT NULL DEFAULT '' COMMENT '知识点名称',
  `question_type`  VARCHAR(16)  NOT NULL DEFAULT '单选' COMMENT '单选/多选',
  `difficulty`     TINYINT      NOT NULL DEFAULT 2 COMMENT '1易 2中 3难',
  `stem`           TEXT         NOT NULL COMMENT '题干快照',
  `options_json`   JSON         NULL COMMENT '选项快照（数组）',
  `answer`         VARCHAR(16)  NOT NULL DEFAULT '' COMMENT '答案字母',
  `explanation`    TEXT         NULL COMMENT '解析快照',
  `product_id`     VARCHAR(32)  NOT NULL DEFAULT '' COMMENT '关联产品',

  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_paper_seq` (`paper_no`, `seq`),
  KEY `idx_paper_no` (`paper_no`),
  KEY `idx_question_id` (`question_id`),
  KEY `idx_knowledge_point` (`knowledge_point`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='试卷题目快照（FR-039：题库变更不影响历史试卷）'
"""),

    ("question_usage", """
CREATE TABLE IF NOT EXISTS `question_usage` (
  `id`           BIGINT      NOT NULL AUTO_INCREMENT,
  `question_id`  VARCHAR(64) NOT NULL COMMENT '题目编号',
  `trainee_id`   VARCHAR(64) NOT NULL DEFAULT '' COMMENT '学员编号（空=不限学员）',
  `paper_no`     VARCHAR(32) NOT NULL DEFAULT '' COMMENT '用在哪个卷子',
  `used_at`      DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '使用时间',

  PRIMARY KEY (`id`),
  -- ★ 防重复窗口的核心索引：按「学员 + 时间」查最近用过的题
  KEY `idx_trainee_time` (`trainee_id`, `used_at`),
  KEY `idx_question` (`question_id`),
  KEY `idx_used_at` (`used_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='题目使用记录（FR-038：30 天防重复的依据）'
"""),

    ("report_task", """
CREATE TABLE IF NOT EXISTS `report_task` (
  `task_id`       VARCHAR(64)  NOT NULL COMMENT '任务 ID（= 幂等键）',
  `trainee_id`    VARCHAR(64)  NOT NULL COMMENT '学员编号',
  `period`        VARCHAR(16)  NOT NULL COMMENT '周期 2026-10',
  `thread_id`     VARCHAR(128) NOT NULL DEFAULT '' COMMENT '图的 thread_id',
  `status`        VARCHAR(24)  NOT NULL DEFAULT 'queued'
                              COMMENT 'queued / running / pending_review / done / failed / dead',
  `attempts`      INT          NOT NULL DEFAULT 0 COMMENT '已尝试次数',
  `max_attempts`  INT          NOT NULL DEFAULT 3 COMMENT '最大尝试次数',
  `last_error`    VARCHAR(512) NOT NULL DEFAULT '' COMMENT '最后一次错误',
  `enqueued_at`   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `started_at`    DATETIME     NULL,
  `finished_at`   DATETIME     NULL,
  `updated_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                              ON UPDATE CURRENT_TIMESTAMP,

  PRIMARY KEY (`task_id`),
  UNIQUE KEY `uk_trainee_period` (`trainee_id`, `period`),
  KEY `idx_status` (`status`),
  KEY `idx_period` (`period`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='报告生成任务（P3 队列）。唯一键 (trainee,period) 保证幂等'
"""),

    ("monthly_report", """
CREATE TABLE IF NOT EXISTS `monthly_report` (
  `id`               BIGINT       NOT NULL AUTO_INCREMENT,
  `trainee_id`       VARCHAR(64)  NOT NULL COMMENT '学员编号',
  `period`           VARCHAR(16)  NOT NULL COMMENT '周期，如 2026-10',
  `thread_id`        VARCHAR(128) NOT NULL DEFAULT ''
                                  COMMENT '图的 thread_id（用于回溯 Redis 中间态）',
  `status`           VARCHAR(24)  NOT NULL DEFAULT 'draft'
                                  COMMENT 'draft / pending_review / approved / rejected',
  `report_json`      JSON         NULL COMMENT '三层结构快照（能力项/案例/违规明细）',
  `suggestion`       TEXT         NULL COMMENT '改进建议正文',
  `suggestion_passed` TINYINT     NOT NULL DEFAULT 1 COMMENT '建议是否通过校验',
  `suggestion_note`  VARCHAR(512) NOT NULL DEFAULT '' COMMENT '校验未通过的原因',
  `total_drills`     INT          NOT NULL DEFAULT 0 COMMENT '演练次数快照',
  `total_critical`   INT          NOT NULL DEFAULT 0 COMMENT '红线违规数快照',
  `weak_json`        JSON         NULL COMMENT '薄弱能力项列表快照',
  `graph_steps`      INT          NOT NULL DEFAULT 0 COMMENT '图执行步数（诊断用）',
  `reviewed_by`      VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '复核人',
  `reviewed_at`      DATETIME     NULL COMMENT '复核时间',
  `review_note`      VARCHAR(512) NOT NULL DEFAULT '' COMMENT '复核意见',
  `created_at`       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                                  ON UPDATE CURRENT_TIMESTAMP,

  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_trainee_period` (`trainee_id`, `period`),
  KEY `idx_status` (`status`),
  KEY `idx_period` (`period`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='月度培训报告定稿（P2）。★ 只存定稿，草稿在 Redis checkpoint 里'
"""),

    ("drill_review", """
CREATE TABLE IF NOT EXISTS `drill_review` (
  `drill_id`         VARCHAR(64)  NOT NULL COMMENT '演练 ID（与 drill_record 对应）',
  `trainee_id`       VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '学员编号',
  `case_id`          VARCHAR(32)  NOT NULL DEFAULT '' COMMENT '案例编号',
  `case_type`        VARCHAR(64)  NOT NULL DEFAULT ''
                                  COMMENT '案例提问类型原文（如 contraindication）
                                           ★ 能力项归并的唯一依据，必须落库',
  `duration_sec`     INT          NOT NULL DEFAULT 0 COMMENT '用时',
  `element_hit`      INT          NOT NULL DEFAULT 0 COMMENT '命中的必答要素数',
  `element_total`    INT          NOT NULL DEFAULT 0 COMMENT '必答要素总数',
  `element_hit_all`  INT          NOT NULL DEFAULT 0 COMMENT '命中要素数（含选答，备用）',
  `element_total_all` INT         NOT NULL DEFAULT 0 COMMENT '要素总数（含选答，备用）',
  `risk_count`       INT          NOT NULL DEFAULT 0 COMMENT '违规总数',
  `high_count`       INT          NOT NULL DEFAULT 0 COMMENT 'high 级违规数',
  `medium_count`     INT          NOT NULL DEFAULT 0 COMMENT 'medium 级违规数',
  `critical_count`   INT          NOT NULL DEFAULT 0 COMMENT 'critical 级违规数（一票否决）',
  `created_at`       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (`drill_id`),
  KEY `idx_trainee_time` (`trainee_id`, `created_at`),
  KEY `idx_case` (`case_id`),
  KEY `idx_case_type` (`case_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='演练评阅结果（月度培训报告的算分依据）'
"""),

    ("drill_risk_item", """
CREATE TABLE IF NOT EXISTS `drill_risk_item` (
  `id`             BIGINT       NOT NULL AUTO_INCREMENT,
  `drill_id`       VARCHAR(64)  NOT NULL COMMENT '演练 ID',
  `turn_index`     INT          NOT NULL COMMENT '第几轮（从 1 开始）',
  `sentence_index` INT          NOT NULL COMMENT '第几句（从 1 开始）',
  `rule_type`      VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '规则类型',
  `matched`        VARCHAR(255) NOT NULL DEFAULT '' COMMENT '命中的原文片段',
  `level`          VARCHAR(16)  NOT NULL DEFAULT '' COMMENT 'critical / high / medium',
  `reason`         VARCHAR(512) NOT NULL DEFAULT '' COMMENT '判定理由',

  PRIMARY KEY (`id`),
  KEY `idx_drill` (`drill_id`),
  KEY `idx_level` (`level`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='演练违规明细（逐句定位，支撑可申诉）'
"""),

    ("drill_record", """
CREATE TABLE IF NOT EXISTS `drill_record` (
  `drill_id`     VARCHAR(64)  NOT NULL COMMENT '演练 ID',
  `trainee_id`   VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '学员编号',
  `case_id`      VARCHAR(32)  NOT NULL DEFAULT '' COMMENT '案例编号',
  `started_at`   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `finished_at`  DATETIME     NULL,
  `duration_sec` INT          NOT NULL DEFAULT 0,
  `status`       VARCHAR(24)  NOT NULL DEFAULT 'in_progress'
                              COMMENT 'in_progress / finished / abandoned',
  `card_json`    JSON         NULL COMMENT '判定摘要卡快照',

  PRIMARY KEY (`drill_id`),
  KEY `idx_trainee_time` (`trainee_id`, `started_at`),
  KEY `idx_case` (`case_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='演练记录（P5）'
"""),

    ("daily_summary", """
CREATE TABLE IF NOT EXISTS `daily_summary` (
  `id`           BIGINT       NOT NULL AUTO_INCREMENT,
  `trainee_no`   VARCHAR(64)  NOT NULL COMMENT '学员编号',
  `summary_date` DATE         NOT NULL COMMENT '日期',
  `content`      TEXT         NULL COMMENT '当日总结正文',
  `review_note`  VARCHAR(512) NOT NULL DEFAULT '' COMMENT '主管评语',
  `status`       VARCHAR(16)  NOT NULL DEFAULT 'draft' COMMENT 'draft/reviewed',
  `submitted_at` DATETIME     NULL COMMENT '提交时间（FR-045 要求记录）',
  `created_at`   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                              ON UPDATE CURRENT_TIMESTAMP,

  PRIMARY KEY (`id`),
  -- ★ 每人每天只能有一条
  UNIQUE KEY `uk_trainee_date` (`trainee_no`, `summary_date`),
  KEY `idx_date` (`summary_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='每日总结（P7）'
"""),

    ("miss_log", """
CREATE TABLE IF NOT EXISTS `miss_log` (
  `id`         BIGINT       NOT NULL AUTO_INCREMENT,
  `query`      VARCHAR(512) NOT NULL COMMENT '用户原始问题',
  `intent`     VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '意图分类',
  `trainee_id` VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '提问人',
  `created_at` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `detail_json` JSON        NULL COMMENT '诊断信息（各路命中数等）',

  PRIMARY KEY (`id`),
  KEY `idx_created_at` (`created_at`),
  KEY `idx_intent` (`intent`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='未命中日志（召回为空时记录，供补内容）'
"""),
]


def parse_db_url(url: str) -> dict:
    p = urlparse(url)
    return {
        "host": p.hostname or "localhost",
        "port": p.port or 3306,
        "user": p.username or "root",
        "password": p.password or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="建库建表")
    parser.add_argument("--dry-run", action="store_true", help="只打印 SQL")
    args = parser.parse_args()

    settings = Settings()
    url = str(getattr(settings, "database_url", "") or "")
    if not url:
        print("[FAIL] .env 里没有 DATABASE_URL")
        return 1

    cfg = parse_db_url(url)

    if args.dry_run:
        print(CREATE_DATABASE.strip(), ";\n")
        for name, ddl in TABLES:
            print(ddl.strip(), ";\n")
        return 0

    try:
        conn = pymysql.connect(
            host=cfg["host"], port=cfg["port"], user=cfg["user"],
            password=cfg["password"], charset="utf8mb4")
    except Exception as exc:                    # noqa: BLE001
        print(f"[FAIL] 连接 MySQL 失败: {type(exc).__name__}: {exc}")
        return 1

    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_DATABASE)
            print(f"[OK] 数据库 `{DB_NAME}` 就绪")

            cur.execute(f"USE `{DB_NAME}`")

            for name, ddl in TABLES:
                cur.execute(ddl)
                cur.execute(f"SHOW TABLES LIKE '{name}'")
                exists = cur.fetchone() is not None
                print(f"[{'OK' if exists else 'FAIL'}] 表 `{name}`")

            cur.execute("SHOW TABLES")
            tables = [r[0] for r in cur.fetchall()]
            print(f"\n库 `{DB_NAME}` 现有 {len(tables)} 张表: {tables}")

        conn.commit()
        print("\n完成。")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
