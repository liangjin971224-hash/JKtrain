"""HTTP 请求 / 响应模型。

★ 设计约束：**响应里不能出现 score / verdict**（约束 #3）。
  Pydantic 模型用 `extra="forbid"` 之类只能防输入，
  输出侧靠「模型里根本没有这个字段」来保证 —— 这是最可靠的。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ===================================================================
# 通用
# ===================================================================
class HealthResponse(BaseModel):
    """健康检查。"""

    status: str = "ok"
    ready_ports: list[str] = Field(default_factory=list)
    pending_ports: dict[str, str] = Field(default_factory=dict)
    llm_endpoints: list[str] = Field(default_factory=list)
    guardrail_rules: dict[str, int] = Field(default_factory=dict)


# ===================================================================
# 知识问答
# ===================================================================
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500,
                          description="员工的问题")
    product_ids: list[str] = Field(
        default_factory=list,
        description="已知产品 ID（如 P001）。前端可从产品列表带过来")
    #: ★ 刻意**没有** allowed_permissions 字段（2026-10-23 修）。
    #:   原实现让前端传，学员传 restricted 就能越权检索受限内容。
    #:   现在检索级别由登录角色决定（`Role.retrieval_levels`）。
    top_k: int = Field(default=20, ge=1, le=50)
    stream: bool = Field(default=False, description="是否用 SSE 流式返回")
    #: 是否启用多跳（拆子查询 / 前一步结果作后一步前提）。
    #:
    #: ★ 默认 **true**：用户明确提出「改善体脂…其中那种最便宜」这类问题，
    #:   不拆就答不了。
    #:   关闭它只对排查有用（对比「拆了」与「没拆」的差异），
    #:   不做成「省性能」的开关 —— 单步问题本来就不会多跳，
    #:   规划器保证简单问题只有一步（有测试守着）。
    multi_hop: bool = Field(default=True, description="是否启用多跳拆解")
    #: 多轮会话 ID。★ 不传 = 单轮，行为完全不变；功能开关关闭时忽略。
    #: 合法性由路由按 `[A-Za-z0-9_-]{8,64}` 再校验，非法按无会话处理。
    session_id: str | None = Field(
        default=None, max_length=64, description="多轮会话 ID（可选）")
    #: 是否清空该会话的历史（用户点「新对话」）。
    #:
    #: ★ 只清历史、不换 session_id：这样前端不用改本地存的值，
    #:   也避免"点了新对话但旧请求还在飞"导致串话。
    #:   与 `session_id` 一起生效；无会话时为 no-op。
    new_conversation: bool = Field(
        default=False,
        description="为 true 时清空该会话历史，作为新对话开始")


class AskSegment(BaseModel):
    """回答里的一句话（含合规判定），供前端渲染标红。

    ★ `text` 可能含**违规原文**。这些内容是为了让学员看见并理解
      「哪句话不能说」，属于培训用途。

    ★ 使用边界（约定，非技术强制）：
      只应在前端页面渲染，不要落日志、不要入库、不要对外转发。
      对外安全的正文在 `AskResponse.answer`。
    """

    text: str = Field(
        description="句子原文。可能含违规表述，仅用于前端标红展示")
    ok: bool = True
    #: 命中的规则类别标签（如「疾病治疗」「替客户判断」）
    rule_types: list[str] = Field(default_factory=list)
    #: 给学员看的原因说明
    reason: str = ""


class AnnotationResponse(BaseModel):
    """违规内容标注。

    ★ 安全约束：
      `AskResponse.answer` 是**剔除违规句后**的合规版本，可直接对外；
      本结构里的 `segments` 含违规原文，**只应喂给前端渲染**。
    """

    has_safe_content: bool = False
    segment_count: int = 0
    violation_count: int = 0
    repaired_warning: bool = False
    segments: list[AskSegment] = Field(default_factory=list)


class HopInfo(BaseModel):
    """一跳的执行摘要（供前端展示「这个问题被拆成了几步」）。

    ★ 只放**过程信息**，不放正文：正文已合并进 `AskResponse.answer`。
      拆开放是为了让学员看得懂「为什么答案是分两段给的」，
      也方便排障时对照「第 2 步的候选集是什么」。
    """

    order: int = 0
    question: str = ""
    kind: str = "simple"
    depends_on_previous: bool = False
    product_ids: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    candidate_count: int = 0
    refused: bool = False


class AskResponse(BaseModel):
    """问答响应。

    ★★ 使用约定（**必须遵守**）：`answer` 与 `annotation` 的用途不同。

    | 字段 | 内容 | 允许用途 |
    |---|---|---|
    | `answer` | 已剔除违规句的**合规版本** | 任何用途：展示、复制、落日志、转发、第三方调用 |
    | `annotation.segments` | 逐句明细，**含违规原文** | **仅限前端渲染标红**。不得落日志、不得入库、不得对外转发 |

    `answer` 是安全版本，任何消费方直接使用都合规。
    `annotation` 是为了培训教学（让学员看见「哪句话不能说、为什么」）
    而额外提供的，其中 `segments[].text` 保留了 LLM 原始违规表述。

    ★ 如果只需要正文，请只读 `answer`，忽略 `annotation`。
    """

    answer: str = Field(
        description="合规版正文：已剔除违规句，可直接对外使用")
    refused: bool = False
    refusal_reason: str = ""
    #: 本次问答所属的会话 ID。前端要存下来，下次请求带回去才能追问。
    #: ★ 单轮（未开多轮/非法/开关关闭）时为 None，且因
    #:   `response_model_exclude_unset=True` 不会出现在响应里。
    session_id: str | None = None
    #: 与原问题相同时留空，减少前端噪声。
    rewritten_question: str = ""
    #: 当前是第几轮（1-based）。前端可用它显示「已记 N 轮」。
    turn: int = 0
    #: 旧字段（主干前端在用）：本会话已积累的历史轮数，值等于 `turn`。
    history_turns: int = Field(
        default=0, description="当前会话已积累的历史轮数（= turn）")
    multi_turn: bool = False
    #: rule | llm | none
    rewrite_source: str = "none"
    cited_chunk_ids: list[str] = Field(default_factory=list)
    product_ids: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    #: 多跳拆解明细（只填「拆成了几步」与每步的候选集大小）。
    #:
    #: ★ 一步时不填（`[]`）：绝大多数问题只有一步，
    #:   给前端一个恒为 1 的数组只会让人误以为「多跳是常态」。
    hops: list[HopInfo] = Field(default_factory=list)
    annotation: AnnotationResponse | None = Field(
        default=None,
        description=(
            "违规句标注（含违规原文）。**仅供前端渲染标红**，"
            "不得写入日志或对外转发。只要正文请读 `answer`。"
        ),
    )


# ===================================================================
# 演练
# ===================================================================
class CaseSummary(BaseModel):
    """案例列表项。"""

    case_id: str
    title: str
    question_type: str = ""
    difficulty: int = 0
    stage: str = ""
    teaching_point: str = ""
    element_count: int = 0
    rule_count: int = 0
    is_red_line: bool = False


class CaseDetail(CaseSummary):
    """案例详情（含脚本与要素）。"""

    opening_line: str = ""
    elements: list[dict[str, Any]] = Field(default_factory=list)
    follow_ups: list[dict[str, str]] = Field(default_factory=list)


class StartDrillRequest(BaseModel):
    case_id: str = Field(..., min_length=1)
    trainee_id: str = Field(default="", max_length=64)


class DrillStateResponse(BaseModel):
    """演练状态（每次操作后返回）。"""

    session_id: str
    case_id: str
    case_title: str = ""
    trainee_id: str = ""

    current_question: str = ""
    current_follow_up_id: str | None = None
    progress_current: int = 1
    progress_total: int = 1
    remaining: int = 0
    turn_count: int = 0
    is_finished: bool = False
    duration_sec: int = 0

    #: 完整对话记录（前端滚动区用）
    messages: list[dict[str, str]] = Field(default_factory=list)


class AnswerDrillRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1, max_length=2000)


class ReviewDrillRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


# ===================================================================
# 摘要卡（四区块）
# ===================================================================
class RiskItemResponse(BaseModel):
    level: str
    rule_type: str
    matched: str
    location: str
    reason: str = ""
    is_critical: bool = False


class ElementCheckResponse(BaseModel):
    element_id: str
    element_name: str
    required: bool
    covered: bool
    evidence: str = ""
    status_mark: str = ""


class SummaryCardResponse(BaseModel):
    """判定摘要卡。

    ★ 没有 score 字段，没有 verdict 字段（约束 #3）。
      这里不是「不填」，是「模型里就没有这个字段」。
    """

    trainee_id: str = ""
    case_id: str = ""
    case_title: str = ""
    duration_sec: int = 0

    risk_items: list[RiskItemResponse] = Field(default_factory=list)
    element_coverage: list[ElementCheckResponse] = Field(default_factory=list)
    pending_human: list[str] = Field(default_factory=list)

    #: 便捷字段（前端用不同颜色标记红线案例）
    has_critical_violation: bool = False
    critical_count: int = 0
    missing_required_count: int = 0


class ErrorResponse(BaseModel):
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


# ===================================================================
# 试卷（P6）
# ===================================================================
class BuildPaperRequest(BaseModel):
    """组卷请求。"""

    total: int = Field(default=20, ge=1, le=100, description="题目总数")
    stage: str = Field(default="S1", max_length=16, description="所属阶段")
    title: str = Field(default="", max_length=128)
    knowledge_points: list[str] = Field(
        default_factory=list,
        description="要求覆盖的知识点；为空则自动覆盖全部")
    #: 难度配比，键为难度 1/2/3，值为题数。为空表示不限配比。
    #:
    #: ★ 踩过的两个坑：
    #:   ① `dict[str, int] = {}` —— Pydantic 把 `{}` 当类属性（不是字段默认值），
    #:      实例上拿到 tuple，`.items()` 直接 AttributeError → 500。
    #:   ② `Field(...)` 后面**不能加逗号** —— 加了会让整个 Field 变成 tuple，
    #:      症状与上一条一样。这类错很难一眼看出来。
    difficulty_quota: dict[str, int] = Field(
        default_factory=dict,
        description="难度配比，键为难度 1/2/3，值为题数。为空表示不限配比",
    )
    product_ids: list[str] = Field(default_factory=list)
    shuffle_options: bool = True
    trainee_id: str = Field(
        default="", max_length=64,
        description="学员编号。防重复按学员隔离；空表示全局去重")
    created_by: str = Field(default="", max_length=64)
    seed: int | None = Field(
        default=None,
        description="随机种子。给定则结果可复现（补考/复核需要）")
    persist: bool = Field(default=True, description="是否落库")


class QuestionView(BaseModel):
    """试卷里的一道题。"""

    seq: int = 0
    question_id: str = ""
    knowledge_point: str = ""
    kp_name: str = ""
    question_type: str = "单选"
    difficulty: int = 2
    stem: str = ""
    options: list[str] = Field(default_factory=list)
    answer: str = Field(default="", description="答案字母；导出试卷页时应隐藏")
    explanation: str = ""
    product_id: str = ""


class PaperView(BaseModel):
    """试卷。"""

    paper_no: str
    stage: str = ""
    title: str = ""
    status: str = "draft"
    total_count: int = 0
    created_by: str = ""
    created_at: str = ""
    questions: list[QuestionView] = Field(default_factory=list)
    #: 组卷警告（难度不足、产品未覆盖等）
    warnings: list[str] = Field(default_factory=list)
    #: 覆盖率信息
    covered_knowledge_points: list[str] = Field(default_factory=list)
    difficulty_distribution: dict[str, int] = Field(default_factory=dict)


class PaperListItem(BaseModel):
    """试卷列表项（不含题目）。"""

    paper_no: str
    stage: str = ""
    title: str = ""
    status: str = ""
    total_count: int = 0
    created_by: str = ""
    created_at: str = ""


class PaperStatusRequest(BaseModel):
    """试卷状态流转请求。"""

    paper_no: str = Field(..., min_length=1)
    target: str = Field(
        ..., description="目标状态：active（生效）/ archived（归档）")


class ExportResponse(BaseModel):
    """导出结果（Markdown 文本）。"""

    paper_no: str
    exam_markdown: str = Field(default="", description="试卷页（不含答案）")
    answer_markdown: str = Field(default="", description="答案与解析页")
    summary_markdown: str = Field(default="", description="组卷报告")


# ===================================================================
# 员工与日报（P7）
# ===================================================================
class EmployeeView(BaseModel):
    """员工档案。"""

    employee_no: str
    name: str
    hire_date: str = ""
    position: str = ""
    team: str = ""
    mentor: str = ""
    active: bool = True


class SetEmployeeActiveRequest(BaseModel):
    """启用/停用员工档案。"""

    employee_no: str = Field(..., min_length=1)
    active: bool = Field(...)


class ImportEmployeesRequest(BaseModel):
    """批量导入员工。

    ★ 刻意**没有** `operator_role` 字段：
      操作者角色从令牌取。原实现让前端传，学员就能冒充主管导入。
    """

    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="原始行，键可用中文或英文表头（工号/姓名/入职日期…）")


class ImportResultResponse(BaseModel):
    ok: bool = False
    imported: int = 0
    duplicates: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    summary: str = ""


class SummaryView(BaseModel):
    """每日总结。"""

    trainee_no: str
    summary_date: str
    content: str = ""
    status: str = "draft"
    submitted_at: str = ""
    review_note: str = ""
    can_edit: bool = False


class SaveSummaryRequest(BaseModel):
    """保存/提交日报。

    ★ 刻意**没有** `requester` / `role` 字段（2026-10-23 修）：
      原实现让前端传工号，学员传别人工号就能写别人的日报。
      现在工号与角色一律从令牌取。
    """

    summary_date: str = Field(default="", description="YYYY-MM-DD，空=今天")
    content: str = Field(default="", max_length=4000)
    submit: bool = Field(default=False, description="是否同时提交（提交后冻结）")


class ReviewRequest(BaseModel):
    """主管评语。

    ★ 没有 `requester` / `role`：主管身份从令牌取。
    """

    trainee_no: str = Field(..., min_length=1, description="被评语的学员工号")
    summary_date: str = Field(..., min_length=1)
    note: str = Field(default="", max_length=1000)


class MissLogsResponse(BaseModel):
    """未命中问题清单。

    ★ 不含学员维度聚合（FR-047：不得用于绩效评价）。
    """

    recent: list[dict[str, Any]] = Field(default_factory=list)
    top: list[dict[str, Any]] = Field(default_factory=list)


# ===================================================================
# 认证与账号（P8）
# ===================================================================
class LoginRequest(BaseModel):
    """登录请求。"""

    employee_no: str = Field(..., min_length=1, max_length=64,
                             description="工号")
    password: str = Field(..., min_length=1, max_length=128,
                          description="密码（明文传输请务必走 HTTPS）")


class LoginResponse(BaseModel):
    """登录响应。"""

    token: str = Field(description="Bearer 令牌，放进 Authorization 头")
    employee_no: str = ""
    role: str = ""
    role_label: str = ""


class AccountView(BaseModel):
    """账号信息（**不含密码哈希**）。"""

    employee_no: str
    role: str = "trainee"
    active: bool = True
    last_login_at: str = ""


class ChangeRoleRequest(BaseModel):
    """改角色。"""

    employee_no: str = Field(..., min_length=1)
    role: str = Field(..., description="trainee / manager")


class SetActiveRequest(BaseModel):
    """启用/停用账号。"""

    employee_no: str = Field(..., min_length=1)
    active: bool = Field(...)


class ResetPasswordRequest(BaseModel):
    """主管重置某员工密码。"""

    employee_no: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6, max_length=128)


class CreateAccountRequest(BaseModel):
    """主管为已存在员工开通账号。"""

    employee_no: str = Field(..., min_length=1)
    role: str = Field("trainee", description="trainee / manager")
    password: str = Field(..., min_length=6, max_length=128)


class ChangePasswordRequest(BaseModel):
    """员工修改自己的密码。"""

    old_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=6, max_length=128)


class AuditLogView(BaseModel):
    """审计日志（**不含任何凭据字段**）。"""

    actor: str
    role: str = ""
    action: str
    target: str = ""
    result: str = "ok"
    detail: dict[str, Any] = Field(default_factory=dict)
    ip: str = ""
    created_at: str = ""


# ===================================================================
# 月度报告（P2：图编排）
# ===================================================================
class GenerateReportRequest(BaseModel):
    """生成月度报告（跑图到挂起）。"""

    trainee_id: str = Field("", description="学员编号，空则用当前用户")
    period: str = Field("", description="周期 YYYY-MM，空则用当前月")
    since: str = Field("", description="起始日期 YYYY-MM-DD（含）")
    until: str = Field("", description="结束日期 YYYY-MM-DD（不含）")


class ReviewReportRequest(BaseModel):
    """复核决定。

    ★ 只有两个取值：
        approve（同意）/ reject（驳回 → 重新生成建议）
      驳回不回上游重算数据（见 P2 设计文档的实测理由）。
    """

    decision: str = Field(..., description="approve 或 reject")
    note: str = Field("", description="复核意见（驳回时建议填写，会进提示词）")


class ReportStateResponse(BaseModel):
    """报告任务状态（前端轮询用）。"""

    thread_id: str
    status: str = Field(..., description="pending_review / approved / rejected / unknown")
    trainee_id: str = ""
    period: str = ""
    #: 待复核时给出的摘要（主管看这个决定要不要通过）
    pending: dict[str, Any] = Field(default_factory=dict)
    #: 已定稿时给出的快照
    report: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = ""


# ===================================================================
# 批量生成（P3 队列）
# ===================================================================
class BatchGenerateRequest(BaseModel):
    """批量生成报告的投递请求。"""

    period: str = Field("", description="周期 YYYY-MM，空则用当前月")
    since: str = Field("", description="起始日期 YYYY-MM-DD（含）")
    until: str = Field("", description="结束日期 YYYY-MM-DD（不含）")
    trainee_ids: list[str] = Field(
        default_factory=list,
        description="学员编号列表。空则取全部在职员工",)


class BatchProgressResponse(BaseModel):
    """批量进度（前端轮询）。"""

    period: str
    total: int = 0
    generated: int = 0
    queued: int = 0
    running: int = 0
    pending_review: int = 0
    done: int = 0
    failed: int = 0
    dead: int = 0
    percent: float = 0.0
    enqueued_now: int = 0
    skipped: int = 0


class BatchTaskItem(BaseModel):
    """批量任务的一条明细。"""

    trainee_no: str
    name: str = ""
    status: str = ""
    status_label: str = ""
    attempts: int = 0
    last_error: str = ""
    thread_id: str = ""


class BatchTaskListResponse(BaseModel):
    """批量任务明细列表。"""

    period: str
    items: list[BatchTaskItem] = Field(default_factory=list)
