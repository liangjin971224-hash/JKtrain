"""领域异常。

三层异常体系的第一层：
    domain/exceptions.py          领域异常（业务规则被违反）
    infrastructure/exceptions.py  基础设施异常（连接、超时）
    interfaces/http/errors.py     协议异常（400/404/500 映射）

★ 重要区分（来自 coding-conventions §5.2）：
    产品不存在        → 异常（调用前提不成立）
    产品对用户禁忌    → 返回值（这是正常业务结果）
    合规检测命中禁用词 → 返回值（正常业务结果）
    Neo4j 连接失败    → 异常（基础设施故障）

把「业务结果」当异常会污染控制流。
"""
from __future__ import annotations


class DomainError(Exception):
    """所有领域异常的基类。"""


class ProductNotFoundError(DomainError):
    """产品不存在。

    只在「ID 查不到具体产品」时抛。注意与禁忌命中区分：
    禁忌命中是返回值，不是异常。
    """

    def __init__(self, product_id: str) -> None:
        super().__init__(f"产品不存在: {product_id}")
        self.product_id = product_id


class CaseNotFoundError(DomainError):
    """演练案例不存在。"""

    def __init__(self, case_id: str) -> None:
        super().__init__(f"案例不存在: {case_id}")
        self.case_id = case_id


class QuestionNotFoundError(DomainError):
    """题目不存在。"""

    def __init__(self, question_id: str) -> None:
        super().__init__(f"题目不存在: {question_id}")
        self.question_id = question_id


class InvalidPermissionLevelError(DomainError):
    """权限级别取值非法。"""

    def __init__(self, value: str) -> None:
        super().__init__(
            f"非法权限级别: {value}（仅允许 public/internal/restricted/confidential）"
        )
        self.value = value


class UnsupportedFileTypeError(DomainError):
    """上传文件类型不在白名单（pdf/jpg/png）。"""

    def __init__(self, suffix: str) -> None:
        super().__init__(f"不支持的文件类型: {suffix}（仅允许 .pdf/.jpg/.jpeg/.png）")
        self.suffix = suffix


class FileTooLargeError(DomainError):
    """上传文件超过大小上限。"""

    def __init__(self, size_bytes: int, limit_bytes: int) -> None:
        super().__init__(
            f"文件超过大小上限: {size_bytes / 1024**2:.1f}MB > "
            f"{limit_bytes / 1024**2:.0f}MB"
        )
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes


class PathTraversalError(DomainError):
    """文件路径逃出了存储根目录。

    `sub_path` 来自用户输入，必须校验解析后仍在 root 内。
    """

    def __init__(self, sub_path: str) -> None:
        super().__init__(f"非法路径（疑似目录穿越）: {sub_path}")
        self.sub_path = sub_path


class ExamPaperLockedError(DomainError):
    """试卷已激活，不可再改题目（P6 的唯一业务不变量）。"""

    def __init__(self, paper_id: str, status: str) -> None:
        super().__init__(f"试卷 {paper_id} 当前状态为 {status}，不允许修改")
        self.paper_id = paper_id
        self.status = status


class DuplicateDailySummaryError(DomainError):
    """同人同天重复提交日报。

    唯一约束 `uk_trainee_date (trainee_no, summary_date)` 的领域表达。
    这是业务不变量，不是技术错误。
    """

    def __init__(self, trainee_no: str, summary_date: str) -> None:
        super().__init__(f"{trainee_no} 在 {summary_date} 已有日报，每人每天仅一条")
        self.trainee_no = trainee_no
        self.summary_date = summary_date


class DraftAlreadySubmittedError(DomainError):
    """已提交的日报不允许再改（P7 约束）。"""

    def __init__(self, trainee_no: str, summary_date: str) -> None:
        super().__init__(f"{trainee_no} 在 {summary_date} 的日报已提交，不可修改")
        self.trainee_no = trainee_no
        self.summary_date = summary_date


class EmbeddingDimensionError(DomainError):
    """向量维度与配置不符。

    维度不符会让 Milvus 检索直接失败，且报错信息往往很晦涩，
    所以在适配器层就拦下并给出可执行的提示。
    """

    def __init__(self, actual: int, expected: int) -> None:
        super().__init__(
            f"向量维度不符: 实际 {actual}, 期望 {expected}。"
            f"请检查 EMBED_MODEL_DIR 与 EXPECTED_EMBED_DIM 是否指向同一模型"
        )
        self.actual = actual
        self.expected = expected


class ExamBuildError(DomainError):
    """组卷失败。

    ★ 报错必须**说清缺什么**（需求 FR-037 原文要求）。
      只说「组卷失败」等于没说，管理员不知道要补哪类题。

    典型原因：
        · 某知识点题库为空
        · 去重窗口内题量不足
        · 题量为 0 / 参数非法
    """


class InvalidExamTransition(DomainError):
    """试卷状态流转非法（FR-040：状态单向，不可回退）。"""


class DuplicateEmployeeError(DomainError):
    """工号重复（员工档案导入时）。"""


class DailySummaryFrozenError(DomainError):
    """每日总结已提交，不可修改。

    ★ FR-045 原文：「草稿可反复修改，提交后不可修改」。
      冻结是硬约束——提交意味着学员确认了内容，
      允许改就等于日报失去「当时真实想法」的价值。
    """


class DataPermissionError(DomainError):
    """数据权限不足。

    ★ 与功能权限区分（FR-048 vs FR-049）：
      功能权限 = 能不能用这个功能（组卷、看日报）
      数据权限 = 能看到谁的数据（自己的 / 所辖的）
    """


class AuthenticationError(DomainError):
    """认证失败（未登录 / 凭据错误 / 令牌失效）。

    ★ 报错信息**不能区分**「用户不存在」和「密码错误」。
      区分了就等于给攻击者一个账号枚举接口。

      统一话术：「工号或密码不正确」。
    """


class PermissionDeniedError(DomainError):
    """授权失败（已登录，但没这个功能的权限）。

    ★ 与 AuthenticationError 分开：
        401 = 你是谁我不知道 → 去登录
        403 = 我知道你是谁，但你不能做这个
      混用会让前端无法正确处理（登录态失效 vs 权限不足）。
    """
