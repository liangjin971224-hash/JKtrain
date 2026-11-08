"""架构门禁测试（P0 阶段核心）。

本文件是「红灯先行」的落点：
P0 开工时先写这里，跑一遍应当**全部失败**，因为分层实现还不存在。
实现完成后转绿，才算骨架真的立住了。

设计要点：
- 不依赖任何外部服务（零 IO），可在无网络、无数据库的机器上跑
- 用 AST 解析而非 grep，避免"注释里提到 fastapi"这类误报
- 违规时打印出**具体文件:行号**，不是只说一句"失败"
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# 项目代码根目录（tests/ 的上一级）
ROOT = Path(__file__).resolve().parents[1]

DOMAIN = ROOT / "domain"
APPLICATION = ROOT / "application"
INTERFACES = ROOT / "interfaces"
INFRASTRUCTURE = ROOT / "infrastructure"
CONFIG = ROOT / "config"


# ===================================================================
# 工具：AST 级 import 解析
# ===================================================================
def _iter_py_files(base: Path) -> list[Path]:
    """递归取 .py 文件。目录不存在时返回空，让测试报"缺失"而不是崩。"""
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*.py") if p.name != "__init__.py")


def _imported_modules(path: Path) -> list[tuple[str, int]]:
    """解析一个文件的所有 import，返回 [(模块全名, 行号)]。

    用 AST 而不是正则：正则会把 docstring/注释里的文字也算命中。
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:      # 语法错误的文件本身就该暴露
        pytest.fail(f"无法解析 {path}: {exc}")

    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            # 相对导入解析不出绝对模块名，单独标记出来（本项目禁止相对导入）
            if node.level and node.level > 0:
                out.append((f"<relative:{'.' * node.level}{node.module or ''}>",
                            node.lineno))
            elif node.module:
                out.append((node.module, node.lineno))
    return out


def _match_prefix(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _format_violations(items: list[tuple[Path, str, int]]) -> str:
    return "\n".join(f"  {p.relative_to(ROOT)}:{ln}  →  {m}" for p, m, ln in items)


# ===================================================================
# 红灯 0：防止「空目录假绿」
# ===================================================================
def test_layers_are_not_empty() -> None:
    """各层必须有实际代码文件，否则门禁是空转的假绿。

    为什么先测这个：P0 开工时四层目录都是空的，
    「domain 无框架依赖」「interfaces 不依赖 infrastructure」这类断言
    在空目录上会**天然通过**——看起来全绿，其实什么都没验证。
    这条测试保证门禁测的是真代码。
    """
    required = {
        "domain": DOMAIN,
        "application": APPLICATION,
        "interfaces": INTERFACES,
        "infrastructure": INFRASTRUCTURE,
        "config": CONFIG,
    }
    empty = [name for name, base in required.items() if not _iter_py_files(base)]
    assert not empty, (
        f"以下层没有任何 .py 文件，架构门禁会空转通过: {empty}\n"
        f"  P0 要求四层都有实际内容（domain/ports 至少 7 个端口定义）"
    )


# ===================================================================
# 红灯 1：domain 层零框架依赖
# ===================================================================
FRAMEWORK_MODULES = (
    "fastapi", "starlette", "uvicorn",
    "sqlalchemy", "pymysql", "pymilvus", "neo4j", "redis",
    "langchain", "langgraph", "openai", "torch",
    "sentence_transformers", "transformers", "numpy",
    "pydantic", "pydantic_settings",
)


def test_domain_has_no_framework_import() -> None:
    """domain 层不得 import 任何框架/SDK。

    为什么：domain 是「去掉技术外壳后剩下的专业逻辑」。
    它必须能在无网、无库、无模型的机器上跑测试。
    一旦引入 pydantic/neo4j，换框架就要改领域代码，架构即失效。

    注意 pydantic 也在禁止列表：它是 Web 层的请求校验工具，
    不是领域概念。领域类型用 dataclass + StrEnum。
    """
    violations = [
        (p, m, ln)
        for p in _iter_py_files(DOMAIN)
        for m, ln in _imported_modules(p)
        if _match_prefix(m, FRAMEWORK_MODULES)
    ]
    assert not violations, (
        "domain 层混入框架依赖（违反架构红线 3）：\n" + _format_violations(violations)
    )


def test_domain_has_no_io() -> None:
    """domain 层不得做任何 IO（读文件、开连接、发请求）。

    与上一条互补：即使不 import 框架，用 open() 读配置也算越界。
    """
    # 只匹配真正的调用，不匹配变量名里带这些词的情况
    forbidden_calls = {"open", "print"}
    forbidden_attrs = {"read_text", "write_text", "read_bytes", "write_bytes",
                       "connect", "urlopen", "getenv"}

    violations: list[str] = []
    for path in _iter_py_files(DOMAIN):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in forbidden_calls:
                    violations.append(
                        f"  {path.relative_to(ROOT)}:{node.lineno}  →  {fn.id}()")
                elif isinstance(fn, ast.Attribute) and fn.attr in forbidden_attrs:
                    violations.append(
                        f"  {path.relative_to(ROOT)}:{node.lineno}  →  .{fn.attr}()")

    assert not violations, "domain 层出现 IO 操作：\n" + "\n".join(violations)


def test_domain_has_no_relative_import() -> None:
    """全项目禁止相对导入（`from ..x import y`）。

    原因：相对导入在 pytest 收集阶段容易出错（RAG 骨架的实测教训）。
    """
    violations = []
    for base in (DOMAIN, APPLICATION, INTERFACES, INFRASTRUCTURE, CONFIG):
        for path in _iter_py_files(base):
            for mod, ln in _imported_modules(path):
                if mod.startswith("<relative:"):
                    violations.append((path, mod, ln))
    assert not violations, (
        "存在相对导入，请改为绝对导入：\n" + _format_violations(violations)
    )


# ===================================================================
# 红灯 2：interfaces 不得依赖 infrastructure
# ===================================================================
def test_interfaces_does_not_import_infrastructure() -> None:
    """接口层不得直接 import 基础设施。

    接口层只能经 application 层调用能力。
    直接 import adapter 会导致换实现要改路由，且无法用假实现测接口。
    """
    violations = [
        (p, m, ln)
        for p in _iter_py_files(INTERFACES)
        for m, ln in _imported_modules(p)
        if _match_prefix(m, ("infrastructure",))
    ]
    assert not violations, (
        "interfaces 直接依赖 infrastructure：\n" + _format_violations(violations)
    )


def test_application_does_not_import_infrastructure() -> None:
    """应用层不得 import 具体 Adapter，只能依赖 Port 抽象。

    这是「能用假实现测试用例」的前提。
    """
    violations = [
        (p, m, ln)
        for p in _iter_py_files(APPLICATION)
        for m, ln in _imported_modules(p)
        if _match_prefix(m, ("infrastructure",))
    ]
    assert not violations, (
        "application 直接依赖 infrastructure（应改为依赖 domain.ports）：\n"
        + _format_violations(violations)
    )


def test_domain_does_not_import_outer_layers() -> None:
    """domain 不得反向依赖任何外层（依赖方向只能由外向内）。"""
    violations = [
        (p, m, ln)
        for p in _iter_py_files(DOMAIN)
        for m, ln in _imported_modules(p)
        if _match_prefix(m, ("application", "infrastructure", "interfaces"))
    ]
    assert not violations, (
        "domain 反向依赖外层：\n" + _format_violations(violations)
    )


# ===================================================================
# 红灯 3：安全判定逻辑唯一
# ===================================================================
def test_safety_logic_not_scattered() -> None:
    """禁忌/合规定义只允许出现在 domain/rules/ 与 domain/services/。

    允许外层「调用」，不允许外层「定义」。

    ★ 词根用 `contraindicat`（不是 `contraindicated`）：
      必须覆盖 contraindication / contraindications / contraindicated
      三种形态。用过去分词会把最常见的 `check_contraindication()` 漏掉
      ——这个缺陷是被 `tests/test_ci_gate.py` 的自证测试抓出来的。

    ★ 端口契约方法放行，但必须**同时匹配文件路径与方法名**：
      只按名字放行的话，任何地方写 `def check_contraindication()`
      都能绕过门禁，等于门禁失效。
      这条约束由 `test_port_contract_method_has_no_decision_logic` 守着。
    """
    definition_pattern = re.compile(
        r"^\s*(?:(?:async\s+)?def|class)\s+"
        r"\w*(?:contraindicat|forbidden_claim|allergen)\w*",
        re.M | re.I,
    )
    # 文件相对路径 → 该文件里允许放行的方法名（与 ci_gate.py 保持一致）
    allowlist = {
        "infrastructure/graph/neo4j_graph_adapter.py": {"check_contraindication"},
    }

    violations: list[str] = []
    for base in (APPLICATION, INTERFACES, INFRASTRUCTURE):
        for path in _iter_py_files(base):
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            allowed_here = allowlist.get(rel, set())

            text = path.read_text(encoding="utf-8")
            for m in definition_pattern.finditer(text):
                full_name = m.group(0).split()[-1].split("(")[0]
                if full_name in allowed_here:
                    continue
                line = text[:m.start()].count("\n") + 1
                violations.append(
                    f"  {rel}:{line}  →  {m.group(0).strip()}")

    assert not violations, (
        "安全判定逻辑定义在 domain 之外（应集中到 domain/rules/）：\n"
        + "\n".join(violations)
    )


def test_port_contract_method_has_no_decision_logic() -> None:
    """放行的端口契约方法，函数体内不得有判定分支。

    这是门禁 3 白名单的守门测试。
    `check_contraindication` 之所以能放行，是因为它只「查数据」——
    若有人在里面加 `if hit.severity.must_block: return ...` 这类判定，
    它就变成了第二处安全判定入口，白名单的前提就不成立了。

    判据：函数体内不得出现 severity / must_block / allowed / blocked
    与条件语句的组合。
    """
    decision_pattern = re.compile(
        r"\bif\b[^\n]*\b(severity|must_block|allowed|blocked)\b"
        r"|\b(severity|must_block)\b[^\n]*\b(return|raise)\b"
    )

    violations: list[str] = []
    target = INFRASTRUCTURE / "graph" / "neo4j_graph_adapter.py"
    if not target.exists():
        pytest.skip("adapter 尚未创建")

    tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            if fn.name not in {"check_contraindication"}:
                continue
            body = ast.get_source_segment(
                target.read_text(encoding="utf-8"), fn) or ""
            if decision_pattern.search(body):
                violations.append(f"{cls.name}.{fn.name}")

    assert not violations, (
        f"端口契约方法里出现了判定分支，白名单前提不成立: {violations}\n"
        f"  判定逻辑应移到 domain/services/safety_validator.py"
    )


def test_no_print_in_non_cli_layers() -> None:
    """application / domain / infrastructure 禁止 print。

    日志用 logging；print 只允许在 CLI 与 main.py。
    """
    violations: list[str] = []
    for base in (DOMAIN, APPLICATION, INFRASTRUCTURE):
        for path in _iter_py_files(base):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "print"):
                    violations.append(
                        f"  {path.relative_to(ROOT)}:{node.lineno}  →  print()")
    assert not violations, "非 CLI 层出现 print()：\n" + "\n".join(violations)


# ===================================================================
# 结构完整性：这些测试保护"骨架真的存在"
# ===================================================================
SEVEN_PORTS = [
    "graph_query_port",
    "vector_search_port",
    "keyword_search_port",
    "language_model_port",
    "repository_port",
    "cache_port",
    "file_storage_port",
]


@pytest.mark.parametrize("port_file", SEVEN_PORTS)
def test_port_defined(port_file: str) -> None:
    """7 个端口文件必须存在。"""
    assert (DOMAIN / "ports" / f"{port_file}.py").exists(), (
        f"缺少端口定义: domain/ports/{port_file}.py"
    )


@pytest.mark.parametrize("port_file", SEVEN_PORTS)
def test_port_has_no_implementation(port_file: str) -> None:
    """Port 里的方法必须全部是 @abstractmethod，不能有实现。

    判据：ABC 子类里除 __init__ / 属性之外的 def 都应有 @abstractmethod。
    """
    path = DOMAIN / "ports" / f"{port_file}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    missing: list[str] = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, (ast.FunctionDef,
                                                         ast.AsyncFunctionDef))]:
            if fn.name.startswith("_"):
                continue        # __init__、私有辅助方法不计
            decorators = {
                d.id if isinstance(d, ast.Name) else
                getattr(d, "attr", "") for d in fn.decorator_list
            }
            if "abstractmethod" not in decorators:
                missing.append(f"{cls.name}.{fn.name}")

    assert not missing, (
        f"{port_file} 中存在未声明 abstractmethod 的公开方法"
        f"（Port 只应声明能力，不含实现）: {missing}"
    )


def test_every_port_has_fake() -> None:
    """每个 Port 必须有对应的测试替身。

    没有假实现，应用层测试就只能连真库，违背「绝大多数测试不依赖外部服务」。
    """
    fakes_dir = ROOT / "tests" / "fakes"
    assert fakes_dir.exists(), "缺少 tests/fakes/ 目录"

    fake_text = "\n".join(
        p.read_text(encoding="utf-8") for p in fakes_dir.rglob("*.py")
    )
    port_classes = [
        "GraphQueryPort", "VectorSearchPort", "KeywordSearchPort",
        "LanguageModelPort", "RepositoryPort", "CachePort", "FileStoragePort",
    ]
    missing = [c for c in port_classes if f"(domain_port" not in fake_text
               and f"({c})" not in fake_text]
    assert not missing, f"以下 Port 缺少测试替身: {missing}"


def test_no_hardcoded_connection_strings() -> None:
    """业务代码不得硬编码连接串/密码。只有 config/settings.py 可以有默认值。"""
    pattern = re.compile(r"(bolt://|redis://|mysql\+pymysql://|api_key\s*=\s*['\"])")
    violations: list[str] = []
    for base in (DOMAIN, APPLICATION, INTERFACES, INFRASTRUCTURE):
        for path in _iter_py_files(base):
            text = path.read_text(encoding="utf-8")
            for m in pattern.finditer(text):
                line = text[:m.start()].count("\n") + 1
                violations.append(
                    f"  {path.relative_to(ROOT)}:{line}  →  {m.group(1)}")
    assert not violations, (
        "硬编码连接串（应走 settings）：\n" + "\n".join(violations)
    )


def test_summary_card_has_no_score_or_verdict() -> None:
    """摘要卡不得含 score / verdict 字段（约束 #3）。

    系统只出卡不出分。这条在第 3 个约束里，P0 就要把值对象定死。
    """
    path = DOMAIN / "value_objects" / "summary_card.py"
    if not path.exists():
        pytest.skip("summary_card.py 尚未创建（P5 阶段交付）")

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    banned: list[str] = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for stmt in cls.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id.lower()
                if "score" in name or "verdict" in name or "grade" in name:
                    banned.append(f"{cls.name}.{stmt.target.id}")

    assert not banned, f"摘要卡出现分数/结论字段（违反约束 #3）: {banned}"
