"""CI 门禁：3 条架构红线。

与 `tests/test_architecture.py` 的关系：
    测试用于开发期（pytest，带详细断言信息）；
    本脚本用于 CI / pre-commit（快速、纯退出码、无 pytest 依赖）。

阈值：全部为「零命中」，有任何一条命中即 exit 1。

用法：
    python scripts/ci_gate.py          # 跑全部
    python scripts/ci_gate.py --fix-hint
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FRAMEWORKS = (
    "fastapi", "starlette", "uvicorn",
    "sqlalchemy", "pymysql", "pymilvus", "neo4j", "redis",
    "langchain", "langgraph", "openai", "torch",
    "sentence_transformers", "transformers", "numpy",
    "pydantic", "pydantic_settings",
)

#: 端口契约规定的方法名 —— 它们是「查数据」，不是「做判定」，必须放行。
#:
#: 为什么需要这个白名单：
#:     `GraphQueryPort.check_contraindication` 是端口契约要求的方法，
#:     adapter 实现它只是「把图谱里的禁忌查出来并转成领域对象」，
#:     不含任何判定分支（不判断该不该拦、不决定措辞）。
#:     真正的判定逻辑在 `domain/rules/` 与 `domain/services/safety_validator.py`。
#:     若不放行，门禁会把「查询」误判为「分散的安全判定」，
#:     逼着开发者改端口契约或绕过门禁 —— 那才是真正的架构腐蚀。
#:
#: ★ 白名单必须**同时限定方法名与所在文件**，不能只按名字放行。
#:   只按名字放行的话，任何地方写 `def check_contraindication()` 都能绕过去，
#:   门禁等于失效。这条是被 `test_gate3_catches_scattered_safety_logic` 抓出来的。
#:
#: 判定标准（改这里前必须满足）：
#:     该函数体内**不得**出现关于 severity / must_block / allowed / blocked
#:     的条件分支。`tests/test_architecture.py` 的
#:     `test_port_contract_method_has_no_decision_logic` 守着这条。
PORT_CONTRACT_ALLOWLIST = {
    # 文件相对路径（正斜杠） → 该文件里允许放行的方法名
    "infrastructure/graph/neo4j_graph_adapter.py": frozenset({
        "check_contraindication",
    }),
}

SAFETY_ROOTS = ("contraindicat", "forbidden_claim", "allergen")


def py_files(layer: str) -> list[Path]:
    base = ROOT / layer
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*.py") if p.name != "__init__.py")


def imports_of(path: Path) -> list[tuple[str, int]]:
    """AST 解析 import。用 AST 而非 grep，避免注释/docstring 误报。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((a.name, node.lineno) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.module, node.lineno))
    return out


def has_prefix(mod: str, prefixes: tuple[str, ...]) -> bool:
    return any(mod == p or mod.startswith(p + ".") for p in prefixes)


# ------------------------------------------------------------------
def gate1_domain_no_framework() -> list[str]:
    """门禁 1：domain 层零框架依赖。"""
    bad = []
    for path in py_files("domain"):
        for mod, line in imports_of(path):
            if has_prefix(mod, FRAMEWORKS):
                bad.append(f"{path.relative_to(ROOT)}:{line}  import {mod}")
    return bad


def gate2_no_layer_shortcut() -> list[str]:
    """门禁 2：interfaces / application 不得直接依赖 infrastructure。"""
    bad = []
    for layer in ("interfaces", "application"):
        for path in py_files(layer):
            for mod, line in imports_of(path):
                if has_prefix(mod, ("infrastructure",)):
                    bad.append(f"{path.relative_to(ROOT)}:{line}  import {mod}")
    return bad


def gate3_safety_not_scattered() -> list[str]:
    """门禁 3：安全判定的**定义**只能在 domain 内。

    只查定义（def / class），不查调用——
    「调用 safety_validator」是允许的，「自己写一套禁忌判定」不允许。

    ★ 词根必须用 `contraindicat` 而不是 `contraindicated`：
      前者能同时覆盖 contraindication / contraindications / contraindicated，
      后者只覆盖过去分词形式。这个差别是「门禁自证测试」抓出来的——
      最初写 `contraindicated`，导致 `check_contraindication()`
      这个**最常见**的命名反而漏过。

    ★ 端口契约方法放行，但**必须同时匹配文件路径与方法名**
      （见 `PORT_CONTRACT_ALLOWLIST`）。只按名字放行会被绕过。
    """
    import re

    roots = SAFETY_ROOTS
    pattern = re.compile(
        r"^\s*(?:(?:async\s+)?def|class)\s+\w*(" + "|".join(roots) + r")\w*",
        re.M | re.I,
    )
    bad = []
    for layer in ("application", "interfaces", "infrastructure"):
        for path in py_files(layer):
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            allowed_here = PORT_CONTRACT_ALLOWLIST.get(rel, frozenset())

            text = path.read_text(encoding="utf-8")
            for m in pattern.finditer(text):
                full_name = m.group(0).split()[-1].split("(")[0]
                if full_name in allowed_here:
                    continue
                line = text[:m.start()].count("\n") + 1
                bad.append(
                    f"{rel}:{line}  {m.group(0).strip()}")
    return bad


GATES = (
    ("门禁1 domain 层零框架", gate1_domain_no_framework,
     "domain 只能依赖标准库与自身。技术细节移到 infrastructure。"),
    ("门禁2 无跨层捷径", gate2_no_layer_shortcut,
     "interfaces/application 只依赖 domain.ports，不 import 具体 Adapter。"),
    ("门禁3 安全判定不分散", gate3_safety_not_scattered,
     "禁忌/合规判定只在 domain/rules 与 domain/services 定义。"),
)


def main() -> int:
    width = 62
    print("=" * width)
    print("架构 CI 门禁（3 条红线）")
    print("=" * width)

    failed = 0
    for name, fn, hint in GATES:
        try:
            violations = fn()
        except Exception as exc:                  # noqa: BLE001
            print(f"[ERROR] {name}: 检查自身异常 {exc}")
            failed += 1
            continue

        if violations:
            failed += 1
            print(f"[FAIL ] {name}  —— {len(violations)} 处违规")
            for v in violations:
                print(f"         {v}")
            print(f"         修法: {hint}")
        else:
            print(f"[PASS ] {name}")

    print("-" * width)
    if failed:
        print(f"结论: {failed} 条门禁未通过，禁止合入")
        return 1
    print("结论: 3 条门禁全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
