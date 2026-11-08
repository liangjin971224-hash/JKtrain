"""门禁自证：证明门禁真的能拦住违规。

为什么需要这个测试：
    一个永远返回 PASS 的门禁和没有门禁是一样的。
    本测试通过「临时注入违规代码 → 跑门禁 → 必须报 FAIL → 清理」
    来证明门禁的有效性。

注入的文件都带 `_gate_probe_` 前缀，测试结束必定删除。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "ci_gate.py"

PROBE_PREFIX = "_gate_probe_"


def _run_gate() -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(GATE)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.fixture()
def inject():
    """注入临时文件，测试结束后清理（即使断言失败也要清）。"""
    created: list[Path] = []

    def _make(layer: str, content: str) -> Path:
        path = ROOT / layer / f"{PROBE_PREFIX}{len(created)}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(path)
        return path

    yield _make

    for p in created:
        p.unlink(missing_ok=True)


class TestGatesActuallyCatchViolations:
    """每条门禁都必须能被真实违规触发。"""

    def test_gate1_catches_framework_import_in_domain(self, inject) -> None:
        """domain 里 import pymilvus → 门禁 1 必须 FAIL。"""
        inject("domain", "import pymilvus\n\n\ndef f():\n    return None\n")

        code, out = _run_gate()
        assert code == 1, f"门禁未能拦住 domain 层框架依赖:\n{out}"
        assert "门禁1" in out and "FAIL" in out

    def test_gate2_catches_infrastructure_import_in_interfaces(self, inject) -> None:
        """interfaces 里 import infrastructure → 门禁 2 必须 FAIL。"""
        inject("interfaces",
               "from infrastructure.graph.neo4j_graph_adapter import "
               "Neo4jGraphAdapter\n")

        code, out = _run_gate()
        assert code == 1, f"门禁未能拦住跨层依赖:\n{out}"
        assert "门禁2" in out and "FAIL" in out

    def test_gate3_catches_scattered_safety_logic(self, inject) -> None:
        """application 里定义禁忌判定函数 → 门禁 3 必须 FAIL。"""
        inject("application",
               "def check_contraindication(pid, cids):\n"
               "    return []\n")

        code, out = _run_gate()
        assert code == 1, f"门禁未能拦住分散的安全判定:\n{out}"
        assert "门禁3" in out and "FAIL" in out

    def test_gate3_still_catches_other_safety_names(self, inject) -> None:
        """白名单只放行端口契约方法，其他安全命名仍必须被拦。

        这条防止「为了让门禁变绿而扩大白名单」——白名单只该有一个名字。
        """
        inject("application",
               "def check_forbidden_claims(text):\n"
               "    return []\n")

        code, out = _run_gate()
        assert code == 1, f"白名单被滥用，漏掉了 forbidden_claims:\n{out}"

    def test_clean_state_passes(self) -> None:
        """无违规时门禁必须 PASS（否则门禁本身坏了）。"""
        code, out = _run_gate()
        assert code == 0, f"干净代码上门禁误报:\n{out}"
        assert "全部通过" in out

    def test_no_probe_files_left(self) -> None:
        """确认探测文件没残留（本测试依赖执行顺序，放最后）。"""
        leftovers = list(ROOT.rglob(f"{PROBE_PREFIX}*.py"))
        assert not leftovers, f"门禁探测文件残留: {leftovers}"
