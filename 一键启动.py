"""本机演示入口：python 一键启动.py；--check 检查，--stop 停止应用。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Callable, TYPE_CHECKING
import urllib.error
import urllib.request
import webbrowser

if TYPE_CHECKING:
    from config.settings import Settings

ROOT = Path(__file__).resolve().parent
APP = ROOT / "项目代码"
PYTHON = APP / ".venv" / "Scripts" / "python.exe"
NEO4J_HOME = (Path.home() / ".Neo4jDesktop2" / "Data" / "dbmss"
              / "dbms-5ee4dc8b-95de-49d0-85c2-387a0d5d70e7")
REQUIRED_PORTS = {"graph", "vector", "keyword", "sparse", "llm",
                  "repository", "cache", "file_storage"}
HIDDEN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class StartupError(RuntimeError):
    """可以直接展示给演示者的启动错误，不包含凭据。"""


def run(command: list[str], timeout: float = 30) -> subprocess.CompletedProcess[str]:
    """以参数列表运行命令，兼容中文路径并避免 shell 转义。"""
    result = subprocess.run(command, cwd=ROOT, capture_output=True,
                            timeout=timeout, creationflags=HIDDEN)
    return subprocess.CompletedProcess(result.args, result.returncode,
                                       decode_output(result.stdout), decode_output(result.stderr))


def decode_output(data: bytes) -> str:
    """兼容 Docker 的 UTF-8 输出与中文 Windows 子进程的 GBK 输出。"""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("gb18030", errors="replace")


def port_open(host: str, port: int) -> bool:
    """只检查 TCP 连通性，凭据和数据在后续独立核验。"""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def wait_for(label: str, check: Callable[[], bool], timeout: int,
             process: subprocess.Popen[bytes] | None = None) -> None:
    """等待就绪；超时或新启动进程提前退出时给出明确错误。"""
    deadline = time.monotonic() + timeout
    last_notice = 0.0
    while time.monotonic() < deadline:
        if check():
            print(f"[就绪] {label}", flush=True)
            return
        if process is not None and process.poll() is not None:
            raise StartupError(f"{label} 进程已退出，请查看项目代码下的启动日志。")
        now = time.monotonic()
        if now - last_notice >= 10:
            print(f"[等待] {label}，请稍候…", flush=True)
            last_notice = now
        time.sleep(1)
    raise StartupError(f"等待 {label} 超时（{timeout} 秒），请检查服务或启动日志。")


def get_json(url: str) -> dict[str, Any] | None:
    """读取本机健康接口，不经过系统代理。"""
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=2) as response:
            data = json.load(response)
        return data if isinstance(data, dict) else None
    except (OSError, urllib.error.URLError, ValueError):
        return None


def app_ready(url: str) -> bool:
    """同时校验应用身份及七个端口，避免把其他服务认成本项目。"""
    health = get_json(url + "/api/health") or {}
    document = get_json(url + "/openapi.json") or {}
    return (document.get("info", {}).get("title") == "健康顾问培训系统"
            and health.get("status") == "ok"
            and REQUIRED_PORTS <= set(health.get("ready_ports", []))
            and not health.get("pending_ports"))


def spawn(command: list[str], log_name: str,
          env: dict[str, str]) -> subprocess.Popen[bytes]:
    """后台运行进程，输出保存到项目目录，关闭本启动窗口后仍可演示。"""
    with (APP / log_name).open("ab") as log:
        return subprocess.Popen(command, cwd=APP, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT,
                                creationflags=HIDDEN)


def ensure_services(settings: Settings, timeout: int, env: dict[str, str]) -> None:
    """按本次部署的服务位置启动依赖，不初始化或重建数据库。"""
    from urllib.parse import urlsplit
    from sqlalchemy.engine import make_url

    mysql = make_url(settings.database_url)
    mysql_host, mysql_port = mysql.host or "localhost", mysql.port or 3306
    neo = urlsplit(settings.neo4j_uri)
    neo_host, neo_port = neo.hostname or "localhost", neo.port or 7687
    if not port_open(mysql_host, mysql_port):
        if mysql_host not in {"localhost", "127.0.0.1"}:
            raise StartupError("远程 MySQL 不可达，请检查数据库服务。")
        result = run(["sc.exe", "start", "MySQL84"])
        if result.returncode != 0:
            raise StartupError("MySQL84 未启动，请用管理员权限启动该服务后重试。")
        wait_for("MySQL", lambda: port_open(mysql_host, mysql_port), timeout)

    if not port_open(neo_host, neo_port):
        if neo_host not in {"localhost", "127.0.0.1"}:
            raise StartupError("远程 Neo4j 不可达，请检查数据库服务。")
        script = NEO4J_HOME / "bin" / "neo4j.ps1"
        runtimes = sorted((Path.home() / ".Neo4jDesktop2" / "Cache" / "runtime").glob("zulu21*"))
        runtimes = [p for p in runtimes if (p / "bin" / "java.exe").is_file()]
        if not script.is_file() or not runtimes:
            raise StartupError("找不到已部署的 Neo4j 或 Java，请在 Neo4j Desktop 中启动 AI0803 实例。")
        neo_env = dict(env, NEO4J_HOME=str(NEO4J_HOME), JAVA_HOME=str(runtimes[-1]))
        process = spawn(["pwsh", "-NoProfile", "-NonInteractive", "-File", str(script), "console"],
                        "demo-neo4j.log", neo_env)
        wait_for("Neo4j", lambda: port_open(neo_host, neo_port), timeout, process)

    if not shutil.which("docker"):
        raise StartupError("找不到 Docker 命令，请确认 Docker Desktop 已安装。")

    def docker_ready() -> bool:
        try:
            return run(["docker", "info", "--format", "{{.ServerVersion}}"], 5).returncode == 0
        except subprocess.TimeoutExpired:
            return False

    if not docker_ready():
        desktop = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Docker/Docker/Docker Desktop.exe"
        if not desktop.is_file():
            raise StartupError("找不到 Docker Desktop，请手动启动后重试。")
        spawn([str(desktop)], "demo-docker.log", env)
        wait_for("Docker Desktop", docker_ready, timeout)
    print("[启动] 本项目的 Milvus 容器", flush=True)
    result = run(["docker", "compose", "-f", str(ROOT / "docker-compose.yml"),
                  "up", "-d", "--pull", "never"], timeout)
    if result.returncode:
        (APP / "demo-compose.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        raise StartupError("Milvus 容器启动失败，请查看 项目代码/demo-compose.log。")
    wait_for("Milvus", lambda: port_open(settings.milvus_host, settings.milvus_port), timeout)


def check_data(settings: Settings) -> None:
    """实连三库检查演示数据，避免仅凭端口或容器状态判断就绪。"""
    from neo4j import GraphDatabase
    from pymilvus import MilvusClient
    import pymysql
    from sqlalchemy.engine import make_url

    target = "Neo4j"
    try:
        with GraphDatabase.driver(settings.neo4j_uri,
                                  auth=(settings.neo4j_user, settings.neo4j_password),
                                  connection_timeout=5) as driver:
            with driver.session(database=settings.neo4j_database) as session:
                row = session.run("MATCH (n) RETURN count(n) AS n").single()
                if row is None or row["n"] == 0:
                    raise StartupError("Neo4j 目标库没有图谱数据，请检查已部署的数据库。")
        target = "MySQL"
        url = make_url(settings.database_url)
        with pymysql.connect(host=url.host or "localhost", port=url.port or 3306,
                             user=url.username, password=url.password, database=url.database,
                             charset="utf8mb4", connect_timeout=5, read_timeout=5) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = {row[0] for row in cursor.fetchall()}
            required = {"employee", "employee_account", "audit_log", "exam_paper",
                        "exam_paper_question", "question_usage", "drill_record",
                        "drill_review", "drill_risk_item", "monthly_report",
                        "report_task", "daily_summary", "miss_log"}
            if not required <= tables:
                raise StartupError("MySQL 业务表不完整，请检查已部署的数据库。")
        target = "Milvus"
        client = MilvusClient(uri=settings.milvus_uri, timeout=5)
        try:
            for name in (settings.coll_training, settings.coll_case,
                         settings.coll_question, settings.coll_self_test):
                if not client.query(collection_name=name, filter="", output_fields=["chunk_id"], limit=1):
                    raise StartupError(f"Milvus 的 {name} 没有数据。")
        finally:
            client.close()
    except StartupError:
        raise
    except Exception as exc:
        # 驱动错误或连接串可能包含凭据，仅显示目标服务与错误类型。
        raise StartupError(f"{target} 验证失败（{type(exc).__name__}），请检查服务、密码或库名。") from None
    print("[通过] Neo4j、MySQL 和四个 Milvus 知识库均可读取", flush=True)


def main() -> int:
    """复用现有应用，或启动并等待就绪后打开演示页面。"""
    parser = argparse.ArgumentParser(description="健康培训系统一键演示启动")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="只检查现有服务与数据，不启动或停止")
    action.add_argument("--stop", action="store_true", help="只停止本项目后台应用，保留数据库服务")
    parser.add_argument("--no-browser", action="store_true", help="就绪后不打开浏览器")
    parser.add_argument("--timeout", type=int, default=180, help="每个启动步骤等待秒数（默认 180）")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout 必须大于 0")
    if args.stop:
        result = run(["pwsh", "-NoProfile", "-File", str(ROOT / "停止项目.ps1")])
        print(result.stdout.strip() or "停止命令已执行。", flush=True)
        if result.returncode:
            raise StartupError("停止应用失败，请在对应终端检查进程状态。")
        return 0
    if not PYTHON.is_file():
        raise StartupError("项目虚拟环境不存在，请先按本机部署记录恢复 .venv。")
    # 用户可直接用系统 Python 双击/运行，由入口切到项目解释器。
    if Path(sys.prefix).resolve() != (APP / ".venv").resolve():
        return subprocess.call([str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.path.insert(0, str(APP))
    from config.settings import Settings

    try:
        settings = Settings()
    except Exception as exc:
        raise StartupError(f"读取项目配置失败（{type(exc).__name__}），请核对 .env。") from None
    model = Path(settings.embed_model_dir)
    if not (model / "1_Pooling/config.json").is_file() or not any(
        (model / name).is_file() for name in ("pytorch_model.bin", "model.safetensors")
    ):
        raise StartupError("bge-m3 模型文件不完整，请检查 .env 的 EMBED_MODEL_DIR。")
    if not settings.llm_enabled:
        raise StartupError("尚未配置可用 LLM 密钥，请检查 .env。")
    host = "127.0.0.1" if settings.http_host in {"0.0.0.0", "localhost"} else settings.http_host
    url = f"http://{host}:{settings.http_port}"
    env = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    print("健康培训系统 · 演示启动", flush=True)
    if not args.check:
        ensure_services(settings, args.timeout, env)
    check_data(settings)
    if not app_ready(url):
        if args.check:
            raise StartupError("三库可读，但应用尚未就绪。请运行 python 一键启动.py。")
        if port_open(host, settings.http_port):
            raise StartupError(f"{settings.http_port} 端口已有进程，但不是就绪的本项目应用；请检查原服务。")
        print("[启动] 应用正在加载 bge-m3 模型…", flush=True)
        process = spawn([str(PYTHON), "-u", "serve.py"], "deployment-server.log", env)
        (APP / "deployment-server.pid").write_text(str(process.pid), encoding="utf-8")
        wait_for("培训应用（7/7 端口）", lambda: app_ready(url), args.timeout, process)
    print(f"\n[成功] {url}/\n学员：E001 / trainee123\n主管：M001 / admin123", flush=True)
    print("停止应用：python 一键启动.py --stop", flush=True)
    if not args.check and not args.no_browser:
        if not webbrowser.open(url + "/"):
            print("浏览器未能自动打开，请点击上面的地址。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (StartupError, OSError, subprocess.SubprocessError) as error:
        print(f"\n[启动失败] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\n启动等待已取消。已启动的后台服务可用 --stop 停止。", flush=True)
        raise SystemExit(130)
