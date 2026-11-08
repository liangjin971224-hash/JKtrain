"""启动自检。

设计原则（来自 RAG 骨架的实践经验）：
    每个失败项都必须给出**可执行的修复提示**，不能只说"连接失败"。

P0 阶段的定位：
    服务还没有业务能力，自检的意义是「环境是否就绪」。
    缺 Neo4j/Milvus/模型时**报出来但不崩**——P0 验收要求"空服务能启动"。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CheckResult:
    """单项自检结果。

    status:
        ok       通过
        warn     不阻塞启动（如 LLM 未配置）
        fail     阻塞启动（环境不可用）
        skip     依赖包未安装，跳过（P0 允许）
    """

    name: str
    status: str
    detail: str
    hint: str = ""

    @property
    def blocking(self) -> bool:
        """是否阻塞启动。"""
        return self.status == "fail"

    def format(self) -> str:
        """一行可读输出。"""
        icon = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}[
            self.status
        ]
        line = f"  [{icon}] {self.name:<24} {self.detail}"
        if self.hint:
            line += f"\n         → {self.hint}"
        return line


# ===================================================================
# 各项检查
# ===================================================================
def check_settings(settings: Settings) -> CheckResult:
    """必填配置项是否齐全。"""
    missing = settings.required_fields_missing()
    if missing:
        return CheckResult(
            name="配置项", status="fail",
            detail=f"缺少必填项: {', '.join(missing)}",
            hint="复制 .env.example 为 .env 并填写，Windows: copy .env.example .env",
        )
    return CheckResult(
        name="配置项", status="ok",
        detail=f"env={settings.app_env}, log={settings.log_level}",
    )


def check_neo4j(settings: Settings) -> CheckResult:
    """Neo4j 连通性与目标库是否有数据。"""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return CheckResult(
            name="Neo4j", status="skip", detail="neo4j 包未安装",
            hint="pip install neo4j",
        )

    driver = None
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            connection_timeout=5,
        )
        with driver.session(database=settings.neo4j_database) as session:
            cnt = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]

        if cnt == 0:
            return CheckResult(
                name="Neo4j", status="fail",
                detail=f"库 {settings.neo4j_database} 为空（0 节点）",
                hint="执行 08_数据入库/neo4j/01_建库与产品图谱.cypher",
            )
        return CheckResult(
            name="Neo4j", status="ok",
            detail=f"{settings.neo4j_database} / {cnt} 节点",
        )
    except Exception as exc:                      # noqa: BLE001 探测阶段统一兜底
        return CheckResult(
            name="Neo4j", status="fail",
            detail=f"连接失败: {type(exc).__name__}",
            hint=(
                f"检查 NEO4J_URI={settings.neo4j_uri}，"
                f"Windows 启动: net start neo4j；"
                f"原始错误: {exc}"
            ),
        )
    finally:
        if driver is not None:
            driver.close()


def check_milvus(settings: Settings) -> CheckResult:
    """Milvus 连通性、collection 是否齐全。"""
    try:
        from pymilvus import MilvusClient
    except ImportError:
        return CheckResult(
            name="Milvus", status="skip", detail="pymilvus 包未安装",
            hint="pip install pymilvus",
        )

    expected = {
        settings.coll_training,
        settings.coll_case,
        settings.coll_question,
        settings.coll_self_test,
    }
    try:
        client = MilvusClient(uri=settings.milvus_uri)
        existing = set(client.list_collections())
        missing = expected - existing

        if missing:
            return CheckResult(
                name="Milvus", status="fail",
                detail=f"{settings.milvus_uri} 缺少 collection: {sorted(missing)}",
                hint="执行 08_数据入库/milvus/build_collections.py 重建",
            )
        return CheckResult(
            name="Milvus", status="ok",
            detail=f"{settings.milvus_uri} / {len(existing)} 个 collection 齐全",
        )
    except Exception as exc:                      # noqa: BLE001
        return CheckResult(
            name="Milvus", status="fail",
            detail=f"连接失败: {type(exc).__name__}",
            hint=f"检查 MILVUS_HOST/PORT={settings.milvus_uri}；原始错误: {exc}",
        )


def check_embedding(settings: Settings) -> CheckResult:
    """模型目录是否存在。P0 不加载模型（加载要 3–6 秒且占显存）。"""
    from pathlib import Path

    model_dir = Path(settings.embed_model_dir)
    if not model_dir.exists():
        return CheckResult(
            name="bge-m3 模型", status="fail",
            detail=f"目录不存在: {model_dir}",
            hint="检查 EMBED_MODEL_DIR，或先下载 bge-m3 到该目录",
        )

    # 只校验结构，不加载权重
    markers = ["config.json", "modules.json"]
    missing = [m for m in markers if not (model_dir / m).exists()]
    if missing:
        return CheckResult(
            name="bge-m3 模型", status="fail",
            detail=f"目录不完整，缺少 {missing}",
            hint="确认 EMBED_MODEL_DIR 指向 sentence-transformers 格式的模型目录",
        )
    return CheckResult(
        name="bge-m3 模型", status="ok",
        detail=f"已就位（{settings.expected_embed_dim} 维，未加载权重）",
    )


def check_llm(settings: Settings) -> CheckResult:
    """LLM 主备双通道配置。

    未配置只警告——启动不需要生成能力。
    但主备状态要明确报出来，因为这是「双重保险」是否真的生效的唯一显式反馈。
    """
    endpoints = settings.llm_endpoints()
    if not endpoints:
        return CheckResult(
            name="LLM", status="warn",
            detail="未配置任何 LLM Key（问答生成不可用）",
            hint="在 .env 填 QWEN_API_KEY（主）和/或 DEEPSEEK_API_KEY（备）",
        )

    primary = endpoints[0]
    backups = endpoints[1:]

    if backups:
        return CheckResult(
            name="LLM 主备", status="ok",
            detail=(
                f"主={primary['name']}({primary['model']}) "
                f"→ 备={','.join(b['name'] for b in backups)}"
            ),
        )

    return CheckResult(
        name="LLM", status="warn",
        detail=f"仅单端点 {primary['name']}({primary['model']})，无兜底",
        hint=(
            "只配了一个 Key，该厂商故障时问答不可用。"
            "建议同时配 QWEN_API_KEY 与 DEEPSEEK_API_KEY 实现主备切换"
        ),
    )


def check_file_storage(settings: Settings) -> CheckResult:
    """文件存储根目录可写。"""
    root = settings.storage_root_path
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:                      # noqa: BLE001
        return CheckResult(
            name="文件存储", status="fail",
            detail=f"目录不可写: {root}",
            hint=f"检查 FILE_STORAGE_ROOT 权限；原始错误: {exc}",
        )
    return CheckResult(
        name="文件存储", status="ok",
        detail=f"{root}（上限 {settings.file_storage_max_mb}MB）",
    )


def check_cache(settings: Settings) -> CheckResult:
    """缓存依赖。redis 未安装时回退内存实现。"""
    try:
        import redis  # noqa: F401
    except ImportError:
        return CheckResult(
            name="缓存", status="warn",
            detail="redis 包未安装，CachePort 回退内存实现",
            hint="需要多实例共享会话态时再 pip install redis",
        )
    return CheckResult(
        name="缓存", status="ok", detail=f"redis 已安装 ({settings.redis_url})",
    )


# ===================================================================
# 汇总
# ===================================================================
ALL_CHECKS = (
    check_settings,
    check_neo4j,
    check_milvus,
    check_embedding,
    check_llm,
    check_file_storage,
    check_cache,
)


def run_startup_checks(
    settings: Settings,
    checks: tuple = ALL_CHECKS,
) -> list[CheckResult]:
    """跑全部自检。

    Args:
        settings: 配置。
        checks: 检查函数序列，可注入替换（测试用）。

    Returns:
        结果列表。调用方负责决定 fail 是否阻塞启动。
    """
    results: list[CheckResult] = []
    for fn in checks:
        try:
            results.append(fn(settings))
        except Exception as exc:                  # noqa: BLE001 单项异常不拖垮整体
            results.append(CheckResult(
                name=getattr(fn, "__name__", "unknown"),
                status="fail",
                detail=f"检查自身异常: {type(exc).__name__}: {exc}",
            ))
    return results


def summarize(results: list[CheckResult]) -> tuple[bool, str]:
    """汇总自检结论。

    Returns:
        (是否可启动, 汇总文本)
    """
    counts = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    blocking = [r for r in results if r.blocking]
    lines = [
        f"自检结果: {counts['ok']} 通过 / {counts['warn']} 警告 / "
        f"{counts['fail']} 失败 / {counts['skip']} 跳过"
    ]
    if blocking:
        lines.append("阻塞项:")
        lines.extend(f"  - {r.name}: {r.detail}" for r in blocking)
        lines.append("服务以「降级模式」启动：以上能力不可用")
    else:
        lines.append("环境就绪，可正常启动")
    return (not blocking), "\n".join(lines)
