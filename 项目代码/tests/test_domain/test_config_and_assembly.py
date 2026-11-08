"""配置与装配根测试。

重点验证两件事：
1. 装配根能用假实现整体替换（这是"绝大多数测试不依赖外部服务"的前提）
2. 启动自检在缺依赖时**报错但不崩**——P0 要求"空服务能启动"
"""
from __future__ import annotations

from config.assembly import Container, build_test_container
from config.settings import Settings
from config.startup_check import (
    CheckResult,
    check_file_storage,
    check_llm,
    check_settings,
    run_startup_checks,
    summarize,
)


def _settings(tmp_path, **overrides) -> Settings:
    """构造测试用配置。存储根指向 pytest 临时目录，避免污染项目。"""
    base = {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_database": "jiankangshujuku",
        "embed_model_dir": str(tmp_path / "model"),
        "file_storage_root": str(tmp_path / "data"),
        "qwen_api_key": "",
        "deepseek_api_key": "",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)      # type: ignore[arg-type]


# ===================================================================
# Settings
# ===================================================================
class TestSettings:
    def test_milvus_uri_derived(self, tmp_path) -> None:
        s = _settings(tmp_path, milvus_host="127.0.0.1", milvus_port=19531)
        assert s.milvus_uri == "http://127.0.0.1:19531"

    def test_llm_disabled_when_no_key(self, tmp_path) -> None:
        assert _settings(tmp_path).llm_enabled is False

    def test_llm_endpoints_primary_first(self, tmp_path) -> None:
        """端点列表必须主在前、备在后——切换适配器按顺序遍历。"""
        s = _settings(tmp_path, qwen_api_key="q1", deepseek_api_key="d1")
        names = [e["name"] for e in s.llm_endpoints()]
        assert names == ["qwen", "deepseek"]

    def test_llm_endpoints_respect_primary_setting(self, tmp_path) -> None:
        """改 `llm_primary` 应改变优先顺序，代码不用动。"""
        s = _settings(tmp_path, qwen_api_key="q1", deepseek_api_key="d1",
                      llm_primary="deepseek")
        names = [e["name"] for e in s.llm_endpoints()]
        assert names == ["deepseek", "qwen"]

    def test_llm_endpoints_skip_missing_key(self, tmp_path) -> None:
        """只有备用了 Key 时，列表里就只有备用。"""
        s = _settings(tmp_path, qwen_api_key="", deepseek_api_key="d1")
        names = [e["name"] for e in s.llm_endpoints()]
        assert names == ["deepseek"]

    def test_fallback_disabled_excludes_backup(self, tmp_path) -> None:
        s = _settings(tmp_path, qwen_api_key="q1", deepseek_api_key="d1",
                      llm_fallback_enabled=False)
        names = [e["name"] for e in s.llm_endpoints()]
        assert names == ["qwen"]

    def test_file_storage_max_bytes(self, tmp_path) -> None:
        s = _settings(tmp_path, file_storage_max_mb=20)
        assert s.file_storage_max_bytes == 20 * 1024 * 1024

    def test_relative_storage_root_resolved(self, tmp_path) -> None:
        """相对路径按项目代码根目录解析，不是按当前工作目录。"""
        s = _settings(tmp_path, file_storage_root="./data")
        assert s.storage_root_path.is_absolute()


# ===================================================================
# 启动自检
# ===================================================================
class TestStartupCheck:
    def test_settings_ok(self, tmp_path) -> None:
        r = check_settings(_settings(tmp_path))
        assert r.status == "ok"

    def test_llm_missing_is_warn_not_fail(self, tmp_path) -> None:
        """LLM 未配置只警告——启动不需要生成能力，不能阻塞。"""
        r = check_llm(_settings(tmp_path))
        assert r.status == "warn"
        assert r.blocking is False

    def test_llm_single_endpoint_warns_no_fallback(self, tmp_path) -> None:
        """只配一个 Key 时要警告「无兜底」，不能报 ok。

        这是「双重保险是否生效」的唯一显式反馈——
        若单端点也报 ok，用户会以为双保险已经上了。
        """
        r = check_llm(_settings(tmp_path, qwen_api_key="q1"))
        assert r.status == "warn"
        assert "无兜底" in r.detail or "无兜底" in r.hint

    def test_llm_dual_endpoint_reports_both(self, tmp_path) -> None:
        """主备都配好时，必须明确报出主是谁、备是谁。"""
        r = check_llm(_settings(tmp_path, qwen_api_key="q1",
                                deepseek_api_key="d1"))
        assert r.status == "ok"
        assert "qwen" in r.detail
        assert "deepseek" in r.detail
        assert "备" in r.detail

    def test_file_storage_creates_dir(self, tmp_path) -> None:
        root = tmp_path / "deep" / "nested" / "data"
        r = check_file_storage(_settings(tmp_path, file_storage_root=str(root)))
        assert r.status == "ok"
        assert root.exists()

    def test_check_exception_does_not_crash_others(self, tmp_path) -> None:
        """单项检查自身抛异常时，不能让整体自检崩掉。"""

        def boom(_s):
            raise RuntimeError("模拟检查器故障")

        def fine(_s):
            return CheckResult(name="ok项", status="ok", detail="fine")

        results = run_startup_checks(_settings(tmp_path), checks=(boom, fine))
        assert len(results) == 2
        assert results[0].status == "fail"
        assert results[1].status == "ok"

    def test_summarize_all_ok(self) -> None:
        ok, text = summarize([
            CheckResult("a", "ok", ""), CheckResult("b", "warn", ""),
        ])
        assert ok is True
        assert "环境就绪" in text

    def test_summarize_with_blocking(self) -> None:
        ok, text = summarize([
            CheckResult("a", "ok", ""),
            CheckResult("neo4j", "fail", "连接失败"),
        ])
        assert ok is False
        assert "降级模式" in text
        assert "neo4j" in text


# ===================================================================
# 装配根
# ===================================================================
class TestAssembly:
    def test_test_container_has_all_ports(self, tmp_path) -> None:
        """假容器必须 7 个端口全就绪——否则应用层测试写不下去。"""
        c = build_test_container(_settings(tmp_path))
        assert isinstance(c, Container)
        assert len(c.ready_ports()) == 7
        assert c.pending == {}

    def test_test_container_does_not_touch_services(self, tmp_path) -> None:
        """假容器不应创建任何真连接。"""
        c = build_test_container(_settings(tmp_path))
        assert c.neo4j_driver is None
        assert c.milvus_client is None

    def test_test_container_ports_are_fakes(self, tmp_path) -> None:
        from tests.fakes import (
            FakeCachePort,
            FakeGraphQueryPort,
            FakeVectorSearchPort,
        )

        c = build_test_container(_settings(tmp_path))
        assert isinstance(c.graph, FakeGraphQueryPort)
        assert isinstance(c.vector, FakeVectorSearchPort)
        assert isinstance(c.cache, FakeCachePort)

    def test_close_is_idempotent(self, tmp_path) -> None:
        """重复 close 不应抛（进程退出路径会被调用多次）。"""
        c = build_test_container(_settings(tmp_path))
        c.close()
        c.close()
