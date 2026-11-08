"""FastAPI 应用入口。

启动：
    .venv\\Scripts\\python.exe -m uvicorn interfaces.http.app:app --reload

或：
    .venv\\Scripts\\python.exe serve.py

浏览器打开 http://127.0.0.1:8000/
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from domain.exceptions import DomainError
from interfaces.http.deps import get_container, lifespan
from interfaces.http.errors import (
    is_infrastructure_error,
    map_domain_error,
    map_infrastructure_error,
)
from interfaces.http.routers import ask, auth, drill, employee, exam, reports
from interfaces.http.schemas.models import HealthResponse

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"


def create_app() -> FastAPI:
    """构造应用（工厂函数，便于测试里多次创建互不干扰）。"""
    settings = get_settings()

    app = FastAPI(
        title="健康顾问培训系统",
        description=(
            "面向连锁健康食品门店的培训系统。\n\n"
            "两条主线：\n"
            "- **知识问答**：三路召回 + 输出护栏，召回为空主动拒答\n"
            "- **演练评阅**：脚本化客户追问 + 判定摘要卡（不给分）"
        ),
        version="0.5.0",
        lifespan=lifespan,
    )

    # ---------------- 中间件 ----------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------- 异常处理 ----------------
    @app.exception_handler(DomainError)
    async def _domain_error(request: Request, exc: DomainError):
        status, body = map_domain_error(exc)
        return JSONResponse(status_code=status, content=body.to_dict())

    @app.exception_handler(Exception)
    async def _uncaught(request: Request, exc: Exception):
        """统一兜底。

        ★ 503 与 500 的区别很重要：
          503 = 下游不可用（Neo4j/Milvus/LLM 挂了）→ 运维该去查服务
          500 = 代码 bug → 开发该去改代码

        判断用 `is_infrastructure_error`（按类名，不 import infrastructure），
        原因见 `interfaces/http/errors.py` 的注释。

        ★ 必须放行 HTTPException：注册 Exception 处理器后，
          FastAPI 自己的 404/422 也会走到这里，
          不放行的话所有 404 都变成 500。
        """
        from fastapi import HTTPException
        from starlette.exceptions import HTTPException as StarletteHTTP

        if isinstance(exc, (HTTPException, StarletteHTTP)):
            return JSONResponse(
                status_code=exc.status_code,          # type: ignore[attr-defined]
                content={
                    "code": f"HTTP_{exc.status_code}",  # type: ignore[attr-defined]
                    "message": str(exc.detail),         # type: ignore[attr-defined]
                    "detail": {},
                },
            )

        if is_infrastructure_error(exc):
            logger.error("基础设施故障: %s", exc, exc_info=True)
            body = map_infrastructure_error(exc)
            return JSONResponse(status_code=503, content=body.to_dict())

        logger.error("未处理异常: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL_ERROR",
                "message": "服务内部错误",
                "detail": {"type": type(exc).__name__},
            },
        )

    # ---------------- 路由 ----------------
    app.include_router(ask.router)
    app.include_router(drill.router)
    app.include_router(exam.router)
    app.include_router(employee.router)
    app.include_router(auth.router)
    app.include_router(reports.router)

    @app.get("/api/health", response_model=HealthResponse, tags=["运维"])
    def health() -> HealthResponse:
        """健康检查。前端用它显示「系统就绪」状态。"""
        from domain.services.output_guardrail import OutputGuardrail

        try:
            c = get_container()
            ready = c.ready_ports()
            pending = dict(c.pending)
            endpoints = [e["name"] for e in settings.llm_endpoints()]
        except RuntimeError:
            ready, pending, endpoints = [], {}, []

        return HealthResponse(
            status="ok" if ready else "degraded",
            ready_ports=ready,
            pending_ports=pending,
            llm_endpoints=endpoints,
            guardrail_rules=OutputGuardrail().describe_rules(),
        )

    # ---------------- 静态前端 ----------------
    if WEB_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(WEB_DIR)),
            name="static",
        )

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            """首页。"""
            return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
