import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

import aio_pika
import psycopg
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from dialogentis.agents import router as agents_router
from dialogentis.api_errors import DomainValidationError, error_response
from dialogentis.auth.routes import router as auth_router
from dialogentis.auth.security import COOKIE
from dialogentis.body_limit import BodyLimitMiddleware
from dialogentis.channels.api_key_routes import router as api_keys_router
from dialogentis.channels.widget_public import router as widget_runtime_router
from dialogentis.channels.widget_routes import router as widget_channels_router
from dialogentis.config import Settings, get_settings
from dialogentis.contracts import ErrorResponse
from dialogentis.datasets.export_routes import router as exports_router
from dialogentis.datasets.import_routes import router as imports_router
from dialogentis.datasets.routes import router as datasets_router
from dialogentis.db import session_factory
from dialogentis.deletions import router as deletions_router
from dialogentis.drafts import router as drafts_router
from dialogentis.flow_routes import router as flow_router
from dialogentis.integrations.routes import router as integrations_router
from dialogentis.organizations import router as organizations_router
from dialogentis.runtime.executor import RuntimeExecutor
from dialogentis.runtime.history_export_routes import router as history_exports_router
from dialogentis.runtime.history_routes import router as history_router
from dialogentis.runtime.public_routes import router as public_runtime_router
from dialogentis.runtime.routes import router as test_chat_router
from dialogentis.runtime.test_chat import TestChatService
from dialogentis.versions.release_routes import router as releases_router
from dialogentis.versions.routes import router as versions_router


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or get_settings()
    factory = (
        session_factory(config.database_url.get_secret_value()) if config.database_url else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            try:
                if app.state.test_chat_service is not None:
                    await app.state.test_chat_service.shutdown()
            finally:
                if factory:
                    await factory.kw["bind"].dispose()

    app = FastAPI(
        title="Dialogentis API",
        version="0.1.0",
        docs_url=(None if config.app_env == "production" else "/api/docs"),
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
        responses={
            status: {"model": ErrorResponse}
            for status in (400, 401, 403, 404, 409, 412, 413, 422, 423, 428, 429, 503)
        },
    )
    app.state.settings = config
    app.state.session_factory = factory
    app.state.test_chat_service = (
        TestChatService(factory, config, RuntimeExecutor()) if factory else None
    )
    app.state.entity_preview_slots = asyncio.Semaphore(4)
    app.add_middleware(BodyLimitMiddleware, max_bytes=config.max_body_bytes)
    app.include_router(auth_router)
    app.include_router(organizations_router)
    app.include_router(deletions_router)
    app.include_router(agents_router)
    app.include_router(drafts_router)
    app.include_router(flow_router)
    app.include_router(integrations_router)
    app.include_router(api_keys_router)
    app.include_router(widget_channels_router)
    app.include_router(widget_runtime_router)
    app.include_router(datasets_router)
    app.include_router(exports_router)
    app.include_router(imports_router)
    app.include_router(versions_router)
    app.include_router(releases_router)
    app.include_router(test_chat_router)
    app.include_router(history_router)
    app.include_router(history_exports_router)
    app.include_router(public_runtime_router)

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[[Request], Awaitable]):
        request.state.request_id = str(uuid4())
        length = request.headers.get("content-length", "0")
        if not length.isdigit() or int(length) > config.max_body_bytes:
            response = error_response(
                request, 413, "payload_too_large", "Request body is too large"
            )
        else:
            response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        session = getattr(request.state, "session_cookie", None)
        if session:
            token, expires = session
            response.set_cookie(
                COOKIE,
                token,
                max_age=max(0, int((expires - datetime.now(UTC)).total_seconds())),
                httponly=True,
                samesite="lax",
                secure=config.public_base_url.startswith("https://"),
            )
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        response = error_response(
            request, exc.status_code, f"http_{exc.status_code}", str(exc.detail)
        )
        response.headers.update(exc.headers or {})
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Do not echo the submitted value (passwords and message texts included).
        return error_response(
            request,
            422,
            "validation_error",
            "Invalid request",
            [
                {"path": list(error["loc"]), "code": error["type"], "message": "Invalid value"}
                for error in exc.errors()
            ],
        )

    @app.exception_handler(DomainValidationError)
    async def domain_validation_error(request: Request, exc: DomainValidationError):
        return error_response(
            request, 422, "validation_error", "Проверьте данные черновика", exc.details
        )

    @app.get("/api/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/health/ready", tags=["health"])
    async def readiness(request: Request):
        checks = {"database": "unavailable", "broker": "unavailable"}
        if config.database_url:
            try:
                async with asyncio.timeout(3):
                    async with await psycopg.AsyncConnection.connect(
                        config.database_url.get_secret_value().replace(
                            "postgresql+psycopg", "postgresql"
                        )
                    ) as connection:
                        await connection.execute("SELECT 1")
                checks["database"] = "ok"
            except TimeoutError, psycopg.Error:
                pass
        if config.broker_url:
            try:
                async with asyncio.timeout(3):
                    connection = await aio_pika.connect(config.broker_url.get_secret_value())
                    await connection.close()
                checks["broker"] = "ok"
            except TimeoutError, ConnectionError, OSError:
                pass
        ready = all(value == "ok" for value in checks.values())
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ready" if ready else "unavailable",
                "checks": checks,
                "request_id": request.state.request_id,
            },
        )

    return app
