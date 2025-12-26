"""Local web interface; evidence is rendered as text, never trusted HTML."""

from datetime import date
import logging
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from .client import Gateway
from .config import Settings
from .pipeline import RAG
from .store import Store


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    companies: list[str] | None = Field(default=None, max_length=20)
    date_from: date | None = None
    date_to: date | None = None
    categories: list[str] | None = Field(default=None, max_length=20)
    top_k: int = Field(default=10, ge=1, le=20)
    rewrite: bool = True
    mode: Literal["hybrid", "lexical", "dense"] = "hybrid"

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("请输入问题")
        return value.strip()

    @field_validator("companies", "categories")
    @classmethod
    def valid_filters(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        if any(not value.strip() or len(value) > 160 for value in values):
            raise ValueError("筛选条件必须为 1–160 个字符")
        return list(dict.fromkeys(value.strip() for value in values)) or None

    @model_validator(mode="after")
    def valid_dates(self) -> "AskRequest":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("开始日期不能晚于结束日期")
        return self


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="公司文档 RAG", docs_url=None, redoc_url=None)
    gateway = Gateway(settings)
    logger = logging.getLogger(__name__)

    @app.middleware("http")
    async def local_access(request: Request, call_next):
        # Local binding plus Host validation prevents DNS rebinding into the API.
        try:
            host = urlsplit("//" + request.headers.get("host", ""))
            allowed = host.hostname in {"localhost", "127.0.0.1", "::1"}
            port = host.port or (443 if request.url.scheme == "https" else 80)
            origin_header = request.headers.get("origin")
            if origin_header:
                origin = urlsplit(origin_header)
                origin_port = origin.port or (443 if origin.scheme == "https" else 80)
                allowed = allowed and (
                    origin.scheme == request.url.scheme
                    and origin.hostname == host.hostname
                    and origin_port == port
                    and not origin.username
                    and not origin.password
                    and not origin.path
                    and not origin.query
                    and not origin.fragment
                )
            allowed = allowed and request.headers.get("sec-fetch-site") != "cross-site"
        except ValueError:
            allowed = False
        if not allowed:
            return JSONResponse({"detail": "仅允许本机同源访问"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/api/stats")
    def stats():
        try:
            with Store(settings.db_path) as store:
                return store.stats()
        except Exception as exc:
            logger.error("Index status failed (%s)", type(exc).__name__)
            raise HTTPException(503, "无法读取本地索引，请检查服务日志及数据库配置。") from None

    @app.post("/api/ask")
    def ask(payload: AskRequest):
        try:
            with Store(settings.db_path) as store:
                return RAG(store, gateway).ask(
                    question=payload.question,
                    companies=payload.companies,
                    date_from=payload.date_from.isoformat() if payload.date_from else None,
                    date_to=payload.date_to.isoformat() if payload.date_to else None,
                    categories=payload.categories,
                    top_k=payload.top_k,
                    rewrite=payload.rewrite,
                    mode=payload.mode,
                )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except Exception as exc:
            # Upstream exception bodies may contain configuration or request data.
            logger.error("Question answering failed (%s)", type(exc).__name__)
            raise HTTPException(
                502, "问答失败，请检查本地模型网关是否可用、凭据配置以及索引状态。"
            ) from None

    @app.get("/api/source/{chunk_id}")
    def source(chunk_id: str):
        if len(chunk_id) > 160:
            raise HTTPException(422, "无效的片段编号")
        with Store(settings.db_path) as store:
            result = store.source(chunk_id)
        if result is None:
            raise HTTPException(404, "未找到该原文片段")
        return result

    return app
