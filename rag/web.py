"""Local web interface with safe Markdown answers and plain-text evidence."""

from datetime import date
import logging
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from .client import Gateway
from .config import Settings
from .pipeline import RAG
from .store import Store


ZH_MESSAGES = {
    'Enter a question.': '请输入问题。',
    'Filters must contain 1–160 characters.': '筛选条件必须为 1–160 个字符。',
    'The start date must not be later than the end date.': '开始日期不能晚于结束日期。',
    'Only local same-origin access is allowed.': '仅允许本机同源访问。',
    'Unable to read the local index. Check the service logs and database configuration.': '无法读取本地索引，请检查服务日志及数据库配置。',
    'Question answering failed. Check the local model gateway, credentials, and index status.': '问答失败，请检查本地模型网关是否可用、凭据配置以及索引状态。',
    'Invalid chunk ID.': '无效的片段编号。',
    'Source passage not found.': '未找到该原文片段。',
    'Embedding index incomplete. Run company-rag embed, or use lexical mode explicitly.': '向量索引不完整。请运行 company-rag embed，或选择关键词检索模式。',
    'Not Found': '未找到该接口。',
    'Method Not Allowed': '不支持此请求方法。',
}


def localized(message, language):
    if language == 'zh-CN':
        if message.startswith('Unknown company filter: '):
            return message.replace('Unknown company filter: ', '未知公司筛选条件：', 1)
        return ZH_MESSAGES.get(message, message)
    return message


def request_language(request):
    language = getattr(request.state, 'language', None) or request.headers.get('accept-language', 'en').split(',')[0].split(';')[0].strip()
    return language if language in ('en', 'zh-CN') else 'en'


def error_response(message, language, status_code, headers=None):
    messages = {code: localized(message, code) for code in ('en', 'zh-CN')}
    return JSONResponse({'detail': messages[language], 'error_messages': messages}, status_code=status_code, headers=headers)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    companies: list[str] | None = Field(default=None, max_length=20)
    date_from: date | None = None
    date_to: date | None = None
    categories: list[str] | None = Field(default=None, max_length=20)
    top_k: int = Field(default=10, ge=1, le=20)
    rewrite: bool = True
    mode: Literal["hybrid", "lexical", "dense"] = "hybrid"
    language: Literal["en", "zh-CN"] = "en"

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter a question.")
        return value.strip()

    @field_validator("companies", "categories")
    @classmethod
    def valid_filters(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        if any(not value.strip() or len(value) > 160 for value in values):
            raise ValueError("Filters must contain 1–160 characters.")
        return list(dict.fromkeys(value.strip() for value in values)) or None

    @model_validator(mode="after")
    def valid_dates(self) -> "AskRequest":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("The start date must not be later than the end date.")
        return self


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Company Evidence RAG", docs_url=None, redoc_url=None)
    gateway = Gateway(settings)
    # Disable raw HTML and remote images; reference definitions must not swallow [S1][S2].
    markdown = MarkdownIt('js-default', {'html': False, 'breaks': True}).disable(['image', 'reference'])
    logger = logging.getLogger(__name__)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return error_response(exc.detail, request_language(request), exc.status_code, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        language = exc.body.get('language') if isinstance(exc.body, dict) else None
        if language not in ('en', 'zh-CN'):
            language = request_language(request)
        messages = {
            'literal_error': '请选择有效选项。', 'missing': '此字段为必填项。',
            'string_type': '请输入文本。', 'string_too_short': '文本过短。',
            'string_too_long': '文本过长。', 'list_type': '请提供列表。',
            'too_long': '项目数量过多。', 'int_parsing': '请输入整数。',
            'int_type': '请输入整数。', 'greater_than_equal': '数值低于允许的最小值。',
            'less_than_equal': '数值超过允许的最大值。',
            'date_from_datetime_parsing': '请输入有效日期（YYYY-MM-DD）。',
            'date_type': '请输入有效日期（YYYY-MM-DD）。',
            'bool_parsing': '请输入有效的布尔值。', 'json_invalid': '请求必须为有效的 JSON。',
        }
        errors = {'en': exc.errors(), 'zh-CN': []}
        for error in errors['en']:
            message = error['msg'].removeprefix('Value error, ')
            errors['zh-CN'].append(dict(error, msg=messages.get(error['type'], localized(message, 'zh-CN'))))
        summaries = {code: ' '.join(error['msg'].removeprefix('Value error, ') for error in values)
                     for code, values in errors.items()}
        return JSONResponse({'detail': jsonable_encoder(errors[language]), 'error_messages': summaries}, status_code=422)

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
            return error_response('Only local same-origin access is allowed.', request_language(request), 403)
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
            raise HTTPException(503, 'Unable to read the local index. Check the service logs and database configuration.') from None

    @app.post("/api/ask")
    def ask(payload: AskRequest, request: Request):
        request.state.language = payload.language
        try:
            with Store(settings.db_path) as store:
                result = RAG(store, gateway).ask(
                    question=payload.question,
                    companies=payload.companies,
                    date_from=payload.date_from.isoformat() if payload.date_from else None,
                    date_to=payload.date_to.isoformat() if payload.date_to else None,
                    categories=payload.categories,
                    top_k=payload.top_k,
                    rewrite=payload.rewrite,
                    mode=payload.mode,
                    language=payload.language,
                )
                result['answer_html'] = markdown.render(result['answer'])
                return result
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except Exception as exc:
            # Upstream exception bodies may contain configuration or request data.
            logger.error("Question answering failed (%s)", type(exc).__name__)
            raise HTTPException(
                502, 'Question answering failed. Check the local model gateway, credentials, and index status.'
            ) from None

    @app.get("/api/source/{chunk_id}")
    def source(chunk_id: str):
        if len(chunk_id) > 160:
            raise HTTPException(422, 'Invalid chunk ID.')
        with Store(settings.db_path) as store:
            result = store.source(chunk_id)
        if result is None:
            raise HTTPException(404, 'Source passage not found.')
        return result

    return app
