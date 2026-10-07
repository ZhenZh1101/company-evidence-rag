"""Web interface with safe Markdown answers and bounded question admission."""

import asyncio
from datetime import date
from ipaddress import ip_address, ip_network
import json
import logging
from pathlib import Path
import sqlite3
import time
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field, field_validator, model_validator
from qdrant_client.http.exceptions import ApiException
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.concurrency import run_in_threadpool

from .admission import AdmissionController, QueueFull, RateLimitExceeded
from .auth import AuthService, COOKIE_NAME, SESSION_SECONDS, InvalidCredentials, LoginBusy, LoginRateLimited
from .client import Gateway
from .config import Settings
from .pipeline import RAG
from .store import Store


ZH_MESSAGES = {
    'Enter a question.': '请输入问题。',
    'Please sign in.': '请先登录。',
    'Invalid username or password.': '用户名或密码错误。',
    'Too many login attempts. Try again later.': '登录尝试过于频繁，请稍后重试。',
    'Login is temporarily busy. Try again later.': '登录服务暂忙，请稍后重试。',
    'Filters must contain 1–160 characters.': '筛选条件必须为 1–160 个字符。',
    'The start date must not be later than the end date.': '开始日期不能晚于结束日期。',
    'Only local same-origin access is allowed.': '仅允许本机同源访问。',
    'Only configured same-origin access is allowed.': '仅允许已配置站点的同源访问。',
    'Question limit reached: 5 per minute and 50 per 24 hours per IP. Try again later.': '已达到提问限制：每个 IP 每分钟最多 5 次、每 24 小时最多 50 次。请稍后重试。',
    'The question queue is full (5 waiting). Try again later.': '提问队列已满（最多等待 5 个请求）。请稍后重试。',
    'Question admission is temporarily unavailable. Try again later.': '提问限流服务暂时不可用，请稍后重试。',
    'The request was disconnected.': '请求连接已断开。',
    'Unable to read the local index. Check the service logs and database configuration.': '无法读取本地索引，请检查服务日志及数据库配置。',
    'Question answering failed. Check the local model gateway, credentials, and index status.': '问答失败，请检查本地模型网关是否可用、凭据配置以及索引状态。',
    'Invalid chunk ID.': '无效的片段编号。',
    'Source passage not found.': '未找到该原文片段。',
    'Embedding index incomplete. Run company-rag embed, or use lexical mode explicitly.': '向量索引不完整。请运行 company-rag embed，或选择关键词检索模式。',
    'Qdrant request failed. Check the vector database service and RAG_QDRANT_URL.': '向量数据库请求失败，请检查 Qdrant 服务和 RAG_QDRANT_URL 配置。',
    'Qdrant index is missing. Run company-rag sync-vectors to restore cached vectors.': 'Qdrant 索引不存在，请运行 company-rag sync-vectors 恢复缓存向量。',
    'Qdrant index is incomplete. Run company-rag sync-vectors to restore cached vectors.': 'Qdrant 索引不完整，请运行 company-rag sync-vectors 恢复缓存向量。',
    'Qdrant source mismatch. Run company-rag sync-vectors before searching.': 'Qdrant 索引与原文不一致，请先运行 company-rag sync-vectors 再检索。',
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


def client_ip(request, trusted_proxies):
    """Walk XFF from the socket backwards, stopping at the first untrusted hop.

    Uvicorn must run with proxy_headers=False so the socket peer is available.
    Header duplicates form one chain; malformed entries never reveal values to
    their left. No X-Real-IP/CF header is trusted implicitly.
    """
    peer = request.client.host if request.client else 'unknown'
    try:
        address = ip_address(peer)
        address = getattr(address, 'ipv4_mapped', None) or address
    except ValueError:
        return peer
    forwarded = ','.join(request.headers.getlist('x-forwarded-for')).split(',')
    for value in reversed(forwarded):
        if not any(address in network for network in trusted_proxies):
            break
        try:
            address = ip_address(value.strip())
            address = getattr(address, 'ipv4_mapped', None) or address
        except ValueError:
            break
    return str(address)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=1024)


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
    if settings.auth_required and not settings.auth_users:
        raise ValueError('Authentication is required; configure RAG_AUTH_USERS_JSON before starting.')
    app = FastAPI(title="Company Evidence RAG", docs_url=None, redoc_url=None, openapi_url=None)
    auth = AuthService(settings.auth_users) if settings.auth_users else None
    gateway = Gateway(settings)
    # Disable raw HTML and remote images; reference definitions must not swallow [S1][S2].
    markdown = MarkdownIt('js-default', {'html': False, 'breaks': True}).disable(['image', 'reference'])
    logger = logging.getLogger(__name__)
    metrics_logger = logging.getLogger('uvicorn.error.rag')
    public_origin = urlsplit(settings.public_origin) if settings.public_origin else None
    trusted_proxies = tuple(ip_network(value.strip()) for value in settings.trusted_proxy_ips.split(',') if value.strip())
    admission = AdmissionController(settings.rate_limit_db_path or settings.db_path.with_name('rate_limits.sqlite3'))
    app.state.admission = admission

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return error_response(exc.detail, request_language(request), exc.status_code, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        if request.url.path == '/api/login':
            return error_response('Invalid username or password.', request_language(request), 401)
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
    async def same_origin_access(request: Request, call_next):
        # Keep DNS rebinding protection locally, and allow only the configured
        # public hostname behind HTTPS termination (not arbitrary forwarded hosts).
        try:
            host = urlsplit("//" + request.headers.get("host", ""))
            allowed = host.hostname in {"localhost", "127.0.0.1", "::1"}
            scheme = request.url.scheme
            if public_origin and host.hostname == public_origin.hostname:
                scheme = public_origin.scheme
                allowed = (host.port or (443 if scheme == 'https' else 80)) == (
                    public_origin.port or (443 if scheme == 'https' else 80))
            allowed = allowed and not (host.username or host.password or host.path or host.query or host.fragment)
            port = host.port or (443 if scheme == "https" else 80)
            origin_header = request.headers.get("origin")
            if origin_header:
                origin = urlsplit(origin_header)
                origin_port = origin.port or (443 if origin.scheme == "https" else 80)
                allowed = allowed and (
                    origin.scheme == scheme
                    and origin.hostname == host.hostname
                    and origin_port == port
                    and not origin.username
                    and not origin.password
                    and not origin.path
                    and not origin.query
                    and not origin.fragment
                )
            # HF embeds the initial document cross-site; its API calls still
            # originate from the Space itself. CSP limits permitted ancestors.
            iframe_document = public_origin and request.method in ('GET', 'HEAD') and request.url.path == '/'
            allowed = allowed and (request.headers.get("sec-fetch-site") != "cross-site" or iframe_document)
        except ValueError:
            allowed = False
        if not allowed:
            message = 'Only configured same-origin access is allowed.' if public_origin else 'Only local same-origin access is allowed.'
            return error_response(message, request_language(request), 403)
        request.state.username = auth.current_user(request.cookies.get(COOKIE_NAME)) if auth else None
        public_route = ((request.method in ('GET', 'HEAD') and request.url.path == '/')
                        or (request.method == 'POST' and request.url.path in ('/api/login', '/api/logout')))
        if auth and not request.state.username and not public_route:
            response = error_response('Please sign in.', request_language(request), 401)
        else:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors "
            + ("'self' https://huggingface.co" if public_origin else "'none'")
        )
        return response

    @app.get("/")
    def index(request: Request):
        filename = 'login.html' if auth and not request.state.username else 'index.html'
        return FileResponse(Path(__file__).parent / 'static' / filename)

    @app.post('/api/login')
    async def login(payload: LoginRequest, request: Request):
        if not auth:
            raise HTTPException(404, 'Not Found')
        try:
            token = await run_in_threadpool(auth.login, payload.username.strip(), payload.password,
                                           client_ip(request, trusted_proxies))
        except InvalidCredentials:
            raise HTTPException(401, 'Invalid username or password.') from None
        except LoginRateLimited as exc:
            raise HTTPException(429, 'Too many login attempts. Try again later.',
                                headers={'Retry-After': str(exc.retry_after)}) from None
        except LoginBusy as exc:
            raise HTTPException(503, 'Login is temporarily busy. Try again later.',
                                headers={'Retry-After': str(exc.retry_after)}) from None
        auth.logout(request.cookies.get(COOKIE_NAME))  # Replace any prior session for this browser.
        response = JSONResponse({'username': payload.username.strip()})
        response.set_cookie(COOKIE_NAME, token, max_age=SESSION_SECONDS, httponly=True,
                            secure=bool(public_origin and public_origin.scheme == 'https') or request.url.scheme == 'https',
                            samesite='lax', path='/')
        return response

    @app.post('/api/logout')
    def logout(request: Request):
        if auth:
            auth.logout(request.cookies.get(COOKIE_NAME))
        response = JSONResponse({'ok': True})
        response.delete_cookie(COOKIE_NAME, path='/')
        return response

    @app.get('/api/session')
    def session(request: Request):
        return {'username': request.state.username, 'authentication_required': auth is not None}

    @app.get("/api/stats")
    def stats():
        try:
            with Store(settings.db_path, settings.qdrant_url) as store:
                return store.stats()
        except Exception as exc:
            logger.error("Index status failed (%s)", type(exc).__name__)
            raise HTTPException(503, 'Unable to read the local index. Check the service logs and database configuration.') from None

    def answer(payload, queued_at, request_id):
        queue_seconds = time.monotonic() - queued_at
        try:
            with Store(settings.db_path, settings.qdrant_url) as store:
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
                result['request_id'] = request_id
                result.setdefault('timings', {}).update(queue_seconds=round(queue_seconds, 4),
                                                       request_seconds=round(time.monotonic() - queued_at, 4))
                # Only timings/counters, never questions, source text, usernames or credentials.
                metrics_logger.info('Query metrics %s', json.dumps(dict(request_id=request_id, mode=payload.mode,
                                    timings=result['timings'], diagnostics=result.get('diagnostics', {}),
                                    warning_codes=result.get('warning_codes', []))))
                return result
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except ApiException as exc:
            logger.error("Vector database request failed (%s)", type(exc).__name__)
            raise HTTPException(503, 'Qdrant request failed. Check the vector database service and RAG_QDRANT_URL.') from None
        except Exception as exc:
            # Upstream exception bodies may contain configuration or request data.
            logger.error("Question answering failed (%s)", type(exc).__name__)
            raise HTTPException(
                502, 'Question answering failed. Check the local model gateway, credentials, and index status.'
            ) from None

    @app.post("/api/ask")
    async def ask(payload: AskRequest, request: Request):
        request.state.language = payload.language
        queued_at, request_id = time.monotonic(), uuid4().hex
        task = asyncio.create_task(admission.run(client_ip(request, trusted_proxies),
                                                 lambda: answer(payload, queued_at, request_id)))

        async def disconnected():
            while (await request.receive())['type'] != 'http.disconnect':
                pass

        disconnect = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait({task, disconnect}, return_when=asyncio.FIRST_COMPLETED)
            if task not in done:
                raise HTTPException(499, 'The request was disconnected.')
            return await task
        except RateLimitExceeded as exc:
            raise HTTPException(429, 'Question limit reached: 5 per minute and 50 per 24 hours per IP. Try again later.',
                                headers={'Retry-After': str(exc.retry_after)}) from None
        except QueueFull:
            raise HTTPException(503, 'The question queue is full (5 waiting). Try again later.',
                                headers={'Retry-After': '5'}) from None
        except sqlite3.Error as exc:
            logger.error('Question admission failed (%s)', type(exc).__name__)
            raise HTTPException(503, 'Question admission is temporarily unavailable. Try again later.',
                                headers={'Retry-After': '5'}) from None
        finally:
            if not task.done():
                task.cancel()
            disconnect.cancel()
            await asyncio.gather(task, disconnect, return_exceptions=True)

    @app.get("/api/source/{chunk_id}")
    def source(chunk_id: str):
        if len(chunk_id) > 160:
            raise HTTPException(422, 'Invalid chunk ID.')
        with Store(settings.db_path, settings.qdrant_url) as store:
            result = store.source(chunk_id)
        if result is None:
            raise HTTPException(404, 'Source passage not found.')
        return result

    return app
