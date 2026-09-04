from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ProviderFailure:
    code: str
    message: str
    request_id: str | None = None
    retryable: bool = False


_PUBLIC_MESSAGES = {
    "authentication_error": "模型访问凭据无效或已失效，请重新配置 API Key。",
    "workspace_denied": "API Key 与模型业务空间不匹配，请检查地域和业务空间。",
    "billing_error": "模型账号计费状态异常，请检查 API Key 所属账号的余额和账单。",
    "rate_limit": "模型服务当前限流，请稍后重试。",
    "provider_timeout": "模型服务响应超时，请稍后重试。",
    "provider_unavailable": "模型服务暂时不可用，请稍后重试。",
    "invalid_provider_response": "模型服务返回了无法解析的响应。",
    "embedding_incomplete": "Embedding 返回数量或向量维度不完整。",
}


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _provider_fields(body: object) -> tuple[str, str, str | None]:
    payload = _mapping(body)
    error = _mapping(payload.get("error"))
    code = str(
        error.get("code")
        or error.get("type")
        or payload.get("code")
        or payload.get("error_code")
        or ""
    ).strip()
    message = str(error.get("message") or payload.get("message") or "").strip()
    request_id = payload.get("request_id") or payload.get("id") or error.get("request_id")
    return code, message, str(request_id) if request_id else None


def classify_provider_failure(
    *,
    status_code: int | None,
    body: object,
    exception: Exception | None = None,
) -> ProviderFailure:
    code, provider_message, request_id = _provider_fields(body)
    normalized_code = code.lower().replace("-", "_")
    normalized_message = provider_message.lower()

    if isinstance(exception, (httpx.TimeoutException, TimeoutError)):
        public_code = "provider_timeout"
        retryable = True
    elif "embedding_count_mismatch" in normalized_code or "embedding-count-mismatch" in normalized_message:
        public_code = "embedding_incomplete"
        retryable = False
    elif "workspace" in normalized_message and (
        "denied" in normalized_message or "mismatch" in normalized_message
    ):
        public_code = "workspace_denied"
        retryable = False
    elif normalized_code in {"arrearage", "insufficient_balance", "billing_error"}:
        public_code = "billing_error"
        retryable = False
    elif status_code in {401, 403} or normalized_code in {
        "invalid_api_key",
        "authentication_error",
        "unauthorized",
    }:
        public_code = "authentication_error"
        retryable = False
    elif status_code == 429 or normalized_code in {"rate_limit", "rate_limit_exceeded"}:
        public_code = "rate_limit"
        retryable = True
    elif isinstance(exception, httpx.TransportError):
        public_code = "provider_unavailable"
        retryable = True
    elif status_code is not None and status_code >= 500:
        public_code = "provider_unavailable"
        retryable = True
    else:
        public_code = "invalid_provider_response"
        retryable = False

    return ProviderFailure(
        code=public_code,
        message=_PUBLIC_MESSAGES[public_code],
        request_id=request_id,
        retryable=retryable,
    )


def failure_from_response(response: httpx.Response) -> ProviderFailure:
    try:
        body: object = response.json()
    except (ValueError, TypeError):
        body = None
    request_id = response.headers.get("x-request-id") or response.headers.get("x-acs-request-id")
    failure = classify_provider_failure(status_code=response.status_code, body=body)
    if failure.request_id or not request_id:
        return failure
    return ProviderFailure(
        code=failure.code,
        message=failure.message,
        request_id=request_id,
        retryable=failure.retryable,
    )
