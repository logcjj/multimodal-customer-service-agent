from __future__ import annotations

import httpx
import pytest

from app.providers.errors import ProviderFailure, classify_provider_failure


@pytest.mark.parametrize(
    ("status", "body", "expected_code", "retryable"),
    [
        (401, {"error": {"code": "invalid_api_key", "message": "bad key"}}, "authentication_error", False),
        (400, {"error": {"code": "Arrearage", "message": "account overdue"}}, "billing_error", False),
        (
            400,
            {"error": {"type": "invalid_request_error", "message": "Workspace endpoint access denied."}},
            "workspace_denied",
            False,
        ),
        (429, {"error": {"message": "too many requests"}}, "rate_limit", True),
        (503, {"error": {"message": "temporarily unavailable"}}, "provider_unavailable", True),
    ],
)
def test_provider_http_failures_are_classified(
    status: int,
    body: dict[str, object],
    expected_code: str,
    retryable: bool,
) -> None:
    failure = classify_provider_failure(status_code=status, body=body)

    assert failure.code == expected_code
    assert failure.retryable is retryable
    assert failure.message


def test_request_id_is_preserved_without_raw_provider_message() -> None:
    secret = "private-provider-secret"
    failure = classify_provider_failure(
        status_code=401,
        body={
            "request_id": "request-123",
            "error": {
                "code": "invalid_api_key",
                "message": f"Authorization Bearer {secret}",
            },
        },
    )

    assert failure.request_id == "request-123"
    assert secret not in failure.message
    assert secret not in repr(failure)


def test_timeout_is_retryable() -> None:
    failure = classify_provider_failure(
        status_code=None,
        body=None,
        exception=httpx.ReadTimeout("provider timed out"),
    )

    assert failure == ProviderFailure(
        code="provider_timeout",
        message="模型服务响应超时，请稍后重试。",
        request_id=None,
        retryable=True,
    )


def test_incomplete_embedding_response_has_specific_error() -> None:
    failure = classify_provider_failure(
        status_code=200,
        body={"error_code": "embedding-count-mismatch", "request_id": "embed-1"},
    )

    assert failure.code == "embedding_incomplete"
    assert failure.request_id == "embed-1"

