from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator

import httpx

from app.contracts.models import ModelKind
from app.models.service import ModelService
from app.providers.errors import (
    ProviderFailure,
    classify_provider_failure,
    failure_from_response,
)


@dataclass(frozen=True)
class LLMOutput:
    text: str
    provider: str
    model: str
    latency_ms: int


GenerateOverride = Callable[[str, str, str, list[str]], str]


class LLMGateway:
    """Runtime gateway for configured LLM/VLM models.

    It intentionally returns ``None`` on provider errors so the frozen champion
    and deterministic evidence path remain available. Raw credentials and
    provider responses never enter Agent traces.
    """

    def __init__(
        self,
        model_service: ModelService,
        *,
        transport: httpx.BaseTransport | None = None,
        generate_override: GenerateOverride | None = None,
    ) -> None:
        self.model_service = model_service
        self.transport = transport
        self.generate_override = generate_override
        self.last_error: str | None = None
        self.last_failure: ProviderFailure | None = None

    def available(self, kind: ModelKind = ModelKind.LLM) -> bool:
        return self.generate_override is not None or self.model_service.get_default_runtime(kind) is not None

    def model_name(self, kind: ModelKind = ModelKind.LLM) -> str | None:
        if self.generate_override is not None:
            return "injected-test-model"
        runtime = self.model_service.get_default_runtime(kind)
        return runtime[0].name if runtime else None

    def generate(
        self,
        *,
        kind: ModelKind,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> LLMOutput | None:
        started = perf_counter()
        self.last_error = None
        self.last_failure = None
        if self.generate_override is not None:
            text = self.generate_override(kind.value, system_prompt, user_prompt, images or []).strip()
            if not text:
                return None
            return LLMOutput(
                text=text,
                provider="injected",
                model="injected-test-model",
                latency_ms=max(1, round((perf_counter() - started) * 1000)),
            )

        runtime = self.model_service.get_default_runtime(kind)
        if runtime is None:
            return None
        record, secret = runtime
        for attempt in range(2):
            try:
                with httpx.Client(timeout=45.0, transport=self.transport, trust_env=False) as client:
                    if record.provider == "anthropic":
                        text = self._anthropic_generate(
                            client,
                            record.base_url,
                            secret,
                            record.name,
                            system_prompt,
                            user_prompt,
                            temperature,
                            max_tokens,
                        )
                    else:
                        text = self._openai_generate(
                            client,
                            record.base_url,
                            secret,
                            record.name,
                            system_prompt,
                            user_prompt,
                            images or [],
                            temperature,
                            max_tokens,
                        )
                return LLMOutput(
                    text=text.strip(),
                    provider=record.provider,
                    model=record.name,
                    latency_ms=max(1, round((perf_counter() - started) * 1000)),
                )
            except Exception as exc:
                failure = (
                    failure_from_response(exc.response)
                    if isinstance(exc, httpx.HTTPStatusError)
                    else classify_provider_failure(
                        status_code=None,
                        body=None,
                        exception=exc,
                    )
                )
                if failure.retryable and attempt == 0:
                    continue
                self.last_error = failure.code
                self.last_failure = failure
                return None
        return None

    def generate_stream(
        self,
        *,
        kind: ModelKind,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> Iterator[str]:
        self.last_error = None
        self.last_failure = None
        if self.generate_override is not None:
            try:
                text = self.generate_override(kind.value, system_prompt, user_prompt, images or [])
            except Exception as exc:
                self.last_error = type(exc).__name__
                self.last_failure = classify_provider_failure(
                    status_code=None,
                    body=None,
                    exception=exc,
                )
                return
            if text:
                yield text
            return

        runtime = self.model_service.get_default_runtime(kind)
        if runtime is None:
            return
        record, secret = runtime
        if record.provider == "anthropic":
            output = self.generate(
                kind=kind,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=images,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if output and output.text:
                yield output.text
            return

        endpoint = (
            record.base_url
            if record.base_url.rstrip("/").endswith("chat/completions")
            else f"{record.base_url.rstrip('/')}/chat/completions"
        )
        user_content: str | list[dict[str, Any]] = user_prompt
        if images:
            user_content = [{"type": "text", "text": user_prompt}]
            user_content.extend(
                {"type": "image_url", "image_url": {"url": image}}
                for image in images
            )
        try:
            with httpx.Client(timeout=45.0, transport=self.transport, trust_env=False) as client:
                with client.stream(
                    "POST",
                    endpoint,
                    headers={"Authorization": f"Bearer {secret}"} if secret else {},
                    json={
                        "model": record.name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": True,
                    },
                ) as response:
                    if response.status_code >= 400:
                        response.read()
                        self.last_failure = failure_from_response(response)
                        self.last_error = self.last_failure.code
                        return
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload:
                            continue
                        if payload == "[DONE]":
                            return
                        try:
                            body = json.loads(payload)
                        except json.JSONDecodeError:
                            self.last_failure = classify_provider_failure(
                                status_code=response.status_code,
                                body=None,
                            )
                            self.last_error = self.last_failure.code
                            return
                        if isinstance(body, dict) and body.get("error"):
                            self.last_failure = classify_provider_failure(
                                status_code=response.status_code,
                                body=body,
                            )
                            self.last_error = self.last_failure.code
                            return
                        try:
                            content = body["choices"][0]["delta"].get("content")
                        except (KeyError, IndexError, TypeError, AttributeError):
                            content = None
                        if isinstance(content, str) and content:
                            yield content
        except Exception as exc:
            self.last_error = type(exc).__name__
            self.last_failure = classify_provider_failure(
                status_code=None,
                body=None,
                exception=exc,
            )
            return

    @staticmethod
    def _openai_generate(
        client: httpx.Client,
        base_url: str,
        secret: str | None,
        model: str,
        system_prompt: str,
        user_prompt: str,
        images: list[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        endpoint = base_url if base_url.rstrip("/").endswith("chat/completions") else f"{base_url.rstrip('/')}/chat/completions"
        user_content: str | list[dict[str, Any]] = user_prompt
        if images:
            user_content = [{"type": "text", "text": user_prompt}]
            user_content.extend({"type": "image_url", "image_url": {"url": image}} for image in images)
        headers = {"Authorization": f"Bearer {secret}"} if secret else {}
        response = client.post(
            endpoint,
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        response.raise_for_status()
        body = response.json()
        return str(body["choices"][0]["message"]["content"])

    @staticmethod
    def _anthropic_generate(
        client: httpx.Client,
        base_url: str,
        secret: str | None,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        endpoint = base_url if base_url.rstrip("/").endswith("messages") else f"{base_url.rstrip('/')}/messages"
        response = client.post(
            endpoint,
            headers={
                "x-api-key": secret or "",
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        body = response.json()
        return str(body["content"][0]["text"])
