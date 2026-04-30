from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or DASHSCOPE_API_KEY is not set.")
        default_base_url = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if os.getenv("DASHSCOPE_API_KEY", "").strip() and not os.getenv("OPENAI_API_KEY", "").strip()
            else "https://api.openai.com/v1"
        )
        base_url = os.getenv("OPENAI_BASE_URL", os.getenv("DASHSCOPE_BASE_URL", default_base_url)).strip().rstrip("/")
        default_model = "qwen-plus" if "dashscope.aliyuncs.com" in base_url else "gpt-4o-mini"
        model = os.getenv("OPENAI_MODEL", os.getenv("DASHSCOPE_MODEL", default_model)).strip()
        return cls(api_key=api_key, base_url=base_url, model=model)


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail[:500]}") from exc

        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response format: {str(body)[:500]}") from exc

