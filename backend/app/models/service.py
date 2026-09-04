from __future__ import annotations

import json
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

import httpx
from fastapi import HTTPException
from sqlmodel import select

from app.contracts.models import (
    ModelConfiguration,
    ModelConfigurationCreate,
    ModelConfigurationUpdate,
    ModelKind,
)
from app.providers.errors import (
    ProviderFailure,
    classify_provider_failure,
    failure_from_response,
)
from app.storage.database import Database, ModelRecord


PROVIDER_CATALOG = [
    {
        "id": "openai",
        "name": "OpenAI",
        "capabilities": ["llm", "embedding", "vlm", "tts", "asr"],
        "accent": "#10a37f",
    },
    {"id": "anthropic", "name": "Anthropic", "capabilities": ["llm", "vlm"], "accent": "#d4a574"},
    {"id": "gemini", "name": "Gemini", "capabilities": ["llm", "embedding", "vlm"], "accent": "#4285f4"},
    {"id": "deepseek", "name": "DeepSeek", "capabilities": ["llm"], "accent": "#4d6bfe"},
    {
        "id": "tongyi-qianwen",
        "name": "Tongyi-Qianwen",
        "capabilities": ["llm", "embedding", "rerank", "vlm", "ocr", "asr", "tts"],
        "accent": "#615ced",
        "model_presets": {
            "rerank": [
                {
                    "name": "qwen3-rerank",
                    "description": "适用于文本语义检索与 RAG 精排",
                },
            ],
            "vlm": [
                {
                    "name": "qwen3-vl-flash",
                    "description": "Qwen3 系列低延迟视觉理解模型",
                },
                {
                    "name": "qwen3-vl-plus",
                    "description": "Qwen3 系列高能力视觉理解模型",
                },
            ],
            "ocr": [
                {
                    "name": "qwen3.5-ocr",
                    "description": "文档解析、文字定位与信息抽取主力版本",
                },
                {
                    "name": "qwen-vl-ocr-latest",
                    "description": "自动指向 Qwen-VL-OCR 最新版本",
                },
                {
                    "name": "qwen-vl-ocr-2025-11-20",
                    "description": "固定日期版本，适合稳定生产回归",
                },
                {
                    "name": "qwen-vl-ocr",
                    "description": "Qwen-VL-OCR 稳定基础版本",
                },
            ]
        },
    },
    {"id": "siliconflow", "name": "SILICONFLOW", "capabilities": ["llm", "embedding", "rerank", "vlm", "asr", "tts"], "accent": "#00a67e"},
    {"id": "volcengine", "name": "VolcEngine", "capabilities": ["llm", "embedding", "vlm"], "accent": "#1664ff"},
    {"id": "ollama", "name": "Ollama", "capabilities": ["llm", "embedding", "vlm"], "accent": "#9ca3af"},
    {
        "id": "openai-compatible",
        "name": "OpenAI-Compatible（自定义接口）",
        "capabilities": ["llm", "embedding", "rerank", "vlm"],
        "accent": "#22c55e",
    },
]


class ModelService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def list_models(self) -> list[ModelConfiguration]:
        with self.database.session() as session:
            records = session.exec(select(ModelRecord).order_by(ModelRecord.created_at)).all()
            return [self._to_contract(record) for record in records]

    def create_model(self, payload: ModelConfigurationCreate) -> ModelConfiguration:
        secret = payload.api_key.get_secret_value() if payload.api_key else None
        with self.database.session() as session:
            existing = session.exec(
                select(ModelRecord).where(ModelRecord.kind == payload.kind.value)
            ).first()
            record = ModelRecord(
                id=str(uuid4()),
                provider=payload.provider,
                name=payload.name,
                kind=payload.kind.value,
                base_url=payload.base_url.rstrip("/"),
                encrypted_secret=self.database.encrypt(secret) if secret else None,
                secret_hint=f"••••{secret[-4:]}" if secret else None,
                capabilities_json=json.dumps(payload.capabilities, ensure_ascii=False),
                enabled=payload.enabled,
                is_default=existing is None,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return self._to_contract(record)

    def get_record(self, model_id: str) -> ModelRecord:
        with self.database.session() as session:
            record = session.get(ModelRecord, model_id)
            if record is None:
                raise HTTPException(status_code=404, detail="模型配置不存在")
            session.expunge(record)
            return record

    def get_default_runtime(self, kind: ModelKind) -> tuple[ModelRecord, str | None] | None:
        with self.database.session() as session:
            record = session.exec(
                select(ModelRecord).where(
                    ModelRecord.kind == kind.value,
                    ModelRecord.is_default == True,  # noqa: E712
                    ModelRecord.enabled == True,  # noqa: E712
                )
            ).first()
            if record is None:
                return None
            session.expunge(record)
        return record, self.database.decrypt(record.encrypted_secret)

    def get_runtime_by_name(
        self,
        kind: ModelKind,
        name: str,
    ) -> tuple[ModelRecord, str | None] | None:
        normalized_name = name.strip().casefold()
        if not normalized_name:
            return None
        with self.database.session() as session:
            records = session.exec(
                select(ModelRecord).where(
                    ModelRecord.kind == kind.value,
                    ModelRecord.enabled == True,  # noqa: E712
                )
            ).all()
            record = next(
                (
                    item
                    for item in records
                    if item.name.strip().casefold() == normalized_name
                ),
                None,
            )
            if record is None:
                return None
            session.expunge(record)
        return record, self.database.decrypt(record.encrypted_secret)

    def set_default(self, model_id: str) -> ModelConfiguration:
        with self.database.session() as session:
            record = session.get(ModelRecord, model_id)
            if record is None:
                raise HTTPException(status_code=404, detail="模型配置不存在")
            existing = session.exec(select(ModelRecord).where(ModelRecord.kind == record.kind)).all()
            for item in existing:
                item.is_default = item.id == model_id
                item.updated_at = datetime.now(UTC)
                session.add(item)
            session.commit()
            session.refresh(record)
            return self._to_contract(record)

    def update_model(
        self,
        model_id: str,
        payload: ModelConfigurationUpdate,
    ) -> ModelConfiguration:
        with self.database.session() as session:
            record = session.get(ModelRecord, model_id)
            if record is None:
                raise HTTPException(status_code=404, detail="模型配置不存在")

            old_kind = record.kind
            new_kind = payload.kind.value
            if old_kind == new_kind:
                return self._to_contract(record)

            was_default = record.is_default
            target_default = session.exec(
                select(ModelRecord).where(
                    ModelRecord.kind == new_kind,
                    ModelRecord.is_default == True,  # noqa: E712
                )
            ).first()

            other_capabilities = [
                item
                for item in record.capabilities
                if item not in {old_kind, new_kind}
            ]
            record.kind = new_kind
            record.capabilities_json = json.dumps(
                [new_kind, *other_capabilities],
                ensure_ascii=False,
            )
            record.is_default = target_default is None
            record.health = "untested"
            record.latency_ms = None
            record.updated_at = datetime.now(UTC)
            session.add(record)

            if was_default:
                replacement = session.exec(
                    select(ModelRecord).where(
                        ModelRecord.kind == old_kind,
                        ModelRecord.id != model_id,
                        ModelRecord.enabled == True,  # noqa: E712
                    ).order_by(ModelRecord.created_at)
                ).first()
                if replacement is not None:
                    replacement.is_default = True
                    replacement.updated_at = datetime.now(UTC)
                    session.add(replacement)

            session.commit()
            session.refresh(record)
            return self._to_contract(record)

    def delete_model(self, model_id: str) -> None:
        with self.database.session() as session:
            record = session.get(ModelRecord, model_id)
            if record is None:
                raise HTTPException(status_code=404, detail="模型配置不存在")
            if record.is_default:
                raise HTTPException(status_code=409, detail="请先为该类型选择新的默认模型")
            session.delete(record)
            session.commit()

    async def test_model(self, model_id: str) -> dict[str, object]:
        record = self.get_record(model_id)
        secret = self.database.decrypt(record.encrypted_secret)
        if not secret:
            return self._save_health(
                model_id,
                "unhealthy",
                None,
                "未配置访问凭据",
                failure=ProviderFailure(
                    code="authentication_error",
                    message="未配置访问凭据",
                    retryable=False,
                ),
            )

        started = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=8.0, trust_env=False) as client:
                headers = {"Authorization": f"Bearer {secret}", "Content-Type": "application/json"}
                if record.kind == ModelKind.EMBEDDING.value:
                    response = await client.post(
                        f"{record.base_url.rstrip('/')}/embeddings",
                        headers=headers,
                        json={"model": record.name, "input": ["connection test"], "encoding_format": "float"},
                    )
                elif record.kind == ModelKind.RERANK.value:
                    endpoint = "reranks" if record.name.startswith("qwen3-") or "compatible-api" in record.base_url else "rerank"
                    response = await client.post(
                        f"{record.base_url.rstrip('/')}/{endpoint}",
                        headers=headers,
                        json={"model": record.name, "query": "test", "documents": ["test", "other"], "top_n": 2},
                    )
                elif record.kind in {ModelKind.LLM.value, ModelKind.VLM.value, ModelKind.OCR.value}:
                    response = await client.post(
                        f"{record.base_url.rstrip('/')}/chat/completions",
                        headers=headers,
                        json={"model": record.name, "messages": [{"role": "user", "content": "Reply OK"}], "max_tokens": 4, "temperature": 0},
                    )
                else:
                    response = await client.get(f"{record.base_url.rstrip('/')}/models", headers=headers)
                response.raise_for_status()
            latency_ms = round((perf_counter() - started) * 1000)
            return self._save_health(model_id, "healthy", latency_ms, "连接正常")
        except httpx.HTTPStatusError as exc:
            latency_ms = round((perf_counter() - started) * 1000)
            failure = failure_from_response(exc.response)
            return self._save_health(
                model_id,
                "unhealthy",
                latency_ms,
                failure.message,
                failure=failure,
            )
        except Exception as exc:
            latency_ms = round((perf_counter() - started) * 1000)
            failure = classify_provider_failure(status_code=None, body=None, exception=exc)
            return self._save_health(
                model_id,
                "unhealthy",
                latency_ms,
                failure.message,
                failure=failure,
            )

    def _save_health(
        self,
        model_id: str,
        health: str,
        latency_ms: int | None,
        message: str,
        *,
        failure: ProviderFailure | None = None,
    ) -> dict[str, object]:
        with self.database.session() as session:
            record = session.get(ModelRecord, model_id)
            if record is None:
                raise HTTPException(status_code=404, detail="模型配置不存在")
            record.health = health
            record.latency_ms = latency_ms
            record.updated_at = datetime.now(UTC)
            session.add(record)
            session.commit()
        payload: dict[str, object] = {
            "health": health,
            "latency_ms": latency_ms,
            "message": message,
        }
        if failure is not None:
            payload.update(
                {
                    "error_code": failure.code,
                    "request_id": failure.request_id,
                    "retryable": failure.retryable,
                }
            )
        return payload

    @staticmethod
    def _to_contract(record: ModelRecord) -> ModelConfiguration:
        return ModelConfiguration(
            id=record.id,
            provider=record.provider,
            name=record.name,
            kind=ModelKind(record.kind),
            base_url=record.base_url,
            secret_configured=bool(record.encrypted_secret),
            secret_hint=record.secret_hint,
            capabilities=record.capabilities,
            enabled=record.enabled,
            is_default=record.is_default,
            health=record.health,
            latency_ms=record.latency_ms,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
