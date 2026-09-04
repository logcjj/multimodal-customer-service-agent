from __future__ import annotations

import os
from pathlib import Path

from app.contracts.models import ModelConfigurationCreate, ModelKind
from app.models.service import ModelService
from app.storage.database import Database


def ensure_model(
    service: ModelService,
    *,
    provider: str,
    name: str,
    kind: ModelKind,
    base_url: str,
    api_key: str,
) -> str:
    existing = next((item for item in service.list_models() if item.kind == kind and item.name == name), None)
    if existing:
        return existing.id
    return service.create_model(
        ModelConfigurationCreate(
            provider=provider,
            name=name,
            kind=kind,
            base_url=base_url,
            api_key=api_key,
            capabilities=[kind.value],
        )
    ).id


def main() -> None:
    data_dir = Path(os.getenv("AKA_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    service = ModelService(Database(data_dir))
    configured: list[str] = []

    llm_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("DEEPSEEK_API_KEY", "").strip()
    )
    if llm_key:
        ensure_model(
            service,
            provider="openai-compatible",
            name=(
                os.getenv("OPENAI_MODEL", "").strip()
                or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
            ),
            kind=ModelKind.LLM,
            base_url=(
                os.getenv("OPENAI_BASE_URL", "").strip()
                or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
            ),
            api_key=llm_key,
        )
        configured.append("LLM")

    vision_key = os.getenv("OPENAI_VISION_API_KEY", "").strip()
    if vision_key:
        ensure_model(
            service,
            provider="openai-compatible",
            name=os.getenv("OPENAI_VISION_MODEL", "qwen3.7-plus").strip(),
            kind=ModelKind.VLM,
            base_url=os.getenv(
                "OPENAI_VISION_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).strip(),
            api_key=vision_key,
        )
        configured.append("VLM")

    aliyun_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if aliyun_key:
        workspace_id = os.getenv("ALIYUN_WORKSPACE_ID", "").strip()
        host = (
            f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com"
            if workspace_id
            else "https://dashscope.aliyuncs.com"
        )
        if workspace_id:
            embedding_base_url = f"{host}/compatible-mode/v1"
            rerank_base_url = f"{host}/compatible-api/v1"
        else:
            embedding_base_url = os.getenv(
                "DASHSCOPE_EMBEDDING_BASE_URL",
                f"{host}/compatible-mode/v1",
            ).strip()
            rerank_base_url = os.getenv(
                "DASHSCOPE_RERANK_BASE_URL",
                f"{host}/compatible-api/v1",
            ).strip()
        ensure_model(
            service,
            provider="Tongyi-Qianwen",
            name=os.getenv("ALIYUN_EMBEDDING_MODEL", "text-embedding-v4").strip(),
            kind=ModelKind.EMBEDDING,
            base_url=embedding_base_url,
            api_key=aliyun_key,
        )
        ensure_model(
            service,
            provider="Tongyi-Qianwen",
            name=os.getenv("ALIYUN_RERANK_MODEL", "qwen3-rerank").strip(),
            kind=ModelKind.RERANK,
            base_url=rerank_base_url,
            api_key=aliyun_key,
        )
        configured.extend(["Embedding", "Rerank"])

        vlm_model = os.getenv("ALIYUN_VLM_MODEL", "").strip()
        if vlm_model and not vision_key:
            ensure_model(
                service,
                provider="Tongyi-Qianwen",
                name=vlm_model,
                kind=ModelKind.VLM,
                base_url=embedding_base_url,
                api_key=aliyun_key,
            )
            configured.append("VLM")

        ocr_model = os.getenv("ALIYUN_OCR_MODEL", "").strip()
        if ocr_model:
            ensure_model(
                service,
                provider="Tongyi-Qianwen",
                name=ocr_model,
                kind=ModelKind.OCR,
                base_url=embedding_base_url,
                api_key=aliyun_key,
            )
            configured.append("OCR")

    print("Configured model roles: " + (", ".join(configured) if configured else "none"))


if __name__ == "__main__":
    main()
