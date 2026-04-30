from __future__ import annotations

import os
from dataclasses import dataclass


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class FlagEmbeddingReranker:
    enabled: bool | None = None
    model_name: str | None = None
    device: str | None = None
    max_length: int = 512
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.enabled is None:
            self.enabled = _env_flag("DF_ENABLE_FLAG_RERANKER", default=False)
        if self.model_name is None:
            self.model_name = os.getenv("DF_FLAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
        if self.device is None:
            self.device = os.getenv("DF_FLAG_RERANKER_DEVICE", "auto")
        self._tokenizer = None
        self._model = None
        self._load_error = ""

    @property
    def active(self) -> bool:
        return bool(self.enabled)

    @property
    def load_error(self) -> str:
        return self._load_error

    def _resolve_device(self, torch_module) -> str:
        if self.device and self.device != "auto":
            return self.device
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    def _ensure_model(self) -> bool:
        if not self.active:
            return False
        if self._model is not None and self._tokenizer is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = self._resolve_device(torch)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(device)
            self._model.eval()
            self.device = device
            return True
        except Exception as exc:
            self._load_error = str(exc)
            self.enabled = False
            self._tokenizer = None
            self._model = None
            return False

    def score_texts(self, query: str, docs: list[str]) -> list[float]:
        if not docs or not self._ensure_model():
            return []

        import torch

        scores: list[float] = []
        assert self._tokenizer is not None
        assert self._model is not None

        with torch.no_grad():
            for start in range(0, len(docs), self.batch_size):
                batch_docs = docs[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [query] * len(batch_docs),
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self._model(**encoded).logits.view(-1).float().cpu().tolist()
                scores.extend(float(item) for item in logits)
        return scores
