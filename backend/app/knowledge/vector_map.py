from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Iterator, Sequence

import joblib
import numpy as np


_SAFE_COMPONENT_MAX_LENGTH = 120
_DIGEST_LENGTH = 12
_GENERATION_RETAIN_COUNT = 2
_CONTROL_LOCK_TIMEOUT_SECONDS = 35.0
_CONTROL_LOCK_STALE_SECONDS = 30.0
_CONTROL_LOCK_POLL_SECONDS = 0.01
_BUILD_LEASE_INITIALIZATION_GRACE_SECONDS = 5.0
_WINDOWS_RESERVED_STEMS = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)


@dataclass(frozen=True)
class VectorSource:
    child_id: str
    dataset_id: str
    document_id: str
    document_name: str
    title: str
    excerpt: str
    page_start: int
    page_end: int
    product: str | None
    embedding: Sequence[float] | None
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.embedding is not None:
            object.__setattr__(self, "embedding", tuple(self.embedding))


class _VectorMapValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _VectorMapBuildCancelled(Exception):
    pass


class VectorMapService:
    def __init__(self, data_dir: str | Path, reducer_factory=None) -> None:
        self.root = Path(data_dir) / "vector_maps"
        self.root.mkdir(parents=True, exist_ok=True)
        self._control_root = self.root / ".control"
        self._control_keys_dir = self._control_root / "keys"
        self._control_locks_dir = self._control_root / "locks"
        self._control_failures_dir = self._control_root / "failures"
        self._build_leases_dir = self._control_root / "build-leases"
        self._control_keys_dir.mkdir(parents=True, exist_ok=True)
        self._control_locks_dir.mkdir(parents=True, exist_ok=True)
        self._control_failures_dir.mkdir(parents=True, exist_ok=True)
        self._build_leases_dir.mkdir(parents=True, exist_ok=True)
        self.reducer_factory = reducer_factory
        self._states: dict[tuple[str, str, str], dict[str, object]] = {}
        self._epochs: dict[tuple[str, str, str], int] = {}
        self._reducer_bundles: dict[
            tuple[str, str, str, str, str, int],
            tuple[dict[str, object], dict[str, object]],
        ] = {}
        self._lock = threading.RLock()

    def build(
        self,
        dataset_id: str,
        published_version: str,
        embedding_model: str,
        sources: Sequence[VectorSource],
    ) -> dict[str, object]:
        key = (dataset_id, published_version, embedding_model)
        try:
            lease_owner = self._acquire_build_lease(key)
        except _VectorMapValidationError as exc:
            return self._failed(key, exc.code, exc.message)
        if lease_owner is None:
            with self._lock:
                self._states[key] = {"status": "building"}
            return {"status": "building"}

        try:
            epoch = self._begin_build_epoch(key)
        except _VectorMapValidationError as exc:
            self._release_build_lease(key, lease_owner)
            return self._failed(key, exc.code, exc.message)

        with self._lock:
            self._epochs[key] = epoch
            self._states[key] = {"status": "building", "epoch": epoch}

        try:
            source_list = list(sources)
            matrix = self._validate_sources(dataset_id, source_list)
            self._raise_if_cancelled(key, epoch)
            if len(source_list) <= 2:
                raw = self._small_sample_projection(len(source_list))
                reducer = None
                reducer_available = False
                small_sample = True
            else:
                reducer = self._make_reducer(len(source_list))
                raw = np.asarray(reducer.fit_transform(matrix), dtype=np.float32)
                reducer_available = hasattr(reducer, "transform")
                small_sample = False
            self._raise_if_cancelled(key, epoch)
            normalized, bounds = self._normalize(raw)
            payload = self._payload(
                dataset_id=dataset_id,
                published_version=published_version,
                embedding_model=embedding_model,
                sources=source_list,
                matrix=matrix,
                normalized=normalized,
                bounds=bounds,
                reducer_available=reducer_available,
                small_sample=small_sample,
            )
            self._atomic_write(
                dataset_id=dataset_id,
                published_version=published_version,
                embedding_model=embedding_model,
                payload=payload,
                reducer=reducer,
                matrix=matrix,
                normalized=normalized,
                reducer_available=reducer_available,
                small_sample=small_sample,
                key=key,
                epoch=epoch,
            )
            try:
                self._raise_if_cancelled(key, epoch)
            except _VectorMapBuildCancelled:
                return self._cancelled(key, epoch)
            except _VectorMapValidationError as exc:
                return self._failed(key, exc.code, exc.message)

            with self._lock:
                self._states[key] = {"status": "ready", "meta": payload["meta"], "epoch": epoch}
            return payload
        except _VectorMapBuildCancelled:
            return self._cancelled(key, epoch)
        except _VectorMapValidationError as exc:
            return self._failed_or_cancelled(key, epoch, exc.code, exc.message)
        except Exception as exc:
            return self._failed_or_cancelled(key, epoch, "projection_failed", str(exc))
        finally:
            self._release_build_lease(key, lease_owner)

    def load(self, dataset_id: str, published_version: str, embedding_model: str) -> dict[str, object]:
        key = (dataset_id, published_version, embedding_model)
        try:
            if self._build_lease_active(key):
                return {"status": "building"}
            failure = self._read_current_failure(key)
            if failure is not None:
                return failure
            generation_dir = self._active_generation_dir(dataset_id, published_version, embedding_model)
        except _VectorMapValidationError as exc:
            return self._failed(key, exc.code, exc.message)

        if generation_dir is None:
            with self._lock:
                state = self._states.get(key)
                if state and state.get("status") == "failed":
                    return self._jsonable(state)
            return {"status": "missing"}

        manifest_path = generation_dir / "manifest.json"
        points_path = generation_dir / "points.json.gz"
        reducer_path = generation_dir / "reducer.joblib"
        if not manifest_path.is_file() or not points_path.is_file() or not reducer_path.is_file():
            return {"status": "missing"}

        try:
            manifest = self._read_json(manifest_path)
            if not self._manifest_epoch_is_current(key, manifest):
                return {"status": "missing"}
            with gzip.open(points_path, "rt", encoding="utf-8") as handle:
                points = json.load(handle)
            if not self._manifest_epoch_is_current(key, manifest):
                return {"status": "missing"}
        except FileNotFoundError:
            return {"status": "missing"}
        except Exception as exc:
            return self._failed(key, "cache_read_failed", str(exc))

        payload = {"status": "ready", "meta": manifest, "points": points}
        with self._lock:
            self._states[key] = {"status": "ready", "meta": manifest, "epoch": self._epochs.get(key, 0)}
        return payload

    def status(self, dataset_id: str, published_version: str, embedding_model: str) -> dict[str, object]:
        key = (dataset_id, published_version, embedding_model)
        try:
            if self._build_lease_active(key):
                return {"status": "building"}
            failure = self._read_current_failure(key)
            if failure is not None:
                return failure
            generation_dir = self._active_generation_dir(dataset_id, published_version, embedding_model)
        except _VectorMapValidationError as exc:
            return self._failed(key, exc.code, exc.message)

        if generation_dir is None:
            with self._lock:
                state = self._states.get(key)
                if state and state.get("status") in {"building", "failed"}:
                    if state.get("status") == "building":
                        return {"status": "building"}
                    return self._jsonable(state)
            return {"status": "missing"}

        manifest_path = generation_dir / "manifest.json"
        points_path = generation_dir / "points.json.gz"
        reducer_path = generation_dir / "reducer.joblib"
        if not manifest_path.is_file() or not points_path.is_file() or not reducer_path.is_file():
            return {"status": "missing"}
        try:
            manifest = self._read_json(manifest_path)
            if not self._manifest_epoch_is_current(key, manifest):
                return {"status": "missing"}
        except FileNotFoundError:
            return {"status": "missing"}
        except Exception as exc:
            return self._failed(key, "cache_read_failed", str(exc))

        with self._lock:
            self._states[key] = {"status": "ready", "meta": manifest, "epoch": self._epochs.get(key, 0)}
        return {"status": "ready", "meta": manifest}

    def invalidate(
        self,
        dataset_id: str,
        published_version: str | None = None,
        embedding_model: str | None = None,
        *,
        remove_cache: bool = True,
    ) -> dict[str, object]:
        removed = 0
        keys_to_invalidate = set(self._matching_control_keys(dataset_id, published_version, embedding_model))
        with self._lock:
            keys_to_invalidate.update(self._matching_state_keys(dataset_id, published_version, embedding_model))
        if published_version is not None and embedding_model is not None:
            keys_to_invalidate.add((dataset_id, published_version, embedding_model))

        new_epochs: dict[tuple[str, str, str], int] = {}
        for key in sorted(keys_to_invalidate):
            try:
                with self._control_lock(key):
                    epoch = self._read_control_epoch_unlocked(key)
                    new_epoch = epoch + 1
                    self._write_control_epoch_unlocked(key, new_epoch)
                    self._clear_failure_unlocked(key)
                    new_epochs[key] = new_epoch
            except _VectorMapValidationError as exc:
                return {
                    "status": "failed",
                    "error": {"code": exc.code, "message": exc.message},
                }

        removal_failures: list[str] = []
        if remove_cache:
            for cache_dir in self._matching_cache_dirs(dataset_id, published_version, embedding_model):
                try:
                    shutil.rmtree(cache_dir)
                except FileNotFoundError:
                    continue
                except OSError:
                    removal_failures.append(str(cache_dir))
                    continue
                if cache_dir.exists():
                    removal_failures.append(str(cache_dir))
                else:
                    removed += 1
            self._cleanup_empty_parents(dataset_id, published_version)
        with self._lock:
            for key, epoch in new_epochs.items():
                self._epochs[key] = epoch
            for key in self._matching_state_keys(dataset_id, published_version, embedding_model):
                self._states.pop(key, None)
            self._drop_reducer_bundles_unlocked(dataset_id, published_version, embedding_model)
        if removal_failures:
            return {
                "status": "failed",
                "removed": removed,
                "error": {
                    "code": "cache_remove_failed",
                    "message": "向量图缓存删除失败，旧投影已通过 epoch 隔离。",
                },
            }
        return {"status": "invalidated", "removed": removed}

    def transform_query(
        self,
        dataset_id: str,
        published_version: str,
        embedding_model: str,
        vector: Sequence[float],
    ) -> dict[str, object] | None:
        key = (dataset_id, published_version, embedding_model)
        try:
            manifest, bundle = self._load_reducer_bundle(dataset_id, published_version, embedding_model)
            query = self._validate_query_vector(vector, int(manifest["vector_dimension"]))
            reducer = bundle.get("reducer")
            reducer_available = bool(bundle.get("reducer_available"))
            small_sample = bool(manifest.get("umap", {}).get("small_sample")) if isinstance(manifest.get("umap"), dict) else False
            if small_sample or not reducer_available:
                return None
            if not hasattr(reducer, "transform"):
                raise _VectorMapValidationError("query_transform_failed", "保存的降维器缺少 transform 能力。")
            try:
                raw = np.asarray(reducer.transform(np.asarray([query], dtype=np.float32)), dtype=np.float32)
                if raw.ndim != 2 or raw.shape != (1, 2) or not np.all(np.isfinite(raw)):
                    raise _VectorMapValidationError("query_transform_failed", "查询向量投影结果不是有效二维坐标。")
                return self._normalize_query(raw[0], manifest["bounds"])
            except _VectorMapValidationError:
                raise
            except Exception as exc:
                raise _VectorMapValidationError("query_transform_failed", "查询向量投影失败。") from exc
        except _VectorMapValidationError as exc:
            return self._failed(key, exc.code, exc.message)
        except Exception as exc:
            return self._failed(key, "query_transform_failed", str(exc))

    def record_failure(
        self,
        dataset_id: str,
        published_version: str,
        embedding_model: str,
        code: str,
        message: str,
    ) -> dict[str, object]:
        key = (dataset_id, published_version, embedding_model)
        try:
            lease_owner = self._acquire_build_lease(key)
        except _VectorMapValidationError as exc:
            return self._failed(key, exc.code, exc.message)
        if lease_owner is None:
            return {"status": "building"}
        try:
            epoch = self._begin_build_epoch(key)
            with self._control_lock(key):
                if self._read_control_epoch_unlocked(key) != epoch:
                    return self._cancelled(key, epoch)
                payload = self._write_failure_unlocked(key, epoch, code, message)
            with self._lock:
                self._states[key] = payload
            return payload
        except _VectorMapValidationError as exc:
            return self._failed(key, exc.code, exc.message)
        finally:
            self._release_build_lease(key, lease_owner)

    def _make_reducer(self, source_count: int):
        if self.reducer_factory is not None:
            return self.reducer_factory()
        return self._default_reducer(source_count)

    def _default_reducer(self, source_count: int):
        from umap import UMAP

        return UMAP(
            n_components=2,
            metric="cosine",
            n_neighbors=min(15, source_count - 1),
            min_dist=0.1,
            random_state=42,
            transform_seed=42,
            low_memory=True,
        )

    def _validate_sources(self, dataset_id: str, sources: list[VectorSource]) -> np.ndarray:
        if not sources:
            raise _VectorMapValidationError("empty_sources", "没有可投影的向量源。")

        vectors: list[list[float]] = []
        expected_dimension: int | None = None
        for item in sources:
            if item.dataset_id != dataset_id:
                raise _VectorMapValidationError("mixed_dataset", "向量源包含其他知识库的数据。")
            if item.embedding is None or len(item.embedding) == 0:
                raise _VectorMapValidationError("missing_embedding", "向量源缺少 Embedding。")
            if expected_dimension is None:
                expected_dimension = len(item.embedding)
            elif len(item.embedding) != expected_dimension:
                raise _VectorMapValidationError("dimension_mismatch", "Embedding 向量维度不一致。")
            vectors.append(self._coerce_finite_vector(item.embedding, "Embedding 含有非数值或非有限数据。"))

        matrix = np.asarray(vectors, dtype=np.float32)
        if not np.all(np.isfinite(matrix)):
            raise _VectorMapValidationError("invalid_embedding", "Embedding 含有非数值或非有限数据。")
        return matrix

    def _validate_query_vector(self, vector: Sequence[float], expected_dimension: int) -> np.ndarray:
        if vector is None or len(vector) == 0:
            raise _VectorMapValidationError("missing_embedding", "查询向量为空。")
        if len(vector) != expected_dimension:
            raise _VectorMapValidationError("dimension_mismatch", "查询向量维度与投影不一致。")
        query = np.asarray(self._coerce_finite_vector(vector, "查询向量含有非数值或非有限数据。"), dtype=np.float32)
        if not np.all(np.isfinite(query)):
            raise _VectorMapValidationError("invalid_embedding", "查询向量含有非数值或非有限数据。")
        return query

    @staticmethod
    def _coerce_finite_vector(vector: Sequence[float], invalid_message: str) -> list[float]:
        values: list[float] = []
        try:
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError("vector value is not a real number")
                values.append(float(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise _VectorMapValidationError("invalid_embedding", invalid_message) from exc
        if not np.all(np.isfinite(values)):
            raise _VectorMapValidationError("invalid_embedding", invalid_message)
        return values

    @staticmethod
    def _small_sample_projection(count: int) -> np.ndarray:
        if count == 1:
            return np.asarray([[0.5, 0.5]], dtype=np.float32)
        return np.asarray([[0.0, 0.5], [1.0, 0.5]], dtype=np.float32)

    def _payload(
        self,
        *,
        dataset_id: str,
        published_version: str,
        embedding_model: str,
        sources: list[VectorSource],
        matrix: np.ndarray,
        normalized: np.ndarray,
        bounds: dict[str, float],
        reducer_available: bool,
        small_sample: bool,
    ) -> dict[str, object]:
        points = [
            {
                "child_id": source.child_id,
                "dataset_id": source.dataset_id,
                "document_id": source.document_id,
                "document_name": source.document_name,
                "title": source.title,
                "excerpt": source.excerpt,
                "page_start": source.page_start,
                "page_end": source.page_end,
                "product": source.product,
                "x": float(normalized[index, 0]),
                "y": float(normalized[index, 1]),
            }
            for index, source in enumerate(sources)
        ]
        meta = {
            "dataset_id": dataset_id,
            "published_version": published_version,
            "embedding_model": embedding_model,
            "vector_dimension": int(matrix.shape[1]),
            "point_count": len(points),
            "bounds": bounds,
            "content_digest": self._content_digest(sources),
            "built_at": datetime.now(UTC).isoformat(),
            "umap": {
                "n_components": 2,
                "metric": "cosine",
                "n_neighbors": min(15, len(sources) - 1) if len(sources) > 1 else 0,
                "min_dist": 0.1,
                "random_state": 42,
                "transform_seed": 42,
                "low_memory": True,
                "reducer_available": reducer_available,
                "small_sample": small_sample,
            },
        }
        return {"status": "ready", "meta": meta, "points": points}

    def _atomic_write(
        self,
        *,
        dataset_id: str,
        published_version: str,
        embedding_model: str,
        payload: dict[str, object],
        reducer,
        matrix: np.ndarray,
        normalized: np.ndarray,
        reducer_available: bool,
        small_sample: bool,
        key: tuple[str, str, str],
        epoch: int,
    ) -> None:
        self._raise_if_cancelled(key, epoch)

        key_dir = self._cache_dir(dataset_id, published_version, embedding_model)
        generation_name = self._new_generation_name()
        temp_parent = self.root / ".tmp"
        temp_parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{generation_name}.", dir=temp_parent))
        generation_dir = key_dir / generation_name
        active_tmp = key_dir / f".active.{generation_name}.json"

        try:
            meta = payload.get("meta")
            if isinstance(meta, dict):
                meta["control_epoch"] = epoch
            self._write_json(temp_dir / "manifest.json", payload["meta"])
            with gzip.open(temp_dir / "points.json.gz", "wt", encoding="utf-8") as handle:
                json.dump(payload["points"], handle, ensure_ascii=False, separators=(",", ":"))
            bundle: dict[str, object] = {
                "reducer": reducer,
                "reducer_available": reducer_available,
            }
            if small_sample or not reducer_available:
                bundle["vectors"] = matrix.astype(float).tolist()
                bundle["points"] = [
                    {"x": float(row[0]), "y": float(row[1])}
                    for row in normalized
                ]
            joblib.dump(bundle, temp_dir / "reducer.joblib")

            self._raise_if_cancelled(key, epoch)
            with self._control_lock(key):
                if self._read_control_epoch_unlocked(key) != epoch:
                    raise _VectorMapBuildCancelled()
                key_dir.mkdir(parents=True, exist_ok=True)
                previous_active = self._read_active_generation_name(key_dir)
                os.replace(temp_dir, generation_dir)
                self._write_json(active_tmp, {"generation": generation_name})
                os.replace(active_tmp, key_dir / "active.json")
                self._clear_failure_unlocked(key)
                self._cleanup_old_generations(key_dir, generation_name, previous_active)
            with self._lock:
                self._drop_reducer_bundles_for_key_unlocked(key)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if active_tmp.exists():
                active_tmp.unlink(missing_ok=True)
            raise

    def _cleanup_old_generations(
        self,
        key_dir: Path,
        active_generation: str,
        previous_generation: str | None = None,
    ) -> None:
        generations = sorted(
            [path for path in key_dir.iterdir() if path.is_dir() and path.name.startswith("generation-")],
            key=lambda path: path.name,
        )
        if len(generations) <= _GENERATION_RETAIN_COUNT:
            return

        keep = {active_generation}
        if previous_generation and previous_generation != active_generation and (key_dir / previous_generation).is_dir():
            keep.add(previous_generation)
        else:
            active_index = next((index for index, path in enumerate(generations) if path.name == active_generation), None)
            if active_index is None:
                return
            if active_index > 0:
                keep.add(generations[active_index - 1].name)

        for generation in generations:
            if generation.name not in keep and generation.name != active_generation:
                shutil.rmtree(generation, ignore_errors=True)

    def _load_reducer_bundle(
        self,
        dataset_id: str,
        published_version: str,
        embedding_model: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        key = (dataset_id, published_version, embedding_model)
        for _ in range(2):
            generation_dir = self._active_generation_dir(dataset_id, published_version, embedding_model)
            if generation_dir is None:
                raise _VectorMapValidationError("cache_missing", "向量图缓存不存在。")
            manifest_path = generation_dir / "manifest.json"
            points_path = generation_dir / "points.json.gz"
            reducer_path = generation_dir / "reducer.joblib"
            if not manifest_path.is_file() or not points_path.is_file() or not reducer_path.is_file():
                raise _VectorMapValidationError("cache_missing", "向量图缓存不存在。")
            manifest = self._read_json(manifest_path)
            if not self._manifest_epoch_is_current(key, manifest):
                raise _VectorMapValidationError("cache_missing", "向量图缓存已失效。")
            digest = str(manifest.get("content_digest") or "")
            control_epoch = manifest.get("control_epoch", 0)
            if isinstance(control_epoch, bool) or not isinstance(control_epoch, int):
                raise _VectorMapValidationError("cache_read_failed", "向量图缓存 epoch 无效。")
            identity = (*key, generation_dir.name, digest, control_epoch)
            with self._lock:
                cached = self._reducer_bundles.get(identity)
                if cached is not None:
                    return cached
                bundle = joblib.load(reducer_path)
                current_generation = self._active_generation_dir(dataset_id, published_version, embedding_model)
                if current_generation == generation_dir and self._manifest_epoch_is_current(key, manifest):
                    self._drop_reducer_bundles_for_key_unlocked(key)
                    value = (manifest, bundle)
                    self._reducer_bundles[identity] = value
                    return value
        raise _VectorMapValidationError("cache_missing", "向量图缓存在加载期间发生切换。")

    def _nearest_saved_point(self, query: np.ndarray, bundle: dict[str, object]) -> dict[str, float]:
        vectors = np.asarray(bundle.get("vectors", []), dtype=np.float32)
        points = list(bundle.get("points", []))
        if len(vectors) == 0 or not points:
            raise _VectorMapValidationError("cache_missing", "向量图缓存缺少原始向量。")

        query_norm = float(np.linalg.norm(query))
        vector_norms = np.linalg.norm(vectors, axis=1)
        if query_norm > 0 and np.all(vector_norms > 0):
            similarities = vectors @ query / (vector_norms * query_norm)
            index = int(np.argmax(similarities))
        else:
            distances = np.linalg.norm(vectors - query, axis=1)
            index = int(np.argmin(distances))
        point = points[index]
        return {"x": float(point["x"]), "y": float(point["y"])}

    @staticmethod
    def _normalize(raw: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        points = np.asarray(raw, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise _VectorMapValidationError("projection_shape", "UMAP 投影结果不是二维坐标。")
        if not np.all(np.isfinite(points)):
            raise _VectorMapValidationError("projection_non_finite", "UMAP 投影结果包含非有限坐标。")
        minimums = points.min(axis=0)
        maximums = points.max(axis=0)
        bounds = {
            "x_min": float(minimums[0]),
            "x_max": float(maximums[0]),
            "y_min": float(minimums[1]),
            "y_max": float(maximums[1]),
        }
        normalized = np.column_stack(
            [
                VectorMapService._normalize_axis(points[:, 0], minimums[0], maximums[0]),
                VectorMapService._normalize_axis(points[:, 1], minimums[1], maximums[1]),
            ]
        ).astype(np.float32)
        return normalized, bounds

    @staticmethod
    def _normalize_axis(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
        span = float(maximum - minimum)
        if span == 0:
            return np.full(values.shape, 0.5, dtype=np.float32)
        return np.clip((values - minimum) / span, 0.0, 1.0)

    @staticmethod
    def _normalize_query(raw: np.ndarray, bounds: dict[str, float]) -> dict[str, float]:
        x = VectorMapService._normalize_scalar(float(raw[0]), float(bounds["x_min"]), float(bounds["x_max"]))
        y = VectorMapService._normalize_scalar(float(raw[1]), float(bounds["y_min"]), float(bounds["y_max"]))
        return {"x": x, "y": y}

    @staticmethod
    def _normalize_scalar(value: float, minimum: float, maximum: float) -> float:
        span = maximum - minimum
        if span == 0:
            return 0.5
        return float(min(1.0, max(0.0, (value - minimum) / span)))

    def _begin_build_epoch(self, key: tuple[str, str, str]) -> int:
        with self._control_lock(key):
            epoch = self._read_control_epoch_unlocked(key) + 1
            self._write_control_epoch_unlocked(key, epoch)
            self._clear_failure_unlocked(key)
        with self._lock:
            self._drop_reducer_bundles_for_key_unlocked(key)
            return epoch

    @staticmethod
    def _control_key_id(key: tuple[str, str, str]) -> str:
        dataset_id, published_version, embedding_model = key
        payload = {
            "dataset_id": dataset_id,
            "embedding_model": embedding_model,
            "published_version": published_version,
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _control_path(self, key: tuple[str, str, str]) -> Path:
        return self._control_keys_dir / f"{self._control_key_id(key)}.json"

    def _control_lock_path(self, key: tuple[str, str, str]) -> Path:
        return self._control_locks_dir / f"{self._control_key_id(key)}.lock"

    def _failure_path(self, key: tuple[str, str, str]) -> Path:
        return self._control_failures_dir / f"{self._control_key_id(key)}.json"

    def _build_lease_path(self, key: tuple[str, str, str]) -> Path:
        return self._build_leases_dir / f"{self._control_key_id(key)}.json"

    def _acquire_build_lease(self, key: tuple[str, str, str]) -> str | None:
        lease_path = self._build_lease_path(key)
        owner = f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
        payload = {
            "owner": owner,
            "pid": os.getpid(),
            "thread": threading.get_ident(),
            "acquired_at": datetime.now(UTC).isoformat(),
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

        for _ in range(3):
            try:
                descriptor = os.open(lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if self._remove_stale_build_lease(lease_path):
                    continue
                return None
            except FileNotFoundError:
                self._build_leases_dir.mkdir(parents=True, exist_ok=True)
                continue
            except OSError as exc:
                raise _VectorMapValidationError("lease_acquire_failed", "向量图构建租约创建失败。") from exc

            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                lease_path.unlink(missing_ok=True)
                raise
            return owner
        return None

    def _build_lease_active(self, key: tuple[str, str, str]) -> bool:
        lease_path = self._build_lease_path(key)
        if not lease_path.exists():
            return False
        if self._remove_stale_build_lease(lease_path):
            return False
        return lease_path.exists()

    def _remove_stale_build_lease(self, lease_path: Path) -> bool:
        try:
            age = time.time() - lease_path.stat().st_mtime
        except FileNotFoundError:
            return True

        owner_pid: int | None = None
        try:
            payload = self._read_json(lease_path)
            candidate_pid = payload.get("pid")
            if isinstance(candidate_pid, int) and not isinstance(candidate_pid, bool) and candidate_pid > 0:
                owner_pid = candidate_pid
        except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
            owner_pid = None

        if owner_pid is not None and self._process_is_alive(owner_pid):
            return False
        if owner_pid is None and age < _BUILD_LEASE_INITIALIZATION_GRACE_SECONDS:
            return False
        try:
            lease_path.unlink()
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return not lease_path.exists()

    def _release_build_lease(self, key: tuple[str, str, str], owner: str) -> None:
        lease_path = self._build_lease_path(key)
        try:
            payload = self._read_json(lease_path)
        except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return
        if payload.get("owner") != owner:
            return
        try:
            lease_path.unlink()
        except FileNotFoundError:
            pass

    def _read_current_failure(self, key: tuple[str, str, str]) -> dict[str, object] | None:
        with self._control_lock(key):
            failure_path = self._failure_path(key)
            try:
                marker = self._read_json(failure_path)
            except FileNotFoundError:
                return None
            except Exception as exc:
                raise _VectorMapValidationError("failure_read_failed", "向量图失败状态读取失败。") from exc

            marker_epoch = marker.get("epoch")
            if (
                isinstance(marker_epoch, bool)
                or not isinstance(marker_epoch, int)
                or marker_epoch < 0
            ):
                raise _VectorMapValidationError("failure_read_failed", "向量图失败状态 epoch 无效。")
            if marker_epoch != self._read_control_epoch_unlocked(key):
                self._clear_failure_unlocked(key)
                return None
            error = marker.get("error")
            if not isinstance(error, dict):
                raise _VectorMapValidationError("failure_read_failed", "向量图失败状态内容无效。")
            code = error.get("code")
            message = error.get("message")
            if not isinstance(code, str) or not isinstance(message, str):
                raise _VectorMapValidationError("failure_read_failed", "向量图失败状态内容无效。")
            return {"status": "failed", "error": {"code": code, "message": message}}

    def _write_failure_unlocked(
        self,
        key: tuple[str, str, str],
        epoch: int,
        code: str,
        message: str,
    ) -> dict[str, object]:
        dataset_id, published_version, embedding_model = key
        payload = {"status": "failed", "error": {"code": code, "message": message}}
        marker = {
            **payload,
            "dataset_id": dataset_id,
            "published_version": published_version,
            "embedding_model": embedding_model,
            "epoch": epoch,
            "failed_at": datetime.now(UTC).isoformat(),
        }
        path = self._failure_path(key)
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        self._write_json(temp_path, marker)
        os.replace(temp_path, path)
        return payload

    def _clear_failure_unlocked(self, key: tuple[str, str, str]) -> None:
        self._failure_path(key).unlink(missing_ok=True)

    @contextmanager
    def _control_lock(self, key: tuple[str, str, str]) -> Iterator[None]:
        lock_dir = self._control_lock_path(key)
        owner = f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
        deadline = time.monotonic() + _CONTROL_LOCK_TIMEOUT_SECONDS
        acquired = False

        while not acquired:
            try:
                lock_dir.mkdir(parents=True)
                acquired = True
            except FileExistsError:
                if self._remove_stale_control_lock(lock_dir):
                    continue
                if time.monotonic() >= deadline:
                    raise _VectorMapValidationError("lock_timeout", "向量图控制锁获取超时。")
                time.sleep(_CONTROL_LOCK_POLL_SECONDS)
            except FileNotFoundError:
                self._control_locks_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._write_json(
                lock_dir / "owner.json",
                {
                    "owner": owner,
                    "pid": os.getpid(),
                    "thread": threading.get_ident(),
                    "acquired_at": datetime.now(UTC).isoformat(),
                },
            )
            yield
        finally:
            self._release_control_lock(lock_dir, owner)

    @staticmethod
    def _remove_stale_control_lock(lock_dir: Path) -> bool:
        try:
            age = time.time() - lock_dir.stat().st_mtime
        except FileNotFoundError:
            return True
        owner_pid: int | None = None
        try:
            owner_payload = json.loads((lock_dir / "owner.json").read_text(encoding="utf-8"))
            candidate_pid = owner_payload.get("pid")
            if isinstance(candidate_pid, int) and not isinstance(candidate_pid, bool) and candidate_pid > 0:
                owner_pid = candidate_pid
        except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
            owner_pid = None
        if owner_pid is not None:
            if VectorMapService._process_is_alive(owner_pid):
                return False
        elif age < _CONTROL_LOCK_STALE_SECONDS:
            return False
        try:
            shutil.rmtree(lock_dir)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return not lock_dir.exists()

    @staticmethod
    def _process_is_alive(pid: int) -> bool:
        if pid == os.getpid():
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _manifest_epoch_is_current(
        self,
        key: tuple[str, str, str],
        manifest: dict[str, object],
    ) -> bool:
        manifest_epoch = manifest.get("control_epoch", 0)
        if isinstance(manifest_epoch, bool) or not isinstance(manifest_epoch, int) or manifest_epoch < 0:
            raise _VectorMapValidationError("cache_read_failed", "向量图缓存 epoch 无效。")
        return self._read_control_epoch_unlocked(key) == manifest_epoch

    def _release_control_lock(self, lock_dir: Path, owner: str) -> None:
        owner_path = lock_dir / "owner.json"
        try:
            owner_payload = self._read_json(owner_path)
        except FileNotFoundError:
            return
        except Exception:
            return
        if owner_payload.get("owner") == owner:
            shutil.rmtree(lock_dir, ignore_errors=True)

    def _read_control_epoch_unlocked(self, key: tuple[str, str, str]) -> int:
        path = self._control_path(key)
        try:
            payload = self._read_json(path)
        except FileNotFoundError:
            return 0
        except Exception as exc:
            raise _VectorMapValidationError("control_read_failed", "向量图控制文件读取失败。") from exc

        epoch = payload.get("epoch", 0)
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise _VectorMapValidationError("control_read_failed", "向量图控制文件 epoch 无效。")
        return epoch

    def _write_control_epoch_unlocked(self, key: tuple[str, str, str], epoch: int) -> None:
        dataset_id, published_version, embedding_model = key
        payload = {
            "dataset_id": dataset_id,
            "embedding_model": embedding_model,
            "epoch": epoch,
            "published_version": published_version,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        path = self._control_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        self._write_json(temp_path, payload)
        os.replace(temp_path, path)

    def _cache_dir(self, dataset_id: str, published_version: str, embedding_model: str) -> Path:
        return (
            self.root
            / self._safe_component(dataset_id)
            / self._safe_component(published_version)
            / self._model_component(embedding_model)
        )

    def _read_active_generation_name(self, key_dir: Path) -> str | None:
        active_path = key_dir / "active.json"
        if not active_path.is_file():
            return None
        try:
            active = self._read_json(active_path)
        except Exception:
            return None
        generation = active.get("generation")
        if (
            isinstance(generation, str)
            and generation
            and Path(generation).name == generation
            and generation not in {".", ".."}
            and (key_dir / generation).is_dir()
        ):
            return generation
        return None

    def _active_generation_dir(
        self,
        dataset_id: str,
        published_version: str,
        embedding_model: str,
    ) -> Path | None:
        key_dir = self._cache_dir(dataset_id, published_version, embedding_model)
        active_path = key_dir / "active.json"
        if not active_path.is_file():
            return None
        try:
            active = self._read_json(active_path)
        except FileNotFoundError:
            return None
        except Exception as exc:
            raise _VectorMapValidationError("cache_read_failed", "向量图缓存指针读取失败。") from exc

        generation = active.get("generation")
        if (
            not isinstance(generation, str)
            or not generation
            or Path(generation).name != generation
            or generation in {".", ".."}
        ):
            raise _VectorMapValidationError("cache_read_failed", "向量图缓存指针无效。")

        generation_dir = key_dir / generation
        if not generation_dir.is_dir():
            return None
        return generation_dir

    @staticmethod
    def _new_generation_name() -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return f"generation-{timestamp}-{os.getpid()}-{uuid.uuid4().hex[:12]}"

    def _raise_if_cancelled(self, key: tuple[str, str, str], epoch: int) -> None:
        with self._control_lock(key):
            current_epoch = self._read_control_epoch_unlocked(key)
        if current_epoch != epoch:
            raise _VectorMapBuildCancelled()

    def _failed_or_cancelled(
        self,
        key: tuple[str, str, str],
        epoch: int,
        code: str,
        message: str,
    ) -> dict[str, object]:
        payload = {"status": "failed", "error": {"code": code, "message": message}}
        try:
            with self._control_lock(key):
                if self._read_control_epoch_unlocked(key) != epoch:
                    return self._cancelled(key, epoch)
                payload = self._write_failure_unlocked(key, epoch, code, message)
                with self._lock:
                    self._states[key] = payload
                return payload
        except _VectorMapValidationError as exc:
            return self._failed(key, exc.code, exc.message)

    def _matching_control_keys(
        self,
        dataset_id: str,
        published_version: str | None,
        embedding_model: str | None,
    ) -> list[tuple[str, str, str]]:
        if not self._control_keys_dir.is_dir():
            return []

        keys: list[tuple[str, str, str]] = []
        for path in self._control_keys_dir.glob("*.json"):
            try:
                payload = self._read_json(path)
            except Exception:
                continue
            control_dataset_id = payload.get("dataset_id")
            control_published_version = payload.get("published_version")
            control_embedding_model = payload.get("embedding_model")
            if not (
                isinstance(control_dataset_id, str)
                and isinstance(control_published_version, str)
                and isinstance(control_embedding_model, str)
            ):
                continue
            if (
                control_dataset_id == dataset_id
                and (published_version is None or control_published_version == published_version)
                and (embedding_model is None or control_embedding_model == embedding_model)
            ):
                keys.append((control_dataset_id, control_published_version, control_embedding_model))
        return keys

    def _matching_state_keys(
        self,
        dataset_id: str,
        published_version: str | None,
        embedding_model: str | None,
    ) -> list[tuple[str, str, str]]:
        return [
            key
            for key in list(self._states)
            if key[0] == dataset_id
            and (published_version is None or key[1] == published_version)
            and (embedding_model is None or key[2] == embedding_model)
        ]

    def _drop_reducer_bundles_for_key_unlocked(self, key: tuple[str, str, str]) -> None:
        for identity in [identity for identity in self._reducer_bundles if identity[:3] == key]:
            self._reducer_bundles.pop(identity, None)

    def _drop_reducer_bundles_unlocked(
        self,
        dataset_id: str,
        published_version: str | None,
        embedding_model: str | None,
    ) -> None:
        for identity in [
            identity
            for identity in self._reducer_bundles
            if identity[0] == dataset_id
            and (published_version is None or identity[1] == published_version)
            and (embedding_model is None or identity[2] == embedding_model)
        ]:
            self._reducer_bundles.pop(identity, None)

    def _matching_cache_dirs(
        self,
        dataset_id: str,
        published_version: str | None,
        embedding_model: str | None,
    ) -> list[Path]:
        dataset_dir = self.root / self._safe_component(dataset_id)
        if published_version is not None and embedding_model is not None:
            path = self._cache_dir(dataset_id, published_version, embedding_model)
            return [path] if path.is_dir() else []
        if published_version is not None:
            version_dir = dataset_dir / self._safe_component(published_version)
            if not version_dir.is_dir():
                return []
            return [path for path in version_dir.iterdir() if path.is_dir()]
        if not dataset_dir.is_dir():
            return []
        model_dir_name = self._model_component(embedding_model) if embedding_model is not None else None
        return [
            path
            for version_dir in dataset_dir.iterdir()
            if version_dir.is_dir()
            for path in version_dir.iterdir()
            if path.is_dir() and (model_dir_name is None or path.name == model_dir_name)
        ]

    def _cleanup_empty_parents(self, dataset_id: str, published_version: str | None) -> None:
        dataset_dir = self.root / self._safe_component(dataset_id)
        if published_version is not None:
            version_dir = dataset_dir / self._safe_component(published_version)
            self._rmdir_if_empty(version_dir)
        for version_dir in list(dataset_dir.iterdir()) if dataset_dir.is_dir() else []:
            self._rmdir_if_empty(version_dir)
        self._rmdir_if_empty(dataset_dir)

    @staticmethod
    def _rmdir_if_empty(path: Path) -> None:
        try:
            path.rmdir()
        except OSError:
            pass

    @staticmethod
    def _safe_component(value: str) -> str:
        text = str(value)
        candidate = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._ ")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
        prefix_budget = _SAFE_COMPONENT_MAX_LENGTH - _DIGEST_LENGTH - 1
        prefix = (candidate or "value")[:prefix_budget].strip("._- ") or "value"
        stem = prefix.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED_STEMS:
            prefix = f"value-{prefix}"[:prefix_budget].strip("._- ") or "value"
        return f"{prefix}-{digest}"

    def _model_component(self, embedding_model: str) -> str:
        return self._safe_component(embedding_model)

    @staticmethod
    def _content_digest(sources: list[VectorSource]) -> str:
        hasher = hashlib.sha256()
        encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(",", ":"))

        def update_text(text: str) -> None:
            hasher.update(text.encode("utf-8"))

        def update_json(value: object) -> None:
            for chunk in encoder.iterencode(value):
                update_text(chunk)

        update_text("[")
        for source_index, source in enumerate(sources):
            if source_index:
                update_text(",")
            update_text('{"child_id":')
            update_json(source.child_id)
            update_text(',"content_hash":')
            update_json(source.content_hash)
            update_text(',"dataset_id":')
            update_json(source.dataset_id)
            update_text(',"document_id":')
            update_json(source.document_id)
            update_text(',"document_name":')
            update_json(source.document_name)
            update_text(',"excerpt":')
            update_json(source.excerpt)
            update_text(',"page_end":')
            update_json(source.page_end)
            update_text(',"page_start":')
            update_json(source.page_start)
            update_text(',"product":')
            update_json(source.product)
            update_text(',"title":')
            update_json(source.title)
            update_text(',"vector_dimension":')
            update_json(len(source.embedding or []))
            update_text(',"embedding":[')
            for value_index, value in enumerate(source.embedding or []):
                if value_index:
                    update_text(",")
                update_json(float(value))
            update_text("]}")
        update_text("]")
        return hasher.hexdigest()

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _failed(self, key: tuple[str, str, str], code: str, message: str) -> dict[str, object]:
        payload = {"status": "failed", "error": {"code": code, "message": message}}
        with self._lock:
            self._states[key] = payload
        return payload

    def _cancelled(self, key: tuple[str, str, str], epoch: int) -> dict[str, object]:
        with self._lock:
            state = self._states.get(key)
            if state and state.get("status") == "building" and state.get("epoch") == epoch:
                self._states.pop(key, None)
        return {
            "status": "failed",
            "error": {
                "code": "build_cancelled",
                "message": "向量图构建已被失效操作取消。",
            },
        }

    @staticmethod
    def _jsonable(payload: dict[str, object]) -> dict[str, object]:
        return json.loads(json.dumps(payload, ensure_ascii=False))
