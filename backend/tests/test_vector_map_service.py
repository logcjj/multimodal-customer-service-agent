from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from app.knowledge import vector_map as vector_map_module
from app.knowledge.vector_map import VectorMapService, VectorSource


class FakeReducer:
    def fit_transform(self, matrix):
        return np.asarray([[row[0], row[1]] for row in matrix], dtype=np.float32)

    def transform(self, matrix):
        return np.asarray([[matrix[0][0], matrix[0][1]]], dtype=np.float32)


class NoTransformReducer:
    def fit_transform(self, matrix):
        return np.asarray([[row[0], row[1]] for row in matrix], dtype=np.float32)


class FailingTransformReducer(FakeReducer):
    def transform(self, matrix):
        raise RuntimeError("controlled transform failure")


class NonFiniteProjectionReducer:
    def __init__(self, raw):
        self.raw = raw

    def fit_transform(self, matrix):
        return np.asarray(self.raw, dtype=np.float32)


WINDOWS_RESERVED_STEMS = {
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


def source(
    child_id: str,
    dataset_id: str,
    vector: list[float] | None,
    *,
    document_name: str | None = None,
) -> VectorSource:
    return VectorSource(
        child_id=child_id,
        dataset_id=dataset_id,
        document_id=f"doc-{dataset_id}",
        document_name=document_name or f"{dataset_id}.json",
        title=f"title-{child_id}",
        excerpt=f"text-{child_id}",
        page_start=1,
        page_end=1,
        product="demo",
        embedding=vector,
    )


def active_generation_dir(
    service: VectorMapService,
    dataset_id: str,
    published_version: str,
    embedding_model: str,
) -> Path:
    key_dir = service._cache_dir(dataset_id, published_version, embedding_model)
    active = json.loads((key_dir / "active.json").read_text(encoding="utf-8"))
    return key_dir / str(active["generation"])


def generation_dirs(
    service: VectorMapService,
    dataset_id: str,
    published_version: str,
    embedding_model: str,
) -> list[Path]:
    key_dir = service._cache_dir(dataset_id, published_version, embedding_model)
    return sorted(
        [path for path in key_dir.iterdir() if path.is_dir() and path.name.startswith("generation-")],
        key=lambda path: path.name,
    )


def control_lock_path(
    root: Path,
    dataset_id: str,
    published_version: str,
    embedding_model: str,
) -> Path:
    payload = {
        "dataset_id": dataset_id,
        "embedding_model": embedding_model,
        "published_version": published_version,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    key_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return root / "vector_maps" / ".control" / "locks" / f"{key_id}.lock"


def test_build_is_dataset_scoped_normalized_and_written_atomically(tmp_path: Path) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)

    payload = service.build(
        dataset_id="dataset-a",
        published_version="v1",
        embedding_model="text-embedding-v4",
        sources=[
            source("a", "dataset-a", [1.0, 4.0]),
            source("b", "dataset-a", [3.0, 8.0]),
            source("c", "dataset-a", [2.0, 6.0]),
        ],
    )

    assert payload["status"] == "ready"
    assert payload["meta"]["dataset_id"] == "dataset-a"
    assert payload["meta"]["published_version"] == "v1"
    assert payload["meta"]["embedding_model"] == "text-embedding-v4"
    assert payload["meta"]["bounds"] == {"x_min": 1.0, "x_max": 3.0, "y_min": 4.0, "y_max": 8.0}
    assert [point["child_id"] for point in payload["points"]] == ["a", "b", "c"]
    assert {point["dataset_id"] for point in payload["points"]} == {"dataset-a"}
    assert payload["points"][0]["x"] == 0.0
    assert payload["points"][1]["x"] == 1.0
    assert payload["points"][2]["x"] == 0.5

    key_dir = service._cache_dir("dataset-a", "v1", "text-embedding-v4")
    assert (key_dir / "active.json").is_file()
    cache_dir = active_generation_dir(service, "dataset-a", "v1", "text-embedding-v4")
    assert (cache_dir / "points.json.gz").is_file()
    assert (cache_dir / "reducer.joblib").is_file()
    json.dumps(payload, ensure_ascii=False)


def test_cache_identity_includes_embedding_model_and_reload_keeps_maps_separate(tmp_path: Path) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    service.build(
        "dataset-a",
        "v1",
        "embedding-small",
        [
            source("small-a", "dataset-a", [1.0, 1.0]),
            source("small-b", "dataset-a", [2.0, 2.0]),
            source("small-c", "dataset-a", [3.0, 3.0]),
        ],
    )
    service.build(
        "dataset-a",
        "v1",
        "embedding-large",
        [
            source("large-a", "dataset-a", [10.0, 10.0]),
            source("large-b", "dataset-a", [20.0, 20.0]),
            source("large-c", "dataset-a", [30.0, 30.0]),
        ],
    )
    service.build(
        "dataset-b",
        "v1",
        "embedding-small",
        [
            source("other-a", "dataset-b", [5.0, 5.0]),
            source("other-b", "dataset-b", [6.0, 6.0]),
            source("other-c", "dataset-b", [7.0, 7.0]),
        ],
    )

    reloaded = VectorMapService(tmp_path, reducer_factory=FakeReducer)

    small = reloaded.load("dataset-a", "v1", "embedding-small")
    large = reloaded.load("dataset-a", "v1", "embedding-large")
    other = reloaded.load("dataset-b", "v1", "embedding-small")

    assert [point["child_id"] for point in small["points"]] == ["small-a", "small-b", "small-c"]
    assert [point["child_id"] for point in large["points"]] == ["large-a", "large-b", "large-c"]
    assert {point["dataset_id"] for point in other["points"]} == {"dataset-b"}
    assert reloaded.status("dataset-a", "v1", "embedding-small")["status"] == "ready"

    small_key_dir = reloaded._cache_dir("dataset-a", "v1", "embedding-small")
    invalidated = reloaded.invalidate("dataset-a", "v1", "embedding-small")

    assert invalidated["status"] == "invalidated"
    assert invalidated["removed"] == 1
    assert not small_key_dir.exists()
    assert reloaded.status("dataset-a", "v1", "embedding-small")["status"] == "missing"
    assert reloaded.status("dataset-a", "v1", "embedding-large")["status"] == "ready"
    json.dumps(small, ensure_ascii=False)
    json.dumps(invalidated, ensure_ascii=False)


def test_cache_identity_includes_published_version_for_same_dataset_and_model(tmp_path: Path) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    service.build(
        "dataset-a",
        "v1",
        "embedding-shared",
        [
            source("v1-a", "dataset-a", [1.0, 1.0]),
            source("v1-b", "dataset-a", [2.0, 2.0]),
            source("v1-c", "dataset-a", [3.0, 3.0]),
        ],
    )
    service.build(
        "dataset-a",
        "v2",
        "embedding-shared",
        [
            source("v2-a", "dataset-a", [10.0, 10.0]),
            source("v2-b", "dataset-a", [20.0, 20.0]),
            source("v2-c", "dataset-a", [30.0, 30.0]),
        ],
    )

    reloaded = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    v1 = reloaded.load("dataset-a", "v1", "embedding-shared")
    v2 = reloaded.load("dataset-a", "v2", "embedding-shared")

    assert [point["child_id"] for point in v1["points"]] == ["v1-a", "v1-b", "v1-c"]
    assert [point["child_id"] for point in v2["points"]] == ["v2-a", "v2-b", "v2-c"]
    json.dumps(v1, ensure_ascii=False)
    json.dumps(v2, ensure_ascii=False)


def test_invalidate_model_only_removes_that_model_across_versions(tmp_path: Path) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    for version in ("v1", "v2"):
        service.build(
            "dataset-a",
            version,
            "embedding-small",
            [
                source(f"{version}-small-a", "dataset-a", [1.0, 1.0]),
                source(f"{version}-small-b", "dataset-a", [2.0, 2.0]),
                source(f"{version}-small-c", "dataset-a", [3.0, 3.0]),
            ],
        )
        service.build(
            "dataset-a",
            version,
            "embedding-large",
            [
                source(f"{version}-large-a", "dataset-a", [10.0, 10.0]),
                source(f"{version}-large-b", "dataset-a", [20.0, 20.0]),
                source(f"{version}-large-c", "dataset-a", [30.0, 30.0]),
            ],
        )

    invalidated = service.invalidate("dataset-a", embedding_model="embedding-small")

    assert invalidated == {"status": "invalidated", "removed": 2}
    assert service.status("dataset-a", "v1", "embedding-small")["status"] == "missing"
    assert service.status("dataset-a", "v2", "embedding-small")["status"] == "missing"
    assert service.status("dataset-a", "v1", "embedding-large")["status"] == "ready"
    assert service.status("dataset-a", "v2", "embedding-large")["status"] == "ready"


def test_query_transform_uses_saved_reducer_and_saved_bounds(tmp_path: Path) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0]),
            source("c", "dataset-a", [5.0, 6.0]),
        ],
    )

    reloaded = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    query = reloaded.transform_query("dataset-a", "v1", "text-embedding-v4", [3.0, 4.0])

    assert query == {"x": 0.5, "y": 0.5}
    json.dumps(query, ensure_ascii=False)


def test_query_transform_requires_complete_active_generation_file_set(tmp_path: Path) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0]),
            source("c", "dataset-a", [5.0, 6.0]),
        ],
    )
    active_dir = active_generation_dir(service, "dataset-a", "v1", "text-embedding-v4")
    (active_dir / "points.json.gz").unlink()

    query = service.transform_query("dataset-a", "v1", "text-embedding-v4", [3.0, 4.0])

    assert query["status"] == "failed"
    assert query["error"]["code"] == "cache_missing"


def test_build_rejects_empty_mixed_missing_and_inconsistent_vectors(tmp_path: Path) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)

    cases = [
        (
            "empty_sources",
            [],
        ),
        (
            "mixed_dataset",
            [source("a", "dataset-a", [1.0, 2.0]), source("b", "dataset-b", [3.0, 4.0])],
        ),
        (
            "missing_embedding",
            [source("a", "dataset-a", [1.0, 2.0]), source("b", "dataset-a", None)],
        ),
        (
            "dimension_mismatch",
            [source("a", "dataset-a", [1.0, 2.0]), source("b", "dataset-a", [3.0])],
        ),
    ]

    for expected_code, sources in cases:
        payload = service.build("dataset-a", "v1", "text-embedding-v4", sources)

        assert payload["status"] == "failed"
        assert payload["error"]["code"] == expected_code
        assert service.status("dataset-a", "v1", "text-embedding-v4")["status"] == "failed"
        json.dumps(payload, ensure_ascii=False)

        service.invalidate("dataset-a", "v1", "text-embedding-v4")


@pytest.mark.parametrize(
    ("vector", "expected_code"),
    [
        ([1.0, "not-a-number"], "invalid_embedding"),
        ([1.0, "1.0"], "invalid_embedding"),
        ([1.0, True], "invalid_embedding"),
        ([1.0, np.nan], "invalid_embedding"),
        ([1.0, np.inf], "invalid_embedding"),
        ([1.0, -np.inf], "invalid_embedding"),
    ],
)
def test_build_rejects_non_numeric_and_non_finite_vectors(
    tmp_path: Path, vector: list[object], expected_code: str
) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)

    payload = service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", vector),
            source("c", "dataset-a", [3.0, 4.0]),
        ],
    )

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == expected_code
    json.dumps(payload, ensure_ascii=False)


def test_one_and_two_point_maps_are_deterministic_and_query_is_unavailable(tmp_path: Path) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)

    single = service.build("dataset-a", "v1", "text-embedding-v4", [source("solo", "dataset-a", [1.0, 0.0])])
    assert single["status"] == "ready"
    assert single["points"] == [
        {
            "child_id": "solo",
            "dataset_id": "dataset-a",
            "document_id": "doc-dataset-a",
            "document_name": "dataset-a.json",
            "title": "title-solo",
            "excerpt": "text-solo",
            "page_start": 1,
            "page_end": 1,
            "product": "demo",
            "x": 0.5,
            "y": 0.5,
        }
    ]
    assert service.transform_query("dataset-a", "v1", "text-embedding-v4", [0.0, 1.0]) is None

    service.build(
        "dataset-a",
        "v2",
        "text-embedding-v4",
        [source("left", "dataset-a", [1.0, 0.0]), source("right", "dataset-a", [0.0, 1.0])],
    )

    assert service.transform_query("dataset-a", "v2", "text-embedding-v4", [0.0, 3.0]) is None


def test_three_point_query_does_not_fall_back_when_saved_transform_fails(tmp_path: Path) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FailingTransformReducer)
    service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0]),
            source("c", "dataset-a", [5.0, 6.0]),
        ],
    )

    query = service.transform_query("dataset-a", "v1", "text-embedding-v4", [3.0, 4.0])

    assert query["status"] == "failed"
    assert query["error"]["code"] == "query_transform_failed"


def test_three_point_query_is_unavailable_when_reducer_transform_is_unavailable(tmp_path: Path) -> None:
    service = VectorMapService(tmp_path, reducer_factory=NoTransformReducer)
    service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 0.0]),
            source("b", "dataset-a", [0.0, 1.0]),
            source("c", "dataset-a", [0.5, 0.5]),
        ],
    )

    assert service.transform_query("dataset-a", "v1", "text-embedding-v4", [0.0, 3.0]) is None


@pytest.mark.parametrize(
    ("query_vector", "expected_code"),
    [
        ([], "missing_embedding"),
        ([1.0], "dimension_mismatch"),
        ([1.0, "not-a-number"], "invalid_embedding"),
        ([1.0, "1.0"], "invalid_embedding"),
        ([1.0, True], "invalid_embedding"),
        ([1.0, np.nan], "invalid_embedding"),
        ([1.0, np.inf], "invalid_embedding"),
        ([1.0, -np.inf], "invalid_embedding"),
    ],
)
def test_query_rejects_empty_mismatched_non_numeric_and_non_finite_vectors(
    tmp_path: Path, query_vector: list[object], expected_code: str
) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0]),
            source("c", "dataset-a", [5.0, 6.0]),
        ],
    )

    query = service.transform_query("dataset-a", "v1", "text-embedding-v4", query_vector)

    assert query["status"] == "failed"
    assert query["error"]["code"] == expected_code
    json.dumps(query, ensure_ascii=False)


@pytest.mark.parametrize("missing_file", ["active.json", "manifest.json", "points.json.gz", "reducer.joblib"])
def test_load_and_status_require_complete_active_generation_file_set(
    tmp_path: Path, missing_file: str
) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0]),
            source("c", "dataset-a", [5.0, 6.0]),
        ],
    )
    cache_dir = active_generation_dir(service, "dataset-a", "v1", "text-embedding-v4")
    key_dir = service._cache_dir("dataset-a", "v1", "text-embedding-v4")
    target = key_dir / missing_file if missing_file == "active.json" else cache_dir / missing_file
    target.unlink()

    assert service.load("dataset-a", "v1", "text-embedding-v4")["status"] == "missing"
    assert service.status("dataset-a", "v1", "text-embedding-v4")["status"] == "missing"


def test_status_reads_manifest_without_decompressing_points(tmp_path: Path, monkeypatch) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0]),
            source("c", "dataset-a", [5.0, 6.0]),
        ],
    )

    def fail_if_points_are_opened(*args, **kwargs):
        raise AssertionError("status must not open points.json.gz")

    monkeypatch.setattr(vector_map_module.gzip, "open", fail_if_points_are_opened)

    status = service.status("dataset-a", "v1", "text-embedding-v4")

    assert status["status"] == "ready"
    assert status["meta"]["point_count"] == 3


def test_cross_instance_load_reports_building_while_active_pointer_switches(
    tmp_path: Path, monkeypatch
) -> None:
    writer = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    reader = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    writer.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("old-a", "dataset-a", [1.0, 1.0]),
            source("old-b", "dataset-a", [2.0, 2.0]),
            source("old-c", "dataset-a", [3.0, 3.0]),
        ],
    )
    real_replace = vector_map_module.os.replace
    pointer_switch_started = threading.Event()
    resume_publish = threading.Event()

    def controlled_replace(src, dst):
        if Path(dst).name == "active.json":
            pointer_switch_started.set()
            assert resume_publish.wait(timeout=5)
        real_replace(src, dst)

    monkeypatch.setattr(vector_map_module.os, "replace", controlled_replace)

    publish_error: list[BaseException] = []

    def publish_new_map() -> None:
        try:
            writer.build(
                "dataset-a",
                "v1",
                "text-embedding-v4",
                [
                    source("new-a", "dataset-a", [10.0, 10.0]),
                    source("new-b", "dataset-a", [20.0, 20.0]),
                    source("new-c", "dataset-a", [30.0, 30.0]),
                ],
            )
        except BaseException as exc:
            publish_error.append(exc)

    publish_thread = threading.Thread(target=publish_new_map)
    publish_thread.start()
    assert pointer_switch_started.wait(timeout=5)

    during_publish = reader.load("dataset-a", "v1", "text-embedding-v4")

    assert during_publish == {"status": "building"}
    resume_publish.set()
    publish_thread.join(timeout=5)

    assert not publish_thread.is_alive()
    assert not publish_error
    after_publish = reader.load("dataset-a", "v1", "text-embedding-v4")
    assert after_publish["status"] == "ready"
    assert [point["child_id"] for point in after_publish["points"]] == ["new-a", "new-b", "new-c"]
    key_dir = writer._cache_dir("dataset-a", "v1", "text-embedding-v4")
    generations = [path for path in key_dir.iterdir() if path.is_dir() and path.name.startswith("generation-")]
    assert len(generations) == 2


def test_cross_instance_concurrent_publish_cannot_delete_pending_generation_before_activation(
    tmp_path: Path, monkeypatch
) -> None:
    service_a = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    service_b = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    reader = VectorMapService(tmp_path, reducer_factory=FakeReducer)

    service_a.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("old-1-a", "dataset-a", [1.0, 1.0]),
            source("old-1-b", "dataset-a", [2.0, 2.0]),
            source("old-1-c", "dataset-a", [3.0, 3.0]),
        ],
    )
    service_a.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("old-2-a", "dataset-a", [4.0, 4.0]),
            source("old-2-b", "dataset-a", [5.0, 5.0]),
            source("old-2-c", "dataset-a", [6.0, 6.0]),
        ],
    )

    name_lock = threading.Lock()
    generation_names = iter(["generation-9999-pending", "generation-0003-publisher"])

    def controlled_generation_name():
        with name_lock:
            return next(generation_names)

    monkeypatch.setattr(VectorMapService, "_new_generation_name", staticmethod(controlled_generation_name))

    real_replace = vector_map_module.os.replace
    pending_generation_moved = threading.Event()
    release_pending_activation = threading.Event()

    def controlled_replace(src, dst):
        real_replace(src, dst)
        if Path(dst).name == "generation-9999-pending":
            pending_generation_moved.set()
            assert release_pending_activation.wait(timeout=5)

    monkeypatch.setattr(vector_map_module.os, "replace", controlled_replace)

    errors: list[BaseException] = []

    def build_pending() -> None:
        try:
            service_a.build(
                "dataset-a",
                "v1",
                "text-embedding-v4",
                [
                    source("pending-a", "dataset-a", [10.0, 10.0]),
                    source("pending-b", "dataset-a", [11.0, 11.0]),
                    source("pending-c", "dataset-a", [12.0, 12.0]),
                ],
            )
        except BaseException as exc:
            errors.append(exc)

    def build_publisher() -> None:
        try:
            service_b.build(
                "dataset-a",
                "v1",
                "text-embedding-v4",
                [
                    source("publisher-a", "dataset-a", [20.0, 20.0]),
                    source("publisher-b", "dataset-a", [21.0, 21.0]),
                    source("publisher-c", "dataset-a", [22.0, 22.0]),
                ],
            )
        except BaseException as exc:
            errors.append(exc)

    pending_thread = threading.Thread(target=build_pending)
    pending_thread.start()
    assert pending_generation_moved.wait(timeout=5)

    publisher_thread = threading.Thread(target=build_publisher)
    publisher_thread.start()
    time.sleep(0.05)
    release_pending_activation.set()

    pending_thread.join(timeout=5)
    publisher_thread.join(timeout=5)

    assert not pending_thread.is_alive()
    assert not publisher_thread.is_alive()
    assert not errors
    key_dir = service_a._cache_dir("dataset-a", "v1", "text-embedding-v4")
    active = json.loads((key_dir / "active.json").read_text(encoding="utf-8"))
    assert (key_dir / str(active["generation"])).is_dir()
    assert reader.status("dataset-a", "v1", "text-embedding-v4")["status"] == "ready"
    assert reader.load("dataset-a", "v1", "text-embedding-v4")["status"] == "ready"
    assert len(generation_dirs(service_a, "dataset-a", "v1", "text-embedding-v4")) <= 2


def test_repeated_rebuilds_keep_only_active_and_previous_generations(tmp_path: Path) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    active_names: list[str] = []

    for index in range(5):
        payload = service.build(
            "dataset-a",
            "v1",
            "text-embedding-v4",
            [
                source(f"{index}-a", "dataset-a", [1.0 + index, 1.0]),
                source(f"{index}-b", "dataset-a", [2.0 + index, 2.0]),
                source(f"{index}-c", "dataset-a", [3.0 + index, 3.0]),
            ],
        )

        active_dir = active_generation_dir(service, "dataset-a", "v1", "text-embedding-v4")
        generations = generation_dirs(service, "dataset-a", "v1", "text-embedding-v4")
        active_names.append(active_dir.name)

        assert payload["status"] == "ready"
        assert active_dir in generations
        assert len(generations) <= 2

    final_generations = generation_dirs(service, "dataset-a", "v1", "text-embedding-v4")
    final_names = {path.name for path in final_generations}

    assert final_names == set(active_names[-2:])
    assert active_names[0] not in final_names
    assert service.load("dataset-a", "v1", "text-embedding-v4")["points"][0]["child_id"] == "4-a"


def test_invalidate_cancels_in_flight_build_before_it_can_publish_ready(tmp_path: Path) -> None:
    build_started = threading.Event()
    resume_build = threading.Event()

    class BlockingReducer(FakeReducer):
        def fit_transform(self, matrix):
            build_started.set()
            assert resume_build.wait(timeout=5)
            return super().fit_transform(matrix)

    service = VectorMapService(tmp_path, reducer_factory=BlockingReducer)
    build_result: list[dict[str, object]] = []
    build_error: list[BaseException] = []

    def build_map() -> None:
        try:
            build_result.append(
                service.build(
                    "dataset-a",
                    "v1",
                    "text-embedding-v4",
                    [
                        source("a", "dataset-a", [1.0, 2.0]),
                        source("b", "dataset-a", [3.0, 4.0]),
                        source("c", "dataset-a", [5.0, 6.0]),
                    ],
                )
            )
        except BaseException as exc:
            build_error.append(exc)

    build_thread = threading.Thread(target=build_map)
    build_thread.start()
    assert build_started.wait(timeout=5)

    invalidated = service.invalidate("dataset-a", "v1", "text-embedding-v4")
    resume_build.set()
    build_thread.join(timeout=5)

    assert not build_thread.is_alive()
    assert not build_error
    assert invalidated == {"status": "invalidated", "removed": 0}
    assert build_result[0]["status"] == "failed"
    assert build_result[0]["error"]["code"] == "build_cancelled"
    assert service.status("dataset-a", "v1", "text-embedding-v4")["status"] == "missing"
    assert service.load("dataset-a", "v1", "text-embedding-v4")["status"] == "missing"


def test_invalidate_then_reducer_failure_does_not_record_stale_failed_state(tmp_path: Path) -> None:
    build_started = threading.Event()
    resume_build = threading.Event()

    class BlockingFailingReducer(FakeReducer):
        def fit_transform(self, matrix):
            build_started.set()
            assert resume_build.wait(timeout=5)
            raise RuntimeError("controlled reducer failure after invalidation")

    service = VectorMapService(tmp_path, reducer_factory=BlockingFailingReducer)
    build_result: list[dict[str, object]] = []
    build_error: list[BaseException] = []

    def build_map() -> None:
        try:
            build_result.append(
                service.build(
                    "dataset-a",
                    "v1",
                    "text-embedding-v4",
                    [
                        source("a", "dataset-a", [1.0, 2.0]),
                        source("b", "dataset-a", [3.0, 4.0]),
                        source("c", "dataset-a", [5.0, 6.0]),
                    ],
                )
            )
        except BaseException as exc:
            build_error.append(exc)

    build_thread = threading.Thread(target=build_map)
    build_thread.start()
    assert build_started.wait(timeout=5)

    invalidated = service.invalidate("dataset-a", "v1", "text-embedding-v4")
    resume_build.set()
    build_thread.join(timeout=5)

    assert not build_thread.is_alive()
    assert not build_error
    assert invalidated == {"status": "invalidated", "removed": 0}
    assert build_result[0]["status"] == "failed"
    assert build_result[0]["error"]["code"] == "build_cancelled"
    assert service.status("dataset-a", "v1", "text-embedding-v4")["status"] == "missing"
    assert service.load("dataset-a", "v1", "text-embedding-v4")["status"] == "missing"


def test_cross_instance_invalidate_cancels_other_instance_in_flight_build(tmp_path: Path) -> None:
    build_started = threading.Event()
    resume_build = threading.Event()

    class BlockingReducer(FakeReducer):
        def fit_transform(self, matrix):
            build_started.set()
            assert resume_build.wait(timeout=5)
            return super().fit_transform(matrix)

    builder = VectorMapService(tmp_path, reducer_factory=BlockingReducer)
    invalidator = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    build_result: list[dict[str, object]] = []
    build_error: list[BaseException] = []

    def build_map() -> None:
        try:
            build_result.append(
                builder.build(
                    "dataset-a",
                    "v1",
                    "text-embedding-v4",
                    [
                        source("a", "dataset-a", [1.0, 2.0]),
                        source("b", "dataset-a", [3.0, 4.0]),
                        source("c", "dataset-a", [5.0, 6.0]),
                    ],
                )
            )
        except BaseException as exc:
            build_error.append(exc)

    build_thread = threading.Thread(target=build_map)
    build_thread.start()
    assert build_started.wait(timeout=5)

    invalidated = invalidator.invalidate("dataset-a", "v1", "text-embedding-v4")
    resume_build.set()
    build_thread.join(timeout=5)

    assert not build_thread.is_alive()
    assert not build_error
    assert invalidated == {"status": "invalidated", "removed": 0}
    assert build_result[0]["status"] == "failed"
    assert build_result[0]["error"]["code"] == "build_cancelled"
    assert builder.status("dataset-a", "v1", "text-embedding-v4")["status"] == "missing"
    assert invalidator.load("dataset-a", "v1", "text-embedding-v4")["status"] == "missing"


def test_status_reports_building_during_initial_build_before_cache_exists(tmp_path: Path) -> None:
    build_started = threading.Event()
    resume_build = threading.Event()

    class BlockingReducer(FakeReducer):
        def fit_transform(self, matrix):
            build_started.set()
            assert resume_build.wait(timeout=5)
            return super().fit_transform(matrix)

    service = VectorMapService(tmp_path, reducer_factory=BlockingReducer)
    build_error: list[BaseException] = []

    def build_map() -> None:
        try:
            service.build(
                "dataset-a",
                "v1",
                "text-embedding-v4",
                [
                    source("a", "dataset-a", [1.0, 2.0]),
                    source("b", "dataset-a", [3.0, 4.0]),
                    source("c", "dataset-a", [5.0, 6.0]),
                ],
            )
        except BaseException as exc:
            build_error.append(exc)

    build_thread = threading.Thread(target=build_map)
    build_thread.start()
    assert build_started.wait(timeout=5)

    assert service.status("dataset-a", "v1", "text-embedding-v4") == {"status": "building"}

    resume_build.set()
    build_thread.join(timeout=5)
    assert not build_thread.is_alive()
    assert not build_error


@pytest.mark.parametrize(
    "raw_projection",
    [
        [[0.0, 0.0], [1.0, np.nan], [2.0, 2.0]],
        [[0.0, 0.0], [1.0, np.inf], [2.0, 2.0]],
        [[0.0, 0.0], [1.0, -np.inf], [2.0, 2.0]],
    ],
)
def test_build_rejects_non_finite_reducer_projection(tmp_path: Path, raw_projection: list[list[float]]) -> None:
    service = VectorMapService(tmp_path, reducer_factory=lambda: NonFiniteProjectionReducer(raw_projection))

    payload = service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0]),
            source("c", "dataset-a", [5.0, 6.0]),
        ],
    )

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "projection_non_finite"


def test_transformable_reducer_bundle_omits_fallback_vectors_and_points(tmp_path: Path) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 0.0]),
            source("b", "dataset-a", [0.0, 1.0]),
            source("c", "dataset-a", [0.5, 0.5]),
        ],
    )

    _, bundle = service._load_reducer_bundle("dataset-a", "v1", "text-embedding-v4")

    assert bundle["reducer_available"] is True
    assert "vectors" not in bundle
    assert "points" not in bundle


@pytest.mark.parametrize(
    ("reducer_factory", "sources"),
    [
        (
            NoTransformReducer,
            [
                source("a", "dataset-a", [1.0, 0.0]),
                source("b", "dataset-a", [0.0, 1.0]),
                source("c", "dataset-a", [0.5, 0.5]),
            ],
        ),
        (
            FakeReducer,
            [
                source("a", "dataset-a", [1.0, 0.0]),
                source("b", "dataset-a", [0.0, 1.0]),
            ],
        ),
    ],
)
def test_fallback_reducer_bundle_keeps_vectors_and_points(tmp_path: Path, reducer_factory, sources) -> None:
    service = VectorMapService(tmp_path, reducer_factory=reducer_factory)
    service.build("dataset-a", "v1", "text-embedding-v4", sources)

    _, bundle = service._load_reducer_bundle("dataset-a", "v1", "text-embedding-v4")

    assert len(bundle["vectors"]) == len(sources)
    assert len(bundle["points"]) == len(sources)


def test_content_digest_streams_without_constructing_full_json(monkeypatch) -> None:
    def forbidden_dumps(*args, **kwargs):
        raise AssertionError("content digest must not build one full JSON document")

    monkeypatch.setattr(vector_map_module.json, "dumps", forbidden_dumps)

    digest = VectorMapService._content_digest(
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0]),
        ]
    )

    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


def test_vector_source_and_digest_cover_content_identity_and_display_metadata() -> None:
    assert "content_hash" in {field.name for field in fields(VectorSource)}

    original = source("a", "dataset-a", [1.0, 2.0])
    changed_title = replace(original, title="updated title")
    changed_excerpt = replace(original, excerpt="updated excerpt")
    changed_page = replace(original, page_end=2)
    changed_hash = replace(original, content_hash="updated-content-hash")

    original_digest = VectorMapService._content_digest([original])

    assert VectorMapService._content_digest([changed_title]) != original_digest
    assert VectorMapService._content_digest([changed_excerpt]) != original_digest
    assert VectorMapService._content_digest([changed_page]) != original_digest
    assert VectorMapService._content_digest([changed_hash]) != original_digest


def test_failed_rebuild_persists_across_instances_and_masks_previous_ready_cache(tmp_path: Path) -> None:
    writer = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    ready = writer.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0]),
            source("c", "dataset-a", [5.0, 6.0]),
        ],
    )
    assert ready["status"] == "ready"

    failed = writer.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0, 5.0]),
        ],
    )
    reloaded = VectorMapService(tmp_path, reducer_factory=FakeReducer)

    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "dimension_mismatch"
    assert reloaded.status("dataset-a", "v1", "text-embedding-v4") == failed
    assert reloaded.load("dataset-a", "v1", "text-embedding-v4") == failed


def test_build_lease_allows_only_one_cross_instance_projection(tmp_path: Path) -> None:
    build_started = threading.Event()
    resume_build = threading.Event()

    first = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    second = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    real_atomic_write = first._atomic_write

    def blocking_atomic_write(**kwargs):
        build_started.set()
        assert resume_build.wait(timeout=5)
        return real_atomic_write(**kwargs)

    first._atomic_write = blocking_atomic_write
    sources = [
        source("a", "dataset-a", [1.0, 2.0]),
        source("b", "dataset-a", [3.0, 4.0]),
        source("c", "dataset-a", [5.0, 6.0]),
    ]
    first_result: list[dict[str, object]] = []

    thread = threading.Thread(
        target=lambda: first_result.append(
            first.build("dataset-a", "v1", "text-embedding-v4", sources)
        )
    )
    thread.start()
    assert build_started.wait(timeout=5)

    competing = second.build("dataset-a", "v1", "text-embedding-v4", sources)

    assert competing == {"status": "building"}
    assert second.status("dataset-a", "v1", "text-embedding-v4") == {"status": "building"}

    resume_build.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert first_result[0]["status"] == "ready"
    assert second.status("dataset-a", "v1", "text-embedding-v4")["status"] == "ready"


def test_reducer_bundle_is_loaded_once_per_active_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    writer = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    writer.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0]),
            source("c", "dataset-a", [5.0, 6.0]),
        ],
    )
    reader = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    real_load = vector_map_module.joblib.load
    load_calls = 0

    def counted_load(path):
        nonlocal load_calls
        load_calls += 1
        return real_load(path)

    monkeypatch.setattr(vector_map_module.joblib, "load", counted_load)

    first_query = reader.transform_query("dataset-a", "v1", "text-embedding-v4", [2.0, 3.0])
    second_query = reader.transform_query("dataset-a", "v1", "text-embedding-v4", [4.0, 5.0])

    assert first_query == {"x": 0.25, "y": 0.25}
    assert second_query == {"x": 0.75, "y": 0.75}
    assert load_calls == 1


def test_dead_build_lease_is_recovered_before_projection(tmp_path: Path, monkeypatch) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    key = ("dataset-a", "v1", "text-embedding-v4")
    lease_path = service._build_lease_path(key)
    lease_path.write_text(
        json.dumps({"owner": "dead-owner", "pid": 987654321, "thread": 1}),
        encoding="utf-8",
    )
    monkeypatch.setattr(service, "_process_is_alive", lambda pid: False)

    payload = service.build(
        *key,
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0]),
            source("c", "dataset-a", [5.0, 6.0]),
        ],
    )

    assert payload["status"] == "ready"
    assert not lease_path.exists()


def test_safe_component_bounds_overlong_identifiers(tmp_path: Path) -> None:
    long_dataset_id = "dataset-" + "x" * 400
    long_version = "version-" + "y" * 400
    long_model = "model-" + "z" * 400

    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    payload = service.build(
        long_dataset_id,
        long_version,
        long_model,
        [
            source("a", long_dataset_id, [1.0, 2.0]),
            source("b", long_dataset_id, [3.0, 4.0]),
            source("c", long_dataset_id, [5.0, 6.0]),
        ],
    )

    key_dir = service._cache_dir(long_dataset_id, long_version, long_model)

    assert payload["status"] == "ready"
    assert service.load(long_dataset_id, long_version, long_model)["status"] == "ready"
    assert len(VectorMapService._safe_component(long_dataset_id)) <= 120
    assert len(VectorMapService._safe_component(long_version)) <= 120
    assert all(len(part) <= 140 for part in key_dir.relative_to(service.root).parts)


def test_safe_component_avoids_windows_reserved_names_and_case_insensitive_collisions() -> None:
    for value in ["CON", "nul", "PRN.txt", "aux.log", "COM1", "lpt9.data"]:
        component = VectorMapService._safe_component(value)
        stem = component.split(".", 1)[0].upper()

        assert stem not in WINDOWS_RESERVED_STEMS
        assert component.upper() != value.upper()
        assert len(component) <= 120

    data_component = VectorMapService._safe_component("Data")
    lower_component = VectorMapService._safe_component("data")

    assert data_component.lower() != lower_component.lower()


def test_control_lock_timeout_returns_stable_failed_payload(tmp_path: Path, monkeypatch) -> None:
    lock_path = control_lock_path(tmp_path, "dataset-a", "v1", "text-embedding-v4")
    lock_path.mkdir(parents=True)
    monkeypatch.setattr(vector_map_module, "_CONTROL_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(vector_map_module, "_CONTROL_LOCK_STALE_SECONDS", 3600.0)
    monkeypatch.setattr(vector_map_module, "_CONTROL_LOCK_POLL_SECONDS", 0.005)
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)

    payload = service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0]),
            source("c", "dataset-a", [5.0, 6.0]),
        ],
    )

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "lock_timeout"
    assert service.status("dataset-a", "v1", "text-embedding-v4")["status"] == "failed"


def test_stale_control_lock_is_recovered_before_build(tmp_path: Path, monkeypatch) -> None:
    lock_path = control_lock_path(tmp_path, "dataset-a", "v1", "text-embedding-v4")
    lock_path.mkdir(parents=True)
    owner_path = lock_path / "owner.json"
    owner_path.write_text("{}", encoding="utf-8")
    old_time = time.time() - 3600.0
    os.utime(owner_path, (old_time, old_time))
    os.utime(lock_path, (old_time, old_time))
    monkeypatch.setattr(vector_map_module, "_CONTROL_LOCK_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(vector_map_module, "_CONTROL_LOCK_STALE_SECONDS", 0.01)
    monkeypatch.setattr(vector_map_module, "_CONTROL_LOCK_POLL_SECONDS", 0.005)
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)

    payload = service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0]),
            source("c", "dataset-a", [5.0, 6.0]),
        ],
    )

    assert payload["status"] == "ready"
    assert not lock_path.exists()


def test_old_but_live_control_lock_is_never_stolen(tmp_path: Path, monkeypatch) -> None:
    lock_path = control_lock_path(tmp_path, "dataset-a", "v1", "text-embedding-v4")
    lock_path.mkdir(parents=True)
    owner_path = lock_path / "owner.json"
    owner_path.write_text(
        json.dumps({"owner": "live-owner", "pid": os.getpid()}),
        encoding="utf-8",
    )
    old_time = time.time() - 3600.0
    os.utime(owner_path, (old_time, old_time))
    os.utime(lock_path, (old_time, old_time))
    monkeypatch.setattr(vector_map_module, "_CONTROL_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(vector_map_module, "_CONTROL_LOCK_STALE_SECONDS", 0.01)
    monkeypatch.setattr(vector_map_module, "_CONTROL_LOCK_POLL_SECONDS", 0.005)
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)

    payload = service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("a", "dataset-a", [1.0, 2.0]),
            source("b", "dataset-a", [3.0, 4.0]),
            source("c", "dataset-a", [5.0, 6.0]),
        ],
    )

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "lock_timeout"
    assert lock_path.is_dir()
    assert json.loads(owner_path.read_text(encoding="utf-8"))["owner"] == "live-owner"


def test_invalidate_can_cancel_builds_without_deleting_last_complete_cache(tmp_path: Path) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("old-a", "dataset-a", [1.0, 2.0]),
            source("old-b", "dataset-a", [3.0, 4.0]),
            source("old-c", "dataset-a", [5.0, 6.0]),
        ],
    )
    key_dir = service._cache_dir("dataset-a", "v1", "text-embedding-v4")
    active_dir = active_generation_dir(service, "dataset-a", "v1", "text-embedding-v4")

    invalidated = service.invalidate(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        remove_cache=False,
    )

    assert invalidated == {"status": "invalidated", "removed": 0}
    assert key_dir.is_dir()
    assert active_dir.is_dir()
    assert service.status("dataset-a", "v1", "text-embedding-v4")["status"] == "missing"
    assert service.load("dataset-a", "v1", "text-embedding-v4")["status"] == "missing"


def test_invalidate_reports_cache_removal_failure_and_never_reexposes_old_epoch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service = VectorMapService(tmp_path, reducer_factory=FakeReducer)
    service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source("old-a", "dataset-a", [1.0, 2.0]),
            source("old-b", "dataset-a", [3.0, 4.0]),
            source("old-c", "dataset-a", [5.0, 6.0]),
        ],
    )
    key_dir = service._cache_dir("dataset-a", "v1", "text-embedding-v4")
    real_rmtree = vector_map_module.shutil.rmtree

    def blocked_rmtree(path, *args, **kwargs):
        if Path(path) == key_dir:
            return None
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(vector_map_module.shutil, "rmtree", blocked_rmtree)

    invalidated = service.invalidate("dataset-a", "v1", "text-embedding-v4")

    assert invalidated["status"] == "failed"
    assert invalidated["error"]["code"] == "cache_remove_failed"
    assert invalidated["removed"] == 0
    assert key_dir.is_dir()
    assert service.status("dataset-a", "v1", "text-embedding-v4")["status"] == "missing"
    assert service.load("dataset-a", "v1", "text-embedding-v4")["status"] == "missing"


@pytest.mark.filterwarnings("ignore:n_jobs value 1 overridden:UserWarning")
def test_default_reducer_fits_serializes_reloads_and_transforms_finite_2d_coordinates(tmp_path: Path) -> None:
    service = VectorMapService(tmp_path)
    matrix = np.asarray(
        [
            [1.0, 0.0, 0.2, 0.4],
            [0.9, 0.1, 0.3, 0.5],
            [0.0, 1.0, 0.4, 0.1],
            [0.1, 0.9, 0.5, 0.2],
            [0.4, 0.4, 1.0, 0.0],
            [0.5, 0.5, 0.9, 0.1],
            [0.2, 0.3, 0.1, 1.0],
            [0.3, 0.2, 0.2, 0.9],
        ],
        dtype=np.float32,
    )
    payload = service.build(
        "dataset-a",
        "v1",
        "text-embedding-v4",
        [
            source(f"child-{index}", "dataset-a", vector.astype(float).tolist())
            for index, vector in enumerate(matrix)
        ],
    )

    reloaded = VectorMapService(tmp_path)
    loaded = reloaded.load("dataset-a", "v1", "text-embedding-v4")
    query = reloaded.transform_query("dataset-a", "v1", "text-embedding-v4", matrix[0].astype(float).tolist())
    active_dir = active_generation_dir(reloaded, "dataset-a", "v1", "text-embedding-v4")

    assert payload["status"] == "ready"
    assert loaded["status"] == "ready"
    assert loaded["meta"]["point_count"] == len(matrix)
    assert (active_dir / "manifest.json").is_file()
    assert (active_dir / "points.json.gz").is_file()
    assert (active_dir / "reducer.joblib").is_file()
    assert all(np.isfinite([point["x"], point["y"]]).all() for point in loaded["points"])
    assert query is not None
    assert np.isfinite([query["x"], query["y"]]).all()
