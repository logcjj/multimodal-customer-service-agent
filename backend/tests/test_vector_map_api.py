from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter
from collections.abc import Callable
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlmodel import select

from app.knowledge.models import ChildChunkRecord
from app.knowledge.vector_map import VectorSource
from app.main import create_app


Embedding = list[float] | None


class IdentityReducer:
    def fit_transform(self, matrix):
        return [[float(row[0]), float(row[1])] for row in matrix]

    def transform(self, matrix):
        return [[float(matrix[0][0]), float(matrix[0][1])]]


def _client(tmp_path) -> TestClient:
    client = TestClient(create_app(data_dir=tmp_path))
    service = client.app.state.knowledge_service
    if hasattr(service, "embedding_model_provider"):
        service.embedding_model_provider = lambda: "test-embedding-v1"
    return client


def _new_dataset(client: TestClient, label: str) -> tuple[str, str]:
    service = client.app.state.knowledge_service
    repository = service.repository
    suffix = uuid4().hex[:8]
    dataset = repository.create_dataset(f"知识库 {label}", parser_profile="manual")
    file = repository.create_file(
        original_name=f"{label}-manual.md",
        content_hash=f"hash-{label}-{suffix}",
        mime_type="text/markdown",
        size_bytes=128,
        storage_path=f"objects/{label}-{suffix}.md",
    )
    document = repository.link_file(dataset.id, file.id, "manual")
    return dataset.id, document.id


def _add_document(client: TestClient, dataset_id: str, label: str) -> str:
    repository = client.app.state.knowledge_service.repository
    suffix = uuid4().hex[:8]
    file = repository.create_file(
        original_name=f"{label}-manual.md",
        content_hash=f"hash-{label}-{suffix}",
        mime_type="text/markdown",
        size_bytes=128,
        storage_path=f"objects/{label}-{suffix}.md",
    )
    return repository.link_file(dataset_id, file.id, "manual").id


def _write_version(
    client: TestClient,
    *,
    dataset_id: str,
    document_id: str,
    version: str,
    label: str,
    embeddings: list[Embedding],
    missing_embedding_ids: set[str] | None = None,
    disabled_ids: set[str] | None = None,
) -> None:
    service = client.app.state.knowledge_service
    children = []
    for index, embedding in enumerate(embeddings, start=1):
        local_id = f"{label}-c{index}"
        text = (
            f"{label} 第 {index} 个 Child Chunk，来自真实 repository 记录。"
            + "这段文本用于验证 excerpt 截断。" * 20
        )
        child = {
            "local_id": local_id,
            "parent_local_id": "p1",
            "title": f"{label} 标题 {index}",
            "text": text,
            "page_start": index,
            "page_end": index,
            "token_count": len(text),
            "product": f"product-{label}",
        }
        if embedding is not None:
            child["embedding"] = embedding
        if local_id not in (missing_embedding_ids or set()):
            children.append(child)
        else:
            children.append({key: value for key, value in child.items() if key != "embedding"})

    service.repository.replace_chunks(
        document_id=document_id,
        dataset_id=dataset_id,
        index_version=version,
        parents=[
            {
                "local_id": "p1",
                "title": f"{label} 父块",
                "text": f"{label} 父块正文",
                "page_start": 1,
                "page_end": max(1, len(children)),
                "token_count": 12,
            }
        ],
        children=children,
    )
    for local_id in disabled_ids or set():
        _set_child_enabled(client, dataset_id, version, local_id, enabled=False)


def _set_child_enabled(
    client: TestClient,
    dataset_id: str,
    version: str,
    local_id: str,
    *,
    enabled: bool,
) -> None:
    database = client.app.state.database
    with database.session() as session:
        record = session.exec(
            select(ChildChunkRecord).where(
                ChildChunkRecord.dataset_id == dataset_id,
                ChildChunkRecord.index_version == version,
                ChildChunkRecord.local_id == local_id,
            )
        ).one()
        record.enabled = enabled
        session.add(record)
        session.commit()


def _publish(client: TestClient, dataset_id: str, version: str) -> None:
    response = client.post(f"/api/datasets/{dataset_id}/publish", json={"index_version": version})
    assert response.status_code == 200


def _published_dataset(
    client: TestClient,
    label: str,
    embeddings: list[Embedding],
    *,
    version: str = "v1",
    missing_embedding_ids: set[str] | None = None,
    disabled_ids: set[str] | None = None,
) -> tuple[str, str]:
    dataset_id, document_id = _new_dataset(client, label)
    _write_version(
        client,
        dataset_id=dataset_id,
        document_id=document_id,
        version=version,
        label=label,
        embeddings=embeddings,
        missing_embedding_ids=missing_embedding_ids,
        disabled_ids=disabled_ids,
    )
    _publish(client, dataset_id, version)
    return dataset_id, document_id


def _wait_for_status(
    client: TestClient,
    dataset_id: str,
    predicate: Callable[[dict[str, object]], bool],
    *,
    timeout: float = 5.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last: dict[str, object] | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/datasets/{dataset_id}/vector-map")
        assert response.status_code == 200
        last = response.json()
        if predicate(last):
            return last
        time.sleep(0.05)
    raise AssertionError(f"vector map did not reach expected status; last={last}")


def test_vector_map_contains_only_requested_dataset_and_real_repository_metadata(tmp_path) -> None:
    client = _client(tmp_path)
    dataset_a, _ = _published_dataset(
        client,
        "A",
        [[1.0, 0.0], [0.0, 1.0], None, [0.5, 0.5]],
        missing_embedding_ids={"A-c3"},
        disabled_ids={"A-c4"},
    )
    dataset_b, _ = _published_dataset(client, "B", [[9.0, 9.0], [8.0, 8.0]])

    rebuild = client.post(f"/api/datasets/{dataset_a}/vector-map/rebuild")
    assert rebuild.status_code == 202

    body = _wait_for_status(client, dataset_a, lambda payload: payload["status"] == "ready")

    assert body["meta"]["dataset_id"] == dataset_a
    assert body["meta"]["published_version"] == "v1"
    assert body["meta"]["embedding_model"] == "test-embedding-v1"
    assert {point["dataset_id"] for point in body["points"]} == {dataset_a}
    assert dataset_b not in {point["dataset_id"] for point in body["points"]}
    assert [point["title"] for point in body["points"]] == ["A 标题 1", "A 标题 2"]
    assert all("embedding" not in point for point in body["points"])
    assert all(point["document_name"] == "A-manual.md" for point in body["points"])
    assert all(len(point["excerpt"]) <= 180 for point in body["points"])
    json.dumps(body, ensure_ascii=False)


def test_ready_vector_map_response_uses_gzip_for_large_point_payloads(tmp_path) -> None:
    client = _client(tmp_path)
    service = client.app.state.knowledge_service
    service.vector_maps.reducer_factory = IdentityReducer
    vectors = [[float(index + 1), float(index + 2)] for index in range(20)]
    dataset_id, _ = _published_dataset(client, "gzip-map", vectors)
    client.post(f"/api/datasets/{dataset_id}/vector-map/rebuild")
    _wait_for_status(client, dataset_id, lambda payload: payload["status"] == "ready")

    response = client.get(
        f"/api/datasets/{dataset_id}/vector-map",
        headers={"Accept-Encoding": "gzip"},
    )

    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"
    assert response.json()["meta"]["point_count"] == 20


def test_get_missing_map_starts_one_background_build_for_same_key(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path)
    dataset_id, _ = _published_dataset(client, "single-build", [[1.0, 0.0], [0.0, 1.0]])
    service = client.app.state.knowledge_service
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    if hasattr(service, "vector_maps"):
        original_build = service.vector_maps.build

        def blocking_build(dataset_id_arg, *args, **kwargs):
            calls.append(dataset_id_arg)
            started.set()
            assert release.wait(timeout=5)
            return original_build(dataset_id_arg, *args, **kwargs)

        monkeypatch.setattr(service.vector_maps, "build", blocking_build)

    first = client.get(f"/api/datasets/{dataset_id}/vector-map")
    try:
        assert first.status_code == 200
        assert first.json()["status"] == "building"
        assert started.wait(timeout=5)

        second = client.get(f"/api/datasets/{dataset_id}/vector-map")

        assert second.status_code == 200
        assert second.json()["status"] == "building"
        assert calls == [dataset_id]
    finally:
        release.set()

    ready = _wait_for_status(client, dataset_id, lambda payload: payload["status"] == "ready")
    assert ready["meta"]["dataset_id"] == dataset_id
    assert calls == [dataset_id]


def test_vector_map_source_metadata_lookups_are_cached_per_document_and_file(
    tmp_path,
    monkeypatch,
) -> None:
    client = _client(tmp_path)
    dataset_id, first_document_id = _new_dataset(client, "cache-doc-a")
    second_document_id = _add_document(client, dataset_id, "cache-doc-b")
    _write_version(
        client,
        dataset_id=dataset_id,
        document_id=first_document_id,
        version="v1",
        label="cache-a",
        embeddings=[[float(index), 0.0] for index in range(1, 9)],
    )
    _write_version(
        client,
        dataset_id=dataset_id,
        document_id=second_document_id,
        version="v1",
        label="cache-b",
        embeddings=[[0.0, float(index)] for index in range(1, 8)],
    )
    _publish(client, dataset_id, "v1")

    service = client.app.state.knowledge_service
    first_document = service.repository.get_document(first_document_id)
    second_document = service.repository.get_document(second_document_id)
    expected_file_ids = {first_document.file_id, second_document.file_id}
    document_calls: Counter[str] = Counter()
    file_calls: Counter[str] = Counter()
    build_finished = threading.Event()
    original_get_document = service.repository.get_document
    original_get_file = service.repository.get_file

    def counted_get_document(document_id: str):
        document_calls[document_id] += 1
        return original_get_document(document_id)

    def counted_get_file(file_id: str):
        file_calls[file_id] += 1
        return original_get_file(file_id)

    def instant_build(dataset_id_arg, published_version, embedding_model, sources):
        build_finished.set()
        return {
            "status": "ready",
            "meta": {
                "dataset_id": dataset_id_arg,
                "published_version": published_version,
                "embedding_model": embedding_model,
                "point_count": len(sources),
            },
            "points": [],
        }

    monkeypatch.setattr(service.repository, "get_document", counted_get_document)
    monkeypatch.setattr(service.repository, "get_file", counted_get_file)
    monkeypatch.setattr(service.vector_maps, "build", instant_build)

    response = client.post(f"/api/datasets/{dataset_id}/vector-map/rebuild")

    assert response.status_code == 202
    assert build_finished.wait(timeout=5)
    assert document_calls == Counter({first_document_id: 1, second_document_id: 1})
    assert file_calls == Counter({file_id: 1 for file_id in expected_file_ids})


def test_vector_map_sources_include_each_documents_own_published_version(tmp_path) -> None:
    client = _client(tmp_path)
    dataset_id, first_document_id = _new_dataset(client, "published-v1")
    _write_version(
        client,
        dataset_id=dataset_id,
        document_id=first_document_id,
        version="v1",
        label="published-v1",
        embeddings=[[1.0, 0.0], [0.5, 0.5]],
    )
    _publish(client, dataset_id, "v1")

    second_document_id = _add_document(client, dataset_id, "published-v2")
    _write_version(
        client,
        dataset_id=dataset_id,
        document_id=second_document_id,
        version="v2",
        label="published-v2",
        embeddings=[[0.0, 1.0]],
    )
    _publish(client, dataset_id, "v2")

    service = client.app.state.knowledge_service
    sources = service._vector_map_sources(dataset_id, "v2")

    assert [item.title for item in sources] == [
        "published-v1 标题 1",
        "published-v1 标题 2",
        "published-v2 标题 1",
    ]
    assert {item.document_id for item in sources} == {first_document_id, second_document_id}


def test_vector_map_unknown_dataset_returns_404(tmp_path) -> None:
    client = _client(tmp_path)

    get_response = client.get("/api/datasets/not-found/vector-map")
    rebuild_response = client.post("/api/datasets/not-found/vector-map/rebuild")

    assert get_response.status_code == 404
    assert rebuild_response.status_code == 404


def test_vector_map_without_published_version_returns_clear_status(tmp_path) -> None:
    client = _client(tmp_path)
    dataset_id, _ = _new_dataset(client, "draft-only")

    response = client.get(f"/api/datasets/{dataset_id}/vector-map")

    assert response.status_code == 200
    assert response.json()["status"] == "no_published_version"

    rebuild = client.post(f"/api/datasets/{dataset_id}/vector-map/rebuild")

    assert rebuild.status_code == 409
    assert "尚未发布" in rebuild.json()["detail"]


def test_vector_map_without_valid_embeddings_returns_no_embeddings(tmp_path) -> None:
    client = _client(tmp_path)
    dataset_id, _ = _published_dataset(client, "no-embeddings", [None, None])

    response = client.get(f"/api/datasets/{dataset_id}/vector-map")

    assert response.status_code == 200
    assert response.json()["status"] == "no_embeddings"

    rebuild = client.post(f"/api/datasets/{dataset_id}/vector-map/rebuild")

    assert rebuild.status_code == 409
    assert "向量化" in rebuild.json()["detail"]


def test_vector_map_reports_failed_payload_for_inconsistent_dimensions(tmp_path) -> None:
    client = _client(tmp_path)
    dataset_id, _ = _published_dataset(client, "bad-dimensions", [[1.0, 0.0], [0.0, 1.0, 2.0]])

    rebuild = client.post(f"/api/datasets/{dataset_id}/vector-map/rebuild")
    assert rebuild.status_code == 202

    body = _wait_for_status(client, dataset_id, lambda payload: payload["status"] == "failed")

    assert body["error"]["code"] == "dimension_mismatch"
    json.dumps(body, ensure_ascii=False)


def test_manual_rebuild_failure_overrides_previous_ready_cache(tmp_path) -> None:
    client = _client(tmp_path)
    dataset_id, _ = _published_dataset(client, "ready-then-bad", [[1.0, 0.0], [0.0, 1.0]])

    client.post(f"/api/datasets/{dataset_id}/vector-map/rebuild")
    ready = _wait_for_status(client, dataset_id, lambda payload: payload["status"] == "ready")
    assert ready["meta"]["point_count"] == 2

    database = client.app.state.database
    with database.session() as session:
        records = session.exec(
            select(ChildChunkRecord).where(
                ChildChunkRecord.dataset_id == dataset_id,
                ChildChunkRecord.index_version == "v1",
            )
        ).all()
        records[1].embedding_json = json.dumps([0.0, 1.0, 2.0])
        session.add(records[1])
        session.commit()

    rebuild = client.post(f"/api/datasets/{dataset_id}/vector-map/rebuild")
    assert rebuild.status_code == 202
    failed = _wait_for_status(client, dataset_id, lambda payload: payload["status"] == "failed")
    follow_up = client.get(f"/api/datasets/{dataset_id}/vector-map")

    assert failed["error"]["code"] == "dimension_mismatch"
    assert follow_up.status_code == 200
    assert follow_up.json()["status"] == "failed"
    assert follow_up.json()["error"]["code"] == "dimension_mismatch"


def test_background_vector_map_exception_becomes_stable_failed_status(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path)
    dataset_id, _ = _published_dataset(client, "background-error", [[1.0, 0.0], [0.0, 1.0]])
    service = client.app.state.knowledge_service

    def broken_build(*args, **kwargs):
        raise RuntimeError("controlled projection crash")

    monkeypatch.setattr(service.vector_maps, "build", broken_build)

    rebuild = client.post(f"/api/datasets/{dataset_id}/vector-map/rebuild")
    assert rebuild.status_code == 202
    failed = _wait_for_status(client, dataset_id, lambda payload: payload["status"] == "failed")
    follow_up = client.get(f"/api/datasets/{dataset_id}/vector-map")

    assert failed["error"]["code"] == "projection_failed"
    assert "controlled projection crash" in failed["error"]["message"]
    assert follow_up.status_code == 200
    assert follow_up.json()["status"] == "failed"


def test_publish_new_version_marks_old_projection_stale_until_latest_build_is_ready(
    tmp_path,
    monkeypatch,
) -> None:
    client = _client(tmp_path)
    dataset_id, document_id = _published_dataset(client, "versioned", [[1.0, 0.0], [0.0, 1.0]], version="v1")

    client.post(f"/api/datasets/{dataset_id}/vector-map/rebuild")
    ready_v1 = _wait_for_status(client, dataset_id, lambda payload: payload["status"] == "ready")
    assert ready_v1["meta"]["published_version"] == "v1"
    assert [point["title"] for point in ready_v1["points"]] == ["versioned 标题 1", "versioned 标题 2"]

    service = client.app.state.knowledge_service
    v1_key_dir = service.vector_maps._cache_dir(dataset_id, "v1", "test-embedding-v1")
    v1_active_name = json.loads((v1_key_dir / "active.json").read_text(encoding="utf-8"))["generation"]
    v1_active_dir = v1_key_dir / v1_active_name
    started = threading.Event()
    release = threading.Event()
    if hasattr(service, "vector_maps"):
        original_build = service.vector_maps.build

        def blocking_v2_build(dataset_id_arg, published_version, *args, **kwargs):
            if dataset_id_arg == dataset_id and published_version == "v2":
                started.set()
                assert release.wait(timeout=5)
            return original_build(dataset_id_arg, published_version, *args, **kwargs)

        monkeypatch.setattr(service.vector_maps, "build", blocking_v2_build)

    _write_version(
        client,
        dataset_id=dataset_id,
        document_id=document_id,
        version="v2",
        label="versioned-new",
        embeddings=[[2.0, 0.0], [0.0, 2.0]],
    )
    _publish(client, dataset_id, "v2")

    stale = client.get(f"/api/datasets/{dataset_id}/vector-map")
    try:
        assert stale.status_code == 200
        assert stale.json()["status"] == "stale"
        assert stale.json()["meta"]["published_version"] == "v1"
        assert [point["title"] for point in stale.json()["points"]] == [
            "versioned 标题 1",
            "versioned 标题 2",
        ]
        assert v1_active_dir.is_dir()
        assert started.wait(timeout=5)
    finally:
        release.set()

    ready_v2 = _wait_for_status(client, dataset_id, lambda payload: payload["status"] == "ready")

    assert ready_v2["meta"]["published_version"] == "v2"
    assert [point["title"] for point in ready_v2["points"]] == [
        "versioned-new 标题 1",
        "versioned-new 标题 2",
    ]


def test_cancelled_old_version_build_does_not_block_current_version_build(
    tmp_path,
    monkeypatch,
) -> None:
    client = _client(tmp_path)
    dataset_id, document_id = _published_dataset(client, "long-old", [[1.0, 0.0], [0.0, 1.0]], version="v1")
    service = client.app.state.knowledge_service
    original_build = service.vector_maps.build
    v1_started = threading.Event()
    v1_release = threading.Event()
    v2_started = threading.Event()

    def controlled_build(dataset_id_arg, published_version, *args, **kwargs):
        if dataset_id_arg == dataset_id and published_version == "v1":
            v1_started.set()
            assert v1_release.wait(timeout=5)
        if dataset_id_arg == dataset_id and published_version == "v2":
            v2_started.set()
        return original_build(dataset_id_arg, published_version, *args, **kwargs)

    monkeypatch.setattr(service.vector_maps, "build", controlled_build)

    client.post(f"/api/datasets/{dataset_id}/vector-map/rebuild")
    try:
        assert v1_started.wait(timeout=5)
        _write_version(
            client,
            dataset_id=dataset_id,
            document_id=document_id,
            version="v2",
            label="long-new",
            embeddings=[[2.0, 0.0], [0.0, 2.0]],
        )
        _publish(client, dataset_id, "v2")

        stale = client.get(f"/api/datasets/{dataset_id}/vector-map")

        assert stale.status_code == 200
        assert stale.json()["status"] == "stale"
        assert v2_started.wait(timeout=1)
    finally:
        v1_release.set()

    ready = _wait_for_status(client, dataset_id, lambda payload: payload["status"] == "ready")
    assert ready["meta"]["published_version"] == "v2"


def test_publish_cancels_queued_vector_map_futures_for_dataset(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path)
    dataset_id, _ = _published_dataset(client, "queued-cancel", [[1.0, 0.0], [0.0, 1.0]], version="v1")
    service = client.app.state.knowledge_service
    sources = service._vector_map_sources(dataset_id, "v1")
    started_versions: list[str] = []
    started_lock = threading.Lock()
    release = threading.Event()

    def blocking_build(dataset_id_arg, published_version, *args, **kwargs):
        if dataset_id_arg == dataset_id:
            with started_lock:
                started_versions.append(published_version)
            assert release.wait(timeout=5)
        return {"status": "failed", "error": {"code": "released", "message": "released"}}

    monkeypatch.setattr(service.vector_maps, "build", blocking_build)

    queued_key = (dataset_id, "queued-old-3", "test-embedding-v1")
    for version in ("queued-old-1", "queued-old-2", "queued-old-3"):
        assert service._ensure_vector_map_build((dataset_id, version, "test-embedding-v1"), sources)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with started_lock:
            if len(started_versions) >= 2:
                break
        time.sleep(0.01)

    queued_future = service._vector_map_futures[queued_key]
    try:
        _publish(client, dataset_id, "v1")

        assert queued_future.cancelled()
    finally:
        release.set()


def test_republishing_same_version_invalidates_content_identity_and_rebuilds(tmp_path) -> None:
    client = _client(tmp_path)
    dataset_id, _ = _published_dataset(client, "identity", [[1.0, 0.0], [0.0, 1.0]], version="v1")

    client.post(f"/api/datasets/{dataset_id}/vector-map/rebuild")
    ready_before = _wait_for_status(client, dataset_id, lambda payload: payload["status"] == "ready")
    digest_before = ready_before["meta"]["content_digest"]
    assert [point["title"] for point in ready_before["points"]] == ["identity 标题 1", "identity 标题 2"]

    database = client.app.state.database
    with database.session() as session:
        records = session.exec(
            select(ChildChunkRecord).where(
                ChildChunkRecord.dataset_id == dataset_id,
                ChildChunkRecord.index_version == "v1",
            )
        ).all()
        for index, record in enumerate(records, start=1):
            text = f"identity-new 第 {index} 个 Child Chunk，内容已经重新发布。"
            record.title = f"identity-new 标题 {index}"
            record.text = text
            record.normalized_text = text.lower()
            record.content_hash = hashlib.sha256(text.encode()).hexdigest()
            record.embedding_json = json.dumps([float(index + 2), float(index + 4)])
            session.add(record)
        session.commit()

    _publish(client, dataset_id, "v1")
    ready_after = _wait_for_status(client, dataset_id, lambda payload: payload["status"] == "ready")

    assert ready_after["meta"]["published_version"] == "v1"
    assert ready_after["meta"]["content_digest"] != digest_before
    assert [point["title"] for point in ready_after["points"]] == [
        "identity-new 标题 1",
        "identity-new 标题 2",
    ]


def test_knowledge_service_shutdown_stops_workers_and_rejects_new_vector_map_work(tmp_path) -> None:
    client = _client(tmp_path)
    dataset_id, _ = _published_dataset(client, "shutdown", [[1.0, 0.0], [0.0, 1.0]], version="v1")
    service = client.app.state.knowledge_service
    workers = list(service._vector_map_executor._threads)

    assert workers
    assert all(worker.daemon for worker in workers)

    service.shutdown()
    service.shutdown()
    result = service.rebuild_vector_map(dataset_id)

    assert all(not worker.is_alive() for worker in workers)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "executor_shutdown"


def test_rebuild_materializes_vector_sources_only_in_background_worker(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path)
    dataset_id, _ = _published_dataset(
        client,
        "background-sources",
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        version="v1",
    )
    service = client.app.state.knowledge_service
    service.vector_maps.reducer_factory = IdentityReducer
    original_sources = service._vector_map_sources
    request_thread_ids: list[int] = []
    source_thread_ids: list[int] = []
    source_loading_started = threading.Event()
    release_source_loading = threading.Event()

    def has_sources(dataset_id_arg: str, published_version: str) -> bool:
        request_thread_ids.append(threading.get_ident())
        return True

    def delayed_sources(dataset_id_arg: str, published_version: str):
        source_thread_ids.append(threading.get_ident())
        source_loading_started.set()
        assert release_source_loading.wait(timeout=1.0)
        return original_sources(dataset_id_arg, published_version)

    monkeypatch.setattr(service.repository, "has_vector_map_sources", has_sources)
    monkeypatch.setattr(service, "_vector_map_sources", delayed_sources)

    started_at = time.monotonic()
    try:
        response = client.post(f"/api/datasets/{dataset_id}/vector-map/rebuild")
        elapsed = time.monotonic() - started_at

        assert response.status_code == 202
        assert response.json()["status"] == "building"
        assert elapsed < 0.5
        assert source_loading_started.wait(timeout=1.0)
        assert request_thread_ids
        assert source_thread_ids
        assert request_thread_ids[0] != source_thread_ids[0]
    finally:
        release_source_loading.set()
        service.shutdown()
