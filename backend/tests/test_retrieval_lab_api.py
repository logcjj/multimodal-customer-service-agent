from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.knowledge.hybrid import IndexedChild, PublishedHybridRetriever
from app.main import create_app


def _published_dataset(client: TestClient) -> str:
    dataset_id = client.post(
        "/api/datasets", json={"name": "洗衣机", "parser_profile": "manual"}
    ).json()["id"]
    file_id = client.post(
        "/api/files",
        files={"file": ("manual.md", b"# E03 drain error\nE03: power off and clean the drain filter.", "text/markdown")},
    ).json()["id"]
    document_id = client.post(
        f"/api/datasets/{dataset_id}/documents",
        json={"file_id": file_id, "parser_profile": "manual"},
    ).json()["id"]
    job = client.post(f"/api/documents/{document_id}/parse").json()
    client.post(f"/api/datasets/{dataset_id}/publish", json={"index_version": job["index_version"]})
    return dataset_id


def test_retrieval_lab_exposes_stages_scores_and_sources(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))
    dataset_id = _published_dataset(client)

    response = client.post(
        "/api/retrieval/test",
        json={"dataset_ids": [dataset_id], "query": "E03 drain filter", "top_n": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert {"lexical", "dense", "rrf", "rerank", "parent"} <= set(body["stages"])
    assert body["results"][0]["document_id"]
    assert body["results"][0]["page_start"] == 1
    assert body["results"][0]["scores"]["lexical"] > 0


def test_chunk_edit_creates_new_draft_version(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))
    dataset_id = _published_dataset(client)
    document_id = client.get(f"/api/datasets/{dataset_id}/documents").json()[0]["id"]
    child = client.get(f"/api/documents/{document_id}/chunks").json()["children"][0]

    response = client.patch(
        f"/api/chunks/{child['id']}",
        json={"text": "E03 时关闭电源，并清理排水过滤器。", "keywords": ["E03", "排水"]},
    )

    assert response.status_code == 200
    assert response.json()["id"] != child["id"]
    assert response.json()["edited"] is True


def test_retrieval_profile_can_be_saved_and_assigned_to_dataset(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))
    dataset_id = _published_dataset(client)

    created = client.post(
        "/api/retrieval/profiles",
        json={"name": "高精度", "rrf_k": 42, "final_top_n": 3, "min_score": 0.02},
    )
    assert created.status_code == 201
    profile = created.json()

    assigned = client.patch(
        f"/api/datasets/{dataset_id}",
        json={"retrieval_profile_id": profile["id"]},
    )

    assert assigned.status_code == 200
    assert assigned.json()["retrieval_profile_id"] == profile["id"]
    assert client.get("/api/retrieval/profiles").json()[0]["rrf_k"] == 42


def test_mcp_knowledge_search_executes_the_same_published_retriever(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))
    dataset_id = _published_dataset(client)

    response = client.post(
        "/api/mcp/tools/knowledge.search",
        json={"dataset_ids": [dataset_id], "query": "E03 drain filter", "top_n": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tool"] == "knowledge.search"
    assert body["is_error"] is False
    assert body["content"][0]["type"] == "text"
    assert body["structured_content"]["results"][0]["document_id"]


def test_supplied_query_vector_preserves_every_retrieval_stage_and_score() -> None:
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[0.9, 0.1]]

    documents = [
        IndexedChild(
            child_id="child-a",
            parent_id="parent-a",
            dataset_id="dataset-a",
            document_id="document-a",
            document_version="v1",
            title="E03 排水故障",
            text="E03 故障需要断电并清理排水过滤器。",
            page_start=1,
            page_end=1,
            embedding=[1.0, 0.0],
        ),
        IndexedChild(
            child_id="child-b",
            parent_id="parent-b",
            dataset_id="dataset-a",
            document_id="document-b",
            document_version="v1",
            title="进水过滤器",
            text="进水缓慢时清洁进水过滤器。",
            page_start=2,
            page_end=2,
            embedding=[0.0, 1.0],
        ),
    ]
    retriever = PublishedHybridRetriever(documents, embed=embed)

    queries = ["E03 排水过滤器", "进水过滤器如何清洁", "过滤器维护"]
    for query in queries:
        internal = retriever.explain(query, top_n=2, use_rerank=False)
        supplied = retriever.explain(
            query,
            top_n=2,
            use_rerank=False,
            query_vector=[0.9, 0.1],
        )

        assert asdict(internal) == asdict(supplied)
    assert calls == [[query] for query in queries]


def test_single_dataset_retrieval_reuses_one_embedding_for_dense_and_umap(tmp_path, monkeypatch) -> None:
    app = create_app(data_dir=tmp_path)
    service = app.state.knowledge_service
    embedding_calls: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        embedding_calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    service.embed_override = embed
    service.rerank_override = None
    service.embedding_configured_provider = lambda: True
    service.embedding_model_provider = lambda: "test-embedding-v1"
    client = TestClient(app)
    dataset_id = _published_dataset(client)
    embedding_calls.clear()

    monkeypatch.setattr(
        service.vector_maps,
        "status",
        lambda dataset_id_arg, published_version, embedding_model: {
            "status": "ready",
            "meta": {
                "dataset_id": dataset_id_arg,
                "published_version": published_version,
                "embedding_model": embedding_model,
                "content_digest": "projection-digest",
            },
        },
    )
    transformed_vectors: list[list[float]] = []

    def transform_query(dataset_id_arg, published_version, embedding_model, vector):
        transformed_vectors.append(list(vector))
        return {"x": 0.61, "y": 0.42}

    monkeypatch.setattr(service.vector_maps, "transform_query", transform_query)

    response = client.post(
        "/api/retrieval/test",
        json={
            "dataset_ids": [dataset_id],
            "query": "E03 drain filter",
            "top_n": 3,
            "use_rerank": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert embedding_calls == [["E03 drain filter"]]
    assert transformed_vectors == [[1.0, 0.0]]
    assert body["visualization"]["query"] == {"x": 0.61, "y": 0.42}
    assert body["visualization"]["dense_top10"][0]["child_id"] == body["stages"]["dense"][0]["id"]
    assert body["visualization"]["dense_top10"][0]["score"] == body["stages"]["dense"][0]["score"]
    assert body["visualization"]["projection_version"].startswith(
        f"{body['results'][0]['document_version']}:"
    )


def test_retrieval_reports_embedding_and_rerank_degradation_without_losing_rrf(
    tmp_path,
    monkeypatch,
) -> None:
    app = create_app(data_dir=tmp_path)
    service = app.state.knowledge_service
    service.embedding_configured_provider = lambda: True
    service.embedding_model_provider = lambda: "test-embedding-v1"
    service.embed_override = lambda texts: [[1.0, 0.0] for _ in texts]
    service.rerank_override = None
    client = TestClient(app)
    dataset_id = _published_dataset(client)

    service.embed_override = lambda texts: []
    service.rerank_override = lambda query, documents: []
    service._retriever_cache.clear()
    monkeypatch.setattr(
        service.vector_maps,
        "status",
        lambda dataset_id_arg, published_version, embedding_model: {
            "status": "ready",
            "meta": {
                "dataset_id": dataset_id_arg,
                "published_version": published_version,
                "embedding_model": embedding_model,
                "content_digest": "degraded-projection",
            },
        },
    )
    response = client.post(
        "/api/retrieval/test",
        json={
            "dataset_ids": [dataset_id],
            "query": "E03 drain filter",
            "top_n": 3,
            "use_rerank": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "lexical-only"
    assert body["stages"]["lexical"]
    assert body["stages"]["rrf"]
    assert body["results"]
    assert body["stages"]["dense"] == []
    assert body["stages"]["rerank"] == []
    assert body["warnings"] == [
        "查询向量生成失败，已降级为 BM25/RRF；红色查询点暂不可用。",
        "Rerank 调用失败，当前使用 RRF 排序。",
    ]
    assert body["visualization"]["query"] is None
    assert body["visualization"]["dense_top10"] == []
    assert body["visualization"]["rerank_top10"] == []
    assert body["visualization"]["rrf_top10"][0]["child_id"] == body["stages"]["rrf"][0]["id"]


def test_app_wires_embedding_model_name_and_shuts_down_vector_workers(tmp_path, monkeypatch) -> None:
    app = create_app(data_dir=tmp_path)
    service = app.state.knowledge_service
    shutdown_calls: list[bool] = []

    monkeypatch.setattr(
        app.state.model_service,
        "get_default_runtime",
        lambda kind: (SimpleNamespace(name="embedding-model-v1"), "secret"),
    )
    monkeypatch.setattr(service, "shutdown", lambda: shutdown_calls.append(True))

    assert service.embedding_model_provider() == "embedding-model-v1"
    with TestClient(app):
        pass
    assert shutdown_calls == [True]
