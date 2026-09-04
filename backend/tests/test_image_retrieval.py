from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

from app.contracts.models import Evidence
from app.config.runtime import RuntimeSettings
from app.knowledge.image_retrieval import (
    IndexedImageChunk,
    PublishedImageRetriever,
    merge_image_evidence,
)
from app.knowledge.service import LiveKnowledgeRetriever
from app.knowledge.service import KnowledgeService
from app.storage.database import Database


def _image(
    image_id: str,
    text: str,
    embedding: list[float],
    *,
    product: str = "吹风机",
) -> IndexedImageChunk:
    return IndexedImageChunk(
        image_chunk_id=image_id,
        dataset_id="manuals",
        document_id="doc-1",
        document_version="idx-1",
        file_id="file-1",
        document_name="吹风机手册.pdf",
        document_mime_type="application/pdf",
        asset_id=f"asset-{image_id}",
        image_id=image_id,
        manual_name="吹风机手册",
        chapter_title="安全说明",
        page_number=8,
        caption=text,
        ocr_text="WARNING",
        retrieval_text=text,
        product=product,
        confidence=0.98,
        related_parent_ids=["parent-1"],
        related_child_ids=["child-1"],
        embedding=embedding,
    )


def test_image_retriever_runs_lexical_dense_rrf_and_rerank() -> None:
    retriever = PublishedImageRetriever(
        [
            _image("guard", "防护罩损坏或缺失时严禁使用设备", [1.0, 0.0]),
            _image("muffler", "消音器高温可能导致烫伤", [0.0, 1.0]),
        ],
        embed=lambda texts: [[1.0, 0.0]],
        rerank=lambda query, documents: [0.95 for _ in documents],
    )

    evidence, explanation = retriever.search("防护罩坏了还能使用吗？", top_k=2)

    assert evidence[0].image_chunk_ids == ["guard"]
    assert evidence[0].source_type == "image"
    assert evidence[0].asset_ids == ["asset-guard"]
    assert explanation.mode == "image-hybrid-rerank"
    assert explanation.stages["image_lexical"]
    assert explanation.stages["image_dense"]
    assert explanation.stages["image_rrf"]
    assert explanation.stages["image_rerank"]


def test_shadow_mode_cannot_change_text_evidence() -> None:
    text = [
        Evidence(
            evidence_id="text-1",
            source_type="manual",
            title="文本证据",
            text="先断电。",
            child_ids=["child-1"],
            asset_ids=[],
        )
    ]
    image = [
        Evidence(
            evidence_id="image-1",
            source_type="image",
            title="图片证据",
            text="警告图片",
            image_chunk_ids=["image-1"],
            asset_ids=["asset-1"],
        )
    ]

    merged, shadow = merge_image_evidence(text, image, mode="shadow", query="看一下图片")

    assert merged == text
    assert shadow == image


def test_on_mode_only_adds_images_for_visual_intent() -> None:
    text = [
        Evidence(
            evidence_id="text-1",
            source_type="manual",
            title="文本证据",
            text="参数说明",
            child_ids=["child-1"],
            asset_ids=[],
        )
    ]
    image = [
        Evidence(
            evidence_id="image-1",
            source_type="image",
            title="铭牌图片",
            text="铭牌位置",
            image_chunk_ids=["image-1"],
            asset_ids=["asset-1"],
        )
    ]

    normal, _ = merge_image_evidence(text, image, mode="on", query="设备重量是多少")
    visual, _ = merge_image_evidence(text, image, mode="on", query="铭牌图片在哪里")

    assert normal == text
    assert visual == [*text, *image]


def test_live_retriever_shadow_keeps_text_result_and_records_image_hits() -> None:
    text = Evidence(
        evidence_id="text-1",
        source_type="manual",
        title="文本",
        text="文本答案",
        child_ids=["child-1"],
        asset_ids=[],
    )
    image = Evidence(
        evidence_id="image-1",
        source_type="image",
        title="图片",
        text="图片说明",
        image_chunk_ids=["image-1"],
        asset_ids=["asset-1"],
    )

    class TextRetriever:
        last_explanation = "text-explanation"

        def search(self, query, products=None, top_k=5):
            return [text]

    class ImageRetriever:
        def search(self, query, top_k=5):
            return [image], "image-explanation"

    class Service:
        settings = RuntimeSettings(
            offline_index_mode="on",
            image_chunk_retrieval="shadow",
            ocr_pipeline="shadow",
            caption_embedding="shadow",
            verified_streaming="on",
            session_memory="on",
            enhanced_verifier="on",
            session_ttl_seconds=3600,
            database_url=None,
        )

        def retriever(self):
            return TextRetriever()

        def image_retriever(self):
            return ImageRetriever()

    retriever = LiveKnowledgeRetriever(Service())

    result = retriever.search("图片在哪里", top_k=3)

    assert result == [text]
    assert retriever.last_explanation == "text-explanation"
    assert retriever.last_image_explanation == "image-explanation"
    assert retriever.last_shadow_image_evidence == [image]


def test_live_retriever_last_explanation_is_request_local_under_concurrency() -> None:
    class TextRetriever:
        last_explanation = None

        def search(self, query, products=None, top_k=5):
            self.last_explanation = SimpleNamespace(query=query)
            return []

    class Service:
        settings = RuntimeSettings(
            offline_index_mode="on",
            image_chunk_retrieval="off",
            ocr_pipeline="shadow",
            caption_embedding="shadow",
            verified_streaming="on",
            session_memory="off",
            enhanced_verifier="on",
            session_ttl_seconds=3600,
            database_url=None,
        )

        def retriever(self):
            return TextRetriever()

    retriever = LiveKnowledgeRetriever(Service())
    barrier = Barrier(2)

    def search(query: str) -> str | None:
        retriever.search(query)
        barrier.wait()
        explanation = retriever.last_explanation
        return explanation.query if explanation is not None else None

    queries = ("请求 A", "请求 B")
    with ThreadPoolExecutor(max_workers=2) as executor:
        observed = list(executor.map(search, queries))

    assert observed == list(queries)


def test_live_retriever_keeps_the_current_text_retriever() -> None:
    class TextRetriever:
        last_explanation = None

        def __init__(self) -> None:
            self.documents = [SimpleNamespace(child_id="request-local")]

        def search(self, query, products=None, top_k=5):
            self.last_explanation = SimpleNamespace(query=query)
            return []

    class Service:
        settings = RuntimeSettings(
            offline_index_mode="on",
            image_chunk_retrieval="off",
            ocr_pipeline="shadow",
            caption_embedding="shadow",
            verified_streaming="on",
            session_memory="off",
            enhanced_verifier="on",
            session_ttl_seconds=3600,
            database_url=None,
        )

        def retriever(self):
            return TextRetriever()

    retriever = LiveKnowledgeRetriever(Service())

    retriever.search("query")

    assert retriever.last_text_retriever is not None
    assert retriever.last_text_retriever.documents[0].child_id == "request-local"


def test_caption_embedding_off_does_not_call_query_embedding_provider(tmp_path) -> None:
    settings = RuntimeSettings(
        offline_index_mode="off",
        image_chunk_retrieval="on",
        ocr_pipeline="off",
        caption_embedding="off",
        verified_streaming="on",
        session_memory="on",
        enhanced_verifier="on",
        session_ttl_seconds=3600,
        database_url=None,
    )
    service = KnowledgeService(Database(tmp_path), settings=settings)
    service.repository.list_datasets = lambda: [
        SimpleNamespace(id="manuals", published_version="v1")
    ]
    service.repository.list_image_chunks = lambda published_only=True: [
        SimpleNamespace(dataset_id="manuals")
    ]
    service._indexed_images = lambda records: [
        _image("guard", "防护罩损坏或缺失时严禁使用设备", [1.0, 0.0])
    ]
    embedding_calls = 0

    def embed(texts):
        nonlocal embedding_calls
        embedding_calls += 1
        return [[1.0, 0.0]]

    service.embed_override = embed
    try:
        evidence, explanation = service.image_retriever(["manuals"]).search("防护罩", top_k=1)
    finally:
        service.shutdown()

    assert evidence
    assert explanation.mode == "image-lexical-only"
    assert embedding_calls == 0
