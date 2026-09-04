from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from app.knowledge.hybrid import IndexedChild, PublishedHybridRetriever
import pytest


def test_exact_error_code_survives_dense_miss_and_parent_is_aggregated() -> None:
    retriever = PublishedHybridRetriever(
        [
            IndexedChild(
                child_id="c-e03-1",
                parent_id="p-e03",
                dataset_id="kb-1",
                document_id="doc-1",
                document_version="v1",
                title="E03 排水故障",
                text="显示 E03 时，先断电，再检查排水管。",
                product="washing-machine",
                page_start=12,
                page_end=12,
            ),
            IndexedChild(
                child_id="c-e03-2",
                parent_id="p-e03",
                dataset_id="kb-1",
                document_id="doc-1",
                document_version="v1",
                title="E03 排水故障",
                text="清理排水过滤器，确认没有堵塞。",
                product="washing-machine",
                page_start=12,
                page_end=13,
            ),
            IndexedChild(
                child_id="c-band",
                parent_id="p-band",
                dataset_id="kb-1",
                document_id="doc-2",
                document_version="v1",
                title="安装表带",
                text="将表带插入接口。",
                product="fitness-tracker",
                page_start=3,
                page_end=3,
            ),
        ],
    )

    result = retriever.explain("洗衣机 E03 怎么处理", top_n=3)

    assert result.results[0].parent_id == "p-e03"
    assert result.results[0].scores.lexical > 0
    assert len(result.results[0].matched_children) == 2
    assert result.stages["lexical"]
    assert "rrf" in result.stages


def test_irrelevant_query_is_rejected_by_evidence_gate() -> None:
    retriever = PublishedHybridRetriever(
        [
            IndexedChild(
                child_id="c1",
                parent_id="p1",
                dataset_id="kb",
                document_id="doc",
                document_version="v1",
                title="空气净化器滤网",
                text="清洁预过滤网。",
                page_start=1,
                page_end=1,
            )
        ]
    )

    result = retriever.explain("量子计算机怎么组装", top_n=3)

    assert result.results == []
    assert result.rejected_reason == "没有达到阈值的证据"


def test_unknown_exact_error_code_cannot_be_satisfied_by_dense_similarity() -> None:
    retriever = PublishedHybridRetriever(
        [
            IndexedChild(
                child_id="coffee-c1",
                parent_id="coffee-p1",
                dataset_id="kb",
                document_id="coffee-doc",
                document_version="v1",
                title="咖啡机常见故障",
                text="检查水箱和电源连接。",
                page_start=1,
                page_end=1,
                embedding=[1.0, 0.0],
            )
        ],
        embed=lambda _: [[1.0, 0.0]],
        rerank=lambda query, texts: [0.99 for _ in texts],
    )

    result = retriever.explain("火星牌咖啡机 ZX999 怎么维修", top_n=3)

    assert result.results == []
    assert result.rejected_reason == "未找到包含指定型号或错误码的证据"


@pytest.mark.parametrize("query_code", ["ERR02", "ERR_02", "Err 02"])
def test_camera_error_code_separator_variants_retrieve_same_exact_parent(
    query_code: str,
) -> None:
    retriever = PublishedHybridRetriever(
        [
            IndexedChild(
                child_id="camera-err-02",
                parent_id="camera-errors",
                dataset_id="kb",
                document_id="camera-manual",
                document_version="v1",
                title="Camera Error Code Countermeasures",
                text="Err 02 means there is a problem with the CF card.",
                parent_text="Err 02 means there is a problem with the CF card.",
                product="Camera",
                page_start=236,
                page_end=236,
            )
        ]
    )

    result = retriever.explain(
        f"佳能相机屏幕显示 {query_code} 怎么处理",
        products=["Camera"],
        top_n=3,
    )

    assert [item.parent_id for item in result.results] == ["camera-errors"]
    assert result.rejected_reason is None


def test_camera_err_020_cannot_borrow_err_02_retrieval_evidence() -> None:
    retriever = PublishedHybridRetriever(
        [
            IndexedChild(
                child_id="camera-err-02",
                parent_id="camera-errors",
                dataset_id="kb",
                document_id="camera-manual",
                document_version="v1",
                title="Camera Error Code Countermeasures",
                text="Err 02 means there is a problem with the CF card.",
                parent_text="Err 02 means there is a problem with the CF card.",
                product="Camera",
                page_start=236,
                page_end=236,
                embedding=[1.0, 0.0],
            )
        ],
        embed=lambda _: [[1.0, 0.0]],
    )

    result = retriever.explain(
        "佳能相机屏幕显示 Err 020 怎么处理",
        products=["Camera"],
        top_n=3,
    )

    assert result.results == []
    assert result.rejected_reason == "未找到包含指定型号或错误码的证据"


def test_dynamic_document_filename_participates_in_lexical_retrieval() -> None:
    retriever = PublishedHybridRetriever(
        [
            IndexedChild(
                child_id="daily-c1",
                parent_id="daily-p1",
                dataset_id="kb",
                document_id="daily-doc",
                document_version="v1",
                file_id="daily-file",
                document_name="徐江涛-2026-07-23 工作日报.pdf",
                document_mime_type="application/pdf",
                title="当日事项汇总",
                text="汇总多个平台数据，并梳理自动化方案。",
                page_start=1,
                page_end=1,
            ),
            IndexedChild(
                child_id="manual-c1",
                parent_id="manual-p1",
                dataset_id="kb",
                document_id="manual-doc",
                document_version="v1",
                document_name="设备说明书.pdf",
                title="工作模式设置",
                text="设备支持定时工作模式。",
                page_start=23,
                page_end=23,
            ),
        ]
    )

    result = retriever.explain("徐江涛 7 月 23 日工作日报内容", top_n=2, use_rerank=False)

    assert result.results
    assert result.results[0].document_id == "daily-doc"


def test_last_explanation_is_isolated_between_concurrent_search_requests() -> None:
    retriever = PublishedHybridRetriever(
        [
            IndexedChild(
                child_id="c1",
                parent_id="p1",
                dataset_id="kb",
                document_id="doc",
                document_version="v1",
                title="设备排水故障",
                text="设备排水故障时检查排水管。",
                page_start=1,
                page_end=1,
            )
        ]
    )
    barrier = Barrier(2)

    def search(query: str) -> str | None:
        retriever.search(query)
        barrier.wait()
        explanation = retriever.last_explanation
        return explanation.query if explanation is not None else None

    queries = ("设备排水故障", "量子计算是什么")
    with ThreadPoolExecutor(max_workers=2) as executor:
        observed = list(executor.map(search, queries))

    assert observed == list(queries)
