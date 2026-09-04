from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

from app.agents.knowledge import KnowledgeAgent
from app.contracts.models import AgentRequest, Evidence, ModelConfigurationCreate, ModelKind
from app.knowledge.retrieval import HybridRetriever, KnowledgeDocument
from app.models.llm_gateway import LLMGateway, LLMOutput
from app.models.service import ModelService
from app.storage.database import Database


def configured_service(tmp_path) -> ModelService:
    service = ModelService(Database(tmp_path))
    created = service.create_model(
        ModelConfigurationCreate(
            provider="deepseek",
            name="deepseek-chat",
            kind=ModelKind.LLM,
            base_url="https://llm.example/v1",
            api_key="sk-test-secret",
            capabilities=["llm"],
        )
    )
    service.set_default(created.id)
    return service


def test_gateway_calls_configured_default_llm_and_returns_model_metadata(tmp_path) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "基于证据生成的回答"}}]},
        )

    gateway = LLMGateway(
        configured_service(tmp_path),
        transport=httpx.MockTransport(handler),
    )

    output = gateway.generate(
        kind=ModelKind.LLM,
        system_prompt="只根据证据回答",
        user_prompt="问题和证据",
    )

    assert output is not None
    assert output.text == "基于证据生成的回答"
    assert output.model == "deepseek-chat"
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer sk-test-secret"
    assert captured["body"]["messages"][0]["content"] == "只根据证据回答"


def test_gateway_is_explicitly_unavailable_without_a_default_model(tmp_path) -> None:
    gateway = LLMGateway(ModelService(Database(tmp_path)))

    assert gateway.available(ModelKind.LLM) is False
    assert gateway.generate(kind=ModelKind.LLM, system_prompt="system", user_prompt="user") is None


def test_gateway_retries_transient_rate_limit_before_degrading(tmp_path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                json={"error": {"code": "rate_limit", "message": "busy"}},
                request=request,
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "重试后成功"}}]},
            request=request,
        )

    gateway = LLMGateway(
        configured_service(tmp_path),
        transport=httpx.MockTransport(handler),
    )

    output = gateway.generate(
        kind=ModelKind.LLM,
        system_prompt="system",
        user_prompt="user",
    )

    assert attempts == 2
    assert output is not None
    assert output.text == "重试后成功"
    assert gateway.last_failure is None


def test_knowledge_agent_uses_llm_but_keeps_claims_bound_to_retrieved_evidence() -> None:
    class FakeGateway:
        def available(self, kind: ModelKind) -> bool:
            return True

        def generate(self, **kwargs) -> LLMOutput:
            return LLMOutput(
                text="E03 表示排水异常。请断电后检查排水管，并清理排水过滤器。",
                provider="fake",
                model="fake-llm",
                latency_ms=320,
            )

    retriever = HybridRetriever(
        [
            KnowledgeDocument(
                child_id="washer-1",
                parent_id="washer-e03",
                title="E03 排水故障",
                text="洗衣机显示 E03 时，先关闭电源，检查排水管，并清理排水过滤器。",
                product="washing-machine",
            )
        ]
    )
    agent = KnowledgeAgent(retriever, llm_gateway=FakeGateway())

    result = agent.run(AgentRequest(question="洗衣机出现 E03 怎么处理？"))

    assert result.answer_fragment.startswith("E03 表示排水异常")
    assert result.model_used == "fake-llm"
    assert result.llm_generated is True
    assert result.claims[0].evidence_ids == ["evidence:washer-e03"]


def test_knowledge_agent_skips_llm_rewrite_after_deterministic_query_hit() -> None:
    class RecordingRetriever:
        def __init__(self) -> None:
            self.products = None
            self.top_k = None

        def search(self, query, products=None, top_k=5):
            self.products = products
            self.top_k = top_k
            return [
                Evidence(
                    evidence_id="manual:air-conditioner",
                    source_type="manual",
                    title="空调等离子净化运行",
                    text="按下等离子键后，遥控器显示屏会显示等离子标识。",
                    product="空调手册",
                    score=0.05,
                )
            ]

    class SequencedGateway:
        def __init__(self) -> None:
            self.calls = []

        def available(self, kind: ModelKind) -> bool:
            return kind == ModelKind.LLM

        def generate(self, **kwargs) -> LLMOutput:
            self.calls.append(kwargs)
            if "查询改写器" in kwargs["system_prompt"]:
                text = "中文：空调 遥控器 小松树 等离子净化图标 | English: air conditioner plasma purification icon"
            else:
                text = "该标识与空调的等离子净化功能有关。"
            return LLMOutput(
                text=text,
                provider="fake",
                model="fake-llm",
                latency_ms=10,
            )

    retriever = RecordingRetriever()
    gateway = SequencedGateway()
    result = KnowledgeAgent(retriever, llm_gateway=gateway).run(
        AgentRequest(
            question=(
                "空调遥控器显示小松树是什么意思？\n"
                "图片可见信息：空调遥控器，背景中有一台白色空气净化器"
            )
        )
    )

    assert retriever.products == ["空调手册"]
    assert retriever.top_k == 8
    assert len(gateway.calls) == 1
    assert gateway.calls[-1]["temperature"] == 0
    assert "图片可见信息" in gateway.calls[-1]["system_prompt"]
    assert "疑似" in gateway.calls[-1]["system_prompt"]
    assert result.query_rewrite_model == "rule-expander-v1"


def test_knowledge_agent_uses_llm_rewrite_only_after_empty_first_search() -> None:
    class RetryRetriever:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query, products=None, top_k=5):
            self.queries.append(query)
            if len(self.queries) == 1:
                return []
            return [
                Evidence(
                    evidence_id="manual:washer-e03",
                    source_type="manual",
                    title="洗衣机 E03 排水故障",
                    text="洗衣机出现 E03 时，先关闭电源并检查排水管。",
                    score=0.05,
                )
            ]

    class SequencedGateway:
        def __init__(self) -> None:
            self.calls = []

        def available(self, kind: ModelKind) -> bool:
            return kind == ModelKind.LLM

        def generate(self, **kwargs) -> LLMOutput:
            self.calls.append(kwargs)
            text = (
                "中文：洗衣机 E03 排水故障 | English: washing machine E03 drain"
                if "查询改写器" in kwargs["system_prompt"]
                else "洗衣机出现 E03 时，请先断电并检查排水管。"
            )
            return LLMOutput(
                text=text,
                provider="fake",
                model="fake-llm",
                latency_ms=10,
            )

    retriever = RetryRetriever()
    gateway = SequencedGateway()
    result = KnowledgeAgent(retriever, llm_gateway=gateway).run(
        AgentRequest(question="洗衣机 E03 怎么处理？")
    )

    assert len(retriever.queries) == 2
    assert len(gateway.calls) == 1
    assert retriever.queries[0] == retriever.queries[1]
    assert result.query_rewrite_model == "rule-expander-v1"


def test_knowledge_agent_uses_broader_evidence_window_for_image_questions() -> None:
    class RecordingRetriever:
        def __init__(self) -> None:
            self.top_k = None

        def search(self, query, products=None, top_k=5):
            self.top_k = top_k
            return [
                Evidence(
                    evidence_id="manual:image",
                    source_type="manual",
                    title="配件说明",
                    text="该配件用于承托物品。",
                    score=0.05,
                )
            ]

    retriever = RecordingRetriever()
    KnowledgeAgent(retriever).run(
        AgentRequest(
            question="图片中的配件是什么？",
            images=["data:image/png;base64,aW1hZ2U="],
        )
    )

    assert retriever.top_k == 8


def test_knowledge_agent_expands_parent_evidence_from_live_retriever() -> None:
    children = [
        SimpleNamespace(
            child_id=f"child-{index}",
            text=f"Related child {index}",
            product="drill",
            dataset_id="manuals",
            document_id="drill-manual",
            file_id="drill-file",
            document_name="drill-manual.pdf",
            document_mime_type="application/pdf",
            document_version="v1",
            page_start=index,
            page_end=index,
            asset_ids=[f"asset-{index}"],
        )
        for index in range(1, 5)
    ]
    retriever = SimpleNamespace(
        documents=[],
        last_text_retriever=SimpleNamespace(documents=children),
    )
    parent = Evidence(
        evidence_id="manual:drill",
        source_type="manual",
        title="Drill belt hook",
        text="Confirm that each accessory is secure before use.",
        child_ids=[child.child_id for child in children],
        score=0.08,
        evidence_confidence=0.8,
    )

    related = KnowledgeAgent(retriever)._supplement_related_evidence([parent])

    assert len(related) == 5
    assert [item.section_id for item in related[1:]] == [
        "child-1",
        "child-2",
        "child-3",
        "child-4",
    ]
    assert KnowledgeAgent._related_assets(related) == [
        "asset-1",
        "asset-2",
        "asset-3",
        "asset-4",
    ]
