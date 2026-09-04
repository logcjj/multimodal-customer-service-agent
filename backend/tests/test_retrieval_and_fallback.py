from __future__ import annotations

import sys
from types import ModuleType

from app.compatibility.legacy_champion import (
    FallbackPolicy,
    LegacyChampionAdapter,
    LegacyLLMRuntime,
)
from app.knowledge.retrieval import HybridRetriever, ImageCandidate, KnowledgeDocument


def build_retriever() -> HybridRetriever:
    return HybridRetriever(
        [
            KnowledgeDocument(
                child_id="air-1",
                parent_id="air-parent",
                title="滤网清洁",
                text="空气净化器的预过滤网建议每两周清洁一次，清洁后完全晾干再安装。",
                product="air-purifier",
                asset_ids=["air-filter.png"],
            ),
            KnowledgeDocument(
                child_id="washer-1",
                parent_id="washer-parent",
                title="E03 排水错误",
                text="洗衣机显示 E03 时，请检查排水管是否弯折，并清理排水过滤器。",
                product="washing-machine",
                asset_ids=["washer-e03.png"],
            ),
            KnowledgeDocument(
                child_id="washer-2",
                parent_id="washer-parent",
                title="E03 排水错误补充",
                text="完成检查后重新启动设备；仍然报错时停止使用并联系售后。",
                product="washing-machine",
            ),
            KnowledgeDocument(
                child_id="camera-1",
                parent_id="camera-parent",
                title="错误提示",
                text="相机无法启动时检查电池安装方向。",
                product="camera",
            ),
        ]
    )


def test_exact_error_code_is_preserved_and_ranked_first() -> None:
    results = build_retriever().search("洗衣机出现 E03 应该怎么办？", products=["washing-machine"])

    assert results[0].document_id == "washer-parent"
    assert "E03" in results[0].text


def test_child_hits_are_aggregated_into_parent_evidence() -> None:
    results = build_retriever().search("E03 排水过滤器重新启动", products=["washing-machine"])

    assert len(results) == 1
    assert "排水管" in results[0].text
    assert "重新启动" in results[0].text
    assert results[0].score is not None and results[0].score > 0


def test_product_filter_prevents_cross_product_recall() -> None:
    results = build_retriever().search("无法启动，检查电池", products=["washing-machine"])

    assert all(item.product == "washing-machine" for item in results)


def test_image_gate_rejects_cross_product_assets() -> None:
    retriever = build_retriever()
    evidence = retriever.search("滤网怎么清洁", products=["air-purifier"])[0]

    approved = retriever.filter_images(
        evidence,
        [
            ImageCandidate(asset_id="air-filter.png", product="air-purifier", related_parent_ids=["air-parent"]),
            ImageCandidate(asset_id="camera-battery.png", product="camera", related_parent_ids=["camera-parent"]),
        ],
    )

    assert approved == ["air-filter.png"]


def test_fallback_policy_and_adapter_keep_champion_available() -> None:
    adapter = LegacyChampionAdapter(answer_func=lambda question, images: f"旧链路：{question}")

    assert FallbackPolicy.should_fallback(confidence=0.3, verification_passed=True, remaining_ms=5000)
    assert FallbackPolicy.should_fallback(confidence=0.9, verification_passed=False, remaining_ms=5000)
    assert FallbackPolicy.should_fallback(confidence=0.9, verification_passed=True, remaining_ms=800)
    assert adapter.answer("测试问题", []) == "旧链路：测试问题"


def test_legacy_adapter_supplies_dynamic_menu_flag_expected_by_frozen_engine(tmp_path) -> None:
    globals().pop("menu_requested", None)

    class FrozenEngine:
        def answer(self, question, images):
            return "menu" if globals()["menu_requested"] else "no-menu"

    adapter = LegacyChampionAdapter(source_root=tmp_path, manual_dir=tmp_path)
    adapter._engine = FrozenEngine()

    assert adapter.answer("common causes of poor reception", []) == "no-menu"
    assert adapter.answer("open the caption menu", []) == "menu"


class _FakeLegacyEngine:
    def __init__(self) -> None:
        self.use_llm = False
        self.use_llm_manual_polish = False
        self.use_llm_query_frame = False
        self.use_llm_query_rewrite = False
        self.use_ann = False
        self.llm = None
        self.customer_llm = None

    def answer(self, question, images):
        if self.use_llm and self.llm is not None:
            return self.llm.chat([{"role": "user", "content": question}])
        return f"确定性冠军答案：{question}"


class _SuccessfulLegacyClient:
    def chat(self, messages, temperature=None):
        return f"Qwen冠军答案：{messages[-1]['content']}"


class _FailingLegacyClient:
    def chat(self, messages, temperature=None):
        raise RuntimeError("provider unavailable")


class _TechnicalLegacyEngine(_FakeLegacyEngine):
    def answer(self, question, images):
        if self.use_llm_manual_polish and self.llm is not None:
            return self.llm.chat([{"role": "user", "content": question}])
        return f"确定性技术答案：{question}"


def _qwen_runtime() -> LegacyLLMRuntime:
    return LegacyLLMRuntime(
        model="qwen3-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="encrypted-at-rest-secret",
        timeout_seconds=20,
    )


def test_legacy_adapter_uses_dedicated_qwen_runtime_when_available(tmp_path) -> None:
    adapter = LegacyChampionAdapter(
        source_root=tmp_path,
        manual_dir=tmp_path,
        llm_runtime_provider=_qwen_runtime,
        llm_client_factory=lambda runtime: _SuccessfulLegacyClient(),
    )
    adapter._engine = _FakeLegacyEngine()

    answer = adapter.answer("测试问题", [])

    assert answer == "Qwen冠军答案：测试问题"
    assert adapter.last_invocation.llm_used is True
    assert adapter.last_invocation.model_used == "qwen3-max"
    assert adapter.last_invocation.fallback_reason is None
    assert adapter._engine.use_llm_query_frame is False
    assert adapter._engine.use_llm_query_rewrite is False
    assert adapter._engine.use_ann is False


def test_legacy_adapter_falls_back_to_deterministic_answer_when_qwen_fails(tmp_path) -> None:
    adapter = LegacyChampionAdapter(
        source_root=tmp_path,
        manual_dir=tmp_path,
        llm_runtime_provider=_qwen_runtime,
        llm_client_factory=lambda runtime: _FailingLegacyClient(),
    )
    adapter._engine = _FakeLegacyEngine()

    answer = adapter.answer("测试问题", [])

    assert answer == "确定性冠军答案：测试问题"
    assert adapter.last_invocation.llm_used is True
    assert adapter.last_invocation.model_used == "qwen3-max"
    assert adapter.last_invocation.fallback_reason == "qwen_error:RuntimeError"


def test_legacy_adapter_stays_deterministic_without_dedicated_runtime(tmp_path) -> None:
    adapter = LegacyChampionAdapter(
        source_root=tmp_path,
        manual_dir=tmp_path,
        llm_runtime_provider=lambda: None,
    )
    adapter._engine = _FakeLegacyEngine()

    answer = adapter.answer("测试问题", [])

    assert answer == "确定性冠军答案：测试问题"
    assert adapter.last_invocation.llm_used is False
    assert adapter.last_invocation.model_used is None
    assert adapter.last_invocation.fallback_reason == "legacy_llm_unavailable"


def test_legacy_adapter_keeps_technical_manual_answer_deterministic(tmp_path) -> None:
    adapter = LegacyChampionAdapter(
        source_root=tmp_path,
        manual_dir=tmp_path,
        llm_runtime_provider=_qwen_runtime,
        llm_client_factory=lambda runtime: _SuccessfulLegacyClient(),
    )
    adapter._engine = _TechnicalLegacyEngine()

    answer = adapter.answer("ERR02 怎么处理", [])

    assert answer == "确定性技术答案：ERR02 怎么处理"
    assert adapter._engine.use_llm is True
    assert adapter._engine.use_llm_manual_polish is False
    assert adapter.last_invocation.llm_used is False
    assert adapter.last_invocation.model_used is None


def test_qwen_profile_restores_missing_customer_profile_attribute(monkeypatch) -> None:
    customer_module = ModuleType("df_kefu_baseline.customer_llm")
    monkeypatch.setitem(
        sys.modules,
        "df_kefu_baseline.customer_llm",
        customer_module,
    )

    class FakeEngine:
        def answer(self, question: str, images: list[str]) -> str:
            assert customer_module.customer_llm_profile() == "qwen_original"
            return question

    answer = LegacyChampionAdapter._answer_with_qwen_profile(
        FakeEngine(),
        "测试问题",
        [],
    )

    assert answer == "测试问题"
    assert not hasattr(customer_module, "customer_llm_profile")
