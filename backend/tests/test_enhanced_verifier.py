from __future__ import annotations

from app.agents.verifier import VerifierAgent
from app.contracts.models import AgentResult, Claim, Evidence


def _result(answer: str, evidence: list[Evidence]) -> AgentResult:
    return AgentResult(
        task_id="knowledge-1",
        agent_id="knowledge",
        status="completed",
        answer_fragment=answer,
        claims=[Claim(text=answer, evidence_ids=[item.evidence_id for item in evidence])],
        evidence=evidence,
        confidence=0.9,
    )


def _manual(text: str, evidence_id: str = "manual:1") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type="manual",
        title="产品说明书",
        text=text,
    )


def test_rejects_reordered_numbered_safety_steps() -> None:
    evidence = [_manual("1. 关闭电源。2. 等待设备冷却。3. 拆下防护罩。")]
    result = _result("先拆下防护罩，再关闭电源，最后等待设备冷却。", evidence)

    report = VerifierAgent().verify([result], {"knowledge"})

    assert report.passed is False
    issue = next(item for item in report.issues if item.code == "safety-step-order-mismatch")
    assert issue.evidence_ids == ["manual:1"]


def test_rejects_answer_that_contradicts_manual_prohibition() -> None:
    evidence = [_manual("防护罩损坏或缺失时严禁继续使用设备。")]
    result = _result("防护罩损坏后仍可以继续使用设备。", evidence)

    report = VerifierAgent().verify([result], {"knowledge"})

    assert report.passed is False
    assert any(item.code == "contradicts-prohibition" for item in report.issues)


def test_rejects_invented_refund_or_free_repair_commitment() -> None:
    evidence = [_manual("是否维修需要由售后检测后确定。")]
    result = _result("我们保证免费维修，并在24小时内完成赔付。", evidence)

    report = VerifierAgent().verify([result], {"knowledge"})

    assert report.passed is False
    assert any(item.code == "unsupported-service-commitment" for item in report.issues)


def test_accepts_service_commitment_explicitly_supported_by_policy() -> None:
    evidence = [
        Evidence(
            evidence_id="policy:1",
            source_type="policy",
            title="售后政策",
            text="符合保修条件时提供免费维修。",
        )
    ]
    result = _result("符合保修条件时提供免费维修。", evidence)

    report = VerifierAgent().verify([result], {"knowledge"})

    assert report.passed is True


def test_accepts_cycle_count_supported_by_escaped_manual_range() -> None:
    evidence = [_manual("重复此循环，直至排水干净（2\\~3 cycles）。")]
    result = _result("重复以上步骤 2～3 次，直到排水干净。", evidence)

    report = VerifierAgent().verify([result], {"knowledge"})

    assert report.passed is True


def test_rejects_visual_observation_promoted_to_certain_fact() -> None:
    evidence = [
        Evidence(
            evidence_id="vision:1",
            source_type="vision",
            title="用户图片视觉观察",
            text="图片中疑似显示 E03。",
        )
    ]
    result = _result("设备已经确认发生 E03 排水故障。", evidence)

    report = VerifierAgent().verify([result], {"knowledge"})

    assert report.passed is False
    assert any(item.code == "visual-observation-promoted" for item in report.issues)


def test_old_evidence_payload_remains_compatible() -> None:
    evidence = Evidence.model_validate(
        {
            "evidence_id": "legacy:1",
            "source_type": "legacy",
            "title": "旧证据",
            "text": "请先关闭电源。",
            "child_ids": [],
            "asset_ids": [],
            "score_breakdown": {},
        }
    )

    assert evidence.parent_id is None
    assert evidence.image_chunk_ids == []
    assert evidence.retrieval_stage is None


def test_basic_verifier_mode_keeps_claim_binding_checks_without_enhanced_fact_rules() -> None:
    evidence = [_manual("是否维修需要由售后检测后确定。")]
    result = _result("我们保证免费维修，并在24小时内完成赔付。", evidence)

    report = VerifierAgent().verify([result], {"knowledge"}, enhanced=False)

    assert report.passed is True
    assert report.action == "accept"
    assert report.issues == []
