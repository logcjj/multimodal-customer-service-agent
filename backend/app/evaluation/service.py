from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import HTTPException
from sqlmodel import select

from app.evaluation.models import EvalCaseRecord, EvalCaseView, EvalRunRecord, EvalRunView
from app.knowledge.service import KnowledgeService
from app.storage.database import Database


class PublishGateError(RuntimeError):
    pass


class EvaluationService:
    def __init__(self, database: Database, knowledge: KnowledgeService) -> None:
        self.database = database
        self.knowledge = knowledge

    def create_case(
        self,
        *,
        question: str,
        dataset_ids: list[str],
        target_parent_ids: list[str],
        reference_answer: str = "",
        required_facts: list[str] | None = None,
        forbidden_facts: list[str] | None = None,
        image_required: bool = False,
        locked: bool = True,
        source: str = "manual",
    ) -> EvalCaseView:
        record = EvalCaseRecord(
            id=str(uuid4()),
            question=question,
            dataset_ids_json=json.dumps(dataset_ids, ensure_ascii=False),
            target_parent_ids_json=json.dumps(target_parent_ids, ensure_ascii=False),
            reference_answer=reference_answer,
            required_facts_json=json.dumps(required_facts or [], ensure_ascii=False),
            forbidden_facts_json=json.dumps(forbidden_facts or [], ensure_ascii=False),
            image_required=image_required,
            locked=locked,
            source=source,
        )
        with self.database.session() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
        return self._case_view(record)

    def list_cases(self) -> list[EvalCaseView]:
        with self.database.session() as session:
            records = session.exec(select(EvalCaseRecord).order_by(EvalCaseRecord.created_at)).all()
            for record in records:
                session.expunge(record)
        return [self._case_view(record) for record in records]

    def run(self, *, candidate_version: str, case_ids: list[str]) -> EvalRunView:
        cases = self._get_cases(case_ids)
        details: list[dict[str, object]] = []
        recall_hits = 0
        reciprocal_rank_sum = 0.0
        required_total = 0
        required_hits = 0
        forbidden_violations = 0
        locked_failure = False
        for case in cases:
            dataset_ids = json.loads(case.dataset_ids_json)
            targets = set(json.loads(case.target_parent_ids_json))
            explanation = self.knowledge.candidate_retriever(
                dataset_ids,
                candidate_version,
            ).explain(
                case.question,
                dataset_ids=dataset_ids,
                top_n=5,
            )
            ranked = [item.parent_id for item in explanation.results]
            first_rank = next((index for index, item in enumerate(ranked, start=1) if item in targets), None)
            hit = bool(first_rank) if targets else bool(ranked)
            recall_hits += int(hit)
            reciprocal_rank_sum += 1 / first_rank if first_rank else 0
            answer_context = "\n".join(item.text for item in explanation.results)
            required = json.loads(case.required_facts_json)
            forbidden = json.loads(case.forbidden_facts_json)
            required_total += len(required)
            current_required_hits = sum(1 for fact in required if fact in answer_context)
            required_hits += current_required_hits
            current_forbidden = sum(1 for fact in forbidden if fact in answer_context)
            forbidden_violations += current_forbidden
            case_passed = hit and current_required_hits == len(required) and current_forbidden == 0
            if case.locked and not case_passed:
                locked_failure = True
            details.append(
                {
                    "case_id": case.id,
                    "hit": hit,
                    "rank": first_rank,
                    "required_hits": current_required_hits,
                    "required_total": len(required),
                    "forbidden_violations": current_forbidden,
                    "passed": case_passed,
                    "rejected_reason": explanation.rejected_reason,
                }
            )
        count = max(1, len(cases))
        metrics = {
            "recall_at_5": recall_hits / count,
            "mrr": reciprocal_rank_sum / count,
            "fact_coverage": required_hits / required_total if required_total else 1.0,
            "forbidden_violation_rate": forbidden_violations / count,
        }
        passed = not locked_failure and metrics["recall_at_5"] == 1.0 and metrics["forbidden_violation_rate"] == 0
        record = EvalRunRecord(
            id=str(uuid4()),
            candidate_version=candidate_version,
            case_ids_json=json.dumps(case_ids),
            metrics_json=json.dumps(metrics),
            details_json=json.dumps(details, ensure_ascii=False),
            passed=passed,
        )
        with self.database.session() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
        return self._run_view(record)

    def get_run(self, run_id: str) -> EvalRunView:
        with self.database.session() as session:
            record = session.get(EvalRunRecord, run_id)
            if record is None:
                raise HTTPException(status_code=404, detail="评测运行不存在")
            session.expunge(record)
        return self._run_view(record)

    def list_runs(self) -> list[EvalRunView]:
        with self.database.session() as session:
            records = session.exec(select(EvalRunRecord).order_by(EvalRunRecord.created_at.desc())).all()
            for record in records:
                session.expunge(record)
        return [self._run_view(record) for record in records]

    def require_publish_approval(
        self,
        dataset_id: str,
        candidate_version: str,
        run_id: str | None,
    ) -> None:
        locked_cases = [
            item
            for item in self.list_cases()
            if item.locked and dataset_id in item.dataset_ids
        ]
        if not locked_cases:
            return
        if not run_id:
            raise PublishGateError("该知识库存在锁定评测用例，发布前必须提供已批准的评测运行")
        run = self.get_run(run_id)
        required_ids = {item.id for item in locked_cases}
        if run.candidate_version != candidate_version:
            raise PublishGateError("评测运行与待发布索引版本不一致")
        if run.status != "approved" or not run.passed:
            raise PublishGateError("评测运行尚未通过并批准")
        if not required_ids.issubset(set(run.case_ids)):
            raise PublishGateError("评测运行没有覆盖全部锁定用例")

    def assess_release_gate(
        self,
        *,
        dataset_id: str,
        candidate_version: str,
        run_id: str | None,
        frozen_score: float | None,
        required_frozen_score: float = 0.88375,
    ) -> dict[str, object]:
        """Return an explicit release state without inventing missing benchmark evidence."""
        if frozen_score is None:
            return {
                "status": "awaiting_approval",
                "reason_code": "frozen_result_missing",
                "dataset_id": dataset_id,
                "candidate_version": candidate_version,
                "evaluation_run_id": run_id,
                "frozen_competition_result": "missing",
            }
        if frozen_score < required_frozen_score:
            return {
                "status": "rejected",
                "reason_code": "frozen_score_regression",
                "dataset_id": dataset_id,
                "candidate_version": candidate_version,
                "evaluation_run_id": run_id,
                "frozen_competition_result": "below_required_threshold",
                "frozen_score": frozen_score,
                "required_frozen_score": required_frozen_score,
            }
        if run_id is None:
            return {
                "status": "awaiting_approval",
                "reason_code": "evaluation_approval_missing",
                "dataset_id": dataset_id,
                "candidate_version": candidate_version,
                "evaluation_run_id": None,
                "frozen_competition_result": "passed",
                "frozen_score": frozen_score,
                "required_frozen_score": required_frozen_score,
            }
        try:
            run = self.get_run(run_id)
        except HTTPException:
            return {
                "status": "awaiting_approval",
                "reason_code": "evaluation_run_missing",
                "dataset_id": dataset_id,
                "candidate_version": candidate_version,
                "evaluation_run_id": run_id,
                "frozen_competition_result": "passed",
            }
        approved = run.candidate_version == candidate_version and run.passed and run.status == "approved"
        return {
            "status": "approved" if approved else "awaiting_approval",
            "reason_code": "all_gates_passed" if approved else "evaluation_approval_missing",
            "dataset_id": dataset_id,
            "candidate_version": candidate_version,
            "evaluation_run_id": run_id,
            "frozen_competition_result": "passed",
            "frozen_score": frozen_score,
            "required_frozen_score": required_frozen_score,
        }

    def approve(self, run_id: str) -> EvalRunView:
        with self.database.session() as session:
            record = session.get(EvalRunRecord, run_id)
            if record is None:
                raise HTTPException(status_code=404, detail="评测运行不存在")
            if not record.passed:
                raise PublishGateError("候选版本未通过锁定回归门禁")
            record.status = "approved"
            record.approved_at = datetime.now(UTC)
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
        return self._run_view(record)

    def _get_cases(self, case_ids: list[str]) -> list[EvalCaseRecord]:
        with self.database.session() as session:
            records = session.exec(select(EvalCaseRecord).where(EvalCaseRecord.id.in_(case_ids))).all()
            if len(records) != len(set(case_ids)):
                raise HTTPException(status_code=404, detail="部分评测用例不存在")
            for record in records:
                session.expunge(record)
            return list(records)

    @staticmethod
    def _case_view(record: EvalCaseRecord) -> EvalCaseView:
        return EvalCaseView(
            id=record.id,
            question=record.question,
            dataset_ids=json.loads(record.dataset_ids_json),
            target_parent_ids=json.loads(record.target_parent_ids_json),
            reference_answer=record.reference_answer,
            required_facts=json.loads(record.required_facts_json),
            forbidden_facts=json.loads(record.forbidden_facts_json),
            image_required=record.image_required,
            locked=record.locked,
            source=record.source,
            created_at=record.created_at,
        )

    @staticmethod
    def _run_view(record: EvalRunRecord) -> EvalRunView:
        return EvalRunView(
            id=record.id,
            candidate_version=record.candidate_version,
            case_ids=json.loads(record.case_ids_json),
            metrics=json.loads(record.metrics_json),
            details=json.loads(record.details_json),
            passed=record.passed,
            status=record.status,
            created_at=record.created_at,
            approved_at=record.approved_at,
        )
