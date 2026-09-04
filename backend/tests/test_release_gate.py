from __future__ import annotations

import json

from app.evaluation.service import EvaluationService
from app.knowledge.service import KnowledgeService
from app.storage.database import Database


def test_missing_frozen_competition_result_stays_awaiting_approval(tmp_path) -> None:
    database = Database(tmp_path)
    service = EvaluationService(database, KnowledgeService(database))

    status = service.assess_release_gate(
        dataset_id="v6-manuals",
        candidate_version="idx-candidate",
        run_id=None,
        frozen_score=None,
    )

    assert status["status"] == "awaiting_approval"
    assert status["frozen_competition_result"] == "missing"
    assert "0.88375" not in json.dumps(status)


def test_unapproved_or_regressed_frozen_score_cannot_be_marked_approved(tmp_path) -> None:
    database = Database(tmp_path)
    service = EvaluationService(database, KnowledgeService(database))

    status = service.assess_release_gate(
        dataset_id="v6-manuals",
        candidate_version="idx-candidate",
        run_id=None,
        frozen_score=0.88,
        required_frozen_score=0.88375,
    )

    assert status["status"] == "rejected"
    assert status["reason_code"] == "frozen_score_regression"
