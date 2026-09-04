from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.evaluation.models import EvalCaseCreate, EvalCaseView, EvalRunCreate, EvalRunView
from app.evaluation.service import PublishGateError


router = APIRouter(prefix="/api/evaluations", tags=["evaluation"])


@router.post("/cases", response_model=EvalCaseView, status_code=status.HTTP_201_CREATED)
def create_case(payload: EvalCaseCreate, request: Request) -> EvalCaseView:
    return request.app.state.evaluation_service.create_case(**payload.model_dump())


@router.get("/cases", response_model=list[EvalCaseView])
def list_cases(request: Request) -> list[EvalCaseView]:
    return request.app.state.evaluation_service.list_cases()


@router.post("/runs", response_model=EvalRunView, status_code=status.HTTP_201_CREATED)
def run_evaluation(payload: EvalRunCreate, request: Request) -> EvalRunView:
    return request.app.state.evaluation_service.run(**payload.model_dump())


@router.get("/runs/{run_id}", response_model=EvalRunView)
def get_run(run_id: str, request: Request) -> EvalRunView:
    return request.app.state.evaluation_service.get_run(run_id)


@router.get("/runs", response_model=list[EvalRunView])
def list_runs(request: Request) -> list[EvalRunView]:
    return request.app.state.evaluation_service.list_runs()


@router.post("/runs/{run_id}/approve", response_model=EvalRunView)
def approve_run(run_id: str, request: Request) -> EvalRunView:
    try:
        return request.app.state.evaluation_service.approve(run_id)
    except PublishGateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
