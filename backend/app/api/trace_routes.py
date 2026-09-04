from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, Request

from app.contracts.models import AgentTrace
from app.observability.traces import TraceStore


router = APIRouter(prefix="/api/traces", tags=["traces"])


def _owner_id(
    user_id: Annotated[str | None, Query(max_length=120)] = None,
    client_id: Annotated[
        str | None,
        Header(alias="X-Client-ID", max_length=120),
    ] = None,
) -> str:
    owner_id = (user_id or client_id or "").strip()
    if not owner_id:
        raise HTTPException(status_code=422, detail="需要 user_id 或 X-Client-ID")
    if owner_id == TraceStore.LEGACY_ANONYMOUS_OWNER:
        raise HTTPException(status_code=422, detail="owner_id 不可使用保留值")
    return owner_id


@router.get("", response_model=list[AgentTrace])
def list_traces(
    request: Request,
    user_id: Annotated[str | None, Query(max_length=120)] = None,
    client_id: Annotated[
        str | None,
        Header(alias="X-Client-ID", max_length=120),
    ] = None,
) -> list[AgentTrace]:
    return request.app.state.trace_store.list(_owner_id(user_id, client_id))


@router.get("/{request_id}", response_model=AgentTrace)
def get_trace(
    request_id: str,
    request: Request,
    user_id: Annotated[str | None, Query(max_length=120)] = None,
    client_id: Annotated[
        str | None,
        Header(alias="X-Client-ID", max_length=120),
    ] = None,
) -> AgentTrace:
    return request.app.state.trace_store.get(
        request_id,
        _owner_id(user_id, client_id),
    )
