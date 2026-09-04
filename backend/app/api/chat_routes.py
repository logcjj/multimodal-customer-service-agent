from __future__ import annotations

import json
from queue import Queue
from threading import Thread

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.contracts.models import AgentRequest, AgentResponse


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=AgentResponse)
def chat(payload: AgentRequest, request: Request) -> AgentResponse:
    metrics = request.app.state.metrics.start_request()
    try:
        response = request.app.state.orchestrator.run(
            payload,
            event_sink=metrics.observe_event,
        )
    except Exception:
        metrics.complete(success=False)
        raise
    metrics.complete(success=True)
    return response


@router.post("/chat/stream")
def stream_chat(payload: AgentRequest, request: Request) -> StreamingResponse:
    event_queue: Queue[dict[str, object] | object] = Queue()
    completed = object()
    metrics = request.app.state.metrics.start_request()

    def emit(event: dict[str, object]) -> None:
        metrics.observe_event(event)
        event_queue.put(event)

    def run() -> None:
        try:
            response = request.app.state.orchestrator.run(payload, event_sink=emit)
            emit(
                {
                    "type": "run.completed",
                    "agent_id": "orchestrator",
                    "status": "completed",
                    "label": "回答生成完成",
                    "summary": f"{response.trace.total_latency_ms} ms",
                    "payload": {"response": response.model_dump(mode="json")},
                }
            )
            metrics.complete(success=True)
        except Exception:
            emit(
                {
                    "type": "run.failed",
                    "agent_id": "orchestrator",
                    "status": "failed",
                    "label": "运行失败",
                    "summary": "服务执行异常",
                    "payload": {},
                }
            )
            metrics.complete(success=False)
        finally:
            event_queue.put(completed)

    Thread(target=run, daemon=True).start()

    def event_stream():
        sequence = 0
        while True:
            event = event_queue.get()
            if event is completed:
                break
            sequence += 1
            assert isinstance(event, dict)
            yield json.dumps({"sequence": sequence, **event}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
        },
    )
