from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from app.knowledge.contracts import RetrievalTestRequest


router = APIRouter(tags=["mcp"])


def _tool_definition() -> dict[str, object]:
    return {
        "name": "knowledge.search",
        "description": "在已发布知识库中执行可解释的只读混合检索。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "query": {"type": "string", "minLength": 1},
                "top_n": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                "use_rerank": {"type": "boolean", "default": True},
            },
            "required": ["dataset_ids", "query"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    }


def _execute_search(request: Request, arguments: dict[str, Any]) -> dict[str, object]:
    try:
        payload = RetrievalTestRequest.model_validate(arguments)
        result = request.app.state.knowledge_service.retrieval_test(payload)
    except ValidationError as exc:
        return {
            "tool": "knowledge.search",
            "is_error": True,
            "content": [{"type": "text", "text": "检索参数不符合工具契约。"}],
            "structured_content": {"errors": exc.errors(include_input=False, include_url=False)},
        }
    titles = [str(item.get("title", "")) for item in result.get("results", [])]
    summary = f"返回 {len(titles)} 条已发布证据"
    if titles:
        summary += "：" + "；".join(titles[:5])
    return {
        "tool": "knowledge.search",
        "is_error": False,
        "content": [{"type": "text", "text": summary}],
        "structured_content": result,
    }


@router.post("/api/mcp/tools/knowledge.search")
def call_knowledge_search(payload: RetrievalTestRequest, request: Request) -> dict[str, object]:
    return _execute_search(request, payload.model_dump())


@router.post("/mcp")
def streamable_http_mcp(payload: dict[str, Any], request: Request) -> dict[str, object]:
    request_id = payload.get("id")
    method = payload.get("method")
    if payload.get("jsonrpc") != "2.0":
        raise HTTPException(status_code=400, detail="MCP 请求必须使用 JSON-RPC 2.0")
    if method == "initialize":
        result: dict[str, object] = {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "aka-customer-service-mcp", "version": "0.6.0"},
        }
    elif method == "notifications/initialized":
        result = {}
    elif method == "tools/list":
        result = {"tools": [_tool_definition()]}
    elif method == "tools/call":
        params = payload.get("params") or {}
        if params.get("name") != "knowledge.search":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "未知 MCP Tool"},
            }
        executed = _execute_search(request, params.get("arguments") or {})
        result = {
            "content": executed["content"],
            "structuredContent": executed["structured_content"],
            "isError": executed["is_error"],
        }
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "MCP 方法不存在"},
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}
