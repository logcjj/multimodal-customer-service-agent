from __future__ import annotations

from fastapi import APIRouter, Request

from app.agents.catalog import AGENT_CATALOG
from app.mcp.registry import MCP_SERVERS
from app.skills.registry import SKILL_REGISTRY
from app.tools.registry import TOOL_REGISTRY


router = APIRouter(prefix="/api", tags=["capabilities"])


@router.get("/agents")
def list_agents() -> list[dict[str, object]]:
    return AGENT_CATALOG


@router.get("/skills")
def list_skills() -> list[dict[str, object]]:
    return SKILL_REGISTRY


@router.get("/tools")
def list_tools() -> list[dict[str, object]]:
    return TOOL_REGISTRY


@router.get("/mcp/servers")
def list_mcp_servers() -> list[dict[str, object]]:
    return MCP_SERVERS


@router.get("/capabilities")
def capabilities(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    return {
        "multi_agent": True,
        "vision": True,
        "skills": True,
        "mcp": True,
        "memory": True,
        "trace": True,
        "legacy_fallback": True,
        "stream_mode": "status-events",
        "agent_count": len(AGENT_CATALOG),
        "skill_count": len(SKILL_REGISTRY),
        "tool_count": len(TOOL_REGISTRY),
        "dynamic_routing": settings.dynamic_routing,
        "conversation_history": settings.conversation_history,
        "layered_memory": settings.layered_memory,
        "general_agent": settings.general_agent,
    }
