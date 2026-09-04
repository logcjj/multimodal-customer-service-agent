from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_agent_catalog_contains_online_and_offline_roles(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get("/api/agents")

    assert response.status_code == 200
    agents = response.json()
    assert len(agents) == 10
    assert {item["id"] for item in agents} == {
        "orchestrator",
        "router",
        "evidence-gap",
        "general",
        "memory-curator",
        "multimodal",
        "knowledge",
        "customer-service",
        "verifier",
        "evolution",
    }
    assert next(item for item in agents if item["id"] == "evolution")["execution_mode"] == "offline"


def test_skills_and_tools_expose_permissions_and_risk(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))

    skills = client.get("/api/skills").json()
    tools = client.get("/api/tools").json()

    assert len(skills) == 12
    assert any(item["id"] == "response-verification" for item in skills)
    assert {"intent-routing", "evidence-clarification", "general-dialogue", "memory-curation"} <= {
        item["id"] for item in skills
    }
    ticket_tool = next(item for item in tools if item["id"] == "ticket.create")
    assert ticket_tool["risk_level"] == "write"
    assert ticket_tool["requires_confirmation"] is True
    knowledge_tool = next(item for item in tools if item["id"] == "knowledge.search")
    assert knowledge_tool["risk_level"] == "read"


def test_mcp_defaults_to_read_only_servers(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))

    servers = client.get("/api/mcp/servers").json()

    assert servers[0]["id"] == "customer-service-mcp"
    assert servers[0]["mode"] == "read-only"
    assert "knowledge.search" in servers[0]["tools"]


def test_capabilities_summarize_runtime_support(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get("/api/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["multi_agent"] is True
    assert body["legacy_fallback"] is True
    assert body["agent_count"] == 10
    assert body["mcp"] is True
    assert body["dynamic_routing"] == "on"
    assert body["conversation_history"] == "on"
    assert body["layered_memory"] == "on"


def test_mcp_streamable_http_lists_and_calls_real_search_tool(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))
    dataset_id = client.post("/api/datasets", json={"name": "MCP 测试知识库"}).json()["id"]
    file_id = client.post(
        "/api/files",
        files={
            "file": (
                "manual.md",
                "# E03 排水故障\n先关闭电源，再检查排水过滤器。".encode(),
                "text/markdown",
            )
        },
    ).json()["id"]
    document_id = client.post(
        f"/api/datasets/{dataset_id}/documents",
        json={"file_id": file_id},
    ).json()["id"]
    version = client.post(f"/api/documents/{document_id}/parse").json()["index_version"]
    client.post(f"/api/datasets/{dataset_id}/publish", json={"index_version": version})
    initialized = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    ).json()
    listed = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ).json()
    called = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "knowledge.search",
                "arguments": {
                    "dataset_ids": [dataset_id],
                    "query": "E03 排水过滤器",
                },
            },
        },
    ).json()

    assert initialized["result"]["protocolVersion"] == "2025-06-18"
    assert listed["result"]["tools"][0]["name"] == "knowledge.search"
    assert called["result"]["isError"] is False
    assert called["result"]["structuredContent"]["results"]
