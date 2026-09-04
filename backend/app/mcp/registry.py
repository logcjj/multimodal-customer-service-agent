from __future__ import annotations


MCP_SERVERS = [
    {
        "id": "customer-service-mcp",
        "name": "Customer Service MCP",
        "description": "向外部 Agent 客户端提供只读说明书、政策和图片元数据。",
        "mode": "read-only",
        "transport": "streamable-http",
        "status": "ready",
        "tools": ["knowledge.search"],
        "resources": [],
    }
]
