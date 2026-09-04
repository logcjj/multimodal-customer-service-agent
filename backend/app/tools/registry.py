from __future__ import annotations


TOOL_REGISTRY = [
    {"id": "knowledge.search", "name": "混合检索", "risk_level": "read", "requires_confirmation": False, "timeout_ms": 2500, "idempotent": True},
    {"id": "knowledge.read", "name": "读取证据章节", "risk_level": "read", "requires_confirmation": False, "timeout_ms": 800, "idempotent": True},
    {"id": "asset.get", "name": "读取图片元数据", "risk_level": "read", "requires_confirmation": False, "timeout_ms": 800, "idempotent": True},
    {"id": "vision.inspect", "name": "图片结构化观察", "risk_level": "read", "requires_confirmation": False, "timeout_ms": 6000, "idempotent": True},
    {"id": "policy.lookup", "name": "查询售后政策", "risk_level": "read", "requires_confirmation": False, "timeout_ms": 1200, "idempotent": True},
    {"id": "order.query", "name": "查询订单", "risk_level": "sensitive-read", "requires_confirmation": False, "timeout_ms": 2000, "idempotent": True},
    {"id": "ticket.create", "name": "创建工单", "risk_level": "write", "requires_confirmation": True, "timeout_ms": 3000, "idempotent": True},
    {"id": "handoff.package", "name": "生成人工接管包", "risk_level": "write", "requires_confirmation": False, "timeout_ms": 1000, "idempotent": True},
    {"id": "verify.claims", "name": "事实与数字验证", "risk_level": "read", "requires_confirmation": False, "timeout_ms": 1600, "idempotent": True},
    {"id": "verify.assets", "name": "图片归属验证", "risk_level": "read", "requires_confirmation": False, "timeout_ms": 1000, "idempotent": True},
    {"id": "eval.run-suite", "name": "运行离线评测", "risk_level": "local-write", "requires_confirmation": False, "timeout_ms": 120000, "idempotent": True},
]

