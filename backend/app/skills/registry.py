from __future__ import annotations


SKILL_REGISTRY = [
    {"id": "route-and-plan", "name": "路由与规划", "owner": "orchestrator", "version": "1.0.0", "status": "active", "description": "将请求转换为有界任务计划。"},
    {"id": "multi-question-decomposition", "name": "多问题拆解", "owner": "orchestrator", "version": "1.0.0", "status": "active", "description": "拆分子问题并记录依赖和覆盖状态。"},
    {"id": "intent-routing", "name": "动态意图路由", "owner": "router", "version": "1.0.0", "status": "active", "description": "结合受约束模型分类和确定性安全覆盖确定候选路由。"},
    {"id": "evidence-clarification", "name": "逐项证据补全", "owner": "evidence-gap", "version": "1.0.0", "status": "active", "description": "一次追问一个缺失字段，最多三轮后安全交接。"},
    {"id": "general-dialogue", "name": "通用对话", "owner": "general", "version": "1.0.0", "status": "active", "description": "处理不依赖产品手册、客服政策或实时订单的开放请求。"},
    {"id": "memory-curation", "name": "分层记忆整理", "owner": "memory-curator", "version": "1.0.0", "status": "active", "description": "维护完整账本、Token 窗口、结构化槽位和可追溯摘要。"},
    {"id": "multimodal-inspection", "name": "多模态检查", "owner": "multimodal", "version": "1.0.0", "status": "active", "description": "区分可见事实、OCR 结果和推测。"},
    {"id": "manual-qa", "name": "说明书问答", "owner": "knowledge", "version": "1.0.0", "status": "active", "description": "基于 EvidenceBundle 回答产品技术问题。"},
    {"id": "troubleshooting", "name": "故障排查", "owner": "knowledge", "version": "1.0.0", "status": "active", "description": "先安全后排查，缺失证据时请求澄清。"},
    {"id": "customer-policy", "name": "客服政策", "owner": "customer-service", "version": "1.0.0", "status": "active", "description": "对高风险承诺强制绑定政策版本。"},
    {"id": "human-handoff", "name": "人工接管", "owner": "customer-service", "version": "1.0.0", "status": "active", "description": "整理上下文、证据、槽位和失败原因。"},
    {"id": "response-verification", "name": "回答验证", "owner": "verifier", "version": "1.0.0", "status": "active", "description": "验证 claim、数字、图片和政策一致性。"},
]
