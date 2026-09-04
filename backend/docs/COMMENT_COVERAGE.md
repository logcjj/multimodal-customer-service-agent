# 代码说明与注释覆盖

本项目以类型标注、Pydantic 契约、模块边界和测试用例作为主要可读性保障；复杂逻辑处保留必要注释，避免把显而易见的语句重复写成噪声注释。

## 代码说明入口

| 模块 | 说明 |
| --- | --- |
| `backend/app/main.py` | 应用创建、依赖注入、模型服务、知识库、会话、评测和智能体初始化 |
| `backend/app/contracts/models.py` | 对话请求、回答、证据、核验、轨迹、模型配置等 API 契约 |
| `backend/app/runtime/orchestrator.py` | 多智能体主编排、事件输出、会话写入、兜底策略 |
| `backend/app/runtime/dynamic_routing.py` | 意图路由、知识覆盖门禁、通用回答安全边界 |
| `backend/app/agents/verifier.py` | 证据绑定、数字型号、禁止项、售后承诺和视觉表述核验 |
| `backend/app/knowledge/hybrid.py` | BM25、向量召回、RRF 融合、Rerank 和父章节聚合 |
| `backend/app/knowledge/image_retrieval.py` | 图片洞察导入、图片 Chunk 归一化和图文检索 |
| `backend/app/models/llm_gateway.py` | OpenAI/Anthropic 兼容模型调用、流式输出和错误降级 |
| `frontend/src/aka/api/types.ts` | 前端接口类型 |
| `frontend/src/aka/pages/` | 工作台、知识库、模型、评测、轨迹等页面 |
| `frontend/src/aka/components/` | 证据抽屉、检索实验、运行轨迹、向量图等组件 |

## 注释覆盖核查方式

如需生成函数/类 docstring 覆盖率，可在后端目录执行以下临时检查命令：

```bash
cd backend
uv run python - <<'PY'
import ast
from pathlib import Path

targets = [Path("app"), Path("scripts")]
modules = []
symbols = []
for root in targets:
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules.append((path, bool(ast.get_docstring(tree))))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_"):
                    continue
                symbols.append((path, node.name, bool(ast.get_docstring(node))))

module_rate = sum(doc for _, doc in modules) / max(1, len(modules)) * 100
symbol_rate = sum(doc for _, _, doc in symbols) / max(1, len(symbols)) * 100
print(f"modules={len(modules)} module_docstring_coverage={module_rate:.2f}%")
print(f"symbols={len(symbols)} symbol_docstring_coverage={symbol_rate:.2f}%")
PY
```

统计口径：

- `module_docstring_coverage`：模块级 docstring 覆盖率。
- `symbol_docstring_coverage`：公开类、函数、异步函数 docstring 覆盖率。
- 私有辅助函数以下划线开头，默认不纳入硬性统计。

## 可读性保障

- 后端接口输入输出均由 Pydantic 模型约束，字段限制集中在契约层。
- 检索、路由、核验和模型调用均有独立测试覆盖，避免只依赖注释说明行为。
- 关键算法已在 `backend/docs/KEY_ALGORITHMS.md` 以工程文档形式说明。
- API 使用方式已在 `backend/docs/API_REFERENCE.md` 和 `backend/docs/chat_api.md` 给出示例。
