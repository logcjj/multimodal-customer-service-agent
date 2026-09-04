<p align="center">
  <img src="frontend/public/logo.svg" width="72" alt="多模态客服智能体系统 logo">
</p>

<h1 align="center">基于自适应检索与自主学习的多模态客服智能体系统</h1>

<p align="center">
  Design of a Multimodal Customer Service Agent Based on Adaptive Retrieval and Self-Learning
</p>

<p align="center">
  面向电子产品说明书、故障排查与售后支持的证据优先客服智能体。
</p>

<p align="center">
  <a href="https://logcjj.github.io/multimodal-customer-service-agent/">在线预览（GitHub Pages）</a>
</p>

<p align="center">
  <a href="https://github.com/logcjj/multimodal-customer-service-agent/actions/workflows/deploy-pages.yml"><img src="https://github.com/logcjj/multimodal-customer-service-agent/actions/workflows/deploy-pages.yml/badge.svg" alt="Deploy frontend"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white" alt="Python 3.13+"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://react.dev/"><img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=111827" alt="React 18"></a>
  <a href="https://www.typescriptlang.org/"><img src="https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white" alt="TypeScript 5"></a>
</p>

## 项目简介

本项目把文本、产品图片和多轮上下文统一为可追溯的客服任务。系统先识别问题意图和产品焦点，再通过混合检索召回说明书证据，最后由验证器检查事实、数值、步骤顺序和引用覆盖率后生成回答。

The repository contains a self-hosted FastAPI backend and a React operations workspace. It is designed for teams that need grounded answers, inspectable retrieval traces, and versioned knowledge indexes instead of a black-box chat endpoint.

## 核心能力

- **多模态输入**：支持文本与产品图片，接入 OCR/VLM 上下文。
- **证据优先 RAG**：Parent/Child 文本块、Image Chunk、BM25、向量检索、RRF 融合与可选重排。
- **动态路由**：区分技术、客服、混合、通用、澄清和敏感问题。
- **回答验证**：核对事实、数值、步骤、引用、图片观察和承诺边界。
- **会话记忆**：保存结构化槽位、澄清轮次与滚动摘要，但不把历史回答当作产品证据。
- **索引版本管理**：离线构建带校验和的索引包，支持增量复用、发布与回滚。
- **可观测工作台**：查看会话、数据集、检索实验、模型、Trace、评测、技能和 MCP 工具。

## 系统架构

```text
浏览器 / REST / SSE
        |
        v
FastAPI API
        |
   Orchestrator
  /      |       \
路由   多模态   记忆
  \      |       /
      混合检索与知识库
              |
          客服 Agent
              |
           Verifier
              |
      回答 + 引用 + Trace
```

一次请求的主要路径：

```text
意图/产品/焦点提取
  -> 词法与向量候选召回
  -> RRF 融合与可选重排
  -> Child 到 Parent 的上下文扩展
  -> 证据覆盖率与风险门控
  -> 带引用的回答与运行 Trace
```

## 快速开始

### 环境要求

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Node.js 20+ 与 npm
- 多实例部署可选 PostgreSQL

### 安装

```bash
git clone https://github.com/logcjj/multimodal-customer-service-agent.git
cd multimodal-customer-service-agent
cp .env.example .env

cd backend
uv sync --dev

cd ../frontend
npm ci
cd ..
```

`.env.example` 只包含空的配置槽位。请把真实凭据放在本地 `.env` 或部署平台的 Secret Manager 中，不要写入源代码、Issue 或日志。

### 启动开发环境

```bash
./run_dev.sh
```

| 服务 | 地址 |
| --- | --- |
| Web UI | <http://127.0.0.1:5175> |
| API | <http://127.0.0.1:8002> |
| OpenAPI | <http://127.0.0.1:8002/docs> |
| MCP | <http://127.0.0.1:8002/mcp> |

也可以分别启动：

```bash
cd backend
uv run uvicorn app.main:app --host 127.0.0.1 --port 8002 --env-file ../.env
```

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5175
```

## 配置

从 `.env.example` 开始，只填写当前部署需要的变量：

| 配置组 | 示例变量 | 用途 |
| --- | --- | --- |
| 运行时 | `AKA_ROLLOUT_MODE`, `AKA_OFFLINE_INDEX_MODE` | 选择回答与索引模式 |
| 存储/跨域 | `AKA_DATABASE_URL`, `AKA_MODEL_SECRET_KEY`, `AKA_CORS_ORIGINS` | SQLite/PostgreSQL、模型凭据加密与浏览器来源 |
| 模型 | `OPENAI_*`, `DASHSCOPE_*`, `ALIYUN_*` | 文本、视觉、向量、重排和 OCR 提供商 |
| 记忆 | `AKA_SESSION_MEMORY`, `AKA_LAYERED_MEMORY` | 会话上下文与分层记忆 |
| 检索 | `AKA_IMAGE_CHUNK_RETRIEVAL`, `AKA_CAPTION_EMBEDDING` | 文本/图片证据路径 |
| 端口 | `BACKEND_PORT`, `FRONTEND_PORT` | 覆盖本地端口 |

模型凭据在写入运行时数据库前会加密，并从 API 响应、Trace、指标和清单中脱敏。

## 知识库与索引

说明书、上传文件、向量和运行时数据库保存在 `backend/data`，默认不会提交到 Git。导入自己的资料后构建索引：

```bash
cd backend
uv run python scripts/import_v6_knowledge.py
uv run python scripts/build_index_bundle.py --dataset-id v6-manuals --data-dir ./data
uv run python scripts/build_vector_map.py --dataset-id v6-manuals --data-dir ./data
```

也可以通过 API 或 Web 工作台创建数据集并上传文件。

## API 速览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/health` | 存活检查 |
| `GET` | `/api/readiness` | 运行时、模型与索引就绪状态 |
| `POST` | `/api/chat` | 返回路由、回答、引用、验证结果和 Trace |
| `POST` | `/api/chat/stream` | 以事件流返回阶段性进度 |
| `GET` | `/api/conversations` | 会话列表 |
| `POST` | `/api/files` | 上传资料 |
| `POST` | `/api/datasets` | 创建知识数据集 |
| `POST` | `/api/retrieval/test` | 检查候选与分数 |
| `GET` | `/api/traces` | 查看运行 Trace |
| `POST` | `/mcp` | Streamable HTTP MCP 端点 |

```bash
curl -X POST http://127.0.0.1:8002/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"设备显示错误代码时应该先检查什么？","images":[],"session_id":"demo"}'
```

完整说明见 [API reference](backend/docs/API_REFERENCE.md) 与 [Chat API guide](backend/docs/chat_api.md)。

## GitHub Pages 部署

`.github/workflows/deploy-pages.yml` 会在 `master` 更新或手动触发时构建并发布 Vite 前端，并自动生成 SPA 的 `404.html` 回退页。

1. 在仓库 Settings → Pages 中将 Source 设为 **GitHub Actions**。
2. 如果后端部署在独立 HTTPS 地址，在 Settings → Secrets and variables → Actions → Variables 添加 `VITE_API_BASE_URL`，值为后端根地址，例如 `https://api.example.com`。
3. 在后端 `.env` 设置 `AKA_CORS_ORIGINS=https://logcjj.github.io`（只填 Origin，不带路径），允许 Pages 前端访问 API。
4. 工作流会把仓库名作为 Pages 子路径注入 `VITE_BASE_URL`，因此前端资源和路由可直接在项目 Pages 地址下工作。

GitHub Pages 只负责静态前端，FastAPI 仍需部署到支持 ASGI 的运行环境。生产环境应启用 TLS、认证、上传限制、日志脱敏和数据库/索引备份。详细命令见 [deployment guide](backend/docs/DEPLOYMENT.md)。

## 仓库结构

```text
.
├── backend/
│   ├── app/       FastAPI、Agent、检索、模型和存储
│   ├── docs/      API、部署和算法文档
│   ├── scripts/   导入、索引和运维脚本
│   └── tests/     后端测试
├── frontend/
│   ├── src/aka/   客服工作台页面、组件和 API 适配器
│   └── src/       共享 UI 基础
├── .github/       CI 与 GitHub Pages 工作流
├── CONTRIBUTING.md
├── SECURITY.md
├── run_dev.sh
└── .env.example
```

## 测试与质量检查

```bash
cd backend
uv run pytest -q
python -m compileall app

cd ../frontend
npm test -- --runInBand
npm run type-check
npm run build
```

依赖外部模型或知识资料的场景，需要在本地配置相应提供商和测试数据。

## 文档

- [部署指南](backend/docs/DEPLOYMENT.md)
- [API 参考](backend/docs/API_REFERENCE.md)
- [Chat API 与流式事件](backend/docs/chat_api.md)
- [关键算法](backend/docs/KEY_ALGORITHMS.md)
- [贡献指南](CONTRIBUTING.md)
- [安全策略](SECURITY.md)
- [前端上游说明](frontend/UPSTREAM.md)

## 分支策略

`master` 是唯一长期分支和发布入口。功能开发使用短生命周期 topic branch，通过 Pull Request 合并，并在合并前完成相关检查。

## 许可证与致谢

前端包含基于 [RAGFlow](https://github.com/infiniflow/ragflow) 改造的组件。再分发前请阅读 [`frontend/UPSTREAM.md`](frontend/UPSTREAM.md) 与 [`frontend/LICENSE.ragflow`](frontend/LICENSE.ragflow)。当前仓库未声明额外的项目级许可证。
