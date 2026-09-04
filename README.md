<p align="center">
  <img src="frontend/public/logo.svg" width="64" alt="AKA Sentinel logo">
</p>

<h1 align="center">AKA Sentinel</h1>

<p align="center">
  Evidence-first multimodal customer service for electronics documentation,
  troubleshooting, and after-sales support.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white" alt="Python 3.13+"></a>
  <a href="https://nodejs.org/"><img src="https://img.shields.io/badge/Node.js-20%2B-339933?logo=node.js&logoColor=white" alt="Node.js 20+"></a>
  <a href="backend/"><img src="https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI backend"></a>
  <a href="frontend/"><img src="https://img.shields.io/badge/frontend-React%20%2B%20TypeScript-61DAFB?logo=react&logoColor=111827" alt="React and TypeScript frontend"></a>
</p>

AKA Sentinel turns a customer question, optionally with product photos, into an
evidence-linked answer. It combines intent routing, multimodal understanding,
hybrid retrieval, conversation memory, verification, and an operations
workspace in one self-hosted repository.

## Features

- **Multimodal input:** accept text and up to three images, with OCR and VLM
  context available to the retrieval pipeline.
- **Evidence-first RAG:** combine Parent/Child chunks, Image Chunks, BM25,
  dense retrieval, reciprocal-rank fusion, and optional reranking.
- **Dynamic routing:** separate technical, customer-service, mixed, general,
  clarification, and safety-sensitive requests.
- **Answer verification:** check claims, numeric values, procedural order,
  citations, image observations, and unsupported promises before returning an
  answer.
- **Conversation memory:** retain turn history, structured slots,
  clarifications, and rolling summaries without treating previous answers as
  authoritative product evidence.
- **Versioned indexes:** build checksummed offline bundles with incremental
  reuse and explicit activation or rollback.
- **Operations workspace:** inspect conversations, knowledge assets, retrieval
  experiments, model health, traces, evaluations, skills, and MCP tools.

## How it works

```text
Client (Web UI / REST / SSE)
              |
              v
       FastAPI application
              |
              v
        Orchestrator
   +----------+----------+
   |          |          |
   v          v          v
Router   Multimodal   Memory
   |          |          |
   +------+---+------+---+
          |          |
          v          v
   Knowledge/RAG  Customer Service
          |          |
          +------+---+
                 |
                 v
              Verifier
                 |
                 v
       Answer + citations + trace
```

The knowledge layer keeps relationships between a document, its parent
sections, child chunks, neighboring chunks, and related images. A typical query
passes through:

```text
intent / product / focus extraction
  -> lexical and dense candidate retrieval
  -> RRF fusion and optional reranking
  -> child-to-parent context expansion
  -> evidence coverage and risk gates
  -> grounded answer with citations
```

The main implementation is in
[`backend/app/knowledge`](backend/app/knowledge),
[`backend/app/runtime`](backend/app/runtime), and
[`backend/app/agents`](backend/app/agents).

## Quick start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Node.js 20+ and npm
- Optional: PostgreSQL for a shared multi-instance deployment

### Install

```bash
git clone https://github.com/logcjj/Electronics-Agent-Customer-Service.git
cd Electronics-Agent-Customer-Service
cp .env.example .env

cd backend
uv sync --dev

cd ../frontend
npm ci
cd ..
```

The committed environment template contains placeholders only. Add provider
credentials to your local `.env` or secret manager, never to source files.

### Run

```bash
./run_dev.sh
```

Default local endpoints:

| Service | URL |
|---|---|
| Web UI | <http://127.0.0.1:5175> |
| API | <http://127.0.0.1:8002> |
| OpenAPI | <http://127.0.0.1:8002/docs> |
| MCP | <http://127.0.0.1:8002/mcp> |

To run the services separately:

```bash
cd backend
uv run uvicorn app.main:app --host 127.0.0.1 --port 8002 --env-file ../.env
```

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5175
```

## Configuration

Copy `.env.example` to `.env` and set only the values required by your
deployment.

| Group | Example variables | Purpose |
|---|---|---|
| Runtime | `AKA_ROLLOUT_MODE`, `AKA_OFFLINE_INDEX_MODE` | Select answer and index behavior |
| Storage | `AKA_DATABASE_URL`, `AKA_MODEL_SECRET_KEY` | Configure SQLite/PostgreSQL and encrypted model credentials |
| Models | `OPENAI_*`, `DASHSCOPE_*`, `ALIYUN_*` | Register LLM, VLM, embedding, reranking, and OCR providers |
| Memory | `AKA_SESSION_MEMORY`, `AKA_LAYERED_MEMORY` | Configure short-term and layered conversation context |
| Retrieval | `AKA_IMAGE_CHUNK_RETRIEVAL`, `AKA_CAPTION_EMBEDDING` | Enable text and image evidence paths |
| Ports | `BACKEND_PORT`, `FRONTEND_PORT` | Override local service ports |

Model credentials are encrypted before persistence in the runtime database and
redacted from API responses, traces, metrics, and manifests.

## Knowledge base and indexing

Runtime databases, uploaded documents, image assets, vectors, and generated
indexes are stored under `backend/data` and excluded from Git. Import your own
source documents before building an index:

```bash
cd backend
uv run python scripts/import_v6_knowledge.py
uv run python scripts/build_index_bundle.py \
  --dataset-id v6-manuals \
  --data-dir ./data
uv run python scripts/build_vector_map.py \
  --dataset-id v6-manuals \
  --data-dir ./data
```

The import script accepts an external source root through `V6_ROOT`. You can
also upload documents and manage datasets through the API or web workspace.

## API overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/readiness` | Runtime, model, and index readiness |
| `POST` | `/api/chat` | Answer with route, citations, verification, and trace |
| `POST` | `/api/chat/stream` | Stream staged answer events |
| `GET` | `/api/conversations` | List persisted conversations |
| `POST` | `/api/files` | Upload a source file |
| `POST` | `/api/datasets` | Create a knowledge dataset |
| `POST` | `/api/retrieval/test` | Inspect retrieval candidates and scores |
| `GET` | `/api/traces` | Review execution traces |
| `POST` | `/mcp` | Streamable HTTP MCP endpoint |

Minimal request:

```bash
curl -X POST http://127.0.0.1:8002/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"The device shows an error code. What should I check?","images":[],"session_id":"demo"}'
```

See the [API reference](backend/docs/API_REFERENCE.md) and
[chat API guide](backend/docs/chat_api.md) for details.

## Repository layout

```text
.
+-- backend/
|   +-- app/                 FastAPI, agents, retrieval, models, and storage
|   +-- docs/                API, deployment, and algorithm documentation
|   +-- scripts/             Import, indexing, evaluation, and setup tools
|   `-- tests/               Backend unit and integration tests
+-- frontend/
|   +-- src/aka/             AKA workspace pages, components, and API client
|   `-- src/                 Shared RAGFlow-based UI foundation
+-- CONTRIBUTING.md
+-- SECURITY.md
+-- run_dev.sh
+-- .env.example
`-- README.md
```

## Tests

```bash
cd backend
uv run pytest -q
python -m compileall app

cd ../frontend
npm test -- --runInBand
npm run type-check
npm run build
```

External-provider and knowledge-dependent scenarios require the corresponding
local configuration or repository fixtures.

## Documentation

- [Deployment guide](backend/docs/DEPLOYMENT.md)
- [API reference](backend/docs/API_REFERENCE.md)
- [Chat API and streaming](backend/docs/chat_api.md)
- [Key algorithms](backend/docs/KEY_ALGORITHMS.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Frontend upstream notice](frontend/UPSTREAM.md)

## Branch policy

`master` is the canonical branch. Use short-lived topic branches and merge them
through pull requests after the relevant checks pass.

## License and acknowledgements

The frontend includes components adapted from RAGFlow. Review
[`frontend/UPSTREAM.md`](frontend/UPSTREAM.md) and
[`frontend/LICENSE.ragflow`](frontend/LICENSE.ragflow) before redistribution.
No separate project-wide license has been declared in this repository.
