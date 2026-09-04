# API reference

The FastAPI application publishes its generated OpenAPI documentation at
`/docs` and its schema at `/openapi.json`.

## Base URL

Local development uses:

```text
http://127.0.0.1:8002
```

All examples below assume this base URL. A production deployment should add
HTTPS and authentication at the application, gateway, or reverse-proxy layer.

## Health and runtime

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/readiness` | Runtime, feature, model, and index readiness |
| `GET` | `/api/metrics` | In-process metrics snapshot |

## Chat

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Return a complete answer and execution metadata |
| `POST` | `/api/chat/stream` | Stream staged answer events |

Example:

```bash
curl -X POST http://127.0.0.1:8002/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I clean the filter?","images":[],"session_id":"demo"}'
```

The response includes an answer, route, citations, related assets,
verification details, and a trace. See [chat_api.md](chat_api.md) for the full
request shape and streaming behavior.

## Knowledge and files

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/files` | Upload a source file |
| `GET` | `/api/files` | List uploaded files |
| `GET` | `/api/files/{file_id}/content` | Download or preview a file |
| `GET` | `/api/assets/{asset_id}` | Retrieve a knowledge asset |
| `POST` | `/api/datasets` | Create a dataset |
| `GET` | `/api/datasets` | List datasets |
| `GET` | `/api/datasets/{dataset_id}` | Read dataset details |
| `PATCH` | `/api/datasets/{dataset_id}` | Update a dataset |
| `POST` | `/api/datasets/{dataset_id}/documents` | Add a file to a dataset |
| `POST` | `/api/documents/{document_id}/parse` | Parse and chunk a document |
| `POST` | `/api/datasets/{dataset_id}/publish` | Publish a dataset version |

## Retrieval and indexes

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/retrieval/test` | Run an explained retrieval query |
| `GET` | `/api/retrieval/profiles` | List retrieval profiles |
| `POST` | `/api/datasets/{dataset_id}/index-builds` | Start an index build |
| `GET` | `/api/index-builds/{build_id}` | Inspect an index build |
| `GET` | `/api/datasets/{dataset_id}/index-manifest` | Read the active manifest |
| `GET` | `/api/index-runtime` | Inspect the loaded offline index |
| `GET` | `/api/datasets/{dataset_id}/image-chunks` | List image chunks |
| `GET` | `/api/datasets/{dataset_id}/vector-map` | Read vector-map data |

## Models

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/providers` | List supported providers |
| `GET` | `/api/models` | List configured models |
| `POST` | `/api/models` | Register a model |
| `PATCH` | `/api/models/{model_id}` | Update a model |
| `POST` | `/api/models/{model_id}/default` | Set the default for its kind |
| `POST` | `/api/models/{model_id}/test` | Test connectivity |
| `DELETE` | `/api/models/{model_id}` | Delete a model configuration |

Pass credentials only in a local request to your own deployment. Model list
responses expose configuration state and a masked hint, never the plaintext
credential.

## Conversations, evaluation, and traces

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/conversations` | List conversations |
| `POST` | `/api/conversations` | Create a conversation |
| `GET` | `/api/conversations/{conversation_id}` | Read a conversation |
| `PATCH` | `/api/conversations/{conversation_id}` | Rename a conversation |
| `DELETE` | `/api/conversations/{conversation_id}` | Delete a conversation |
| `GET` | `/api/evaluations/runs` | List evaluation runs |
| `GET` | `/api/traces` | List traces |
| `GET` | `/api/traces/{request_id}` | Read one trace |

## MCP and capabilities

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/agents` | Agent catalog |
| `GET` | `/api/skills` | Skill catalog |
| `GET` | `/api/tools` | Tool catalog |
| `POST` | `/api/mcp/tools/knowledge.search` | REST wrapper for knowledge search |
| `POST` | `/mcp` | Streamable HTTP MCP endpoint |

For the authoritative schema of a specific build, use the generated OpenAPI
document rather than duplicating request fields in client code.
