# Deployment guide

## Requirements

- Python 3.13+
- uv
- Node.js 20+ and npm
- Optional: PostgreSQL for multiple application instances

## Local development

```bash
cp .env.example .env

cd backend
uv sync --dev

cd ../frontend
npm ci
cd ..

./run_dev.sh
```

The default frontend and backend ports are `5175` and `8002`.

## Backend

```bash
cd backend
uv sync
uv run uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8002 \
  --env-file ../.env
```

Check the service after startup:

```bash
curl http://127.0.0.1:8002/api/health
curl http://127.0.0.1:8002/api/readiness
```

## Frontend

```bash
cd frontend
npm ci
npm run build
```

Serve `frontend/dist` as static files and proxy `/api` and `/mcp` to the
FastAPI service. Keep the browser and API on the same HTTPS origin when
possible.

## Configuration

Start from `.env.example`. Leave optional values blank until the corresponding
provider or feature is enabled. Store live credentials in the deployment
platform's secret manager.

The default database is a local SQLite file under `backend/data`. Use
`AKA_DATABASE_URL` with PostgreSQL for multiple instances. Every instance that
reads encrypted model configuration must share the same
`AKA_MODEL_SECRET_KEY`.

## Knowledge data

The repository does not include environment-specific manuals, uploads, runtime
databases, or generated indexes. Import documents and build the index after
deploying:

```bash
cd backend
uv run python scripts/import_v6_knowledge.py
uv run python scripts/build_index_bundle.py --dataset-id v6-manuals --data-dir ./data
uv run python scripts/build_vector_map.py --dataset-id v6-manuals --data-dir ./data
```

## Production checklist

- Terminate TLS at a trusted proxy or gateway.
- Require authentication for public and administrative endpoints.
- Apply request-body and upload-size limits.
- Run external connectors with least-privilege identities.
- Keep logs and traces free of credentials and define a retention period.
- Back up the database, object data, and active index manifest together.
- Run backend tests and the frontend type check/build before rollout.
