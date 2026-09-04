# Contributing to AKA Sentinel

## Development workflow

1. Create a short-lived topic branch from `master`.
2. Keep each change focused and add tests for behavior changes.
3. Run the relevant backend and frontend checks.
4. Open a pull request with the motivation, implementation notes,
   configuration changes, and verification results.

## Setup

```bash
cp .env.example .env
cd backend && uv sync --dev
cd ../frontend && npm ci
```

Use local or test credentials only. Never commit `.env`, provider keys,
database passwords, callback secrets, private datasets, or runtime output.

## Checks

```bash
cd backend
uv run pytest -q
python -m compileall app

cd ../frontend
npm test -- --runInBand
npm run type-check
npm run build
```

## Pull requests

- Explain user-visible behavior and API contract changes.
- Document new environment variables in `.env.example` with blank values.
- Include migrations for persistence changes.
- Keep generated data, build output, caches, and local model assets out of Git.
- Note rollback steps for changes to indexes, storage, or deployment behavior.
