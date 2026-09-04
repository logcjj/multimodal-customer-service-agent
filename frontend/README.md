# AKA Sentinel frontend

The frontend is a React and TypeScript operations workspace. AKA-specific
pages and components live under `src/aka`; shared UI foundations include code
adapted from RAGFlow.

## Development

```bash
npm ci
npm run dev
```

## Checks

```bash
npm test -- --runInBand
npm run type-check
npm run build
```

For the default full-stack setup, run `../run_dev.sh` from the repository root.
