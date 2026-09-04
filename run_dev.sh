#!/bin/zsh
set -e

ROOT_DIR="${0:A:h}"
BACKEND_PORT="${BACKEND_PORT:-8002}"
FRONTEND_PORT="${FRONTEND_PORT:-5175}"
AKA_ROLLOUT_MODE="${AKA_ROLLOUT_MODE:-agent_first}"

cleanup() {
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$ROOT_DIR/backend"
AKA_ROLLOUT_MODE="$AKA_ROLLOUT_MODE" uv run python -m uvicorn app.main:app --host 127.0.0.1 --port "$BACKEND_PORT" &
BACKEND_PID=$!

cd "$ROOT_DIR/frontend"
npm run dev -- --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

wait "$BACKEND_PID" "$FRONTEND_PID"
