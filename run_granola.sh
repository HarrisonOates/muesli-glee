#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -f .env ]]; then
  echo "Missing $PROJECT_DIR/.env" >&2
  echo "Add GLEE_API_KEY=glee_... before launching Granola." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

if [[ -z "${GLEE_API_KEY:-}" ]]; then
  echo "GLEE_API_KEY is not set in $PROJECT_DIR/.env" >&2
  exit 1
fi

if [[ ! -x .venv/bin/muesli-glee ]]; then
  echo "Project environment is missing; run:" >&2
  echo "  python3 -m venv .venv" >&2
  echo "  .venv/bin/python -m pip install -e '.[dev]'" >&2
  exit 1
fi

CONCURRENCY="${GLEE_CONCURRENCY:-4}"

echo "Launching Granola with MUESLI (concurrency=$CONCURRENCY)."
echo "Private game telemetry: $PROJECT_DIR/.local/glee-games.sqlite3"
echo "Press Ctrl-C once to stop queueing and let in-flight games drain."

exec .venv/bin/muesli-glee --concurrency "$CONCURRENCY" "$@"
