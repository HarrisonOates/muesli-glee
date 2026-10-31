#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -f .env.millet ]]; then
  echo "Missing $PROJECT_DIR/.env.millet" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env.millet
set +a

if [[ -z "${GLEE_API_KEY:-}" ]]; then
  echo "GLEE_API_KEY is not set in $PROJECT_DIR/.env.millet" >&2
  exit 1
fi

if [[ ! -x .venv/bin/muesli-glee ]]; then
  echo "Project environment is missing; install the project dependencies first." >&2
  exit 1
fi

CONCURRENCY="${GLEE_MILLET_CONCURRENCY:-4}"
DATABASE="$PROJECT_DIR/.local/millet-games.sqlite3"

echo "Launching Millet challenger (concurrency=$CONCURRENCY)."
echo "Private Millet telemetry: $DATABASE"
echo "Press Ctrl-C once to stop queueing and let in-flight games drain."

exec .venv/bin/muesli-glee \
  --profile millet \
  --telemetry-db "$DATABASE" \
  --concurrency "$CONCURRENCY" \
  "$@"
