#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -f .env.buckwheat ]]; then
  echo "Missing $PROJECT_DIR/.env.buckwheat" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env.buckwheat
set +a

if [[ -z "${GLEE_API_KEY:-}" ]]; then
  echo "GLEE_API_KEY is not set in $PROJECT_DIR/.env.buckwheat" >&2
  exit 1
fi

if [[ ! -x .venv/bin/muesli-glee ]]; then
  echo "Project environment is missing; install the project dependencies first." >&2
  exit 1
fi

CONCURRENCY="${GLEE_BUCKWHEAT_CONCURRENCY:-4}"
DATABASE="$PROJECT_DIR/.local/buckwheat-games.sqlite3"

echo "Launching Buckwheat challenger (concurrency=$CONCURRENCY)."
echo "Private Buckwheat telemetry: $DATABASE"
echo "Press Ctrl-C once to stop queueing and let in-flight games drain."

exec .venv/bin/muesli-glee \
  --profile buckwheat \
  --telemetry-db "$DATABASE" \
  --concurrency "$CONCURRENCY" \
  "$@"
