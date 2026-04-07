#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${JOBS_DIR:-/app/data/jobs}"

if [ "${AUTO_DOWNLOAD_ON_START:-false}" = "true" ]; then
  /app/scripts/download_model.sh
fi

exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
