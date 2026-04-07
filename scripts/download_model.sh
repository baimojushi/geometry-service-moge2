#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Ruicheng/moge-2-vitb-normal}"
MODEL_DIR="${MODEL_DIR:-/models/moge2}"
READY_FILE="${MODEL_DIR}/READY"
LOCK_DIR="${MODEL_DIR}.lock"

mkdir -p "${MODEL_DIR}"
mkdir -p "$(dirname "${MODEL_DIR}")"

if [ -f "${READY_FILE}" ]; then
  echo "[download_model] model already prepared in ${MODEL_DIR}"
  exit 0
fi

if mkdir "${LOCK_DIR}" 2>/dev/null; then
  trap 'rmdir "${LOCK_DIR}" >/dev/null 2>&1 || true' EXIT
else
  echo "[download_model] waiting for other worker to finish download"
  for i in $(seq 1 180); do
    if [ -f "${READY_FILE}" ]; then
      echo "[download_model] model became ready"
      exit 0
    fi
    sleep 5
  done
  echo "[download_model] timeout waiting for READY file"
  exit 1
fi

TMP_DIR="${MODEL_DIR}.tmp"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

export HF_HOME="${HF_HOME:-/models/.hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/models/.hf/hub}"
export TORCH_HOME="${TORCH_HOME:-/models/.torch}"

python - <<'PY'
import os
from pathlib import Path
from moge.model.v2 import MoGeModel

model_name = os.environ["MODEL_NAME"]
model_dir = Path(os.environ["MODEL_DIR"])
tmp_dir = Path(str(model_dir) + ".tmp")

print(f"[download_model] downloading {model_name} into {tmp_dir}")
MoGeModel.from_pretrained(model_name)
print("[download_model] warmup download finished")
PY

cat > "${READY_FILE}" <<EOF
model_name=${MODEL_NAME}
prepared_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

echo "[download_model] ready"
