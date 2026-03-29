#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${1:-${PROJECT_ROOT}/data/processed}"
MODEL="${2:-${PROJECT_ROOT}/yolo26s-cls.pt}"
IMGSZ="${3:-512}"
EPOCHS="${4:-80}"
AMP="${AMP:-False}"
MODEL_INPUT="${MODEL}"

if [[ ! -f "${MODEL}" ]]; then
  MODEL="${PROJECT_ROOT}/${MODEL}"
fi

if [[ "${MODEL_INPUT##*/}" != *-cls.pt ]]; then
  echo "Expected a classification checkpoint ending with -cls.pt, got: ${MODEL}" >&2
  exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
  if [[ "${MODEL_INPUT}" == */* ]]; then
    echo "Model file not found: ${MODEL_INPUT}" >&2
    exit 1
  fi
  MODEL="${MODEL_INPUT}"
fi

case "${AMP,,}" in
  1|true|yes|on) PY_AMP="True" ;;
  0|false|no|off) PY_AMP="False" ;;
  *)
    echo "Unsupported AMP value: ${AMP}. Use true/false." >&2
    exit 1
    ;;
esac

if [[ -x "${PROJECT_ROOT}/../.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/../.venv/bin/python"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

echo "Training classification model: ${MODEL}"
echo "Using python: ${PYTHON_BIN}"

"${PYTHON_BIN}" - <<PY
from ultralytics import YOLO
model = YOLO("${MODEL}", task="classify")
if model.task != "classify":
    raise SystemExit(f"Expected classify task, got: {model.task}")

model.train(
    data="${DATA_ROOT}",
    imgsz=${IMGSZ},
    epochs=${EPOCHS},
    batch=16,
    workers=4,
    patience=15,
    amp=${PY_AMP},
)
PY
