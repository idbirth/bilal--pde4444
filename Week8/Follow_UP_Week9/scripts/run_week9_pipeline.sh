#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FOLLOWUP_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd -- "${FOLLOWUP_ROOT}/../.." && pwd)"

OUTPUT_DIR="${FOLLOWUP_ROOT}/output"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}/reports" "${OUTPUT_DIR}/inference"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/week9_pipeline_${TIMESTAMP}.log"

{
  echo "[INFO] project root: ${PROJECT_ROOT}"
  echo "[INFO] output dir: ${OUTPUT_DIR}"

  python "${SCRIPT_DIR}/split_dataset.py"

  python "${SCRIPT_DIR}/train_yolo_classify.py" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "week9_train_${TIMESTAMP}"

  BEST_WEIGHTS="$(find "${OUTPUT_DIR}" -type f -path '*/weights/best.pt' | sort | tail -n 1)"
  if [[ -z "${BEST_WEIGHTS}" ]]; then
    echo "[ERROR] best.pt not found under ${OUTPUT_DIR}"
    exit 1
  fi
  echo "[INFO] using best weights: ${BEST_WEIGHTS}"

  python "${SCRIPT_DIR}/evaluate_yolo_classify.py" \
    --weights "${BEST_WEIGHTS}" \
    --split test \
    --output-dir "${OUTPUT_DIR}/reports"

  echo "[DONE] week9 pipeline complete"
} 2>&1 | tee "${LOG_FILE}"

echo "[DONE] pipeline log: ${LOG_FILE}"

