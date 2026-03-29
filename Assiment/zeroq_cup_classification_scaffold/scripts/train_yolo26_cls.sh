#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

pick_python() {
  if [[ -x "${PROJECT_ROOT}/../.venv/bin/python" ]]; then
    printf '%s\n' "${PROJECT_ROOT}/../.venv/bin/python"
    return
  fi
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    printf '%s\n' "${PROJECT_ROOT}/.venv/bin/python"
    return
  fi
  printf '%s\n' "python"
}

resolve_project_path() {
  local input_path="$1"
  if [[ "${input_path}" = /* ]]; then
    printf '%s\n' "${input_path}"
    return
  fi
  if [[ -e "${PROJECT_ROOT}/${input_path}" ]]; then
    printf '%s\n' "${PROJECT_ROOT}/${input_path}"
    return
  fi
  printf '%s\n' "${input_path}"
}

DATA_ROOT="$(resolve_project_path "${1:-data/processed}")"
MODEL="$(resolve_project_path "${2:-yolo26s-cls.pt}")"
IMGSZ="${3:-512}"
EPOCHS="${4:-80}"
MODEL_INPUT="${2:-yolo26s-cls.pt}"
PYTHON_BIN="$(pick_python)"
RUNS_ROOT="${PROJECT_ROOT}/runs/classify"
MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
AMP="${AMP:-false}"

if [[ "${MODEL_INPUT##*/}" != *-cls.pt ]]; then
  echo "Expected a classification checkpoint ending with -cls.pt, got: ${MODEL_INPUT}" >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Dataset root not found: ${DATA_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${MODEL}" && "${MODEL_INPUT}" == */* ]]; then
  echo "Model checkpoint not found: ${MODEL_INPUT}" >&2
  exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
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

mkdir -p "${RUNS_ROOT}" "${MPLCONFIGDIR}"

echo "[INFO] python: ${PYTHON_BIN}"
echo "[INFO] training classification model from ${MODEL}"
echo "[INFO] dataset root: ${DATA_ROOT}"
echo "[INFO] imgsz=${IMGSZ}, epochs=${EPOCHS}"
echo "[INFO] amp=${PY_AMP}"
echo "[INFO] Ultralytics online augmentation: disabled"
echo "[INFO] runs root: ${RUNS_ROOT}"
echo "[INFO] matplotlib config: ${MPLCONFIGDIR}"

export DATA_ROOT MODEL IMGSZ EPOCHS RUNS_ROOT MPLCONFIGDIR PY_AMP

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer

data_root = Path(os.environ["DATA_ROOT"])
model_path = Path(os.environ["MODEL"])
imgsz = int(os.environ["IMGSZ"])
epochs = int(os.environ["EPOCHS"])
runs_root = Path(os.environ["RUNS_ROOT"])
amp = os.environ["PY_AMP"] == "True"


class NoAugClassificationTrainer(ClassificationTrainer):
    """Force the training split onto the non-augmented classification transform path."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=mode)

model = YOLO(str(model_path), task="classify")
if model.task != "classify":
    raise SystemExit(
        f"Expected a classification checkpoint, but got task={model.task!r} from {model_path}"
    )

model.train(
    trainer=NoAugClassificationTrainer,
    data=str(data_root),
    imgsz=imgsz,
    epochs=epochs,
    batch=16,
    workers=4,
    patience=15,
    project=str(runs_root),
    amp=amp,
)
PY
