#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

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

export DATA_ROOT MODEL IMGSZ EPOCHS

python - <<'PY'
import os
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer

data_root = Path(os.environ["DATA_ROOT"])
model_path = Path(os.environ["MODEL"])
imgsz = int(os.environ["IMGSZ"])
epochs = int(os.environ["EPOCHS"])


class NoAugClassificationTrainer(ClassificationTrainer):
    """Force the training split onto the non-augmented classification transform path."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=mode)

if not data_root.exists():
    raise SystemExit(f"Dataset root not found: {data_root}")
if not model_path.exists():
    raise SystemExit(f"Model checkpoint not found: {model_path}")

model = YOLO(str(model_path))
if model.task != "classify":
    raise SystemExit(
        f"Expected a classification checkpoint, but got task={model.task!r} from {model_path}"
    )

print(f"[INFO] training classification model from {model_path}")
print(f"[INFO] dataset root: {data_root}")
print(f"[INFO] imgsz={imgsz}, epochs={epochs}")
print("[INFO] Ultralytics online augmentation: disabled")
model.train(
    trainer=NoAugClassificationTrainer,
    data=str(data_root),
    imgsz=imgsz,
    epochs=epochs,
    batch=16,
    workers=4,
    patience=15,
)
PY
