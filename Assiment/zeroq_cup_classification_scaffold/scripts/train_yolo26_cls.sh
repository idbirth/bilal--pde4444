#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/processed}"
MODEL="${2:-yolo26s-cls.pt}"
IMGSZ="${3:-224}"
EPOCHS="${4:-80}"

python - <<PY
from ultralytics import YOLO
model = YOLO("${MODEL}")
model.train(data="${DATA_ROOT}", imgsz=${IMGSZ}, epochs=${EPOCHS}, batch=16, workers=4, patience=15)
PY
