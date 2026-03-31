# Follow_UP_Week9

This folder contains a reusable Week 9 script pack based on your existing classification workflow.

## Structure
- `scripts/` -> runnable scripts for split, train, evaluate, and inference
- `output/` -> all run artifacts, reports, logs, and prediction CSV files

## Scripts
1. `scripts/split_dataset.py`
- Group-aware split (`train/val/test`) to reduce leakage across related images.

2. `scripts/train_yolo_classify.py`
- Trains YOLO classification model and writes run artifacts to `output/`.

3. `scripts/evaluate_yolo_classify.py`
- Evaluates model on selected split and writes:
  - `classification_report_<split>.txt`
  - `confusion_matrix_<split>.csv`
  - `predictions_<split>.csv`
  into `output/reports/`.

4. `scripts/infer_yolo_classify.py`
- Runs inference on one image or a folder, writes prediction CSV to `output/inference/`.

5. `scripts/run_week9_pipeline.sh`
- Convenience runner for split -> train -> evaluate.
- Writes console logs to `output/logs/`.

## Default Data Paths
Scripts default to your existing scaffold paths under:
- `Assiment/zeroq_cup_classification_scaffold/...`

You can override all paths via CLI arguments.

## Quick Start
From `PDE4444` root:

```bash
source .venv/bin/activate
bash Week8/Follow_UP_Week9/scripts/run_week9_pipeline.sh
```

