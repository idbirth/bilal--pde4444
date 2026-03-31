# Week 8 README

This folder contains the Week 8 data-cleaning and feature-engineering workflow converted from `Week 8.ipynb` into a standalone script:

- `process_energy_data.py`

## Goal
Prepare `Energy_Data.csv` for modeling by:
- cleaning missing/invalid values,
- removing very sparse columns,
- reducing outliers,
- engineering transformed and categorical features,
- reducing collinearity,
- splitting scored data into train/test files.

## Input Data
- `Energy_Data.csv` (raw benchmark data, shape: **11746 x 60**)

## What Was Done To The Data (and Why)
1. Replaced `"Not Available"` with `NaN`.
- Why: standard missing-value token so pandas/numpy can process fields consistently.

2. Converted energy/area/water/score-like columns to numeric.
- Why: ensures calculations (stats, logs, filtering) are mathematically valid.

3. Dropped columns with more than 50% missing values.
- Why: very sparse columns add noise and weak signal for downstream modeling.

4. Renamed `ENERGY STAR Score` to `score`.
- Why: standard target column naming for modeling scripts.

5. Removed extreme outliers in `Site EUI (kBtu/ft²)` using a 3*IQR range.
- Why: prevents extreme values from dominating patterns and model fit.

6. Built engineered feature table:
- kept numeric columns,
- added log-transformed versions (`log_*`) for positive-valued numeric columns,
- one-hot encoded categorical columns (`Borough`, `Largest Property Use Type`).
- Why: improves model compatibility with mixed numeric/categorical data and skewed distributions.

7. Removed collinear features at correlation threshold `0.6` + explicit high-collinearity drops from notebook logic.
- Why: reduces redundancy/multicollinearity and stabilizes feature space.

8. Removed columns that became all-NaN after transformations.
- Why: these columns contain no usable information.

9. Split processed dataset into:
- rows with known `score` (for supervised training),
- rows without `score` (for future inference/analysis),
- train/test split of scored rows (70/30).
- Why: mirrors notebook ML-prep pipeline for reproducible experimentation.

## Generated Outputs
- `processed_energy_data.csv` (**11319 x 65**) -> processed master table (includes `score`, may contain null score rows)
- `no_score.csv` (**1858 x 65**) -> processed rows with missing target
- `training_features.csv` (**6622 x 64**)
- `testing_features.csv` (**2839 x 64**)
- `training_labels.csv` (**6622 x 1**)
- `testing_labels.csv` (**2839 x 1**)

## How To Run
From project root (`PDE4444`):

```bash
source .venv/bin/activate
python Week8/process_energy_data.py
```

## Tracking Folder
A history/log folder is maintained at:
- `Week8/hisory/`

This stores run logs and change summaries so future agents can trace what was done.

## Week 8 and Week 9 Integration
Week 9 work was added as a **follow-up stage to Week 8**, so the flow is now:

1. Week 8 prepares and cleans data (`process_energy_data.py` and generated CSV outputs).
2. Week 9 uses a structured script pipeline for model experiments and reporting.
3. Results are stored in one place for reproducibility and handoff.

### Integration folder
- `Week8/Follow_UP_Week9/`
- `Week8/Follow_UP_Week9/scripts/`
- `Week8/Follow_UP_Week9/output/`

### Integration purpose
- Keep Week 9 automation next to Week 8 outputs, so preprocessing and modeling are linked.
- Standardize experiment steps (split, train, evaluate, inference) as scripts.
- Keep logs/reports/predictions in a single output tree for easy tracking.

### Scripts included
1. `Week8/Follow_UP_Week9/scripts/split_dataset.py`
- Prepares train/validation/test data partitions.

2. `Week8/Follow_UP_Week9/scripts/train_yolo_classify.py`
- Runs classification training as the Week 9 model step.

3. `Week8/Follow_UP_Week9/scripts/evaluate_yolo_classify.py`
- Runs evaluation and writes reports/metrics files.

4. `Week8/Follow_UP_Week9/scripts/infer_yolo_classify.py`
- Runs inference and exports prediction results.

5. `Week8/Follow_UP_Week9/scripts/run_week9_pipeline.sh`
- Runs the end-to-end Week 9 pipeline in order.

### Output policy
All Week 9 follow-up outputs are centralized under:
- `Week8/Follow_UP_Week9/output/`

This includes run artifacts, reports, inference outputs, and pipeline logs.
