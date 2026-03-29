# ZEROQ Cup Defect Classification Scaffold

This repo is a starter scaffold for preparing a **binary image classification dataset** for an assessment that requires a final **PASS/FAIL** decision from a camera image.

Default quality check in this scaffold:
- `defective`  -> **FAIL**
- `non_defective` -> **PASS**

This default keeps the task aligned to a single, clearly verifiable rule: **surface defect present or not present**.

## Assessment alignment

This scaffold is designed to match a brief that asks for:
- a fixed inspection area
- one image per object
- one ML decision per image: **PASS** or **FAIL**
- one clearly defined quality rule
- a working demo on **5 new samples**
- a short report with problem definition, model/training design, quantitative results, and failure cases
- code, data, README, and commits over time

See:
- `docs/assessment_alignment.md`
- `docs/report_template.md`
- `docs/demo_test_sheet.csv`

## Choose exactly one quality rule

Do **not** mix multiple inspection rules in the same submission unless your lecturer explicitly approves it.

Good options are:
1. **Defective vs non-defective**
2. Correct vs incorrect orientation
3. Correct vs incorrect size or shape

For this cup project, the safest path is usually:
- **Problem**: detect visible defect on a single cup image
- **Ground truth**: manually verified label from human inspection
- **Output**: PASS if no visible defect, FAIL if visible defect

## Recommended approach

Use **Albumentations** for dataset preparation and saving augmented files to disk.
Use **YOLO26 classification** as the main model.
Use the scikit-learn baseline only as a quick sanity check.
Use a **single binary class rule** throughout the pipeline.

## Why this layout

YOLO classification expects:

```text
data/processed/
├── train/
│   ├── defective/
│   └── non_defective/
├── val/
│   ├── defective/
│   └── non_defective/
└── test/
    ├── defective/
    └── non_defective/
```

At presentation time, map predictions to engineering decisions:
- `defective` -> **FAIL**
- `non_defective` -> **PASS**

## Suggested raw capture targets

Start with at least:
- 40-80 real `defective` images
- 40-80 real `non_defective` images

Then grow to roughly:
- **200-300 images/class** after offline augmentation
- total **400-600 images** across both classes

## Capture rules for assessment consistency

- Use a **fixed inspection area**.
- Keep camera distance and angle as constant as possible.
- Use consistent lighting and background.
- Keep exactly **one object** centered in the frame.
- Record the ground-truth label before augmentation.
- Keep the final `test/` split as realistic and as untouched as possible.
- Reserve at least **5 genuinely new samples** for the demo.

## Class-specific augmentation intent

### Defective cups
Increase variation in:
- defect position
- lighting
- mild rotation
- mild scale shift
- focus / blur
- contrast

Avoid strong transforms that erase or hide the defect.

### Non-defective cups
Increase variation in:
- orientation
- rotation
- small viewpoint change
- brightness/white balance
- small crop / zoom

Avoid transforms that create fake scratches or unrealistic artifacts.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

### 1) Put original images here

```text
data/raw/defective/
data/raw/non_defective/
```

### 2) Generate augmented images

```bash
python scripts/augment_dataset.py \
  --config configs/augment_offline.yaml \
  --input-root data/raw \
  --output-root data/interim/augmented
```

### 3) Clean duplicates before splitting

```bash
python scripts/cleanup_presplit_duplicates.py \
  --input-root data/interim/augmented \
  --output-root data/interim/cleaned
```

This removes:
- exact duplicates across each class
- near-duplicate variants within each source group by default

It also writes a CSV manifest at `data/interim/cleaned/cleanup_report.csv`.

### 4) Split into train/val/test folders

```bash
python scripts/split_dataset.py \
  --input-root data/interim/cleaned \
  --output-root data/processed \
  --train 0.70 --val 0.15 --test 0.15
```

### 5) Audit the images for duplicates / blur / exposure issues

```bash
python scripts/audit_dataset.py --data-root data/processed
```

### 6) Optional: launch FiftyOne for visual dataset review

```bash
python scripts/launch_fiftyone.py --data-root data/processed
```

### 7) Train YOLO26 classification

```bash
bash scripts/train_yolo26_cls.sh data/processed yolo26n-cls.pt 224 80
```

For more capacity, try:
- `yolo26s-cls.pt`
- `yolo26m-cls.pt`

I would **not** start with `yolo26x-cls.pt` for a 400-600 image dataset.

### 8) Train a scikit-learn baseline

```bash
python scripts/train_sklearn_baseline.py --data-root data/processed
```

### 9) Export quantitative results for the report

```bash
python scripts/evaluate_classification.py \
  --data-root data/processed \
  --predictions runs/classify/train/weights/best.pt \
  --split test
```

### 10) Run a demo prediction on new samples

```bash
python scripts/infer_yolo26_cls.py \
  --model runs/classify/train/weights/best.pt \
  --input demo_samples
```

## Practical notes

- Keep `test/` as real images only. Do **not** fill test with synthetic near-duplicates.
- If you create many variants from one source image, split at the **source-group level**, not randomly by file, to avoid leakage.
- Keep augmentation stronger in training than in validation/test.
- Review bad predictions in FiftyOne after every training run.
- In the final report and demo, always show the engineering output as **PASS/FAIL**, even if internal folder names remain `defective` and `non_defective`.

## Suggested submission structure

```text
README.md
data/
scripts/
docs/
└── report_template.md
```

## Next extension

If binary classification works but you later need **where** the defect is, switch the dataset to:
- object detection, or
- anomaly localization / segmentation.
