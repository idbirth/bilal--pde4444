# ZEROQ Cup Defect Classification Scaffold

This repo is a starter scaffold for preparing a **binary image classification dataset**:
- `defective`
- `non_defective`

It is optimized for:
1. **offline dataset generation** with Albumentations
2. **dataset auditing** with CleanVision
3. **dataset inspection** with FiftyOne
4. **transfer-learning training** with Ultralytics YOLO26 classification
5. a lightweight **scikit-learn baseline**
6. smooth use inside **Codex** via `.codex/config.toml` and `AGENTS.md`

## Recommended approach

Use **Albumentations** for dataset preparation and saving augmented files to disk.
Use **YOLO26 classification** as the main model.
Use the scikit-learn baseline only as a quick sanity check.

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

## Suggested raw capture targets

Start with at least:
- 40-80 real `defective` images
- 40-80 real `non_defective` images

Then grow to roughly:
- **200-300 images/class** after offline augmentation
- total **400-600 images** across both classes

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

### 3) Split into train/val/test folders

```bash
python scripts/split_dataset.py \
  --input-root data/interim/augmented \
  --output-root data/processed \
  --train 0.70 --val 0.15 --test 0.15
```

### 4) Audit the images for duplicates / blur / exposure issues

```bash
python scripts/audit_dataset.py --data-root data/processed
```

### 5) Optional: launch FiftyOne for visual dataset review

```bash
python scripts/launch_fiftyone.py --data-root data/processed
```

### 6) Train YOLO26 classification

```bash
bash scripts/train_yolo26_cls.sh data/processed yolo26n-cls.pt 224 80
```

For more capacity, try:
- `yolo26s-cls.pt`
- `yolo26m-cls.pt`

I would **not** start with `yolo26x-cls.pt` for a 400-600 image dataset.

### 7) Train a scikit-learn baseline

```bash
python scripts/train_sklearn_baseline.py --data-root data/processed
```

## Practical notes

- Keep `test/` as real images only. Do **not** fill test with synthetic near-duplicates.
- If you create many variants from one source image, split at the **source-group level**, not randomly by file, to avoid leakage.
- Keep augmentation stronger in training than in validation/test.
- Review bad predictions in FiftyOne after every training run.

## Next extension

If binary classification works but you later need **where** the defect is, switch the dataset to:
- object detection, or
- anomaly localization / segmentation.
