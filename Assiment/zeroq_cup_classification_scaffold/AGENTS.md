# AGENTS.md

## Project goal
Prepare and audit a binary classification dataset for cup quality inspection:
- defective
- non_defective

Then train a transfer-learning classifier.

## Preferred stack
- Python 3.11+
- Albumentations for offline file-based augmentation
- CleanVision for image quality auditing
- FiftyOne for visual review
- Ultralytics YOLO26 classification for main training
- scikit-learn baseline for quick checks

## Rules
1. Preserve the original files under `data/raw/`.
2. Never augment directly into `data/raw/`.
3. Keep `data/processed/test/` free from synthetic near-duplicates.
4. Split by source image group whenever possible to avoid leakage.
5. Prefer small, reviewable changes.
6. Run lint-like sanity checks by executing the scripts before proposing broad edits.

## Useful commands
```bash
python scripts/augment_dataset.py --config configs/augment_offline.yaml --input-root data/raw --output-root data/interim/augmented
python scripts/split_dataset.py --input-root data/interim/augmented --output-root data/processed --train 0.70 --val 0.15 --test 0.15
python scripts/audit_dataset.py --data-root data/processed
python scripts/train_sklearn_baseline.py --data-root data/processed
bash scripts/train_yolo26_cls.sh data/processed yolo26n-cls.pt 224 80
```

## Review checklist
- Did augmentation preserve label semantics?
- Is the class count balanced enough?
- Are train/val/test source groups separated?
- Are there duplicates or near-duplicates across splits?
- Are the test images real and representative?
