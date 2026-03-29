# AGENTS.md

## Project goal
Prepare and audit a binary classification dataset for cup quality inspection.

Default class mapping:
- `defective` -> FAIL
- `non_defective` -> PASS

The submission must stay aligned to a **single** quality rule and end in a **PASS/FAIL** engineering decision.

## Preferred stack
- Python 3.11+
- Albumentations for offline file-based augmentation
- CleanVision for image quality auditing
- FiftyOne for visual review
- Ultralytics YOLO26 classification for main training
- scikit-learn baseline for quick checks

## Assessment constraints
1. Keep the inspection setup fixed: same region, stable framing, stable lighting where possible.
2. Choose one quality check only unless explicitly told otherwise.
3. Preserve a clear mapping from class label to PASS/FAIL.
4. Keep at least 5 unseen samples for the demonstration.
5. Quantitative results must be exportable for the short report.
6. Document failure cases and limitations.
7. Keep the README ready for submission review.

## Rules
1. Preserve the original files under `data/raw/`.
2. Never augment directly into `data/raw/`.
3. Keep `data/processed/test/` free from synthetic near-duplicates.
4. Split by source image group whenever possible to avoid leakage.
5. Prefer small, reviewable changes.
6. Run sanity checks by executing the scripts before proposing broad edits.
7. Avoid adding augmentations that change the true class.

## Useful commands
```bash
python scripts/augment_dataset.py --config configs/augment_offline.yaml --input-root data/raw --output-root data/interim/augmented
python scripts/cleanup_presplit_duplicates.py --input-root data/interim/augmented --output-root data/interim/cleaned
python scripts/balance_cleaned_dataset.py --input-root data/interim/cleaned --output-root data/interim/balanced --reference-class non_defective
python scripts/split_dataset.py --input-root data/interim/balanced --output-root data/processed --train 0.70 --val 0.15 --test 0.15
python scripts/audit_dataset.py --data-root data/processed
python scripts/train_sklearn_baseline.py --data-root data/processed
bash scripts/train_yolo26_cls.sh data/processed yolo26s-cls.pt 512 80
python scripts/evaluate_classification.py --data-root data/processed --predictions runs/classify/train/weights/best.pt --split test
python scripts/infer_yolo26_cls.py --model runs/classify/train/weights/best.pt --input demo_samples
```

## Review checklist
- Did augmentation preserve label semantics?
- Is the task still one clearly defined PASS/FAIL rule?
- Is the class count balanced enough?
- Are train/val/test source groups separated?
- Are there duplicates or near-duplicates across splits?
- Are the test images real and representative?
- Are there at least 5 unseen demo samples ready?
