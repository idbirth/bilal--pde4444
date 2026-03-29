# Assessment alignment

This project is aligned to a brief that asks for a machine-learning quality inspection system with a **PASS/FAIL** result.

## Selected quality rule

**Visible cup defect present or absent**

- PASS = `non_defective`
- FAIL = `defective`

This keeps the problem definition clear, binary, and easy to verify.

## Pipeline mapping

1. Object is placed in a fixed inspection area
2. Camera captures one image
3. Model predicts `defective` or `non_defective`
4. System maps prediction to FAIL or PASS

## Deliverable mapping

### Demonstration
- Use `scripts/infer_yolo26_cls.py`
- Prepare 5 unseen samples
- Show image -> class -> PASS/FAIL

### Report
Include:
- Problem definition and ground truth
- Model design and training loop pseudo-code
- Quantitative results
- Failure cases and limitations

Use `docs/report_template.md` as a starting point.

### Code
Include:
- data folder structure
- scripts
- README
- commit history over time

## What to avoid

- Mixing defect detection and orientation in the same submission
- Using synthetic images in the final 5-sample demo without saying so
- Reporting only accuracy without confusion matrix or example failures
- Using test leakage from augmented siblings across splits
