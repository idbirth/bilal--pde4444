# Short report template (2 pages)

## 1. Problem definition and ground truth
We built a visual quality inspection model for cups.
The task is binary classification:
- PASS: cup is non-defective
- FAIL: cup is defective

Ground truth was assigned by manual inspection under the fixed capture setup.

## 2. Model design and training
### Dataset
- Raw classes: `defective`, `non_defective`
- Train/val/test split: 70/15/15 by source group
- Augmentation used only for training data generation

### Model
- Main model: YOLO26 classification
- Baseline: HOG + Logistic Regression

### Training loop pseudo-code
```text
load train and validation datasets
initialize pretrained classification model
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        predictions = model(images)
        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with no_grad():
        run validation predictions
        compute validation metrics
save best model weights
```

## 3. Quantitative results
Report at minimum:
- accuracy
- precision
- recall
- F1-score
- confusion matrix

Also state how PASS/FAIL is mapped from class predictions.

## 4. Failure cases and limitations
Show a few misclassified examples and explain why they likely failed.
Typical causes:
- glare
- low contrast defect
- unusual cup pose
- background leakage
- insufficient real data diversity
