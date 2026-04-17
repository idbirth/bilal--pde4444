# Defect / Non-Defect Image Classification Stack

This package trains and compares three families of image classification models from the same two-folder dataset:

1. **scikit-learn baselines**
   - HOG + SVM
   - HOG + MLP
2. **TensorFlow / Keras baseline MLP**
   - Random Search hyperparameter tuning with Keras Tuner
3. **TensorFlow CNN activation sweep**
   - Compares `relu`, `elu`, `gelu`, `selu`, and `leaky_relu`
4. **Ultralytics YOLO26n-cls**
   - Image classification model, not bounding-box detection

## Raw input format

Your raw dataset can live anywhere. The scripts expect you to point to two folders:

- `Defect` or `Fail`
- `Non-Defect` or `Pass`

Example:

```text
raw_data/
├── defect_fail/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── non_defect_pass/
    ├── img101.jpg
    ├── img102.jpg
    └── ...
```

## Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Prepare a shared dataset split

This creates one dataset layout that works for scikit-learn, TensorFlow, and YOLO classification.

```bash
python prepare_dataset.py \
  --defect-dir /path/to/raw_data/defect_fail \
  --pass-dir /path/to/raw_data/non_defect_pass \
  --output-dir /path/to/processed_dataset \
  --img-size 224 224
```

Output layout:

```text
processed_dataset/
├── train/
│   ├── defect/
│   └── non_defect/
├── val/
│   ├── defect/
│   └── non_defect/
└── test/
    ├── defect/
    └── non_defect/
```

## Step 3: Run baselines

### A. scikit-learn HOG baselines

```bash
python train_sklearn_baseline.py \
  --data-dir /path/to/processed_dataset \
  --output-dir runs/sklearn_baseline
```

### B. TensorFlow/Keras MLP with Random Search

```bash
python train_keras_mlp_random_search.py \
  --data-dir /path/to/processed_dataset \
  --output-dir runs/keras_mlp \
  --max-trials 20 \
  --epochs 25
```

### C. CNN activation-function sweep

```bash
python train_cnn_activation_sweep.py \
  --data-dir /path/to/processed_dataset \
  --output-dir runs/cnn_activation_sweep \
  --epochs 20
```

### D. YOLO26n classification

```bash
python train_yolo26_cls.py \
  --data-dir /path/to/processed_dataset \
  --output-dir runs/yolo26_cls \
  --epochs 30 \
  --imgsz 224
```

## Step 4: Compare all results

```bash
python compare_results.py \
  --sklearn-dir runs/sklearn_baseline \
  --keras-dir runs/keras_mlp \
  --cnn-dir runs/cnn_activation_sweep \
  --yolo-dir runs/yolo26_cls \
  --output-file runs/model_comparison.csv
```

## Notes

- The Keras MLP is included because you asked for a baseline deep learning model.
- The scikit-learn script includes both **SVM** and **MLP**. If you literally meant **SPN**, that is a different model family and is not part of standard scikit-learn.
- YOLO26 is run in **classification mode**, which is appropriate for whole-image labels like `defect` vs `non_defect`.
- If your classes are imbalanced, the scripts compute class weights where supported.
- The scripts save metrics, confusion matrices, and model artifacts in their output folders.
