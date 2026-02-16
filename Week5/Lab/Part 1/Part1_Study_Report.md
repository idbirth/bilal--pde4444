# Week 5 Part 1 Study Report

## Scope
This report analyzes:
- `Week5/Lab/Part 1/Task1.py` (SVM)
- `Week5/Lab/Part 1/Task2.py` (MLP)

Both scripts were executed and results below are from those runs.

## Dataset and Setup
- Dataset: Iris (150 samples, 4 numerical features, 3 classes)
- Features: `sepal length`, `sepal width`, `petal length`, `petal width`
- Target classes: `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`
- Split in both scripts: 80% train (120), 20% test (30), stratified, `random_state=42`

## Task1.py (SVM) Parameter Study

### Parameters used
- `kernel`: compared `poly`, `sigmoid`, `linear`
- `C=1`
- `decision_function_shape='ovr'`
- `random_state=42`

### Why these parameters matter
- `kernel` controls decision boundary shape:
- `linear` uses a straight hyperplane
- `poly` uses polynomial curved boundaries
- `sigmoid` behaves like a neural activation boundary and is often sensitive
- `C` controls regularization:
- small `C` = softer margin (more tolerance)
- large `C` = harder margin (less tolerance, possible overfit)
- `ovr` (one-vs-rest) handles multi-class classification by training one classifier per class.

### Observed results
- `poly` accuracy: **0.9667**
- `sigmoid` accuracy: **0.1000**
- `linear` accuracy: **1.0000**

Interpretation:
- `linear` performed best on this split, suggesting Iris is close to linearly separable in this feature space.
- `sigmoid` failed badly for this setup; this is common when kernel and hyperparameters are not well aligned.

### GridSearchCV tuning (SVM)
Search space:
- `C`: `[0.1, 1, 10, 100]`
- `gamma`: `[1, 0.1, 0.01, 0.001]` for `rbf` and `sigmoid`
- `kernel`: `rbf`, `sigmoid`, `linear`
- `cv=2`

Best found:
- Best params: `{'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}`
- Best CV score: **0.9833**
- Train accuracy: **0.9833**
- Test accuracy: **0.9333**

Interpretation:
- The tuned model optimized 2-fold CV score, not the single test split score.
- Because `cv=2` is small, ranking can be unstable. That is why tuned test accuracy can be lower than the earlier `linear` test result.

## Task2.py (MLP) Parameter Study

### Baseline MLP parameters
- `hidden_layer_sizes=(10,)` (one hidden layer with 10 neurons)
- `activation='logistic'` (sigmoid activation)
- `solver='sgd'` (stochastic gradient descent)
- `learning_rate='constant'`
- `learning_rate_init=0.001`
- `max_iter=1000`
- `random_state=42`

### Why these parameters matter
- `hidden_layer_sizes=(10,)` controls model capacity. Too small can underfit, too large can overfit.
- `activation='logistic'` can work, but may train slower and saturate compared with `relu` on many problems.
- `solver='sgd'` is sensitive to learning rate and can converge slowly.
- `learning_rate_init=0.001` is conservative; may be too low for fast convergence.
- `max_iter=1000` caps training epochs. If optimization is slow, model may still underfit.

### Observed baseline results
- Accuracy: **0.7333**
- Confusion matrix:
  - `[[10, 0, 0], [0, 2, 8], [0, 0, 10]]`

Interpretation:
- `Iris-setosa` and `Iris-virginica` were strong.
- Major weakness on `Iris-versicolor` recall (`2/10`), mostly confused as `Iris-virginica`.
- This indicates the baseline optimizer/learning-rate setup is not finding a good boundary for the overlapping middle class.

### GridSearchCV tuning (MLP)
Search space:
- `hidden_layer_sizes`: `(10,)`, `(20,)`, `(20,10)`
- `activation`: `logistic`, `tanh`, `relu`
- `solver`: `sgd`, `adam`
- `learning_rate_init`: `0.001`, `0.01`
- `max_iter`: `1000`
- `cv=3`, `n_jobs=1`

Best found:
- Best params: `{'activation': 'logistic', 'hidden_layer_sizes': (10,), 'learning_rate_init': 0.01, 'max_iter': 1000, 'solver': 'sgd'}`
- Best CV score: **0.9750**
- Tuned test accuracy: **1.0000**

Interpretation:
- The main winning change was increasing learning rate from `0.001` to `0.01`.
- Same architecture/activation/solver remained best in this search.
- Baseline was likely under-trained at the smaller step size.

## Key Takeaways (Part 1)
1. Model family and hyperparameters strongly affect performance, even on a small dataset.
2. For SVM here, `linear` performed best on the held-out split, while CV tuning picked `rbf`.
3. For MLP here, training dynamics (especially learning rate) were the biggest driver of improvement.
4. CV best model and single-split best model are not always identical.

## Practical Recommendations
1. Increase CV folds to 5 (or use repeated stratified CV) for more stable tuning.
2. Add feature scaling (`StandardScaler`) before MLP and SVM tuning for more robust optimization.
3. Report both CV mean and test score together, as done here, to avoid over-interpreting one metric.

## Generated artifacts
- `Week5/Lab/Part 1/svm_kernel_accuracy.png`
- `Week5/Lab/Part 1/Part1_Study_Report.md` (this report)
