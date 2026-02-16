# Week 5 README

This README documents **Week 5 Lab Part 1 and Part 2**.

## Files
- `Week5/Lab/Part 1/Task1.py` (SVM)
- `Week5/Lab/Part 1/Task2.py` (MLP)
- `Week5/Lab/Part 1/Part1_Study_Report.md` (analysis report)
- `Week5/Lab/Part 2/Part2_CNN.py` (direct CNN conversion from notebook/html)
- `Week5/Lab/Part 2/Part2_CNN_tuned.py` (tuning pipeline for CNN)

## Part 1 Summary
- `Task1.py`:
SVM on Iris, compares `poly`/`sigmoid`/`linear`, plots kernel accuracy, then applies `GridSearchCV`.
- `Task2.py`:
MLP on Iris with baseline parameters, then `GridSearchCV` tuning and evaluation.

## Part 2 Summary
- `Part2_CNN.py`:
Straight conversion of the lab CNN flow using dataset `Week5/Lab/Part 2/D1`.
- `Part2_CNN_tuned.py`:
Runs predefined tuning iterations to improve validation accuracy and logs each run.

## Part 2 Tuning History
The tuned script writes artifacts in:
- `Week5/Lab/Part 2/tuing_history`

Important structure:
- `Week5/Lab/Part 2/tuing_history/changing_values.json`
- `Week5/Lab/Part 2/tuing_history/search_space.json`
- `Week5/Lab/Part 2/tuing_history/run/iteration_01/config.json`
- `Week5/Lab/Part 2/tuing_history/run/iteration_01/metrics.json`
- `Week5/Lab/Part 2/tuing_history/run/iteration_01/history.csv` (created after training)
- `Week5/Lab/Part 2/tuing_history/run/iteration_01/model_summary.txt` (created after training)
- `Week5/Lab/Part 2/tuing_history/run/iteration_01/val_accuracy_curve.png` (created after training)
- `Week5/Lab/Part 2/tuing_history/leaderboard.csv` (created after training all iterations)
- `Week5/Lab/Part 2/tuing_history/best_run.json` (created after training all iterations)
- `Week5/Lab/Part 2/tuing_history/val_accuracy_per_iteration.png` (created after training all iterations)

## Run Commands
From repo root:

```bash
python "Week5/Lab/Part 1/Task1.py"
python "Week5/Lab/Part 1/Task2.py"
python "Week5/Lab/Part 2/Part2_CNN.py"
```

Initialize Part 2 tuning folders and iteration configs without training:

```bash
python "Week5/Lab/Part 2/Part2_CNN_tuned.py" --init-only --max-iterations 5
```

Run actual tuning (requires TensorFlow installed):

```bash
python "Week5/Lab/Part 2/Part2_CNN_tuned.py" --max-iterations 5 --epochs 20
```

## Dependencies
- `pandas`
- `scikit-learn`
- `matplotlib`
- `tensorflow` (required for Part 2 training scripts)
- `Pillow`

Install:

```bash
pip install pandas scikit-learn matplotlib tensorflow pillow
```
