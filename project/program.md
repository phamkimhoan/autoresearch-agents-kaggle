# autoresearch — Titanic Kaggle Competition

## Competition

**Task**: Binary classification — predict whether a passenger survived the Titanic disaster.
**Metric**: Accuracy (higher is better). Maximize `val_accuracy`.
**Data**: 891 training samples, 418 test samples.
**Baseline**: Logistic Regression → val_accuracy = 0.798883

## Setup

1. **Branch**: `autoresearch/titanic-apr26`
2. **In-scope files**:
   - `prepare.py` — fixed: data loading, feature engineering, `evaluate_accuracy`. DO NOT MODIFY.
   - `train.py` — the file you modify. Model, hyperparameters, preprocessing.
3. **Results**: `results.tsv` (tab-separated, not committed to git)

## Experimentation

**What you CAN do** (modify `train.py` only):
- Change the classifier (RandomForest, XGBoost, LightGBM, SVM, etc.)
- Add/tune hyperparameters
- Add cross-validation for model selection
- Ensemble multiple classifiers
- Add feature selection or dimensionality reduction in the pipeline

**What you CANNOT do**:
- Modify `prepare.py` — the feature engineering and evaluation are fixed.
- Install new packages beyond what's in the venv.

**Goal**: Maximize `val_accuracy` printed by `uv run train.py`.

## Output format

The script prints:

```
---
train_accuracy: 0.818820
val_accuracy:   0.798883
```

Extract key metric:
```
grep "^val_accuracy:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	val_accuracy	status	description
```

1. git commit hash (short, 7 chars)
2. val_accuracy (e.g. 0.812345)
3. status: `keep`, `discard`, or `crash`
4. short text description

Example:
```
commit	val_accuracy	status	description
a1b2c3d	0.798883	keep	baseline logistic regression
b2c3d4e	0.820000	keep	random forest 100 trees
c3d4e5f	0.785000	discard	svm rbf kernel
```

## Experiment loop

LOOP for 20 experiments total (including baseline):

1. Check git state (current branch/commit)
2. Edit `train.py` with a new idea
3. `git commit`
4. Run: `python train.py > run.log 2>&1`
5. Read result: `grep "^val_accuracy:" run.log`
6. If empty → crash. Read `tail -50 run.log` and fix or discard.
7. Log to `results.tsv`
8. If val_accuracy improved → keep commit (advance branch)
9. If val_accuracy equal or worse → `git reset --hard HEAD~1` (discard)

**Never stop** between experiments. Run all 20 without pausing.

**Crashes**: Fix trivial bugs and re-run. Discard fundamentally broken ideas.

## Ideas to try (in order of promise)

1. Baseline: Logistic Regression (done)
2. Random Forest (n_estimators=100-300, tune max_depth)
3. Gradient Boosting (XGBoost or LightGBM)
4. SVM with RBF kernel
5. Ensemble: VotingClassifier (LR + RF + XGB)
6. XGBoost with tuned hyperparameters
7. LightGBM with tuned hyperparameters
8. Extra Trees classifier
9. Stacking ensemble
10. RandomForest + feature selection
11. GradientBoosting (sklearn)
12. Bagging + tuned base estimator
13. AdaBoost
14. HistGradientBoosting
15. RandomForest with class_weight='balanced'
16. XGBoost + scale_pos_weight
17. LightGBM with DART boosting
18. Stacking with meta-learner
19. VotingClassifier (hard vs soft voting)
20. Best model with re-tuned hyperparameters
