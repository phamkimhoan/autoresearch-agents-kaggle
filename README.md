# Autoresearch Kaggle — Titanic Tutorial

A hands-on tutorial showing how to use **Claude Code + the autoresearch loop** to autonomously compete in a Kaggle competition — from raw data to a leaderboard submission — with zero manual model tuning.

---

## What This Is

This project adapts [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) pattern — originally designed for language model research — to Kaggle tabular ML competitions. The idea: Claude Code acts as an autonomous researcher that proposes experiments, runs them, keeps improvements, discards failures, and submits the best result.

**Competition**: [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview)  
**Metric**: Accuracy (higher is better)  
**Result**: 79.89% → **84.92%** validation accuracy, submitted to leaderboard

---

## Prerequisites

- [Claude Code](https://claude.ai/code) CLI
- [uv](https://github.com/astral-sh/uv) Python package manager
- Kaggle account with API key configured (`~/.kaggle/kaggle.json`)
- macOS / Linux
- `git`

> **Note:** XGBoost and LightGBM require `libomp` on macOS (`brew install libomp`). This tutorial uses sklearn alternatives that work without it.

---

## Repository Structure

```
autoresearch-agents-kaggle/
├── CLAUDE.md                   # Rules for Claude Code in this project
├── requirements.txt            # Python dependencies
├── original/                   # Karpathy's autoresearch repo (cloned, read-only)
├── project/
│   ├── prepare.py              # FIXED: data loading, feature engineering, evaluate_accuracy()
│   ├── train.py                # MODIFY THIS: model architecture (the experiment file)
│   ├── submit.py               # Generate Kaggle submission from best model
│   ├── program.md              # Instructions adapted for Titanic (Claude reads this)
│   ├── results.tsv             # Experiment log (not committed)
│   ├── run.log                 # Latest experiment output (not committed)
│   ├── submission.csv          # Final Kaggle submission (not committed)
│   └── data/
│       ├── train.csv
│       ├── test.csv
│       └── gender_submission.csv
└── .claude/
    └── settings.json           # Permissions (bypass all prompts in this repo)
```

---

## How to Run It Yourself

### Step 1 — Clone and install

```bash
git clone <this-repo>
cd autoresearch-agents-kaggle
uv venv .venv
uv pip install -r requirements.txt
```

### Step 2 — Configure Kaggle API

Make sure `~/.kaggle/kaggle.json` exists with your credentials. Test with:

```bash
.venv/bin/kaggle competitions list
```

### Step 3 — Launch Claude Code and run the skill

Open Claude Code in this directory:

```bash
claude
```

Then type:

```
/autoresearch-compete-kaggle competition_url=https://www.kaggle.com/competitions/titanic/overview metric=accuracy max_experiments=20 submit=true
```

Claude will handle everything from here — no further input needed.

---

## What Claude Does Automatically

### Phase 1: Setup
1. Checks git, kaggle CLI, and Python environment
2. Clones the original autoresearch repo to `original/`
3. Downloads competition data via `kaggle competitions download`
4. Reads competition structure to understand the problem

### Phase 2: Autoresearch scaffolding
1. Creates `project/prepare.py` — the **fixed evaluation harness** with:
   - Feature engineering (title extraction, age imputation, family size, fare bins, etc.)
   - `evaluate_accuracy(model, X_val, y_val)` — the locked metric function
   - `load_data()` — stratified 80/20 train/val split
2. Reads `original/program.md` and creates `project/program.md` adapted for Titanic
3. Creates `project/train.py` with the baseline model

### Phase 3: Baseline
- Trains **Logistic Regression** as baseline → **79.89% val accuracy**
- Commits baseline to branch `autoresearch/titanic-apr26`
- Initializes `results.tsv`

### Phase 4: Experiment loop (20 experiments)
For each experiment:
1. Modifies `train.py` with a new idea
2. `git commit`
3. Runs `python train.py > run.log 2>&1`
4. Reads `val_accuracy` from log
5. If improved → keep the commit, advance the branch
6. If worse/same → `git reset --hard HEAD~1`, log as discard
7. Records result in `results.tsv`

### Phase 5: Submission
- Retrains best model on **full training data** (no val split held out)
- Generates `submission.csv`
- Submits via `kaggle competitions submit`

---

## Experiment Results

| # | Commit | Val Accuracy | Status | What Was Tried |
|---|--------|-------------|--------|----------------|
| 1 | e6f78ea | 0.7989 | ✅ keep | Baseline: Logistic Regression |
| 2 | a3e744d | 0.8101 | ✅ keep | Random Forest (200 trees, max_depth=8) |
| 3 | 13e97c4 | crash | ❌ crash | XGBoost — missing libomp.dylib on macOS |
| 4 | dbedc28 | 0.8156 | ✅ keep | HistGradientBoosting (sklearn, no libomp) |
| 5 | 96bb063 | 0.8268 | ✅ keep | VotingClassifier: LR + RF + HistGB (soft) |
| 6 | d7fd371 | 0.8268 | ⏭ discard | HistGB tuned — same score as voting |
| 7 | fd993ba | 0.8101 | ⏭ discard | StackingClassifier RF+HistGB+SVC — worse |
| 8 | fcf0ed3 | 0.8156 | ⏭ discard | Voting + ExtraTrees, weighted — worse |
| 9 | fe9e1fa | 0.8045 | ⏭ discard | Voting SVC+RF+HistGB weighted — worse |
| 10 | 7fbab53 | 0.8045 | ⏭ discard | GridSearchCV on HistGB (learned best params) |
| 11 | b923f0e | 0.8324 | ✅ keep | Voting with GridSearch-tuned HistGB params |
| 12 | 59de36f | 0.8156 | ⏭ discard | Voting + GradientBoosting — worse |
| 13 | 9dcf31b | 0.8101 | ⏭ discard | Voting weights [1,2,4] — worse |
| 14 | ed0ea27 | 0.8436 | ✅ keep | **+MLP(128,64) to voting** — big jump |
| 15 | 3bf512d | 0.8436 | ⏭ discard | Deeper MLP (256,128,64) — same |
| 16 | 348ba39 | **0.8492** | ✅ **best** | **Two MLPs, different random seeds** |
| 17 | fd04cb6 | 0.8436 | ⏭ discard | Three MLPs — diminishing returns |
| 18 | 44c1580 | 0.8324 | ⏭ discard | Two HistGBs + two MLPs — worse |
| 19 | 923bb70 | 0.8380 | ⏭ discard | Up-weighted MLPs [1,2,2,3,3] — worse |
| 20 | 2fd776d | 0.8436 | ⏭ discard | Diverse MLP arch+alpha — worse |

**Best model**: Experiment 16 — `0.849162` accuracy

---

## Best Model Architecture

```python
VotingClassifier(voting="soft", estimators=[
    ("lr",   Pipeline([StandardScaler, LogisticRegression(C=1.0)])),
    ("rf",   RandomForestClassifier(n_estimators=300, max_depth=8)),
    ("hgb",  HistGradientBoostingClassifier(max_iter=400, lr=0.03, max_depth=5)),
    ("mlp1", Pipeline([StandardScaler, MLPClassifier((128,64), alpha=0.01, seed=42)])),
    ("mlp2", Pipeline([StandardScaler, MLPClassifier((128,64), alpha=0.01, seed=123)])),
])
```

**Why this works:**
- Soft voting averages probability estimates — better than hard majority vote
- Tree models (RF, HistGB) and neural networks (MLP) capture different patterns
- Two MLPs with different random seeds add diversity without adding complexity
- Equal weights — tuning weights hurt in every experiment tried

---

## Feature Engineering (in `prepare.py`)

All features are fixed — `prepare.py` is the locked evaluation harness.

| Feature | Description |
|---------|-------------|
| `Pclass` | Passenger class (1/2/3) |
| `Sex` | Encoded as 0/1 |
| `Age` | Filled with median per Pclass+Sex group |
| `SibSp`, `Parch` | Siblings/spouses and parents/children aboard |
| `Fare` | Ticket price (median-filled) |
| `Embarked` | Port encoded S=0, C=1, Q=2 |
| `Title` | Extracted from name (Mr/Miss/Mrs/Master/Rare) |
| `FamilySize` | SibSp + Parch + 1 |
| `IsAlone` | 1 if travelling alone |
| `FareBin` | Fare quartile (0–3) |
| `AgeBin` | Age bucket (child/teen/adult/middle/senior) |
| `HasCabin` | 1 if cabin number known |

---

## Key Lessons Learned

1. **Ensemble > single model** — every kept experiment was an ensemble improvement
2. **MLPs complement tree ensembles** — the biggest accuracy jump (+1.2%) came from adding a neural network to a tree-based voting classifier
3. **Seed diversity in MLPs** — two MLPs with different random seeds beat one, but three didn't help further
4. **Equal weights are hard to beat** — all attempts at custom voting weights made things worse
5. **GridSearch is a research tool, not a final model** — the CV-tuned HistGB alone scored lower than the ensemble, but its best hyperparameters were reused inside the voting ensemble
6. **Stacking < Voting for small datasets** — stacking introduced overfitting on 891 samples; soft voting was more robust
7. **XGBoost/LightGBM on macOS** — require `brew install libomp`; sklearn's `HistGradientBoostingClassifier` is a drop-in alternative

---

## Reproducing the Submission

```bash
# Activate venv
source .venv/bin/activate   # or use .venv/bin/python directly

# Generate predictions (retrains on full data)
python project/submit.py

# Submit to Kaggle
kaggle competitions submit -c titanic -f project/submission.csv \
  -m "VotingClassifier LR+RF+HistGB+MLP2seeds val_acc=0.8492"
```

---

## Customising for Another Competition

1. Change the `competition_url` and `metric` when invoking the skill
2. Claude will read the competition page, infer the metric, and adapt `prepare.py` accordingly
3. `train.py` is the only file that gets modified during experiments
4. `program.md` controls the experiment strategy — edit it to guide Claude toward domain-specific ideas

---

## Project Configuration

**`CLAUDE.md`** sets the ground rules for Claude in this repo:
- Use `uv` + `.venv` for all Python work
- Never touch system Python
- Work only in this directory

**`.claude/settings.json`** grants full auto-approval for all tool operations so the experiment loop runs without interruption:

```json
{
  "permissions": {
    "allow": ["Bash(*)", "Read(*)", "Write(*)", "Edit(*)", "Glob(*)", "Grep(*)", "WebFetch(*)", "WebSearch(*)"],
    "defaultMode": "bypassPermissions"
  }
}
```
