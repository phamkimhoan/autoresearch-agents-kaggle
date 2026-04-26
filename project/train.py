"""
Experiment 4: VotingClassifier (LR + RF + HistGB) — soft voting
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from prepare import load_data, evaluate_accuracy

def build_model():
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=4,
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
    hgb = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=42)
    return VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("hgb", hgb)],
        voting="soft",
    )

def main():
    X_train, X_val, y_train, y_val = load_data()

    model = build_model()
    model.fit(X_train, y_train)

    val_accuracy = evaluate_accuracy(model, X_val, y_val)
    train_accuracy = evaluate_accuracy(model, X_train, y_train)

    print("---")
    print(f"train_accuracy: {train_accuracy:.6f}")
    print(f"val_accuracy:   {val_accuracy:.6f}")

    return model, val_accuracy

if __name__ == "__main__":
    main()
