"""
Generate submission using best model (exp15) retrained on full training data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from prepare import engineer_features, FEATURE_COLS, TRAIN_PATH, TEST_PATH

def build_model():
    lr = Pipeline([("scaler", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))])
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=4,
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
    hgb = HistGradientBoostingClassifier(max_iter=400, max_depth=5, learning_rate=0.03,
                                          min_samples_leaf=15, l2_regularization=0.5, random_state=42)
    mlp1 = Pipeline([("scaler", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                           alpha=0.01, random_state=42, early_stopping=True))])
    mlp2 = Pipeline([("scaler", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                           alpha=0.01, random_state=123, early_stopping=True))])
    return VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("hgb", hgb), ("mlp1", mlp1), ("mlp2", mlp2)],
        voting="soft",
    )

if __name__ == "__main__":
    # Train on full training data (no val split)
    train_df = pd.read_csv(TRAIN_PATH)
    train_df = engineer_features(train_df)
    X_all = train_df[FEATURE_COLS].values
    y_all = train_df["Survived"].values

    test_df = pd.read_csv(TEST_PATH)
    passenger_ids = test_df["PassengerId"].values
    test_df = engineer_features(test_df)
    X_test = test_df[FEATURE_COLS].values

    model = build_model()
    model.fit(X_all, y_all)

    preds = model.predict(X_test)
    submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": preds})
    submission.to_csv("submission.csv", index=False)
    print(f"Saved submission.csv ({len(submission)} rows)")
    print(submission["Survived"].value_counts().to_string())
