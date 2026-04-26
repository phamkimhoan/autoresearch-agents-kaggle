"""
Data preparation and evaluation for Titanic Kaggle competition.
Fixed constants, feature engineering, train/val split, and evaluation.

DO NOT MODIFY — this is the fixed evaluation harness.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
VAL_SPLIT = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Feature engineering (fixed)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Title extraction from Name
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    rare_titles = df["Title"].value_counts()
    rare_titles = rare_titles[rare_titles < 10].index
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df["Title"] = df["Title"].map(title_map).fillna(5).astype(int)

    # Age: fill missing with median per Pclass/Sex group
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Fare: fill missing with median
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False, duplicates="drop")

    # Age bins
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, 120], labels=False)
    df["AgeBin"] = df["AgeBin"].fillna(2).astype(int)

    # Cabin: has cabin or not
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # Embarked: fill missing, encode
    df["Embarked"] = df["Embarked"].fillna("S")
    embarked_map = {"S": 0, "C": 1, "Q": 2}
    df["Embarked"] = df["Embarked"].map(embarked_map).fillna(0).astype(int)

    # Sex encode
    df["Sex"] = (df["Sex"] == "female").astype(int)

    return df


FEATURE_COLS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked", "Title", "FamilySize", "IsAlone", "FareBin", "AgeBin", "HasCabin"
]


def load_data():
    """Load and split data into train/val sets. Returns (X_train, X_val, y_train, y_val)."""
    df = pd.read_csv(TRAIN_PATH)
    df = engineer_features(df)
    X = df[FEATURE_COLS].values
    y = df["Survived"].values
    return train_test_split(X, y, test_size=VAL_SPLIT, random_state=RANDOM_STATE, stratify=y)


def load_test_data():
    """Load test data for submission. Returns (passenger_ids, X_test)."""
    df = pd.read_csv(TEST_PATH)
    passenger_ids = df["PassengerId"].values
    df = engineer_features(df)
    X_test = df[FEATURE_COLS].values
    return passenger_ids, X_test


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_accuracy(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Accuracy: fraction of correct predictions on validation set.
    Higher is better (maximize).
    """
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()
    print(f"Train size: {X_train.shape}, Val size: {X_val.shape}")
    print(f"Features: {FEATURE_COLS}")
    print(f"Survival rate (train): {y_train.mean():.3f}")
    print(f"Survival rate (val):   {y_val.mean():.3f}")
