"""
Baseline model: Logistic Regression with original features.
Modify this file to improve val_accuracy.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from prepare import load_data, evaluate_accuracy

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

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
