"""
Experiment 2: Random Forest classifier
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from prepare import load_data, evaluate_accuracy

def build_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
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
