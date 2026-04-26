"""
Experiment 3: sklearn HistGradientBoostingClassifier (XGBoost-like, no libomp needed)
"""

from sklearn.ensemble import HistGradientBoostingClassifier

from prepare import load_data, evaluate_accuracy

def build_model():
    return HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
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
