import pickle
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.preprocessing import (
    FEATURE_COLUMNS,
    clean_sensor_data,
    create_feature_matrix,
    create_target_vector,
    generate_synthetic_failure_data,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "model.pkl"
SCALER_PATH = ROOT_DIR / "scaler.pkl"
CSV_PATH = DATA_DIR / "synthetic_data.csv"


def train():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic dataset...")
    df = generate_synthetic_failure_data(n_samples=8200, random_state=42)
    df = clean_sensor_data(df)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved synthetic dataset to {CSV_PATH}")

    X = create_feature_matrix(df)
    y = create_target_vector(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training RandomForestClassifier...")
    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
    )
    classifier.fit(X_train_scaled, y_train)

    y_pred = classifier.predict(X_test_scaled)
    y_proba = classifier.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["Healthy", "Fail"])
    matrix = confusion_matrix(y_test, y_pred)

    print("\nModel evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Classification report:\n", report)
    print("Confusion matrix:\n", matrix)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(classifier, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved trained model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")

    # Optional visual summary saved to root for quick inspection.
    plt.figure(figsize=(6, 4))
    plt.title("Failure Rate by Sensor Feature")
    plt.bar(["temperature", "rpm", "load", "vibration"], df[FEATURE_COLUMNS].mean(), color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.ylabel("Average sensor reading")
    plt.tight_layout()
    summary_plot_path = ROOT_DIR / "feature_summary.png"
    plt.savefig(summary_plot_path)
    print(f"Saved simple feature summary chart to {summary_plot_path}")


if __name__ == "__main__":
    train()
