from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "customer_payment_risk_dataset.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "payment_risk_model.joblib"
METRICS_PATH = REPORT_DIR / "model_metrics.json"

TARGET = "late_payment"
# Only use information expected to be known at prediction time.
# Do NOT use payment_date or outstanding_amount: these may reveal the outcome
# or depend on events that occur after the prediction point.
NUMERIC_FEATURES = [
    "invoice_amount",
    "payment_terms",
    "previous_late_payments",
    "invoice_month",
    "invoice_quarter",
]
CATEGORICAL_FEATURES = []
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nPut the CSV in data/raw/ "
            "and name it customer_payment_risk_dataset.csv."
        )

    df = pd.read_csv(path)
    required = [
        "invoice_date", "invoice_amount", "payment_terms",
        "previous_late_payments", TARGET
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"], dayfirst=True, errors="coerce"
    )
    df["invoice_amount"] = pd.to_numeric(df["invoice_amount"], errors="coerce")
    df["payment_terms"] = pd.to_numeric(df["payment_terms"], errors="coerce")
    df["previous_late_payments"] = pd.to_numeric(
        df["previous_late_payments"], errors="coerce"
    )
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET, "invoice_date"]).copy()
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    df["invoice_month"] = df["invoice_date"].dt.month
    df["invoice_quarter"] = df["invoice_date"].dt.quarter

    if df[TARGET].nunique() != 2:
        raise ValueError("Target late_payment must contain both 0 and 1.")
    if len(df) < 30:
        raise ValueError("Dataset is too small for a meaningful train/test split.")
    return df


def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ], remainder="drop")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_and_prepare_data(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
            random_state=42
        ),
    }

    results = []
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        }
        results.append(metrics)
        fitted[name] = pipeline
        print(f"{name}: {metrics}")

    # Select by ROC-AUC; for real deployment, also review business costs and recall.
    best_result = max(results, key=lambda row: row["roc_auc"])
    best_name = best_result["model"]
    best_model = fitted[best_name]
    pred = best_model.predict(X_test)

    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET,
        "threshold": 0.50,
    }, MODEL_PATH)

    report = {
        "dataset_rows_used": int(len(df)),
        "dataset_columns": list(pd.read_csv(DATA_PATH, nrows=1).columns),
        "target_distribution": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "selection_metric": "ROC-AUC",
        "best_model": best_name,
        "best_model_metrics": best_result,
        "all_model_metrics": results,
        "classification_report": classification_report(
            y_test, pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "feature_columns_used": FEATURES,
        "warning": (
            "Offline evaluation only. Validate on a later time period and real "
            "company data before production use."
        ),
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nBest model: {best_name}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
