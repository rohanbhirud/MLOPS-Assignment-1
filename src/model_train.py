"""
Feature engineering and model training for the Heart Disease classifier.

Usage:
    python -m src.model_train --data data/raw/heart.csv --models-dir models --reports-dir reports
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from .data_utils import CleanConfig, clean_dataframe, infer_column_types, load_raw_csv


def build_preprocessor(cat_cols, num_cols) -> ColumnTransformer:
    """Assemble preprocessing: scale numeric features and one-hot encode categoricals."""
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )
    return preprocessor


def cross_validate_model(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    cv_splits: int = 5,
) -> Dict:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall": make_scorer(recall_score, average="weighted", zero_division=0),
        "roc_auc": make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )
    metrics = {f"{m}_mean": float(np.mean(scores)) for m, scores in results.items() if m.startswith("test_")}
    return {"name": name, "metrics": metrics, "estimator": pipeline}


def _sanitize_params(params: Dict) -> Dict:
    """Ensure params are JSON/MLflow friendly."""
    safe_params = {}
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe_params[k] = v
        else:
            safe_params[k] = str(v)
    return safe_params


def train_models(
    data_path: str, models_dir: str, reports_dir: str, cv_splits: int = 5
) -> None:
    models_path = pathlib.Path(models_dir)
    reports_path = pathlib.Path(reports_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_csv(data_path)
    clean_df = clean_dataframe(raw_df, CleanConfig())
    target_col = CleanConfig().target_column
    # Convert multiclass label (0-4) to binary presence/absence for clarity.
    y = (clean_df[target_col] > 0).astype(int)
    X = clean_df.drop(columns=[target_col])

    cat_cols, num_cols = infer_column_types(clean_df, target_col)
    preprocessor = build_preprocessor(cat_cols, num_cols)

    candidates = [
        ("log_reg", LogisticRegression(max_iter=200, class_weight="balanced")),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ]

    reports = []
    best_estimator = None
    for name, model in candidates:
        result = cross_validate_model(name, model, X, y, preprocessor, cv_splits=cv_splits)
        reports.append(result)
        print(f"{name} metrics: {result['metrics']}")
        # Log each model run to MLflow.
        with mlflow.start_run(run_name=f"{name}_cv"):
            mlflow.set_tag("stage", "model_selection")
            mlflow.log_params(_sanitize_params(model.get_params()))
            mlflow.log_metric("cv_splits", cv_splits)
            mlflow.log_metrics(result["metrics"])

    # Select best by ROC-AUC mean.
    best = max(reports, key=lambda r: r["metrics"].get("roc_auc_mean", 0))
    best_estimator = best["estimator"]
    # Drop estimator before persisting metrics to keep JSON clean.
    for r in reports:
        r.pop("estimator", None)

    best_model_path = models_path / f"{best['name']}_pipeline.joblib"
    joblib.dump(best_estimator, best_model_path)
    print(f"Saved best model ({best['name']}) to {best_model_path.resolve()}")

    metrics_path = reports_path / "metrics.json"
    json.dump(reports, metrics_path.open("w"), indent=2)
    print(f"Saved cross-validation metrics to {metrics_path.resolve()}")

    # Log the best model and artifacts to MLflow.
    with mlflow.start_run(run_name="best_model"):
        mlflow.set_tag("stage", "packaging")
        mlflow.log_param("selected_model", best["name"])
        mlflow.log_params(
            {
                "model_path": str(best_model_path),
                "model_size_bytes": best_model_path.stat().st_size,
                "cv_splits": cv_splits,
            }
        )
        mlflow.log_metrics(best["metrics"])
        mlflow.sklearn.log_model(best_estimator, artifact_path="model")
        mlflow.log_artifact(best_model_path, artifact_path="artifacts")
        if reports_path.exists():
            mlflow.log_artifacts(reports_path, artifact_path="reports")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifiers with feature engineering")
    parser.add_argument("--data", default="data/raw/heart.csv", help="Path to cleaned CSV input")
    parser.add_argument("--models-dir", default="models", help="Where to store trained pipelines")
    parser.add_argument("--reports-dir", default="reports", help="Where to store metrics")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of CV folds")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    train_models(args.data, args.models_dir, args.reports_dir, args.cv_splits)


if __name__ == "__main__":
    main()
