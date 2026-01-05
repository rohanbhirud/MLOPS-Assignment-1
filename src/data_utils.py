"""
Data loading, cleaning, and preprocessing helpers for the Heart Disease dataset.

These utilities keep transformations reusable between EDA and model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


# Column names for the common heart.csv dataset
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


@dataclass
class CleanConfig:
    numeric_impute_strategy: str = "median"
    categorical_impute_value: str = "missing"
    target_column: str = "target"


def load_raw_csv(path: str) -> pd.DataFrame:
    """Load the CSV and enforce known column names."""
    df = pd.read_csv(path)
    if list(df.columns) != COLUMNS:
        # Align to expected schema for downstream steps.
        df.columns = COLUMNS[: len(df.columns)]
    return df


def clean_dataframe(df: pd.DataFrame, config: CleanConfig) -> pd.DataFrame:
    """Handle missing values and basic type coercion."""
    df = df.copy()

    # Replace common missing markers and strip whitespace.
    df.replace(["?", "NA", "NaN", "nan", ""], np.nan, inplace=True)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    # Identify columns by simple heuristics; adjust if schema changes.
    cat_cols, num_cols = infer_column_types(df, config.target_column)

    # Impute missing values.
    if num_cols:
        num_impute_val = (
            df[num_cols].median() if config.numeric_impute_strategy == "median" else df[num_cols].mean()
        )
        df[num_cols] = df[num_cols].fillna(num_impute_val)
    if cat_cols:
        df[cat_cols] = df[cat_cols].fillna(config.categorical_impute_value)

    # Coerce features to numeric where possible to avoid downstream errors.
    feature_cols = [c for c in df.columns if c != config.target_column]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Fill new NaNs (from coercion) with median or most frequent? 
        # Median is safer for now.
        df[col] = df[col].fillna(df[col].median())

    return df


def infer_column_types(
    df: pd.DataFrame, target_col: str
) -> Tuple[Iterable[str], Iterable[str]]:
    """Infer categorical vs numeric columns, excluding the target."""
    cat_cols = []
    num_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == "object" or df[col].nunique() < 10:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return cat_cols, num_cols
