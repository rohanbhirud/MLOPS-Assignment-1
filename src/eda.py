"""
Run exploratory data analysis and save plots to disk.

Usage:
    python -m src.eda --data data/raw/heart.csv --out reports/figures
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_utils import CleanConfig, clean_dataframe, load_raw_csv


def plot_histograms(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    for ax, col in zip(axes, num_cols):
        sns.histplot(df[col], bins=20, kde=True, ax=ax)
        ax.set_title(col)
    for ax in axes[len(num_cols) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "histograms.png", dpi=200)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=200)
    plt.close()


def plot_class_balance(df: pd.DataFrame, target_col: str, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x=target_col)
    plt.title("Class Balance")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=200)
    plt.close()


def run_eda(data_path: str, output_dir: str) -> None:
    output_dir_path = pathlib.Path(output_dir)
    raw_df = load_raw_csv(data_path)
    clean_df = clean_dataframe(raw_df, CleanConfig())
    plot_histograms(clean_df, output_dir_path)
    plot_correlation(clean_df, output_dir_path)
    plot_class_balance(clean_df, CleanConfig().target_column, output_dir_path)
    summary_path = output_dir_path / "data_summary.csv"
    clean_df.describe(include="all").transpose().to_csv(summary_path)
    print(f"Saved EDA figures and summary to {output_dir_path.resolve()}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA and save plots")
    parser.add_argument("--data", default="data/raw/heart.csv", help="Path to input CSV")
    parser.add_argument("--out", default="reports/figures", help="Directory to store plots")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run_eda(args.data, args.out)


if __name__ == "__main__":
    main()
