"""
Download the Heart Disease dataset from a public source.

Usage:
    python data/download_data.py --output data/raw/heart.csv

The script keeps dependencies minimal and provides clear logging so users can
verify the download location or quickly swap the URL if needed.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import requests


# UCI Heart Disease (Cleveland) processed dataset.
DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def download_file(url: str, output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        print(f"Saved dataset to {output_path.resolve()}")
    except requests.RequestException as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Heart Disease dataset")
    parser.add_argument(
        "--url",
        default=DATA_URL,
        help=f"Source URL (default: {DATA_URL})",
    )
    parser.add_argument(
        "--output",
        default="data/raw/heart.csv",
        help="Path to save the CSV (default: data/raw/heart.csv)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    output_path = pathlib.Path(args.output)
    download_file(args.url, output_path)


if __name__ == "__main__":
    main()
