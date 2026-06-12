"""
annomi_eda.py
Exploratory Data Analysis for the AnnoMI dataset.

Outputs:
  - Basic dataset statistics (shape, dtypes, missing values)
  - Numeric and categorical distributions
  - Session-level conversation statistics
  - Speaker distribution
  - Text length analysis
  - Label distribution
  - Summary CSV

Usage:
    python src/eda/annomi_eda.py --annomi-dir data/annomi
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Argument parsing ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="AnnoMI EDA")
parser.add_argument("--annomi-dir", type=str, default="data/annomi",
                    help="Path to directory containing AnnoMI CSV files")
parser.add_argument("--output-dir", type=str, default="results",
                    help="Directory to save outputs")
args, _ = parser.parse_known_args()

BASE_DIR = Path.cwd()
ANNOMI_DIR = Path(args.annomi_dir)
OUT_DIR = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANNOMI_FULL = ANNOMI_DIR / "AnnoMI-full.csv"
ANNOMI_SIMPLE = ANNOMI_DIR / "AnnoMI-simple.csv"

SEP = "=" * 78


def load_data() -> pd.DataFrame:
    if ANNOMI_FULL.exists():
        df = pd.read_csv(ANNOMI_FULL)
        print(f"Loaded: AnnoMI-full.csv")
    elif ANNOMI_SIMPLE.exists():
        df = pd.read_csv(ANNOMI_SIMPLE)
        print(f"Loaded: AnnoMI-simple.csv")
    else:
        raise FileNotFoundError(
            f"AnnoMI CSV not found in {ANNOMI_DIR}. "
            "See data/README.md for download instructions."
        )
    return df


def print_basic_info(df: pd.DataFrame):
    print(f"\n[1] BASIC INFORMATION\n{'-'*60}")
    print(f"Shape              : {df.shape}")
    print(f"Memory Usage (MB)  : {df.memory_usage(deep=True).sum()/1024**2:.2f}")
    print("\nColumns:")
    for c in df.columns:
        print(f"  - {c} [{df[c].dtype}]")


def print_missing(df: pd.DataFrame):
    print(f"\n[2] MISSING VALUES\n{'-'*60}")
    missing = df.isnull().sum().sort_values(ascending=False)
    miss_df = pd.DataFrame({
        "missing_count": missing,
        "missing_ratio(%)": (missing / len(df) * 100).round(2),
    })
    print(miss_df[miss_df.missing_count > 0])


def print_numeric_stats(df: pd.DataFrame):
    print(f"\n[3] NUMERIC DESCRIPTIVE STATISTICS\n{'-'*60}")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        print(df[num_cols].describe().T.to_string())
    else:
        print("No numeric columns.")


def print_categorical_stats(df: pd.DataFrame):
    print(f"\n[4] CATEGORICAL COLUMN DISTRIBUTIONS\n{'-'*60}")
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    for col in cat_cols:
        print(f"\n### {col}  ({df[col].nunique(dropna=False)} unique values)")
        print(df[col].value_counts(dropna=False).head(10))


def print_session_stats(df: pd.DataFrame):
    print(f"\n[5] SESSION STATISTICS\n{'-'*60}")
    if "transcript_id" in df.columns:
        utt_per = df.groupby("transcript_id").size()
        print(f"Sessions      : {df.transcript_id.nunique()}")
        print(f"Mean utt/sess : {utt_per.mean():.2f}")
        print(f"Median        : {utt_per.median():.2f}")
        print(f"Min / Max     : {utt_per.min()} / {utt_per.max()}")


def text_length_analysis(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[6] TEXT LENGTH ANALYSIS\n{'-'*60}")
    if "utterance_text" in df.columns:
        df = df.copy()
        df["char_len"] = df["utterance_text"].astype(str).apply(len)
        df["word_len"] = df["utterance_text"].astype(str).apply(lambda x: len(x.split()))
        print("Character length:")
        print(df["char_len"].describe())
        print("\nWord length:")
        print(df["word_len"].describe())
    return df


def print_label_distribution(df: pd.DataFrame):
    print(f"\n[7] LABEL DISTRIBUTION\n{'-'*60}")
    for col in ["mi_quality", "client_talk_type", "main_therapist_behaviour"]:
        if col in df.columns:
            print(f"\n### {col}")
            print(df[col].value_counts(dropna=False))


def save_summary(df: pd.DataFrame):
    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [df[c].isnull().sum() for c in df.columns],
        "missing_ratio": [df[c].isnull().mean() for c in df.columns],
        "n_unique": [df[c].nunique(dropna=False) for c in df.columns],
    })
    path = OUT_DIR / "annomi_eda_summary.csv"
    summary.to_csv(path, index=False)
    print(f"\nEDA summary saved: {path}")


def main():
    print(SEP)
    print("AnnoMI EDA")
    print(SEP)

    df = load_data()
    print_basic_info(df)
    print_missing(df)
    print_numeric_stats(df)
    print_categorical_stats(df)
    print_session_stats(df)
    df = text_length_analysis(df)
    print_label_distribution(df)
    save_summary(df)

    print(f"\n{SEP}\nEDA complete.\n{SEP}")


if __name__ == "__main__":
    main()
