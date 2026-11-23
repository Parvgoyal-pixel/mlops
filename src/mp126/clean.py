"""
clean.py — Data cleaning pipeline for MP126 College Canteen Lunch Rush.

Reads the original (read-only) train.csv and writes a clean version to
data/processed/train_clean.csv. Rules applied (in order):

  1. Drop the leaking column ``plates_counted_by_cashier`` (missing from
     holdout; correlation with target = 0.9986).
  2. Fix sentinel values: replace special_dishes == -999 with median
     of valid non-negative rows.
  3. Fix target typos: divide plates_sold values above the threshold by 10.
  4. Remove duplicate rows (extra copies after the first).

Nothing in this file modifies the original CSVs.

Author: G224 — Ekta Mishra (S24CSEU1362)
"""

from __future__ import annotations

import pathlib
import pandas as pd
import numpy as np

from mp126.config import get_params, project_root


def load_raw(params: dict) -> pd.DataFrame:
    """Load the original train.csv (read-only)."""
    root = project_root()
    path = root / params["paths"]["train_csv"]
    return pd.read_csv(path)


def drop_leaking_column(df: pd.DataFrame, col: str = "plates_counted_by_cashier") -> pd.DataFrame:
    """Remove the leaking column if present.

    ``plates_counted_by_cashier`` is filled in after lunch closes — it is
    unavailable at prediction time and must never be used as a feature.
    """
    if col in df.columns:
        df = df.drop(columns=[col])
    return df


def fix_sentinel_values(
    df: pd.DataFrame,
    col: str = "special_dishes",
    sentinel: int = -999,
) -> pd.DataFrame:
    """Replace sentinel values with the median of valid non-negative rows.

    -999 was used as a placeholder for missing data. We impute with the
    median of the rows where ``special_dishes`` is valid (>= 0).
    Handles pandas dtype casting safely when median is fractional.
    """
    df = df.copy()
    mask = df[col] == sentinel
    if not mask.any():
        return df

    valid_median = df.loc[~mask, col].median()
    if pd.isna(valid_median):
        return df

    # Safely handle int column receiving float median in pandas 3+
    if df[col].dtype.kind in "iu":
        if float(valid_median).is_integer():
            valid_median = int(valid_median)
        else:
            df[col] = df[col].astype(float)

    df.loc[mask, col] = valid_median
    return df


def fix_target_typos(
    df: pd.DataFrame,
    target_col: str = "plates_sold",
    threshold: float = 2000,
) -> pd.DataFrame:
    """Divide target values that are ~10x too large by 10.

    Values above ``threshold`` sit in the range 2940–5180 while the normal
    maximum is 576. Dividing by 10 restores them to the normal range.
    """
    df = df.copy()
    mask = df[target_col] > threshold
    df.loc[mask, target_col] = (df.loc[mask, target_col] / 10).round().astype(int)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop extra copies of duplicated rows (keep first occurrence).

    Duplicates are identified on all columns except ``record_id`` because
    two insertions of the same record can get different auto-generated IDs.
    """
    cols = [c for c in df.columns if c != "record_id"]
    return df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)


def run_cleaning_pipeline(params: dict) -> pd.DataFrame:
    """Run the full cleaning pipeline and return the cleaned DataFrame.

    Steps applied:
      1. Drop leaking column
      2. Fix sentinel values
      3. Fix target typos
      4. Remove duplicates

    The original file is never modified.
    """
    chk = params.get("checks", {})
    sentinel_val = chk.get("sentinel_candidates", [-999])[0]
    typo_thresh = chk.get("target_typo_threshold", 2000)

    df = load_raw(params)
    print(f"Raw rows: {len(df)}")

    df = drop_leaking_column(df)
    print(f"After drop leaking col: {list(df.columns)}")

    df = fix_sentinel_values(df, sentinel=sentinel_val)
    print(f"After fix sentinel: special_dishes min={df['special_dishes'].min()}")

    df = fix_target_typos(df, threshold=typo_thresh)
    print(f"After fix typos: plates_sold max={df['plates_sold'].max()}")

    df = remove_duplicates(df)
    print(f"After remove dups: {len(df)} rows")

    return df


def save_clean(df: pd.DataFrame, params: dict) -> pathlib.Path:
    """Save cleaned DataFrame to data/processed/train_clean.csv."""
    root = project_root()
    out_dir = root / params["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"Clean data saved to: {out_path} ({len(df)} rows)")
    return out_path


if __name__ == "__main__":
    params = get_params()
    df_clean = run_cleaning_pipeline(params)
    save_clean(df_clean, params)

