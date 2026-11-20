"""
data_checks.py — Seven data-quality checks for MP126 (College Canteen Lunch Rush).

Each function:
  - Accepts pandas DataFrames (never touches the CSV files directly).
  - Returns a dict shaped exactly like the matching block of findings.yaml.
  - Decides present: True/False from the data, never hard-coded.
  - Is unit-testable on a tiny synthetic DataFrame.
  - Reads thresholds from params via the ``params`` argument so nothing is
    baked into the logic (rule 6 / params.yaml requirement).

Author: G224 — Data Lead (Hiten Mittal, S24CSEU1356)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# 1. Leaking column
# ---------------------------------------------------------------------------

def check_leaking_column(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    target_col: str = "plates_sold",
    typo_threshold: float = 2000,
) -> dict[str, Any]:
    """Detect a column that is present in train but absent from holdout.

    A leaking column is one that is only filled *after* the event
    (e.g. the cashier count is recorded at closing time), so the model
    cannot use it at prediction time.  Its absence from holdout_features.csv
    is the structural signal.  We also check correlation: a genuine leak
    will correlate very strongly with the target once typo rows are removed.

    Returns a dict matching the ``leaking_column`` block of findings.yaml.
    """
    # Columns in train that are not in holdout and not the target
    extra = set(train.columns) - set(holdout.columns) - {target_col}

    if not extra:
        return {"present": False, "column": None}

    # Pick the extra column with the highest absolute correlation to the target
    # (after removing target typos which distort correlation).
    clean = train[train[target_col] < typo_threshold].copy()
    best_col = None
    best_corr = 0.0
    for col in extra:
        if pd.api.types.is_numeric_dtype(train[col]):
            corr = abs(clean[col].corr(clean[target_col]))
            if corr > best_corr:
                best_corr = corr
                best_col = col

    # Only flag if correlation is very high (> 0.95) — a true leak
    if best_col is None or best_corr < 0.95:
        return {"present": False, "column": None}

    return {"present": True, "column": best_col}


# ---------------------------------------------------------------------------
# 2. Unit mixup
# ---------------------------------------------------------------------------

def check_unit_mixup(
    train: pd.DataFrame,
    group_col: str = "block",
    ratio_threshold: float = 5.0,
) -> dict[str, Any]:
    """Detect when one group recorded a column in a different unit.

    For every numeric column we compare the mean of each group to the
    overall median of group means.  If one group's mean is ``ratio_threshold``
    times larger or smaller than the smallest group's mean, we flag it and
    report the exact integer conversion factor (e.g. 1000 for km→m).

    Returns a dict matching the ``unit_mixup`` block of findings.yaml.
    """
    numeric_cols = train.select_dtypes(include="number").columns.tolist()
    # Exclude id-like or target columns
    exclude = {"record_id", "plates_sold"}
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    for col in numeric_cols:
        group_means = train.groupby(group_col)[col].mean()
        min_mean = group_means.min()
        max_mean = group_means.max()

        if min_mean <= 0:
            continue  # skip columns with non-positive means (e.g. sentinel polluted)

        ratio = max_mean / min_mean
        if ratio >= ratio_threshold:
            # Exact factor: round to nearest standard conversion
            raw_factor = ratio
            # Choose the closest standard factor
            standard_factors = [10, 12, 60, 100, 1000]
            factor = min(standard_factors, key=lambda f: abs(f - raw_factor))
            affected_group = group_means.idxmax()
            return {
                "present": True,
                "column": col,
                "group_value": affected_group,
                "factor": factor,
            }

    return {"present": False, "column": None, "group_value": None, "factor": None}


# ---------------------------------------------------------------------------
# 3. Sentinel value
# ---------------------------------------------------------------------------

def check_sentinel_value(
    train: pd.DataFrame,
    sentinel_candidates: list[float] | None = None,
) -> dict[str, Any]:
    """Detect impossible repeated sentinel values used instead of blanks.

    Checks every numeric column for values in ``sentinel_candidates``
    (e.g. -999, -1, 999, 9999).  Reports the column, the value, and
    how many rows contain it.

    Returns a dict matching the ``sentinel_value`` block of findings.yaml.
    """
    if sentinel_candidates is None:
        sentinel_candidates = [-999, -1, 999, 9999]

    numeric_cols = train.select_dtypes(include="number").columns

    best_col = None
    best_val = None
    best_count = 0

    for col in numeric_cols:
        vc = train[col].value_counts()
        for sentinel in sentinel_candidates:
            if sentinel in vc.index and vc[sentinel] > best_count:
                best_col = col
                best_val = int(sentinel)
                best_count = int(vc[sentinel])

    if best_col is None:
        return {"present": False, "column": None, "value": None, "count": None}

    return {
        "present": True,
        "column": best_col,
        "value": best_val,
        "count": best_count,
    }


# ---------------------------------------------------------------------------
# 4. Duplicate rows
# ---------------------------------------------------------------------------

def check_duplicate_rows(train: pd.DataFrame) -> dict[str, Any]:
    """Count extra copies of duplicated rows.

    We check duplicates on all columns *except* record_id, because two rows
    can have different auto-generated IDs yet be the same real-world record.
    ``df.duplicated()`` marks every copy after the first, so the count is
    exactly the number of rows you would delete.

    Returns a dict matching the ``duplicate_rows`` block of findings.yaml.
    """
    cols_excl_id = [c for c in train.columns if c != "record_id"]
    n_dups = int(train.duplicated(subset=cols_excl_id).sum())

    if n_dups == 0:
        return {"present": False, "count": None}

    return {"present": True, "count": n_dups}


# ---------------------------------------------------------------------------
# 5. Stuck sensor
# ---------------------------------------------------------------------------

def check_stuck_sensor(
    train: pd.DataFrame,
    date_col: str = "date",
    min_days: int = 7,
) -> dict[str, Any]:
    """Detect a sensor that transmitted the same value for many consecutive days.

    Algorithm:
      1. Sort by date.
      2. For each sensor-like column, compute the per-date median value and
         count of rows per date.
      3. A date is "stuck" if ALL rows that day share the same value
         (nunique == 1) AND there are multiple rows on that date (meaning the
         sensor reported a constant despite seeing different blocks/records).
         Single-row dates are excluded because there is nothing to compare.
      4. Among stuck dates, find the longest run of consecutive calendar dates
         where the per-date median is also constant (same value across days).
      5. Flag if the run is >= ``min_days`` long.

    Returns a dict matching the ``stuck_sensor`` block of findings.yaml.
    """
    sensor_cols = [
        "temp_c",
        "queue_minutes_yesterday",
        "students_on_campus",
        "menu_price_rs",
        "exam_days_left",
        "special_dishes",
    ]
    sensor_cols = [c for c in sensor_cols if c in train.columns]

    df = train.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    for col in sensor_cols:
        per_date = df.groupby(date_col)[col].agg(["nunique", "count", "median"])

        # Require >= 2 rows per date so "nunique == 1" is meaningful
        # (a date with only 1 row trivially has nunique == 1)
        multi_row_dates = per_date[per_date["count"] >= 2]
        stuck_dates = sorted(
            multi_row_dates[multi_row_dates["nunique"] == 1].index.tolist()
        )

        if not stuck_dates:
            continue

        # Among consecutive stuck dates, also require the same median value
        def _find_long_runs(dates_with_medians):
            """Find longest run where consecutive dates share the same median."""
            if not dates_with_medians:
                return 0, None, None
            best_len = 0
            best_start = None
            best_end = None
            run_start = dates_with_medians[0][0]
            run_end = dates_with_medians[0][0]
            run_val = dates_with_medians[0][1]

            for d, v in dates_with_medians[1:]:
                if (d - run_end).days == 1 and v == run_val:
                    run_end = d
                else:
                    rlen = (run_end - run_start).days + 1
                    if rlen > best_len:
                        best_len = rlen
                        best_start = run_start
                        best_end = run_end
                    run_start = d
                    run_end = d
                    run_val = v

            rlen = (run_end - run_start).days + 1
            if rlen > best_len:
                best_len = rlen
                best_start = run_start
                best_end = run_end
            return best_len, best_start, best_end

        date_med_pairs = [(d, per_date.loc[d, "median"]) for d in stuck_dates]
        max_run_len, max_run_start, max_run_end = _find_long_runs(date_med_pairs)

        if max_run_len >= min_days:
            return {
                "present": True,
                "column": col,
                "start_date": max_run_start.strftime("%Y-%m-%d"),
                "end_date": max_run_end.strftime("%Y-%m-%d"),
            }

    return {
        "present": False,
        "column": None,
        "start_date": None,
        "end_date": None,
    }


# ---------------------------------------------------------------------------
# 6. Target typos
# ---------------------------------------------------------------------------

def check_target_typos(
    train: pd.DataFrame,
    target_col: str = "plates_sold",
    typo_threshold: float = 2000,
) -> dict[str, Any]:
    """Detect target values that were typed with an extra zero (10× too big).

    Method:
      - Sort the target column.
      - Find values above ``typo_threshold`` that sit in a group separated
        from the rest by a large gap.
      - Verify that dividing the suspect values by 10 puts them inside the
        normal range.

    Returns a dict matching the ``target_typos`` block of findings.yaml.
    """
    vals = train[target_col].sort_values().dropna()
    normal = vals[vals <= typo_threshold]
    suspects = vals[vals > typo_threshold]

    if suspects.empty:
        return {"present": False, "count": None}

    # Verify: dividing by 10 should land inside the normal range
    normal_max = normal.max()
    normal_min = normal.min()
    corrected = suspects / 10
    all_in_range = corrected.between(normal_min, normal_max * 1.5).all()

    if not all_in_range:
        return {"present": False, "count": None}

    return {"present": True, "count": int(len(suspects))}


# ---------------------------------------------------------------------------
# 7. Spelling variants
# ---------------------------------------------------------------------------

def check_spelling_variants(
    train: pd.DataFrame,
    later: pd.DataFrame,
) -> dict[str, Any]:
    """Detect text columns with the same category spelled differently.

    For every text column shared between train and later:
      - Strip whitespace and lowercase both.
      - Compare the resulting sets of unique values.
      - If the raw train levels differ from the cleaned levels, spellings vary.
      - ``true_levels`` is the number of real categories after cleaning.

    Returns a dict matching the ``spelling_variants`` block of findings.yaml.
    """
    text_cols = train.select_dtypes(include=["object", "str"]).columns.tolist()
    shared_text = [c for c in text_cols if c in later.columns and c not in ("record_id", "date")]

    for col in shared_text:
        raw_levels_train = set(train[col].dropna().unique())
        clean_levels_train = {v.strip().lower() for v in raw_levels_train}
        clean_levels_later = {v.strip().lower() for v in later[col].dropna().unique()}

        # If cleaning collapses levels or the raw train levels differ from later's
        raw_later = set(later[col].dropna().unique())
        if raw_levels_train != raw_later or len(raw_levels_train) != len(clean_levels_train):
            true_levels = len(clean_levels_train | clean_levels_later)
            return {
                "present": True,
                "column": col,
                "true_levels": true_levels,
            }

    return {"present": False, "column": None, "true_levels": None}
