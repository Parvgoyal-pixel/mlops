"""
test_data_checks.py — Pytest tests for all 7 data check functions.

IMPORTANT: These tests build tiny synthetic DataFrames inline.
They do NOT read any CSV from disk. This means they will pass even
if data/raw/ is not present (required by the marking tool).

Run with:  pytest -q   (from project root)

Author: G224 — Yash Mudgal (S24CSEU0246)
"""

import pandas as pd
import numpy as np
import pytest
import sys
import pathlib

# Add src/ to path so we can import mp126
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mp126.data_checks import (
    check_leaking_column,
    check_unit_mixup,
    check_sentinel_value,
    check_duplicate_rows,
    check_stuck_sensor,
    check_target_typos,
    check_spelling_variants,
)


# ---------------------------------------------------------------------------
# Helper: minimal valid train-like DataFrame
# ---------------------------------------------------------------------------

def _base_train(n: int = 10) -> pd.DataFrame:
    """Return a small clean DataFrame with all expected columns."""
    return pd.DataFrame({
        "record_id": [f"R{i:04d}" for i in range(n)],
        "date": pd.date_range("2026-01-01", periods=n, freq="D").astype(str),
        "students_on_campus": np.random.randint(1000, 5000, n),
        "menu_price_rs": np.random.randint(50, 120, n),
        "temp_c": np.random.uniform(20, 40, n),
        "exam_days_left": np.random.randint(0, 60, n),
        "special_dishes": np.random.randint(0, 6, n),
        "queue_minutes_yesterday": np.random.uniform(5, 35, n),
        "day_type": ["weekday"] * n,
        "block": ["A-Block"] * n,
        "plates_counted_by_cashier": np.random.randint(200, 500, n),
        "plates_sold": np.random.randint(200, 500, n),
    })


# ===========================================================================
# TEST 1: Leaking column detected
# ===========================================================================

class TestLeakingColumn:
    def test_leaking_column_detected(self):
        """plates_counted_by_cashier is in train but not holdout → leak."""
        train = _base_train(50)
        # Make the leaking col nearly identical to target (high correlation)
        train["plates_counted_by_cashier"] = train["plates_sold"] + np.random.randint(-2, 3, 50)
        # holdout has no plates_counted_by_cashier
        holdout = train.drop(columns=["plates_counted_by_cashier", "plates_sold"])
        result = check_leaking_column(train, holdout)
        assert result["present"] is True
        assert result["column"] == "plates_counted_by_cashier"

    def test_no_leaking_column_when_all_cols_in_holdout(self):
        """When holdout contains all train columns (minus target), no leak."""
        train = _base_train(20)
        holdout = train.drop(columns=["plates_sold"])
        result = check_leaking_column(train, holdout)
        assert result["present"] is False
        assert result["column"] is None

    def test_leaking_column_low_correlation_not_flagged(self):
        """A column missing from holdout but uncorrelated with target is not a leak."""
        train = _base_train(50)
        # random_noise column: missing from holdout but not correlated with target
        train["random_noise"] = np.random.randint(0, 1000, 50)
        holdout = train.drop(columns=["random_noise", "plates_sold"])
        result = check_leaking_column(train, holdout)
        # random_noise should not be flagged (correlation << 0.95)
        if result["present"]:
            assert result["column"] != "random_noise"


# ===========================================================================
# TEST 2: Unit mixup
# ===========================================================================

class TestUnitMixup:
    def test_unit_mixup_detected_1000x(self):
        """One block records a column 1000x larger → flagged."""
        rows = []
        for i in range(30):
            rows.append({
                "block": "A-Block",
                "students_on_campus": 3000,
                "menu_price_rs": 70,
                "plates_sold": 400,
            })
        for i in range(30):
            rows.append({
                "block": "B-Block",
                "students_on_campus": 3000000,  # 1000× A-Block
                "menu_price_rs": 70,
                "plates_sold": 400,
            })
        train = pd.DataFrame(rows)
        result = check_unit_mixup(train, group_col="block", ratio_threshold=5.0)
        assert result["present"] is True
        assert result["column"] == "students_on_campus"
        assert result["factor"] == 1000

    def test_no_unit_mixup_when_similar(self):
        """Blocks with similar means → no unit mixup."""
        rows = [{"block": "A-Block", "students_on_campus": 3000, "plates_sold": 400} for _ in range(20)]
        rows += [{"block": "B-Block", "students_on_campus": 3100, "plates_sold": 410} for _ in range(20)]
        train = pd.DataFrame(rows)
        result = check_unit_mixup(train, group_col="block", ratio_threshold=5.0)
        assert result["present"] is False


# ===========================================================================
# TEST 3: Sentinel value
# ===========================================================================

class TestSentinelValue:
    def test_sentinel_minus999_detected(self):
        """-999 repeated in special_dishes → flagged."""
        train = _base_train(20)
        train.loc[0:4, "special_dishes"] = -999  # 5 sentinel rows
        result = check_sentinel_value(train, sentinel_candidates=[-999, -1])
        assert result["present"] is True
        assert result["column"] == "special_dishes"
        assert result["value"] == -999
        assert result["count"] == 5

    def test_no_sentinel_in_clean_data(self):
        """Clean data with no sentinel values → not present."""
        train = _base_train(20)
        result = check_sentinel_value(train, sentinel_candidates=[-999, -1, 999])
        assert result["present"] is False
        assert result["count"] is None


# ===========================================================================
# TEST 4: Duplicate rows
# ===========================================================================

class TestDuplicateRows:
    def test_duplicates_detected(self):
        """Adding 3 exact copies of a row → 3 extra copies detected."""
        base = _base_train(10)
        # Duplicate row 0 three times (keep same record_id to test logic)
        dup_row = base.iloc[[0]].copy()
        train = pd.concat([base, dup_row, dup_row, dup_row], ignore_index=True)
        result = check_duplicate_rows(train)
        assert result["present"] is True
        assert result["count"] == 3

    def test_no_duplicates_in_clean_data(self):
        """Unique rows → not present."""
        train = _base_train(10)
        result = check_duplicate_rows(train)
        assert result["present"] is False
        assert result["count"] is None


# ===========================================================================
# TEST 5: Stuck sensor
# ===========================================================================

class TestStuckSensor:
    def test_stuck_sensor_detected(self):
        """temp_c same value across multiple rows on each of 10 consecutive days → flagged.

        A stuck sensor means every reading on a date is identical (nunique==1)
        despite there being multiple rows (multiple blocks), and the constant
        repeats for many consecutive days.
        """
        n = 10
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        rows = []
        for d in dates:
            # 3 rows per day, all with temp_c = 25.0  → nunique == 1 each day
            for block in ["A-Block", "B-Block", "Hostel"]:
                rows.append({
                    "date": str(d.date()),
                    "temp_c": 25.0,                      # same value every row
                    "queue_minutes_yesterday": 20.0 + hash(str(d)) % 5,  # varies
                    "students_on_campus": 3000,
                    "menu_price_rs": 70,
                    "exam_days_left": 30,
                    "special_dishes": 2,
                    "block": block,
                })
        train = pd.DataFrame(rows)
        result = check_stuck_sensor(train, min_days=7)
        assert result["present"] is True
        assert result["column"] == "temp_c"
        assert result["start_date"] == "2026-01-01"
        assert result["end_date"] == "2026-01-10"

    def test_no_stuck_sensor_when_values_vary(self):
        """Every sensor column changes from day to day → not present.

        Uses single row per date but each date has a different value for every
        sensor column, so the longest any column is 'stuck' is 1 day — well
        below the 7-day threshold.
        """
        n = 14  # 14 unique days
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        rows = []
        for i, d in enumerate(dates):
            # All sensor values vary with i, so no run >= 2 days
            rows.append({
                "date": str(d.date()),
                "temp_c": 20.0 + i,                     # changes every day
                "queue_minutes_yesterday": 10.0 + i,    # changes every day
                "students_on_campus": 3000 + i * 10,    # changes every day
                "menu_price_rs": 70 + i,                # changes every day
                "exam_days_left": 30 + i,               # changes every day
                "special_dishes": (i % 6),              # changes every day
            })
        train = pd.DataFrame(rows)
        result = check_stuck_sensor(train, min_days=7)
        assert result["present"] is False


# ===========================================================================
# TEST 6: Target typos
# ===========================================================================

class TestTargetTypos:
    def test_target_typos_detected(self):
        """3 values 10× too big → 3 typos detected."""
        train = _base_train(20)
        train.loc[0, "plates_sold"] = 3500   # typo: should be 350
        train.loc[1, "plates_sold"] = 4200   # typo: should be 420
        train.loc[2, "plates_sold"] = 2900   # typo: should be 290
        result = check_target_typos(train, typo_threshold=2000)
        assert result["present"] is True
        assert result["count"] == 3

    def test_no_target_typos_in_clean_data(self):
        """All values in normal range → not present."""
        train = _base_train(20)
        # Ensure max is well below threshold
        train["plates_sold"] = np.random.randint(200, 500, 20)
        result = check_target_typos(train, typo_threshold=2000)
        assert result["present"] is False
        assert result["count"] is None


# ===========================================================================
# TEST 7: Spelling variants
# ===========================================================================

class TestSpellingVariants:
    def test_spelling_variants_detected(self):
        """'Weekday' vs 'weekday' → spelling variant detected."""
        train = pd.DataFrame({
            "day_type": ["Weekday", "saturday", "holiday"],
            "block": ["A-Block", "B-Block", "Hostel"],
        })
        later = pd.DataFrame({
            "day_type": ["weekday", "saturday", "holiday"],
            "block": ["A-Block", "B-Block", "Hostel"],
        })
        result = check_spelling_variants(train, later)
        assert result["present"] is True
        assert result["column"] == "day_type"

    def test_no_spelling_variants_when_spellings_match(self):
        """Identical spellings in train and later → not present."""
        train = pd.DataFrame({
            "day_type": ["weekday", "saturday", "holiday"],
            "block": ["A-Block", "B-Block", "Hostel"],
        })
        later = train.copy()
        result = check_spelling_variants(train, later)
        assert result["present"] is False
