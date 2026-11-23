"""
test_clean.py — Pytest tests for the cleaning pipeline (Stage 2).

All tests use small synthetic DataFrames — no CSV files needed.

Author: G224 — Yash Mudgal (S24CSEU0246)
"""

from __future__ import annotations

import sys
import pathlib

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mp126.clean import (
    drop_leaking_column,
    fix_sentinel_values,
    fix_target_typos,
    remove_duplicates,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_df(**kwargs) -> pd.DataFrame:
    """Build a minimal DataFrame with sensible defaults, overridable via kwargs."""
    base = {
        "record_id": ["R001", "R002", "R003", "R004", "R005"],
        "date": ["2026-01-01"] * 5,
        "students_on_campus": [3000, 3100, 2900, 3200, 3050],
        "menu_price_rs": [70, 72, 68, 75, 71],
        "temp_c": [28.0, 29.0, 27.0, 30.0, 28.5],
        "exam_days_left": [30, 29, 28, 27, 26],
        "special_dishes": [2, 3, 1, 4, 2],
        "queue_minutes_yesterday": [20.0, 22.0, 18.0, 25.0, 21.0],
        "day_type": ["weekday"] * 5,
        "block": ["A-Block"] * 5,
        "plates_counted_by_cashier": [350, 360, 340, 370, 355],
        "plates_sold": [348, 362, 338, 372, 353],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


# ── Tests: drop_leaking_column ────────────────────────────────────────────────

class TestDropLeakingColumn:
    def test_drops_cashier_column(self):
        """plates_counted_by_cashier must be removed."""
        df = _make_df()
        assert "plates_counted_by_cashier" in df.columns
        result = drop_leaking_column(df)
        assert "plates_counted_by_cashier" not in result.columns

    def test_other_columns_preserved(self):
        """All other columns survive the drop."""
        df = _make_df()
        result = drop_leaking_column(df)
        expected_cols = [c for c in df.columns if c != "plates_counted_by_cashier"]
        assert list(result.columns) == expected_cols

    def test_no_error_if_col_absent(self):
        """Should not raise if the leaking column is already gone."""
        df = _make_df().drop(columns=["plates_counted_by_cashier"])
        result = drop_leaking_column(df)
        assert "plates_counted_by_cashier" not in result.columns


# ── Tests: fix_sentinel_values ────────────────────────────────────────────────

class TestFixSentinelValues:
    def test_sentinel_replaced_by_median(self):
        """-999 is replaced by the median of valid rows."""
        df = _make_df(special_dishes=[2, 3, -999, 4, 2])
        result = fix_sentinel_values(df, col="special_dishes", sentinel=-999)
        # Median of [2, 3, 4, 2] = 2.5
        assert (result["special_dishes"] == -999).sum() == 0
        assert result["special_dishes"].iloc[2] == pytest.approx(2.5, abs=0.1)

    def test_no_sentinel_data_unchanged(self):
        """When no sentinel present, column is returned as-is."""
        df = _make_df(special_dishes=[2, 3, 1, 4, 2])
        result = fix_sentinel_values(df, col="special_dishes", sentinel=-999)
        pd.testing.assert_series_equal(result["special_dishes"], df["special_dishes"])

    def test_multiple_sentinels_all_replaced(self):
        """All occurrences of -999 are replaced."""
        df = _make_df(special_dishes=[-999, -999, 3, 4, 2])
        result = fix_sentinel_values(df, col="special_dishes", sentinel=-999)
        assert (result["special_dishes"] < 0).sum() == 0


# ── Tests: fix_target_typos ───────────────────────────────────────────────────

class TestFixTargetTypos:
    def test_typos_divided_by_10(self):
        """Values above threshold are divided by 10."""
        df = _make_df(plates_sold=[350, 362, 3500, 4200, 353])
        result = fix_target_typos(df, target_col="plates_sold", threshold=2000)
        assert result["plates_sold"].iloc[2] == 350
        assert result["plates_sold"].iloc[3] == 420

    def test_normal_values_untouched(self):
        """Values within normal range are not changed."""
        df = _make_df(plates_sold=[350, 362, 338, 372, 353])
        result = fix_target_typos(df, target_col="plates_sold", threshold=2000)
        pd.testing.assert_series_equal(result["plates_sold"], df["plates_sold"])

    def test_no_typos_no_change(self):
        """No values above threshold → DataFrame returned unchanged."""
        df = _make_df(plates_sold=[400, 410, 390, 420, 400])
        result = fix_target_typos(df, threshold=2000)
        assert result["plates_sold"].max() <= 2000


# ── Tests: remove_duplicates ──────────────────────────────────────────────────

class TestRemoveDuplicates:
    def test_extra_copies_removed(self):
        """3 duplicates of one row → only 1 kept."""
        base = _make_df()
        dup_row = base.iloc[[0]].copy()
        df = pd.concat([base, dup_row, dup_row, dup_row], ignore_index=True)
        result = remove_duplicates(df)
        assert len(result) == len(base)

    def test_different_record_ids_still_deduplicated(self):
        """Different record_ids on otherwise identical rows → still deduplicated."""
        base = _make_df()
        dup = base.iloc[[0]].copy()
        dup["record_id"] = "R999"  # different ID, same data
        df = pd.concat([base, dup], ignore_index=True)
        result = remove_duplicates(df)
        assert len(result) == len(base)

    def test_clean_data_untouched(self):
        """No duplicates in input → all rows preserved."""
        df = _make_df()
        result = remove_duplicates(df)
        assert len(result) == len(df)


# ── Integration: full pipeline produces sane output ──────────────────────────

class TestFullPipeline:
    def test_pipeline_output_shape(self):
        """After all cleaning steps, output has expected fewer rows and cols."""
        df = _make_df(
            plates_sold=[350, 362, 3500, 372, 353],  # 1 typo
            special_dishes=[2, -999, 1, 4, 2],  # 1 sentinel
        )
        # Duplicate row 0
        dup = df.iloc[[0]].copy()
        df = pd.concat([df, dup], ignore_index=True)

        df = drop_leaking_column(df)
        df = fix_sentinel_values(df, sentinel=-999)
        df = fix_target_typos(df, threshold=2000)
        df = remove_duplicates(df)

        assert "plates_counted_by_cashier" not in df.columns
        assert df["special_dishes"].min() >= 0
        assert df["plates_sold"].max() <= 2000
        assert len(df) == 5  # 6 - 1 duplicate

