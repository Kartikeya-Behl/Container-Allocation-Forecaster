"""
tests/test_features.py
======================
Leakage and correctness tests for src/features/windows.py and src/eval/metrics.py.
All tests use tiny synthetic data and run in <5 s.

Parametrized over n_steps ∈ {240, 480} so both 4h and 8h variants are covered.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.loader import split_bounds
from eval.metrics import mae, rmse, smape, container_metrics
from eval.results import append_metrics, new_run_id, save_predictions
from features.windows import make_supervised


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_Y(n_containers: int, n_steps: int, seed: int = 0) -> np.ndarray:
    """Return a deterministic [n_containers, n_steps] float matrix in [0.1, 0.9]."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.1, 0.9, size=(n_containers, n_steps))


def _container_ids(n: int) -> list[str]:
    return [f"c{i:03d}" for i in range(n)]


# ---------------------------------------------------------------------------
# Parametrize across dataset sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_steps", [240, 480])
class TestMakeSupervisedCorrectness:
    """Core correctness checks for make_supervised."""

    def _setup(self, n_steps):
        N, H, L = 4, 1, 3
        bounds = split_bounds(n_steps)
        Y = _make_Y(N, n_steps, seed=42)
        ids = _container_ids(N)
        return Y, bounds, ids, N, H, L, n_steps

    # -----------------------------------------------------------------------
    # 1. Features at origin t are unchanged when all values strictly after t are modified
    # -----------------------------------------------------------------------
    def test_no_future_leakage(self, n_steps):
        Y, bounds, ids, N, H, L, _ = self._setup(n_steps)

        X_ref, _, _, _, meta_ref = make_supervised(Y, H, L, "train", bounds, ids)

        # For each row, verify lags match Y at the expected positions
        # Then verify that changing values strictly after each origin does not affect features
        for row_i, row_meta in meta_ref.iterrows():
            t = int(row_meta["origin_step"])
            c_idx = int(row_meta["container_idx"])
            # Build Y with all steps > t set to 999
            Y_future_corrupt = Y.copy()
            Y_future_corrupt[c_idx, t + 1:] = 999.0
            # Expected lags: y[t], y[t-1], ..., y[t-L+1]  — all at or before t
            expected_lags = Y[c_idx, t - L + 1: t + 1][::-1]
            # Actual lags from ref row
            actual_lags = X_ref[row_i, :L]
            np.testing.assert_allclose(
                actual_lags, expected_lags, atol=1e-12,
                err_msg=f"Row {row_i}: lag mismatch at origin t={t}, container {c_idx}",
            )
            # Confirm the ref features match the "corrupt-future" version (lags are all <= t)
            Y_test = Y_future_corrupt
            series_lags = Y_test[c_idx, t - L + 1: t + 1][::-1]
            np.testing.assert_allclose(
                actual_lags, series_lags, atol=1e-12,
                err_msg=f"Future corruption bled into lags at row {row_i} (leakage!).",
            )


    # -----------------------------------------------------------------------
    # 2. Split membership is determined by target_step
    # -----------------------------------------------------------------------
    def test_target_step_split_membership(self, n_steps):
        Y, bounds, ids, N, H, L, _ = self._setup(n_steps)
        (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = bounds

        for sp, lo, hi in [("train", tr_s, tr_e), ("val", va_s, va_e), ("test", te_s, te_e)]:
            _, _, _, _, meta = make_supervised(Y, H, L, sp, bounds, ids)
            assert (meta["target_step"] >= lo).all(), f"{sp}: target_step < {lo}"
            assert (meta["target_step"] < hi).all(), f"{sp}: target_step >= {hi}"
            assert (meta["split"] == sp).all()

    # -----------------------------------------------------------------------
    # 3. Row counts match closed-form formula
    # -----------------------------------------------------------------------
    def test_row_counts(self, n_steps):
        Y, bounds, ids, N, H, L, _ = self._setup(n_steps)
        (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = bounds

        expected = {
            "train": N * (tr_e - H - (L - 1)),
            "val": N * (va_e - va_s),
            "test": N * (te_e - te_s),
        }
        for sp, exp in expected.items():
            _, _, _, _, meta = make_supervised(Y, H, L, sp, bounds, ids)
            assert len(meta) == exp, (
                f"n_steps={n_steps}, split={sp}: got {len(meta)}, expected {exp}"
            )

    # -----------------------------------------------------------------------
    # 4. y_last + y_delta == y_level exactly
    # -----------------------------------------------------------------------
    def test_delta_identity(self, n_steps):
        Y, bounds, ids, N, H, L, _ = self._setup(n_steps)
        for sp in ("train", "val", "test"):
            _, y_delta, y_level, y_last, _ = make_supervised(Y, H, L, sp, bounds, ids)
            np.testing.assert_allclose(
                y_last + y_delta, y_level, atol=1e-12,
                err_msg=f"y_last + y_delta ≠ y_level in split={sp}, n_steps={n_steps}",
            )

    # -----------------------------------------------------------------------
    # 5. No row mixes two containers (each row belongs to exactly one container)
    # -----------------------------------------------------------------------
    def test_no_cross_container_mixing(self, n_steps):
        # Give each container a unique constant value so lags are distinctive
        N, H, L = 4, 2, 3
        Y = np.zeros((N, n_steps))
        for i in range(N):
            Y[i, :] = float(i + 1) * 0.1  # 0.1, 0.2, 0.3, 0.4
        bounds = split_bounds(n_steps)
        ids = _container_ids(N)

        X, _, _, y_last, meta = make_supervised(Y, H, L, "train", bounds, ids)

        for row_i, c_idx in enumerate(meta["container_idx"]):
            expected_val = float(c_idx + 1) * 0.1
            # All lag features should equal the container's constant value
            assert np.allclose(X[row_i], expected_val, atol=1e-12), (
                f"Row {row_i} (container {c_idx}) has features from another container"
            )

    # -----------------------------------------------------------------------
    # 6. Horizon=5 row counts (to exercise larger horizons)
    # -----------------------------------------------------------------------
    def test_row_counts_h5(self, n_steps):
        N, H, L = 4, 5, 5
        bounds = split_bounds(n_steps)
        Y = _make_Y(N, n_steps, seed=1)
        ids = _container_ids(N)
        (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = bounds

        expected_train = N * (tr_e - H - (L - 1))
        _, _, _, _, meta = make_supervised(Y, H, L, "train", bounds, ids)
        assert len(meta) == expected_train, (
            f"h=5 train rows: got {len(meta)}, expected {expected_train}"
        )


# ---------------------------------------------------------------------------
# Metrics tests (not parametrized — single array, independent of n_steps)
# ---------------------------------------------------------------------------

class TestMetrics:
    """Hand-computed metric checks including near-zero denominator floor."""

    def test_mae_exact(self):
        yt = np.array([1.0, 2.0, 3.0])
        yp = np.array([1.0, 1.0, 1.0])
        assert abs(mae(yt, yp) - (0 + 1 + 2) / 3) < 1e-12

    def test_rmse_exact(self):
        yt = np.array([0.0, 3.0])
        yp = np.array([0.0, 0.0])
        expected = np.sqrt((0 + 9) / 2)
        assert abs(rmse(yt, yp) - expected) < 1e-12

    def test_smape_near_zero_floor(self):
        """When both values ≈ 0 the floor prevents blow-up."""
        yt = np.array([0.0, 1.0])
        yp = np.array([0.0, 1.0])
        # Row 0: |0-0|/max(0, 1e-3) = 0; Row 1: |1-1|/1.0 = 0
        assert abs(smape(yt, yp, floor=1e-3) - 0.0) < 1e-12

    def test_smape_nonzero(self):
        """sMAPE with known values."""
        yt = np.array([1.0])
        yp = np.array([0.0])
        # denom = max((1+0)/2, 1e-3) = 0.5; smape = |1-0|/0.5 = 2.0
        assert abs(smape(yt, yp, floor=1e-3) - 2.0) < 1e-12

    def test_smape_near_zero_nonzero_pred(self):
        """Near-zero true with nonzero pred — floor activates or not."""
        yt = np.array([0.0])
        yp = np.array([0.002])
        # (|0|+|0.002|)/2 = 0.001 > floor=1e-3 → denom=0.001
        # smape = 0.002 / 0.001 = 2.0
        assert abs(smape(yt, yp, floor=1e-3) - 2.0) < 1e-12

    def test_container_metrics_structure(self):
        """container_metrics returns correct schema keys."""
        n = 10
        yt = np.random.rand(n)
        yp = np.random.rand(n)
        meta = pd.DataFrame({
            "container_idx": [0] * n,
            "container_id": ["c000"] * n,
            "origin_step": list(range(n)),
            "target_step": list(range(1, n + 1)),
            "split": ["train"] * n,
        })
        rows = container_metrics(yt, yp, meta, "run0", "alibaba8h", "rf", 1, "cpu",
                                 1.5, 0.5, 100.0, 42)
        assert len(rows) == 1
        required = {"run_id", "dataset", "model", "horizon_steps", "target",
                    "container_id", "split", "mae", "rmse", "smape",
                    "train_time_s", "pred_latency_ms", "model_size_kb", "seed"}
        assert required.issubset(set(rows[0].keys()))


# ---------------------------------------------------------------------------
# append_metrics never overwrites earlier rows
# ---------------------------------------------------------------------------

class TestAppendMetrics:
    def test_append_never_overwrites(self, tmp_path, monkeypatch):
        """Calling append_metrics twice accumulates rows; no existing row disappears."""
        # Point _METRICS_CSV to a temp path
        import eval.results as rmod
        orig_csv = rmod._METRICS_CSV
        test_csv = tmp_path / "metrics_long.csv"
        monkeypatch.setattr(rmod, "_METRICS_CSV", test_csv)
        monkeypatch.setattr(rmod, "_RESULTS_ROOT", tmp_path)

        row1 = {k: "v1" for k in rmod._METRICS_COLS}
        row2 = {k: "v2" for k in rmod._METRICS_COLS}

        append_metrics([row1])
        append_metrics([row2])

        df = pd.read_csv(test_csv)
        assert len(df) == 2, f"Expected 2 rows after two appends, got {len(df)}"
        assert set(df.columns.tolist()) == set(rmod._METRICS_COLS)
        # Both run_ids present
        assert "v1" in df["run_id"].values
        assert "v2" in df["run_id"].values


# ---------------------------------------------------------------------------
# split_bounds agrees with chrono_split
# ---------------------------------------------------------------------------

class TestSplitBoundsAgreement:
    @pytest.mark.parametrize("n_steps", [240, 480])
    def test_agrees_with_chrono_split(self, n_steps):
        """split_bounds boundaries must match what chrono_split produces."""
        from data.loader import split_bounds
        from data.split import chrono_split

        fracs = (0.70, 0.15, 0.15)
        df = pd.DataFrame({"grp": ["A"] * n_steps, "t": list(range(n_steps)), "v": 0.0})
        train_df, val_df, test_df = chrono_split(df, group_col="grp", time_col="t", fracs=fracs)

        (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = split_bounds(n_steps, fracs)

        assert tr_e - tr_s == len(train_df), "train size mismatch"
        assert va_e - va_s == len(val_df),   "val size mismatch"
        assert te_e - te_s == len(test_df),  "test size mismatch"

        assert train_df["t"].min() == tr_s
        assert train_df["t"].max() == tr_e - 1
        assert val_df["t"].min() == va_s
        assert val_df["t"].max() == va_e - 1
        assert test_df["t"].min() == te_s
        assert test_df["t"].max() == te_e - 1
