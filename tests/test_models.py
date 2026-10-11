"""
tests/test_models.py
====================
Tests for the four classical forecasting models.

Fixtures
--------
* Tiny synthetic 3-container AR-like series; 60 steps total (train/val/test).
* ETS and ARIMA tests use only a small number of steps to stay fast.

Tests
-----
1. invariance    -- forecasts at origin t are unchanged when values after t are altered.
2. selection     -- altering val/test values does not change chosen k (MA), ETS config, or ARIMA order.
3. correctness   -- naive equals y[t]; MA equals hand-computed mean; output shape correct.
4. runner        -- writes rows with expected columns; appending a second run does NOT overwrite.
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.naive import NaiveForecaster
from models.ma import MAForecaster
from models.ets import ETSForecaster
from models.arima import ARIMAForecaster
from data.loader import split_bounds
from eval.results import append_metrics, save_predictions

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CONTAINERS = 3
N_STEPS = 60  # small; train=42, val=9, test=9
MAX_H = 5
RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def Y_synth() -> np.ndarray:
    """Synthetic AR(1)-like series: y[t] = 0.8*y[t-1] + noise."""
    Y = np.zeros((N_CONTAINERS, N_STEPS))
    Y[:, 0] = RNG.uniform(0.2, 0.8, N_CONTAINERS)
    for t in range(1, N_STEPS):
        Y[:, t] = 0.8 * Y[:, t - 1] + RNG.normal(0, 0.02, N_CONTAINERS)
    Y = np.clip(Y, 0.0, 1.0)
    return Y


@pytest.fixture(scope="module")
def ids_synth() -> list[str]:
    return [f"c{i}" for i in range(N_CONTAINERS)]


@pytest.fixture(scope="module")
def split_info():
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = split_bounds(N_STEPS)
    return (tr_s, tr_e), (va_s, va_e), (te_s, te_e)


@pytest.fixture(scope="module")
def naive_fitted(Y_synth, ids_synth, split_info):
    (tr_s, tr_e), _, _ = split_info
    m = NaiveForecaster()
    m.fit(Y_synth[:, tr_s:tr_e], ids_synth)
    return m


@pytest.fixture(scope="module")
def ma_fitted(Y_synth, ids_synth, split_info):
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = split_info
    m = MAForecaster()
    m.fit(Y_synth[:, tr_s:tr_e], ids_synth)
    # build val origins
    val_origins: set[int] = set()
    for ts in range(va_s, va_e):
        for h in range(1, MAX_H + 1):
            o = ts - h
            if o >= 0:
                val_origins.add(o)
    m.select_k(Y_synth, sorted(val_origins), target="cpu", val_start=va_s, val_end=va_e)
    return m


@pytest.fixture(scope="module")
def ets_fitted(Y_synth, ids_synth, split_info):
    (tr_s, tr_e), _, _ = split_info
    m = ETSForecaster()
    m.fit(Y_synth[:, tr_s:tr_e], ids_synth)
    return m


@pytest.fixture(scope="module")
def arima_fitted(Y_synth, ids_synth, split_info):
    (tr_s, tr_e), _, _ = split_info
    m = ARIMAForecaster(n_jobs=1)  # sequential for test isolation
    m.fit(Y_synth[:, tr_s:tr_e], ids_synth)
    return m


# ---------------------------------------------------------------------------
# Helper: alter future values
# ---------------------------------------------------------------------------

def _alter_future(Y: np.ndarray, t: int) -> np.ndarray:
    """Return a copy of Y with all values at columns > t replaced by large noise."""
    Y2 = Y.copy()
    Y2[:, t + 1 :] = 999.0
    return Y2


# ---------------------------------------------------------------------------
# 1. Invariance: prediction at origin t unchanged when future values altered
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name,fixture_name", [
    ("naive", "naive_fitted"),
    ("ma", "ma_fitted"),
    ("ets", "ets_fitted"),
    ("arima", "arima_fitted"),
])
def test_invariance(request, model_name, fixture_name, Y_synth, split_info):
    """Forecasts at origin t must be identical when Y[:, t+1:] is changed."""
    model = request.getfixturevalue(fixture_name)
    (tr_s, tr_e), (va_s, _), _ = split_info
    # Use a mid-val origin
    t = va_s + 2

    preds_original = model.predict_origins(Y_synth, [t], max_h=MAX_H)
    Y_altered = _alter_future(Y_synth, t)
    preds_altered = model.predict_origins(Y_altered, [t], max_h=MAX_H)

    np.testing.assert_array_almost_equal(
        preds_original, preds_altered, decimal=10,
        err_msg=f"{model_name}: prediction changed when future values were altered",
    )


# ---------------------------------------------------------------------------
# 2. Selection uses train only
# ---------------------------------------------------------------------------

def test_ma_selection_uses_val(Y_synth, ids_synth, split_info):
    """MA k is chosen on validation MAE (by spec); altering only TEST values
    must not change k because test targets are not used in k selection.
    """
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = split_info

    val_origins: set[int] = set()
    for ts in range(va_s, va_e):
        for h in range(1, MAX_H + 1):
            o = ts - h
            if o >= 0:
                val_origins.add(o)
    val_origins_sorted = sorted(val_origins)

    m1 = MAForecaster()
    m1.fit(Y_synth[:, tr_s:tr_e], ids_synth)
    m1.select_k(Y_synth, val_origins_sorted, target="cpu", val_start=va_s, val_end=va_e)

    # Alter ONLY test values; val targets and origins are unchanged.
    Y2 = Y_synth.copy()
    Y2[:, te_s:] = 999.0
    m2 = MAForecaster()
    m2.fit(Y2[:, tr_s:tr_e], ids_synth)
    m2.select_k(Y2, val_origins_sorted, target="cpu", val_start=va_s, val_end=va_e)

    assert m1.k_ == m2.k_, (
        f"MA k changed from {m1.k_} to {m2.k_} when test values were altered"
    )


def test_ets_selection_uses_train_only(Y_synth, ids_synth, split_info):
    """ETS AICc is computed on train; altering val/test must not change chosen configs."""
    (tr_s, tr_e), _, _ = split_info

    m1 = ETSForecaster()
    m1.fit(Y_synth[:, tr_s:tr_e], ids_synth)

    Y2 = Y_synth.copy()
    Y2[:, tr_e:] = 999.0  # alter only val+test
    m2 = ETSForecaster()
    m2.fit(Y2[:, tr_s:tr_e], ids_synth)  # same train slice

    assert m1._chosen_cfg == m2._chosen_cfg, (
        "ETS chosen configs changed when val/test values were altered"
    )


def test_arima_selection_uses_train_only(Y_synth, ids_synth, split_info):
    """ARIMA AIC is computed on train; altering val/test must not change chosen orders."""
    (tr_s, tr_e), _, _ = split_info

    m1 = ARIMAForecaster(n_jobs=1)
    m1.fit(Y_synth[:, tr_s:tr_e], ids_synth)

    Y2 = Y_synth.copy()
    Y2[:, tr_e:] = 999.0
    m2 = ARIMAForecaster(n_jobs=1)
    m2.fit(Y2[:, tr_s:tr_e], ids_synth)

    assert m1._chosen_order == m2._chosen_order, (
        "ARIMA chosen orders changed when val/test values were altered"
    )


# ---------------------------------------------------------------------------
# 3. Correctness
# ---------------------------------------------------------------------------

def test_naive_equals_last_value(naive_fitted, Y_synth, split_info):
    """Naive prediction at origin t must equal Y[:, t] for all h."""
    (tr_s, tr_e), (va_s, _), _ = split_info
    t = va_s + 1
    preds = naive_fitted.predict_origins(Y_synth, [t], max_h=MAX_H)
    # shape [3, 1, 5]
    for h in range(MAX_H):
        np.testing.assert_array_almost_equal(
            preds[:, 0, h], Y_synth[:, t],
            err_msg=f"Naive h={h+1}: prediction != Y[:, t]",
        )


def test_ma_equals_hand_computed(ma_fitted, Y_synth, split_info):
    """MA prediction at origin t must equal mean(Y[:, t-k+1:t+1])."""
    (tr_s, tr_e), (va_s, _), _ = split_info
    t = va_s + 3
    k = ma_fitted.k_
    preds = ma_fitted.predict_origins(Y_synth, [t], max_h=MAX_H)
    start = max(0, t - k + 1)
    expected = Y_synth[:, start : t + 1].mean(axis=1)
    for h in range(MAX_H):
        np.testing.assert_array_almost_equal(
            preds[:, 0, h], expected,
            err_msg=f"MA h={h+1}: prediction != hand-computed mean",
        )


def test_output_shape_all_models(
    naive_fitted, ma_fitted, ets_fitted, arima_fitted, Y_synth, split_info
):
    """All models must return shape [n_containers, n_origins, max_h]."""
    (tr_s, tr_e), (va_s, va_e), _ = split_info
    origins = [va_s, va_s + 1, va_s + 2]
    expected_shape = (N_CONTAINERS, len(origins), MAX_H)
    for name, model in [
        ("naive", naive_fitted),
        ("ma", ma_fitted),
        ("ets", ets_fitted),
        ("arima", arima_fitted),
    ]:
        out = model.predict_origins(Y_synth, origins, max_h=MAX_H)
        assert out.shape == expected_shape, (
            f"{name}: expected shape {expected_shape}, got {out.shape}"
        )


# ---------------------------------------------------------------------------
# 4. Runner: writes rows with expected columns, no overwrite on second run
# ---------------------------------------------------------------------------

def test_runner_writes_and_appends(tmp_path, monkeypatch):
    """append_metrics writes expected columns; calling twice does not overwrite."""
    import importlib
    import eval.results as results_mod

    # Redirect results root to tmp_path
    monkeypatch.setattr(results_mod, "_RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(results_mod, "_METRICS_CSV", tmp_path / "metrics_long.csv")
    monkeypatch.setattr(results_mod, "_PREDICTIONS_DIR", tmp_path / "predictions")

    _METRICS_COLS = [
        "run_id", "dataset", "model", "horizon_steps", "target",
        "container_id", "split", "mae", "rmse", "smape", "mase",
        "train_time_s", "pred_latency_ms", "model_size_kb", "seed",
    ]

    row1 = {c: "r1" if c in ("run_id", "dataset", "model", "target", "container_id", "split") else 1.0
            for c in _METRICS_COLS}
    row1["horizon_steps"] = 1
    row1["seed"] = 42

    row2 = dict(row1)
    row2["run_id"] = "r2"

    append_metrics([row1])
    append_metrics([row2])

    csv_path = tmp_path / "metrics_long.csv"
    assert csv_path.exists(), "metrics_long.csv not created"
    df = pd.read_csv(csv_path)
    assert set(_METRICS_COLS).issubset(set(df.columns)), "Missing columns in metrics_long.csv"
    assert len(df) == 2, f"Expected 2 rows after two appends, got {len(df)}"
    assert set(df["run_id"]) == {"r1", "r2"}, "Both run_ids must be present"
