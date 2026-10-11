"""
src/eval/metrics.py
===================
Forecasting accuracy metrics and result-row builder.

Metrics
-------
MAE:   mean absolute error.
RMSE:  root mean squared error.
sMAPE: symmetric mean absolute percentage error with a denominator floor.

  sMAPE = (1/n) * sum( |y_true - y_pred| / max((|y_true| + |y_pred|)/2, floor) )

The floor (default 1e-3 on normalised [0,1] data) prevents blow-up when both
true and predicted values are near zero.  It is documented here so the formula
is reproducible without reading the code.

Functions
---------
mae(y_true, y_pred)
rmse(y_true, y_pred)
smape(y_true, y_pred, floor=1e-3)
container_metrics(y_true, y_pred, meta, run_id, dataset, model, horizon, target,
                  train_time_s, pred_latency_ms, model_size_kb, seed)
    -> list[dict]  (one dict per container per split, matching AGENTS.md schema)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Denominator floor for sMAPE (prevents 0/0 on near-zero normalised values)
_SMAPE_FLOOR = 1e-3


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error.

    Parameters
    ----------
    y_true, y_pred : array-like of float

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error.

    Parameters
    ----------
    y_true, y_pred : array-like of float

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float = _SMAPE_FLOOR,
) -> float:
    """Symmetric mean absolute percentage error with denominator floor.

    Formula:
      sMAPE = mean( |y_true - y_pred| / max((|y_true| + |y_pred|) / 2, floor) )

    The floor prevents division by zero when both values are near zero
    (common in normalised utilisation data where containers can be idle).

    Parameters
    ----------
    y_true, y_pred : array-like of float
    floor : float, default 1e-3
        Minimum denominator value to avoid blow-up near zero.

    Returns
    -------
    float  (value in [0, 1] when |y_true|, |y_pred| <= 1 and floor >= 0)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, floor)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def container_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    run_id: str,
    dataset: str,
    model: str,
    horizon: int,
    target: str,
    train_time_s: float,
    pred_latency_ms: float,
    model_size_kb: float,
    seed: int,
) -> list[dict]:
    """Compute per-container per-split metrics and format as AGENTS.md schema rows.

    Parameters
    ----------
    y_true : array-like, shape [n_rows]
        Actual target values (y_level from make_supervised).
    y_pred : array-like, shape [n_rows]
        Predicted target values.
    meta : pd.DataFrame
        Meta DataFrame from make_supervised with columns:
        container_idx, container_id, origin_step, target_step, split.
    run_id : str
        Unique run identifier (from new_run_id()).
    dataset : str
        Dataset tag (e.g. 'alibaba8h').
    model : str
        Model name (e.g. 'rf', 'xgb', 'naive').
    horizon : int
        Forecast horizon in steps.
    target : str
        'cpu' or 'mem'.
    train_time_s : float
        Training time in seconds.
    pred_latency_ms : float
        Median prediction latency per single origin in milliseconds.
    model_size_kb : float
        Serialised model size in kilobytes.
    seed : int
        Random seed used.

    Returns
    -------
    list[dict]
        One dict per (container_id, split) combination.  Matches the
        metrics_long.csv schema in AGENTS.md §12.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    rows = []
    for (c_id, sp), grp_idx in meta.groupby(["container_id", "split"]).groups.items():
        yt = y_true[grp_idx]
        yp = y_pred[grp_idx]
        rows.append({
            "run_id": run_id,
            "dataset": dataset,
            "model": model,
            "horizon_steps": horizon,
            "target": target,
            "container_id": c_id,
            "split": sp,
            "mae": mae(yt, yp),
            "rmse": rmse(yt, yp),
            "smape": smape(yt, yp),
            "train_time_s": train_time_s,
            "pred_latency_ms": pred_latency_ms,
            "model_size_kb": model_size_kb,
            "seed": seed,
        })
    return rows
