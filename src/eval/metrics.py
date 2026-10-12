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
MASE:  mean absolute scaled error with a denominator floor.
       MASE = MAE / max(naive_mae1_<target>, floor)
       where naive_mae1 is in-sample one-step naive MAE on train.

The floor (default 1e-3 on normalised [0,1] data) prevents blow-up when both
true and predicted values (or train naive errors) are near zero.

Functions
---------
mae(y_true, y_pred)
rmse(y_true, y_pred)
smape(y_true, y_pred, floor=1e-3)
mase(y_true, y_pred, naive_mae1=None, floor=1e-3)
container_metrics(y_true, y_pred, meta, run_id, dataset, model, horizon, target,
                  train_time_s, pred_latency_ms, model_size_kb, seed,
                  naive_mae1_map=None)
    -> list[dict]  (one dict per container per split, matching AGENTS.md schema)
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

# Denominator floor for sMAPE and MASE (prevents 0/0 on near-zero normalised values)
_FLOOR = 1e-3


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
    floor: float = _FLOOR,
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


def mase(
    y_true: np.ndarray | float,
    y_pred: np.ndarray | float,
    naive_mae1: float | None = None,
    floor: float = _FLOOR,
) -> float:
    """Mean Absolute Scaled Error (MASE) with denominator floor.

    Formula:
      MASE = MAE / max(naive_mae1, floor)

    Supports two calling conventions:
      1. mase(y_true, y_pred, naive_mae1=val, floor=1e-3)
      2. mase(mae_val, naive_mae1=val, floor=1e-3) (where y_true is a precomputed scalar MAE)

    Parameters
    ----------
    y_true : array-like or float
        Actual values, or precomputed scalar MAE.
    y_pred : array-like or float
        Predicted values, or scalar naive_mae1 if y_true is a precomputed scalar MAE.
    naive_mae1 : float, optional
        In-sample one-step naive MAE on train (the scale denominator).
    floor : float, default 1e-3
        Minimum denominator value to avoid division by zero.

    Returns
    -------
    float
    """
    if naive_mae1 is None:
        mae_val = float(y_true)
        denom = float(y_pred)
    else:
        mae_val = mae(y_true, y_pred)
        denom = float(naive_mae1)
    return float(mae_val / max(denom, floor))


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
    naive_mae1_map: Mapping[str, float] | pd.DataFrame | None = None,
    fallback_rate: float = 0.0,
    n_models: int = 1,
    n_jobs: int = 1,
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
        Serialised model size of ONE model instance in kilobytes.
    seed : int
        Random seed used.
    naive_mae1_map : mapping or pd.DataFrame, optional
        Map of container_id -> naive_mae1 on train. If DataFrame, looks for column
        f"naive_mae1_{target}" or "naive_mae1". If None, attempts to load from
        data/processed/container_meta_8h.parquet if present.
    fallback_rate : float, default 0.0
        Fallback rate over all forecasts for this model and target.
    n_models : int, default 1
        Number of fitted model instances needed to cover all containers.
    n_jobs : int, default 1
        Number of parallel workers used in fitting.

    Returns
    -------
    list[dict]
        One dict per (container_id, split) combination.  Matches the
        metrics_long.csv schema in AGENTS.md §12.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Resolve naive_mae1 mapping
    n_map: dict[str, float] = {}
    if isinstance(naive_mae1_map, pd.DataFrame):
        col = f"naive_mae1_{target}" if f"naive_mae1_{target}" in naive_mae1_map.columns else "naive_mae1"
        if col in naive_mae1_map.columns and "container_id" in naive_mae1_map.columns:
            n_map = dict(zip(naive_mae1_map["container_id"], naive_mae1_map[col].astype(float)))
    elif isinstance(naive_mae1_map, Mapping):
        n_map = {str(k): float(v) for k, v in naive_mae1_map.items()}
    elif naive_mae1_map is None:
        cm_path = Path("data/processed/container_meta_8h.parquet")
        if cm_path.exists():
            try:
                cm_df = pd.read_parquet(cm_path)
                col = f"naive_mae1_{target}"
                if col in cm_df.columns and "container_id" in cm_df.columns:
                    n_map = dict(zip(cm_df["container_id"], cm_df[col].astype(float)))
            except Exception:
                n_map = {}

    rows = []
    for (c_id, sp), grp_idx in meta.groupby(["container_id", "split"]).groups.items():
        yt = y_true[grp_idx]
        yp = y_pred[grp_idx]
        mae_val = mae(yt, yp)
        rmse_val = rmse(yt, yp)
        smape_val = smape(yt, yp)
        naive_denom = n_map.get(str(c_id), 0.0)
        mase_val = mase(mae_val, naive_denom)

        rows.append({
            "run_id": run_id,
            "dataset": dataset,
            "model": model,
            "horizon_steps": horizon,
            "target": target,
            "container_id": c_id,
            "split": sp,
            "mae": mae_val,
            "rmse": rmse_val,
            "smape": smape_val,
            "mase": mase_val,
            "train_time_s": train_time_s,
            "pred_latency_ms": pred_latency_ms,
            "model_size_kb": model_size_kb,
            "n_models": n_models,
            "n_jobs": n_jobs,
            "fallback_rate": fallback_rate,
            "seed": seed,
        })
    return rows
