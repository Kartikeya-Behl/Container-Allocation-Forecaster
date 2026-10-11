"""
src/eval/results.py
===================
Append-only persistence of metrics and predictions; timing helpers.

All writes are strictly append-only (never overwrite).  If a predictions
parquet already exists the new rows are appended and the file is rewritten;
this preserves old runs while adding the new run_id.

Functions
---------
new_run_id()                          -> str  (timestamp-based unique ID)
machine_info()                        -> dict (OS, CPU, RAM, Python, library versions)
time_fit(fn)                          -> (result, elapsed_s: float)
measure_latency(predict_fn, n=100)    -> float  (median ms per single-origin prediction)
append_metrics(rows)                  -> None  (appends to results/metrics_long.csv)
save_predictions(df, dataset, model, horizon, target) -> None
"""

from __future__ import annotations

import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_RESULTS_ROOT = Path("results")
_METRICS_CSV = _RESULTS_ROOT / "metrics_long.csv"
_PREDICTIONS_DIR = _RESULTS_ROOT / "predictions"
_TUNING_DIR = _RESULTS_ROOT / "tuning"

# Expected columns for metrics_long.csv (AGENTS.md §12)
_METRICS_COLS = [
    "run_id", "dataset", "model", "horizon_steps", "target",
    "container_id", "split", "mae", "rmse", "smape", "mase",
    "train_time_s", "pred_latency_ms", "model_size_kb", "seed",
]

# Expected columns for predictions parquet (AGENTS.md §12)
_PRED_COLS = ["run_id", "container_id", "origin_step", "target_step", "split", "y_true", "y_pred"]


def new_run_id() -> str:
    """Generate a unique run ID combining a UTC timestamp and a short random hex suffix.

    Returns
    -------
    str
        e.g. '20261011T183045_a3f9'
    """
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    suffix = uuid.uuid4().hex[:4]
    return f"{ts}_{suffix}"


def machine_info() -> dict:
    """Collect OS, CPU, RAM, Python version, and key library versions.

    Returns
    -------
    dict
        Keys: os, cpu, ram_gb, python, numpy, pandas, sklearn, xgboost.
    """
    info: dict = {
        "os": f"{platform.system()} {platform.release()} {platform.machine()}",
        "cpu": f"{os.cpu_count()} logical cores",
        "python": sys.version.split()[0],
    }
    # RAM (best-effort; psutil optional)
    try:
        import psutil  # type: ignore
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        info["ram_gb"] = "unknown (psutil not installed)"

    # Library versions
    for lib in ("numpy", "pandas", "sklearn", "xgboost"):
        try:
            mod = __import__(lib if lib != "sklearn" else "sklearn")
            info[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[lib] = "not installed"
    return info


def time_fit(fn: Callable) -> tuple:
    """Call fn() and return (result, elapsed_seconds).

    Parameters
    ----------
    fn : callable
        Zero-argument callable (typically a lambda wrapping model.fit(...)).

    Returns
    -------
    result : any
        Whatever fn() returns.
    elapsed_s : float
        Wall-clock seconds elapsed.
    """
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    return result, elapsed


def measure_latency(predict_fn: Callable, n: int = 100) -> float:
    """Measure median single-origin prediction latency in milliseconds.

    predict_fn should accept a single row (1-D array or 2-D [1, n_features])
    and return a scalar prediction.  It is called n times and the median
    wall-clock time per call is returned.

    Parameters
    ----------
    predict_fn : callable
        Zero-argument callable that performs one single-row prediction.
    n : int, default 100
        Number of repeated calls.

    Returns
    -------
    float
        Median latency in milliseconds.
    """
    times_ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        predict_fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times_ms))


def append_metrics(rows: list[dict]) -> None:
    """Append metric rows to results/metrics_long.csv (never overwrites).

    Creates the file with a header if it does not exist; otherwise appends
    without writing the header again.

    Parameters
    ----------
    rows : list[dict]
        Each dict must have keys matching _METRICS_COLS (AGENTS.md §12).
    """
    _RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows, columns=_METRICS_COLS)
    write_header = not _METRICS_CSV.exists()
    df_new.to_csv(_METRICS_CSV, mode="a", header=write_header, index=False)


def save_predictions(
    df: pd.DataFrame,
    dataset: str,
    model: str,
    horizon: int,
    target: str,
) -> None:
    """Save val+test predictions to a parquet file (append if file exists).

    File path: results/predictions/<dataset>_<model>_h<horizon>_<target>.parquet
    Columns: run_id, container_id, origin_step, target_step, split, y_true, y_pred.

    If the file already exists both the old and new rows are kept (different run_ids).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns in _PRED_COLS.
    dataset, model : str
    horizon : int
    target : str  ('cpu' or 'mem')
    """
    _PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{dataset}_{model}_h{horizon}_{target}.parquet"
    dest = _PREDICTIONS_DIR / filename

    missing = set(_PRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"save_predictions: df missing required columns: {missing}")

    if dest.exists():
        existing = pd.read_parquet(dest)
        df = pd.concat([existing, df[_PRED_COLS]], ignore_index=True)
    else:
        df = df[_PRED_COLS].copy()

    df.to_parquet(dest, index=False, engine="pyarrow")
