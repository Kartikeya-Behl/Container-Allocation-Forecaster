"""
src/run_forecast.py
===================
Runner script for the four classical forecasting models.

Usage
-----
    python src/run_forecast.py --models naive ma ets arima \
        --targets cpu mem --n-containers 20 --run-tag val20

Each (model, target) combination is run independently and can be resumed:
if a row with the same run_tag and combination already exists in
results/metrics_long.csv it is skipped.

Workflow per (model, target)
-----------------------------
1. Load data; select n_containers containers (seeded).
2. Compute split bounds (never hard-coded).
3. fit() on train columns only.
4. Compute all required origins for val + test (target steps h=1..5).
5. predict_origins() over those origins; record fit time and latency.
6. Build prediction rows and metric rows per (container, horizon, split).
7. Append to results/metrics_long.csv and save predictions parquet.
8. Save model choices / fallback counts to results/tuning/.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Make `src/` importable when run as `python src/run_forecast.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import get_logger, set_seed, save_run
from data.loader import load_long, split_bounds, to_matrix
from eval.metrics import mae, rmse, smape, mase
from eval.results import (
    append_metrics,
    machine_info,
    measure_latency,
    new_run_id,
    save_predictions,
)

# Import all model modules to populate MODELS registry
import models.naive  # noqa: F401
import models.ma  # noqa: F401
import models.ets  # noqa: F401
import models.arima  # noqa: F401
from models.base import MODELS

logger = get_logger(__name__)

_DATASET_DEFAULT = "data/processed/alibaba_v2022_8h.parquet"
_DATASET_TAG = "alibaba8h"
_MAX_H = 5
_SEED = 42

# ---- helpers ----------------------------------------------------------------


def _load_naive_mae1(target: str) -> dict[str, float]:
    """Load container_meta_8h naive_mae1 for MASE denominator.

    Parameters
    ----------
    target : str
        'cpu' or 'mem'.

    Returns
    -------
    dict mapping container_id -> naive_mae1.
    """
    cm_path = Path("data/processed/container_meta_8h.parquet")
    if not cm_path.exists():
        logger.warning("container_meta_8h.parquet not found; MASE denominator = 0")
        return {}
    cm = pd.read_parquet(cm_path)
    col = f"naive_mae1_{target}"
    if col not in cm.columns:
        logger.warning("Column %s not in container_meta; MASE denominator = 0", col)
        return {}
    return dict(zip(cm["container_id"].astype(str), cm[col].astype(float)))


def _origins_for_split(
    val_start: int, val_end: int, test_start: int, test_end: int, max_h: int
) -> tuple[list[int], list[int]]:
    """Compute forecast origins needed so that target steps cover val and test.

    For a target step ts in [val_start, val_end) or [test_start, test_end),
    the origin is ts - h for each h in 1..max_h.  We collect the union of all
    needed origins, deduplicated and sorted.

    Parameters
    ----------
    val_start, val_end, test_start, test_end : int
    max_h : int

    Returns
    -------
    (val_origins, test_origins) : both are sorted lists of int.
    """
    val_origins: set[int] = set()
    for ts in range(val_start, val_end):
        for h in range(1, max_h + 1):
            o = ts - h
            if o >= 0:
                val_origins.add(o)

    test_origins: set[int] = set()
    for ts in range(test_start, test_end):
        for h in range(1, max_h + 1):
            o = ts - h
            if o >= 0:
                test_origins.add(o)

    return sorted(val_origins), sorted(test_origins)


def _build_pred_rows(
    preds: np.ndarray,
    Y: np.ndarray,
    ids: list[str],
    origins: list[int],
    split_name: str,
    run_id: str,
    val_start: int,
    val_end: int,
    test_start: int,
    test_end: int,
    max_h: int,
) -> list[dict]:
    """Build prediction rows for the predictions parquet schema.

    Parameters
    ----------
    preds : ndarray, shape [n_containers, n_origins, max_h]
    Y : ndarray, shape [n_containers, n_steps]
    ids : list[str]
    origins : list[int]
    split_name : str  ('val' or 'test')
    run_id : str
    val_start, val_end, test_start, test_end : int
    max_h : int

    Returns
    -------
    list of dicts with columns: run_id, container_id, origin_step, target_step,
    split, y_true, y_pred.
    """
    n_steps = Y.shape[1]
    rows = []
    for i, c_id in enumerate(ids):
        for j, t in enumerate(origins):
            for h in range(1, max_h + 1):
                ts = t + h
                if ts >= n_steps:
                    continue
                # Determine split membership by target step (AGENTS.md §14)
                if val_start <= ts < val_end:
                    sp = "val"
                elif test_start <= ts < test_end:
                    sp = "test"
                else:
                    continue
                if sp != split_name:
                    continue
                rows.append({
                    "run_id": run_id,
                    "container_id": c_id,
                    "origin_step": t,
                    "target_step": ts,
                    "split": sp,
                    "y_true": float(Y[i, ts]),
                    "y_pred": float(preds[i, j, h - 1]),
                })
    return rows


def _compute_metrics(
    pred_rows: list[dict],
    run_id: str,
    dataset: str,
    model_name: str,
    target: str,
    train_time_s: float,
    pred_latency_ms: float,
    model_size_kb: float,
    seed: int,
    naive_mae1_map: dict[str, float],
) -> list[dict]:
    """Compute per-container per-horizon per-split metrics from pred_rows.

    Parameters
    ----------
    pred_rows : list of dicts (run_id, container_id, origin_step, target_step,
                               split, y_true, y_pred).
    run_id, dataset, model_name, target : str
    train_time_s, pred_latency_ms, model_size_kb : float
    seed : int
    naive_mae1_map : dict[str, float]

    Returns
    -------
    list of metric dicts matching AGENTS.md §12 schema.
    """
    if not pred_rows:
        return []
    df = pd.DataFrame(pred_rows)
    metric_rows = []
    for (c_id, h, sp), grp in df.groupby(["container_id", "target_step", "split"]):
        # Compute horizon from the group (all rows have same target_step)
        # We need per-horizon metrics: group by container_id, horizon, split.
        pass
    # Re-group correctly: horizon = target_step - origin_step
    df["horizon"] = df["target_step"] - df["origin_step"]
    for (c_id, h, sp), grp in df.groupby(["container_id", "horizon", "split"]):
        yt = grp["y_true"].to_numpy(dtype=np.float64)
        yp = grp["y_pred"].to_numpy(dtype=np.float64)
        mae_val = mae(yt, yp)
        rmse_val = rmse(yt, yp)
        smape_val = smape(yt, yp)
        naive_d = naive_mae1_map.get(str(c_id), 0.0)
        mase_val = mase(mae_val, naive_d)
        metric_rows.append({
            "run_id": run_id,
            "dataset": dataset,
            "model": model_name,
            "horizon_steps": int(h),
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
            "seed": seed,
        })
    return metric_rows


def _already_done(
    run_tag: str, model_name: str, target: str, metrics_csv: Path
) -> bool:
    """Return True if this (run_tag, model, target) already has rows in metrics_long.csv.

    Parameters
    ----------
    run_tag : str
    model_name : str
    target : str
    metrics_csv : Path

    Returns
    -------
    bool
    """
    if not metrics_csv.exists():
        return False
    df = pd.read_csv(metrics_csv, usecols=["run_id", "model", "target"])
    mask = (
        df["run_id"].str.startswith(run_tag)
        & (df["model"] == model_name)
        & (df["target"] == target)
    )
    return bool(mask.any())


# ---- main run loop ----------------------------------------------------------


def run(
    model_names: list[str],
    targets: list[str],
    dataset_path: str,
    n_containers: int | None,
    run_tag: str,
    seed: int = _SEED,
) -> None:
    """Execute the full forecast run for each (model, target) combination.

    Parameters
    ----------
    model_names : list[str]
    targets : list[str]  e.g. ['cpu', 'mem']
    dataset_path : str
    n_containers : int or None  (None = all)
    run_tag : str
    seed : int
    """
    set_seed(seed)
    metrics_csv = Path("results/metrics_long.csv")

    df_long = load_long(dataset_path)
    all_ids_set = sorted(df_long["container_id"].unique())

    if n_containers is not None and n_containers < len(all_ids_set):
        rng = random.Random(seed)
        selected_ids = sorted(rng.sample(all_ids_set, n_containers))
    else:
        selected_ids = all_ids_set

    # Filter long df to selected containers
    df_long = df_long[df_long["container_id"].isin(selected_ids)].copy()

    minfo = machine_info()
    logger.info("Machine: %s", minfo)
    logger.info("Containers: %d, models: %s, targets: %s", len(selected_ids), model_names, targets)

    for model_name in model_names:
        if model_name not in MODELS:
            logger.error("Unknown model '%s'. Available: %s", model_name, list(MODELS))
            continue

        for target in targets:
            if _already_done(run_tag, model_name, target, metrics_csv):
                logger.info("SKIP %s/%s (run_tag=%s already in metrics_long.csv)", model_name, target, run_tag)
                continue

            logger.info("=== %s / %s ===", model_name, target)
            run_id = f"{run_tag}_{model_name}_{target}"

            # Build matrix
            ids, Y = to_matrix(df_long, target)

            n_steps = Y.shape[1]
            (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = split_bounds(n_steps)

            Y_train = Y[:, tr_s:tr_e]

            # ---- fit --------------------------------------------------------
            model = MODELS[model_name]()
            t0_fit = time.perf_counter()

            if model_name == "ma":
                # MA needs fit + select_k on val origins (target steps strictly in val)
                model.fit(Y_train, ids)
                val_origins_for_k, _ = _origins_for_split(va_s, va_e, te_s, te_e, _MAX_H)
                model.select_k(Y, val_origins_for_k, target=target, val_start=va_s, val_end=va_e)
            else:
                model.fit(Y_train, ids)

            fit_time_s = time.perf_counter() - t0_fit
            logger.info("Fit done in %.2fs", fit_time_s)

            # Save choices / tuning logs
            if hasattr(model, "save_choices"):
                model.save_choices(target)

            # ---- predict all needed origins ----------------------------------
            val_origins, test_origins = _origins_for_split(va_s, va_e, te_s, te_e, _MAX_H)
            all_origins = sorted(set(val_origins) | set(test_origins))

            preds_all = model.predict_origins(Y, all_origins, max_h=_MAX_H)

            # ---- latency measurement ----------------------------------------
            # Pick 100 random (container, origin) pairs
            rng_lat = np.random.default_rng(seed)
            n_lat = min(100, len(ids) * len(all_origins))
            lat_i = rng_lat.integers(0, len(ids), n_lat)
            lat_j = rng_lat.integers(0, len(all_origins), n_lat)
            lat_times = []
            for li, lj in zip(lat_i, lat_j):
                t_single = all_origins[lj]
                t0_lat = time.perf_counter()
                model.predict_origins(Y[li:li+1], [t_single], max_h=_MAX_H)
                lat_times.append((time.perf_counter() - t0_lat) * 1000.0)
            pred_latency_ms = float(np.median(lat_times))
            logger.info("Median latency: %.3f ms", pred_latency_ms)

            model_size_kb = model.size_kb()

            # ---- build prediction rows and save ------------------------------
            # Map origin index in all_origins -> index in preds_all
            origin_to_j = {o: j for j, o in enumerate(all_origins)}

            for split_name, split_origins in [("val", val_origins), ("test", test_origins)]:
                s_j = [origin_to_j[o] for o in split_origins]
                preds_split = preds_all[:, s_j, :]

                rows = _build_pred_rows(
                    preds_split, Y, ids, split_origins, split_name,
                    run_id, va_s, va_e, te_s, te_e, _MAX_H,
                )
                if not rows:
                    continue
                pred_df = pd.DataFrame(rows)
                for h in range(1, _MAX_H + 1):
                    h_df = pred_df[pred_df["target_step"] - pred_df["origin_step"] == h].copy()
                    if h_df.empty:
                        continue
                    save_predictions(h_df, _DATASET_TAG, model_name, h, target)

            # ---- metrics ----------------------------------------------------
            naive_mae1_map = _load_naive_mae1(target)

            # Rebuild all pred rows for metric computation
            all_pred_rows = _build_pred_rows(
                preds_all, Y, ids, all_origins, "all",
                run_id, va_s, va_e, te_s, te_e, _MAX_H,
            )
            # fix split labels (already correct in rows)
            all_pred_rows_fixed: list[dict] = []
            for r in all_pred_rows:
                ts = r["target_step"]
                if va_s <= ts < va_e:
                    r["split"] = "val"
                elif te_s <= ts < te_e:
                    r["split"] = "test"
                else:
                    continue
                all_pred_rows_fixed.append(r)

            metric_rows = _compute_metrics(
                all_pred_rows_fixed,
                run_id=run_id,
                dataset=_DATASET_TAG,
                model_name=model_name,
                target=target,
                train_time_s=fit_time_s,
                pred_latency_ms=pred_latency_ms,
                model_size_kb=model_size_kb,
                seed=seed,
                naive_mae1_map=naive_mae1_map,
            )

            if metric_rows:
                append_metrics(metric_rows)
                logger.info("Appended %d metric rows", len(metric_rows))

            # Log fallbacks
            if hasattr(model, "fallback_count_"):
                logger.info("Fallbacks: %d", model.fallback_count_)

            # Save run record
            save_run(
                result_dict={
                    "model": model_name,
                    "target": target,
                    "n_containers": len(ids),
                    "fit_time_s": fit_time_s,
                    "pred_latency_ms": pred_latency_ms,
                    "model_size_kb": model_size_kb,
                    "fallbacks": getattr(model, "fallback_count_", 0),
                    "machine": minfo,
                },
                config={
                    "run_tag": run_tag,
                    "model": model_name,
                    "target": target,
                    "dataset": dataset_path,
                    "n_containers": n_containers,
                    "seed": seed,
                },
                name=f"{run_tag}_{model_name}_{target}",
            )


# ---- CLI --------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run classical forecasting models.")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    p.add_argument("--targets", nargs="+", default=["cpu", "mem"])
    p.add_argument("--n-containers", type=int, default=None)
    p.add_argument("--dataset", type=str, default=_DATASET_DEFAULT)
    p.add_argument("--run-tag", type=str, default="run")
    p.add_argument("--seed", type=int, default=_SEED)
    return p.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = _parse_args()
    run(
        model_names=args.models,
        targets=args.targets,
        dataset_path=args.dataset,
        n_containers=args.n_containers,
        run_tag=args.run_tag,
        seed=args.seed,
    )
