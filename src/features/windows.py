"""
src/features/windows.py
=======================
Build supervised (X, y) datasets from container utilisation matrices.

Design rules (AGENTS.md §2, §14)
---------------------------------
- A row has origin t and target y[t+h].  The row belongs to the split of
  its TARGET step (t+h), never of its origin step.
- Lag features at origin t are y[t], y[t-1], ..., y[t-n_lags+1].
  All lag values are strictly in the past relative to the target.
- Features NEVER cross container boundaries.
- Val/test rows may use lag values from earlier splits; that is legitimate
  past information.

Main function: make_supervised()
Optional extra features (use_extra=True, off by default): rolling stats,
cross-target lags, minute-of-hour indicator.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from data.loader import split_bounds


# Split name literal type for clarity
Split = Literal["train", "val", "test"]


def _split_label(target_step: int, bounds: tuple) -> Split:
    """Return the split name for a given target step.

    Parameters
    ----------
    target_step : int
        The step index of the prediction target.
    bounds : tuple
        ((train_start, train_end), (val_start, val_end), (test_start, test_end))

    Returns
    -------
    str : 'train', 'val', or 'test'
    """
    (ts, te), (vs, ve), (qs, qe) = bounds
    if ts <= target_step < te:
        return "train"
    if vs <= target_step < ve:
        return "val"
    if qs <= target_step < qe:
        return "test"
    raise ValueError(f"target_step={target_step} outside all split ranges {bounds}")


def make_supervised(
    Y: np.ndarray,
    horizon: int,
    n_lags: int,
    split: Split,
    bounds: tuple,
    container_ids: list[str] | None = None,
    use_extra: bool = False,
    Y_other: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build supervised feature matrices from a container × step utilisation matrix.

    Parameters
    ----------
    Y : np.ndarray, shape [n_containers, n_steps]
        Utilisation values (cpu or mem) ordered by container_id then step.
    horizon : int
        Forecast horizon in steps (h >= 1).
    n_lags : int
        Number of lag features.  lag_1 = y[t], lag_2 = y[t-1], ..., lag_n = y[t-n+1].
    split : {'train', 'val', 'test'}
        Which split to build rows for.
    bounds : tuple
        Output of split_bounds(n_steps, fracs): ((ts, te), (vs, ve), (qs, qe)).
    container_ids : list[str] or None
        Container ID strings aligned with rows of Y.  If None, integer indices are used.
    use_extra : bool, default False
        If True, add rolling stats (mean/std over 5 and 10 steps ending at t),
        lags of Y_other (same count), and minute_of_hour = origin_step % 60.
        NOT used by default; off until ablation stage.
    Y_other : np.ndarray or None
        Second utilisation channel (e.g. mem when Y=cpu). Required if use_extra=True.

    Returns
    -------
    X : np.ndarray, shape [n_rows, n_features]
        Feature matrix (lag features, optionally + extra).
    y_delta : np.ndarray, shape [n_rows]
        y[t+h] - y[t]  (change from last observed to target).
    y_level : np.ndarray, shape [n_rows]
        y[t+h]  (absolute target value).
    y_last : np.ndarray, shape [n_rows]
        y[t]  (last observed value at origin; y_last + y_delta == y_level exactly).
    meta : pd.DataFrame
        One row per supervised row with columns:
        container_idx, container_id, origin_step, target_step, split.

    Notes
    -----
    Row counts (per container):
      train: n_rows = train_end - horizon - (n_lags - 1)
             origins t in [n_lags-1, train_end-1-horizon]  (inclusive both ends)
      val:   one row per target step s in [val_start, val_end),  origin t = s - horizon
      test:  one row per target step s in [test_start, test_end), origin t = s - horizon
    For val/test the origin t may fall inside the train or val range; that is intentional.
    """
    n_containers, n_steps = Y.shape
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = bounds

    if container_ids is None:
        container_ids = [str(i) for i in range(n_containers)]

    if use_extra and Y_other is None:
        raise ValueError("use_extra=True requires Y_other to be provided.")

    # Determine origin step range for this split
    if split == "train":
        # Origin t must satisfy: t >= n_lags-1  AND  t+h < tr_e  => t <= tr_e-1-h
        t_min = n_lags - 1
        t_max = tr_e - 1 - horizon
        if t_max < t_min:
            raise ValueError(
                f"No train rows possible: n_lags={n_lags}, horizon={horizon}, train_end={tr_e}"
            )
        origin_steps = list(range(t_min, t_max + 1))
    elif split == "val":
        # One row per target step s in val range; origin = s - horizon
        origin_steps = [s - horizon for s in range(va_s, va_e)]
    elif split == "test":
        origin_steps = [s - horizon for s in range(te_s, te_e)]
    else:
        raise ValueError(f"split must be 'train', 'val', or 'test'; got {repr(split)}")

    # Validate origins are within series bounds and have enough history
    valid_origins = [t for t in origin_steps if t >= n_lags - 1 and t + horizon < n_steps]
    if len(valid_origins) != len(origin_steps):
        raise ValueError(
            f"Some origin steps are out of range for n_lags={n_lags}, horizon={horizon}, "
            f"n_steps={n_steps}. Split={split}, first origin={origin_steps[0]}, "
            f"last origin={origin_steps[-1]}"
        )

    n_origins = len(valid_origins)
    n_rows_total = n_containers * n_origins

    # Build base lag feature names: lag_1..lag_n
    n_base_features = n_lags
    if use_extra:
        # rolling mean(5), rolling std(5), rolling mean(10), rolling std(10)
        # + n_lags lags of Y_other + minute_of_hour
        n_extra_features = 4 + n_lags + 1
    else:
        n_extra_features = 0
    n_features = n_base_features + n_extra_features

    X = np.empty((n_rows_total, n_features), dtype=np.float64)
    y_delta = np.empty(n_rows_total, dtype=np.float64)
    y_level = np.empty(n_rows_total, dtype=np.float64)
    y_last = np.empty(n_rows_total, dtype=np.float64)

    meta_rows = []
    row_idx = 0

    for c_idx in range(n_containers):
        series = Y[c_idx]  # shape [n_steps]
        c_id = container_ids[c_idx]

        for t in valid_origins:
            target_step = t + horizon
            y_t = series[t]
            y_tph = series[target_step]

            # lag_1 = y[t], lag_2 = y[t-1], ..., lag_n = y[t-n+1]
            lags = series[t - n_lags + 1 : t + 1][::-1]  # most recent first

            feat = list(lags)

            if use_extra:
                # Rolling stats over strictly past window (steps t-4..t inclusive for w=5)
                win5 = series[max(0, t - 4) : t + 1]
                win10 = series[max(0, t - 9) : t + 1]
                feat.append(float(np.mean(win5)))
                feat.append(float(np.std(win5, ddof=0)))
                feat.append(float(np.mean(win10)))
                feat.append(float(np.std(win10, ddof=0)))
                # Lags of Y_other
                other_lags = Y_other[c_idx][t - n_lags + 1 : t + 1][::-1]
                feat.extend(other_lags)
                # Minute of hour (origin step modulo 60)
                feat.append(float(t % 60))

            X[row_idx] = feat
            y_delta[row_idx] = y_tph - y_t
            y_level[row_idx] = y_tph
            y_last[row_idx] = y_t

            meta_rows.append({
                "container_idx": c_idx,
                "container_id": c_id,
                "origin_step": t,
                "target_step": target_step,
                "split": split,
            })
            row_idx += 1

    meta = pd.DataFrame(meta_rows)
    return X, y_delta, y_level, y_last, meta
