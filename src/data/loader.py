"""
src/data/loader.py
==================
Utilities for loading container time-series datasets and computing canonical
temporal split boundaries.

Functions
---------
split_bounds(n_steps, fracs)  -> ((train_start, train_end), (val_start, val_end), (test_start, test_end))
load_long(path)               -> pd.DataFrame  (long format, sorted by container_id, step)
to_matrix(df, target)         -> (container_ids: list[str], Y: ndarray [n_containers, n_steps])
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def split_bounds(
    n_steps: int,
    fracs: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Compute half-open step index boundaries [start, end) for train, val, and test splits.

    Uses the same integer-truncation semantics as chrono_split:
      n_train = int(n_steps * fracs[0])
      n_val   = int(n_steps * fracs[1])
      n_test  = n_steps - (n_train + n_val)

    Parameters
    ----------
    n_steps : int
        Total number of steps per container (e.g. 240 for 4 h, 480 for 8 h).
    fracs : tuple of float, default (0.70, 0.15, 0.15)
        Fractions for (train, val, test). Must sum to 1.0 within 1e-6.

    Returns
    -------
    tuple of ((train_start, train_end), (val_start, val_end), (test_start, test_end))
        Half-open index intervals [start, end).  For 480 steps and default fracs:
        train=(0,336), val=(336,408), test=(408,480).
    """
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {fracs} (sum={sum(fracs)})")

    train_frac, val_frac, _ = fracs
    n_train = int(n_steps * train_frac)
    n_val = int(n_steps * val_frac)

    train_bound = (0, n_train)
    val_bound = (n_train, n_train + n_val)
    test_bound = (n_train + n_val, n_steps)

    return train_bound, val_bound, test_bound


def load_long(path: str | Path) -> pd.DataFrame:
    """Load a processed container parquet file into a long-format DataFrame.

    The file must contain at least columns: container_id, step, cpu, mem.
    Rows are sorted by (container_id, step) so downstream code can assume
    contiguous per-container blocks in ascending step order.

    Parameters
    ----------
    path : str or Path
        Path to the parquet file produced by build_dataset.py.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame sorted by (container_id, step).
    """
    df = pd.read_parquet(Path(path))
    df = df.sort_values(["container_id", "step"]).reset_index(drop=True)
    required = {"container_id", "step", "cpu", "mem"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet file missing required columns: {missing}")
    return df


def to_matrix(
    df: pd.DataFrame,
    target: str,
) -> Tuple[list, np.ndarray]:
    """Pivot long-format container data into a 2-D matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame from load_long (must have container_id, step, and `target`).
    target : str
        Column name to use as values, e.g. 'cpu' or 'mem'.

    Returns
    -------
    container_ids : list[str]
        Sorted list of container_id values (row labels of Y).
    Y : np.ndarray of shape [n_containers, n_steps]
        Matrix with Y[i, t] = df value for container_ids[i] at step t.
        Containers and steps are both sorted in ascending order.
    """
    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found in DataFrame.")

    pivot = df.pivot(index="container_id", columns="step", values=target)
    pivot = pivot.sort_index()  # alphabetical order of container_ids

    container_ids = list(pivot.index)
    Y = pivot.values.astype(np.float64)

    if np.isnan(Y).any():
        raise ValueError("to_matrix: NaN values found in matrix. Check dataset completeness.")

    return container_ids, Y
