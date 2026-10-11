"""
src/data/split.py
=================
Chronological train/val/test splitting per time-series group without shuffling.
Enforces non-negotiable temporal splitting rules from AGENTS.md.
"""

from typing import Tuple, Optional
import pandas as pd


def chrono_split(
    df: pd.DataFrame,
    group_col: Optional[str],
    time_col: str,
    fracs: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-series data chronologically per group into train, val, and test.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing time series.
    group_col : Optional[str]
        Column name identifying individual series/containers. If None,
        the entire DataFrame is treated as a single chronological series.
    time_col : str
        Column name representing timestamp or chronological sequence.
    fracs : tuple of float, default (0.7, 0.15, 0.15)
        Fractions for (train, val, test). Must sum to 1.0 within 1e-6.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        DataFrames corresponding to train, val, and test splits.
    """
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {fracs} (sum={sum(fracs)})")

    train_frac, val_frac, _ = fracs

    if group_col is None:
        sorted_df = df.sort_values(by=time_col)
        n = len(sorted_df)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        return (
            sorted_df.iloc[:n_train].copy(),
            sorted_df.iloc[n_train : n_train + n_val].copy(),
            sorted_df.iloc[n_train + n_val :].copy(),
        )

    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in df.groupby(group_col, sort=False):
        sorted_group = group.sort_values(by=time_col)
        n = len(sorted_group)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_parts.append(sorted_group.iloc[:n_train])
        val_parts.append(sorted_group.iloc[n_train : n_train + n_val])
        test_parts.append(sorted_group.iloc[n_train + n_val :])

    train_df = pd.concat(train_parts, axis=0) if train_parts else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_parts, axis=0) if val_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts, axis=0) if test_parts else pd.DataFrame(columns=df.columns)

    return train_df, val_df, test_df
