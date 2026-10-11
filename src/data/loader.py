"""
src/data/loader.py
==================
Shared utilities for loading container time-series datasets and computing
canonical temporal split boundaries.
"""

from typing import Tuple


def split_bounds(
    n_steps: int,
    fracs: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Compute half-open step index boundaries [start, end) for train, val, and test splits.

    Matches the exact integer truncation semantics of chrono_split:
      n_train = int(n_steps * fracs[0])
      n_val   = int(n_steps * fracs[1])
      n_test  = n_steps - (n_train + n_val)

    Parameters
    ----------
    n_steps : int
        Total number of steps in the series (e.g. 240 for 4h, 480 for 8h).
    fracs : tuple of float, default (0.70, 0.15, 0.15)
        Fractions for train, val, and test.

    Returns
    -------
    tuple of ((train_start, train_end), (val_start, val_end), (test_start, test_end))
        Half-open index intervals [start, end).
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
