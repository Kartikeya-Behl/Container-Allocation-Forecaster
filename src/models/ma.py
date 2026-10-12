"""
src/models/ma.py
================
Moving-average forecaster.

For every horizon h:
    y_hat[t + h] = mean(y[t-k+1 : t+1])  (the last k observations at origin t).

The window k is the same for all horizons and all containers.
k is chosen from the candidate set {2, 3, 5, 10, 15, 20} by minimising the
mean validation MAE averaged over all containers and all horizons h=1..max_h.

No per-container fitting; only k is global and selected on val.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models.base import MODELS, Forecaster

logger = logging.getLogger(__name__)

_K_CANDIDATES = [2, 3, 5, 10, 15, 20]


def _ma_preds_for_k(
    Y: np.ndarray,
    origins: list[int],
    k: int,
    max_h: int,
) -> np.ndarray:
    """Compute MA(k) forecasts for given origins without look-ahead.

    Parameters
    ----------
    Y : ndarray, shape [n_containers, n_steps]
    origins : list[int]
    k : int
        Window size.
    max_h : int

    Returns
    -------
    ndarray, shape [n_containers, n_origins, max_h]
    """
    n_c, _ = Y.shape
    n_o = len(origins)
    preds = np.empty((n_c, n_o, max_h), dtype=np.float64)
    for j, t in enumerate(origins):
        # Use at most k values strictly <= t (no look-ahead)
        start = max(0, t - k + 1)
        window = Y[:, start : t + 1]  # shape [n_c, actual_window]
        mean_val = window.mean(axis=1, keepdims=True)  # [n_c, 1]
        preds[:, j, :] = mean_val  # same mean for all h
    return preds


class MAForecaster(Forecaster):
    """Moving-average forecaster with k chosen on the validation split.

    Parameters
    ----------
    name : str, default 'ma'
    max_h : int, default 5
    k_candidates : list[int], default [2, 3, 5, 10, 15, 20]
    tuning_dir : str or Path, optional
        Directory for saving the k-selection log.
        Default: 'results/tuning'.
    """

    def __init__(
        self,
        name: str = "ma",
        max_h: int = 5,
        k_candidates: list[int] | None = None,
        tuning_dir: str | Path = "results/tuning",
    ) -> None:
        """Initialise the MA forecaster."""
        super().__init__(name)
        self.max_h = max_h
        self.k_candidates = k_candidates if k_candidates is not None else _K_CANDIDATES
        self.tuning_dir = Path(tuning_dir)
        self.k_: int | None = None  # chosen after select_k

    def fit(self, Y_train: np.ndarray, ids: list[str]) -> None:
        """Store training data for future reference; k is chosen in select_k.

        Parameters
        ----------
        Y_train : ndarray, shape [n_containers, n_train_steps]
        ids : list[str]
        """
        self._Y_train = Y_train
        self._ids = list(ids)

    def select_k(
        self,
        Y_full: np.ndarray,
        val_origins: list[int],
        target: str = "target",
        val_start: int | None = None,
        val_end: int | None = None,
        run_tag: str | None = None,
    ) -> int:
        """Choose the best k on validation by mean MAE (h-averaged, container-averaged).

        Only (origin, h) pairs whose target_step = origin + h falls strictly
        within [val_start, val_end) are used, so k depends only on val targets.

        Parameters
        ----------
        Y_full : ndarray, shape [n_containers, n_steps]
            Full observation matrix (train + val + test).
        val_origins : list[int]
            Origin indices into Y_full that generate validation targets.
        target : str
            Target name for logging (e.g. 'cpu').
        val_start : int or None
            First step index of the validation split (inclusive).
            If None, no filtering is applied (all target steps accepted).
        val_end : int or None
            One-past-the-end step index of the validation split.
            If None, no filtering is applied.
        run_tag : str, optional
            Run tag directory identifier.

        Returns
        -------
        int
            Chosen k value.
        """
        best_k = self.k_candidates[0]
        best_mae = np.inf
        mae_by_k: dict[int, float] = {}

        for k in self.k_candidates:
            preds = _ma_preds_for_k(Y_full, val_origins, k, self.max_h)
            errors = []
            for j, t in enumerate(val_origins):
                for h in range(1, self.max_h + 1):
                    target_step = t + h
                    if target_step >= Y_full.shape[1]:
                        continue
                    # Skip if target step is outside val split (prevents test leakage)
                    if val_start is not None and target_step < val_start:
                        continue
                    if val_end is not None and target_step >= val_end:
                        continue
                    y_true = Y_full[:, target_step]
                    y_pred = preds[:, j, h - 1]
                    errors.append(float(np.mean(np.abs(y_true - y_pred))))
            mae_k = float(np.mean(errors)) if errors else np.inf
            mae_by_k[k] = mae_k
            if mae_k < best_mae:
                best_mae = mae_k
                best_k = k

        self.k_ = best_k
        logger.info("MA: chose k=%d (val MAE=%.6f) for target=%s", best_k, best_mae, target)

        # Save tuning log
        dest_dir = self.tuning_dir / run_tag if run_tag else self.tuning_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "k": k,
                "val_mae": v,
                "chosen": k == best_k,
                **({"run_tag": run_tag} if run_tag else {}),
            }
            for k, v in mae_by_k.items()
        ]
        pd.DataFrame(rows).to_csv(
            dest_dir / f"ma_{target}_k_search.csv", index=False
        )
        return best_k

    def size_kb(self, container_idx: int = 0) -> float:
        """Return serialised size of pickled k in kilobytes."""
        import io
        import pickle
        buf = io.BytesIO()
        pickle.dump(self.k_, buf)
        return buf.tell() / 1024.0

    def predict_origins(
        self,
        Y: np.ndarray,
        origins: list[int],
        max_h: int = 5,
    ) -> np.ndarray:
        """Return MA(k) forecasts.

        Parameters
        ----------
        Y : ndarray, shape [n_containers, n_steps]
            Full observation matrix.
        origins : list[int]
        max_h : int, default 5

        Returns
        -------
        ndarray, shape [n_containers, n_origins, max_h]
        """
        if self.k_ is None:
            raise RuntimeError("Call select_k before predict_origins.")
        return _ma_preds_for_k(Y, origins, self.k_, max_h)


MODELS["ma"] = MAForecaster
