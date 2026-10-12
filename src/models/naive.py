"""
src/models/naive.py
===================
Naive last-value forecaster: y_hat[t+h] = y[t] for all h.

No hyperparameters; always produces the last observed value.
"""

from __future__ import annotations

import numpy as np

from models.base import MODELS, Forecaster


class NaiveForecaster(Forecaster):
    """Naive (last-value) forecaster.

    For every container and every forecast horizon h, the prediction equals
    the last observed value at the origin step:
        y_hat[t + h] = y[t]

    Parameters
    ----------
    name : str
        Model name tag, default 'naive'.
    """

    def __init__(self, name: str = "naive") -> None:
        """Initialise the naive forecaster."""
        super().__init__(name)
        self._n_containers: int = 0
        self._ids: list[str] = []

    def fit(self, Y_train: np.ndarray, ids: list[str]) -> None:
        """Record the number of containers; no parameters to estimate.

        Parameters
        ----------
        Y_train : ndarray, shape [n_containers, n_train_steps]
        ids : list[str]
            Container identifiers.
        """
        self._n_containers = Y_train.shape[0]
        self._ids = list(ids)

    def size_kb(self, container_idx: int = 0) -> float:
        """Return serialised size of minimal parameters (0 KB for naive)."""
        return 0.0

    def predict_origins(
        self,
        Y: np.ndarray,
        origins: list[int],
        max_h: int = 5,
    ) -> np.ndarray:
        """Return last-value forecasts for all containers, origins, and horizons.

        Parameters
        ----------
        Y : ndarray, shape [n_containers, n_steps]
            Full observation matrix.
        origins : list[int]
            Origin step indices.
        max_h : int, default 5
            Number of forecast horizons.

        Returns
        -------
        ndarray, shape [n_containers, n_origins, max_h]
            preds[i, j, h-1] = Y[i, origins[j]]  (same for all h).
        """
        n_c = Y.shape[0]
        n_o = len(origins)
        preds = np.empty((n_c, n_o, max_h), dtype=np.float64)
        for j, t in enumerate(origins):
            # last observed value at step t (column t); h does not matter
            preds[:, j, :] = Y[:, t : t + 1]  # broadcast over max_h axis
        return preds


MODELS["naive"] = NaiveForecaster
