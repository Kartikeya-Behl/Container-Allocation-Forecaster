"""
src/models/base.py
==================
Abstract base class for all forecasting models plus the global model registry.

Contract
--------
* fit(Y_train, ids)  -- train on Y_train [n_containers, n_train_steps].
* predict_origins(Y, origins, max_h) -> ndarray [n_containers, n_origins, max_h]
  - Y is the full matrix [n_containers, n_steps].
  - The prediction at origin t may read ONLY Y[:, :t+1] (columns 0..t inclusive).
  - This constraint is verified by the invariance test in tests/test_models.py.
* size_kb()  -- serialised byte size of all fitted parameters / 1024.

Registry
--------
MODELS : dict[str, type[Forecaster]]
    Maps model name strings (e.g. 'naive', 'ma', 'ets', 'arima') to their class.
    Models register themselves by calling MODELS[name] = cls at module level.
"""

from __future__ import annotations

import io
import pickle
from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    """Abstract base for container-level univariate forecasters.

    Subclasses must implement fit, predict_origins, and (optionally) size_kb.

    Parameters
    ----------
    name : str
        Short identifier used in results files (e.g. 'naive', 'arima').
    """

    def __init__(self, name: str) -> None:
        """Initialise with a model name string."""
        self.name = name

    @abstractmethod
    def fit(self, Y_train: np.ndarray, ids: list[str]) -> None:
        """Fit the model on training data.

        Parameters
        ----------
        Y_train : ndarray, shape [n_containers, n_train_steps]
            Training observations; each row is one container's series.
        ids : list[str]
            Container identifiers aligned with the rows of Y_train.
        """

    @abstractmethod
    def predict_origins(
        self,
        Y: np.ndarray,
        origins: list[int],
        max_h: int = 5,
    ) -> np.ndarray:
        """Produce multi-horizon forecasts for every container at each origin.

        Contract: the forecast at origin t uses ONLY Y[:, :t+1].
        The full matrix Y is passed so the model can access the history
        up to the origin without receiving pre-sliced arrays; slicing is
        the model's responsibility.

        Parameters
        ----------
        Y : ndarray, shape [n_containers, n_steps]
            Full observation matrix (train + val + test combined).
        origins : list[int]
            Step indices at which to produce forecasts.
        max_h : int, default 5
            Number of steps ahead to forecast (h = 1 .. max_h).

        Returns
        -------
        ndarray, shape [n_containers, n_origins, max_h]
            preds[i, j, h-1] = forecast for container i at origin origins[j],
            horizon h.
        """

    def size_kb(self) -> float:
        """Return serialised size of fitted parameters in kilobytes.

        Uses pickle to serialise the instance.  Subclasses may override
        with a lighter measurement if pickle is too slow.

        Returns
        -------
        float
            Size in KB.
        """
        buf = io.BytesIO()
        pickle.dump(self, buf)
        return buf.tell() / 1024.0


# ---------------------------------------------------------------------------
# Global model registry
# ---------------------------------------------------------------------------

MODELS: dict[str, type[Forecaster]] = {}
"""Maps model name strings to Forecaster subclasses.

Populated at module import time by each model module (naive, ma, ets, arima)
calling ``MODELS['name'] = MyClass`` at the bottom of their file.
"""
