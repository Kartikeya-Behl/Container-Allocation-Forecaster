"""
src/models/arima.py
===================
ARIMA forecaster — per-container.

Model selection
---------------
Grid: p in {0,1,2,3}, d in {0,1}, q in {0,1,2}  => 24 orders.
Trend: 'c' (constant) when d=0, 'n' (none) when d=1.
The order with the lowest AIC on TRAIN ONLY is chosen per container.

Prediction at origin t
-----------------------
The fitted result is applied to the history y[:t+1] with refit=False:
    fitted_res.apply(y_hist, refit=False).forecast(max_h)
This fixes the ARIMA parameters at their train-estimated values while
extending the Kalman filter over the new history — no re-estimation.

Failures
--------
On any failure (convergence, non-stationarity, etc.) the container falls
back to the naive last-value forecast.  All fallbacks are counted.

Parallelism
-----------
joblib n_jobs=-1 across containers in fit().

Tuning logs
-----------
results/tuning/arima_<target>_choices.csv
    container_id, order (p,d,q), trend, aic, fallback (bool)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.arima.model import ARIMA

from models.base import MODELS, Forecaster

logger = logging.getLogger(__name__)

# Candidate (p, d, q) orders
_ORDERS: list[tuple[int, int, int]] = [
    (p, d, q)
    for p in range(4)
    for d in range(2)
    for q in range(3)
]  # 24 combinations


def _trend_for_d(d: int) -> str:
    """Return trend term: 'c' for d=0 (allow constant), 'n' for d>0."""
    return "c" if d == 0 else "n"


def _fit_order(
    y: np.ndarray,
    order: tuple[int, int, int],
) -> tuple[object | None, float]:
    """Fit ARIMA with given order on y. Return (result, AIC) or (None, inf).

    Parameters
    ----------
    y : ndarray, shape [n_train]
    order : (p, d, q)

    Returns
    -------
    (result, aic) or (None, inf)
    """
    p, d, q = order
    trend = _trend_for_d(d)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ARIMA(y, order=order, trend=trend)
            res = m.fit()
        aic = float(res.aic)
        return res, aic if np.isfinite(aic) else (None, np.inf)
    except Exception:
        return None, np.inf


def _fit_one_container(
    i: int, y: np.ndarray
) -> tuple[int, object | None, tuple[int, int, int]]:
    """Select best ARIMA order for container i by AIC on train.

    Parameters
    ----------
    i : int
        Container row index.
    y : ndarray, shape [n_train]

    Returns
    -------
    (i, best_result, best_order)
    """
    best_res = None
    best_order = (1, 0, 0)
    best_aic = np.inf

    for order in _ORDERS:
        res, aic = _fit_order(y, order)
        if aic < best_aic:
            best_aic = aic
            best_res = res
            best_order = order

    return i, best_res, best_order


class ARIMAForecaster(Forecaster):
    """Per-container ARIMA forecaster with AIC-based order selection on train.

    Parameters
    ----------
    name : str, default 'arima'
    max_h : int, default 5
    tuning_dir : str or Path, default 'results/tuning'
    n_jobs : int, default -1
        Joblib parallelism for fitting.
    """

    def __init__(
        self,
        name: str = "arima",
        max_h: int = 5,
        tuning_dir: str | Path = "results/tuning",
        n_jobs: int = -1,
    ) -> None:
        """Initialise the ARIMA forecaster."""
        super().__init__(name)
        self.max_h = max_h
        self.tuning_dir = Path(tuning_dir)
        self.n_jobs = n_jobs
        self._fitted: list[object | None] = []
        self._chosen_order: list[tuple[int, int, int]] = []
        self._ids: list[str] = []
        self.fallback_count_: int = 0

    def fit(self, Y_train: np.ndarray, ids: list[str]) -> None:
        """Select ARIMA order per container using AIC on train; fit in parallel.

        Parameters
        ----------
        Y_train : ndarray, shape [n_containers, n_train_steps]
        ids : list[str]
        """
        self._ids = list(ids)
        n_c = Y_train.shape[0]

        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_fit_one_container)(i, Y_train[i]) for i in range(n_c)
        )

        # Sort by i to preserve container order
        results.sort(key=lambda x: x[0])
        self._fitted = [r for _, r, _ in results]
        self._chosen_order = [o for _, _, o in results]

    def save_choices(self, target: str) -> None:
        """Write per-container choices to results/tuning/arima_<target>_choices.csv.

        Parameters
        ----------
        target : str
            'cpu' or 'mem'.
        """
        rows = []
        for i, c_id in enumerate(self._ids):
            res = self._fitted[i]
            p, d, q = self._chosen_order[i]
            trend = _trend_for_d(d)
            aic = float(res.aic) if res is not None and hasattr(res, "aic") else None
            rows.append({
                "container_id": c_id,
                "order": f"({p},{d},{q})",
                "trend": trend,
                "aic": aic,
                "fallback": res is None,
            })
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            self.tuning_dir / f"arima_{target}_choices.csv", index=False
        )

    def predict_origins(
        self,
        Y: np.ndarray,
        origins: list[int],
        max_h: int = 5,
    ) -> np.ndarray:
        """Forecast at each origin using fixed fitted ARIMA parameters.

        Parameters
        ----------
        Y : ndarray, shape [n_containers, n_steps]
        origins : list[int]
        max_h : int, default 5

        Returns
        -------
        ndarray, shape [n_containers, n_origins, max_h]
        """
        n_c = Y.shape[0]
        n_o = len(origins)
        preds = np.full((n_c, n_o, max_h), np.nan, dtype=np.float64)
        fallbacks = 0

        for i in range(n_c):
            res = self._fitted[i]
            for j, t in enumerate(origins):
                y_hist = Y[i, : t + 1]
                fc = self._forecast_one(res, y_hist, max_h)
                if fc is None:
                    fallbacks += 1
                    fc = np.full(max_h, Y[i, t], dtype=np.float64)
                preds[i, j, :] = fc

        self.fallback_count_ += fallbacks
        return preds

    def _forecast_one(
        self, res: object | None, y_hist: np.ndarray, max_h: int
    ) -> np.ndarray | None:
        """Apply fixed ARIMA params to y_hist and forecast max_h steps.

        Parameters
        ----------
        res : fitted ARIMAResults or None
        y_hist : ndarray, shape [t+1]
        max_h : int

        Returns
        -------
        ndarray, shape [max_h] or None on failure
        """
        if res is None:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                applied = res.apply(y_hist, refit=False)
                fc = applied.forecast(max_h)
            return np.asarray(fc, dtype=np.float64)
        except Exception:
            return None


MODELS["arima"] = ARIMAForecaster
