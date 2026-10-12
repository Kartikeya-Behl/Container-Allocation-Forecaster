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
    i: int,
    y_train: np.ndarray,
    y_full: np.ndarray | None = None,
    val_origins: list[int] | None = None,
    max_h: int = 5,
) -> tuple[int, object | None, tuple[int, int, int], float | None, float | None, list[tuple[int, int, int]]]:
    """Fit all 24 ARIMA orders on train, shortlist lowest-AIC + (0,1,0)/(0,1,1), and select by val MAE.

    Parameters
    ----------
    i : int
        Container row index.
    y_train : ndarray, shape [n_train]
    y_full : ndarray, shape [n_steps], optional
    val_origins : list[int], optional
    max_h : int, default 5

    Returns
    -------
    (i, best_result, best_order, best_val_score, best_aic, shortlist)
    """
    results: dict[tuple[int, int, int], tuple[object | None, float]] = {}

    for order in _ORDERS:
        res, aic = _fit_order(y_train, order)
        results[order] = (res, aic)

    # 5 lowest-AIC orders
    valid_orders = [o for o in _ORDERS if np.isfinite(results[o][1])]
    valid_orders.sort(key=lambda o: results[o][1])
    shortlist = valid_orders[:5]

    # Add (0,1,0) and (0,1,1) with trend 'n' (dedupe)
    for special in [(0, 1, 0), (0, 1, 1)]:
        if special in results and results[special][0] is not None and special not in shortlist:
            shortlist.append(special)

    if not shortlist:
        shortlist = [(1, 0, 0)]

    best_order = shortlist[0]
    best_res, best_aic = results.get(best_order, (None, np.inf))
    best_val_score: float | None = None

    if val_origins and y_full is not None:
        min_score = np.inf
        for order in shortlist:
            res, aic = results[order]
            if res is None:
                continue
            errors = []
            for t in val_origins:
                y_hist = y_full[: t + 1]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        applied = res.apply(y_hist, refit=False)
                        fc = applied.forecast(max_h)
                    target_vals = y_full[t + 1 : t + 1 + max_h]
                    errors.append(float(np.mean(np.abs(fc - target_vals))))
                except Exception:
                    errors.append(np.inf)
                    break
            score = float(np.mean(errors)) if errors else np.inf
            if score < min_score:
                min_score = score
                best_order = order
                best_res = res
                best_aic = aic
        best_val_score = min_score if np.isfinite(min_score) else None

    return i, best_res, best_order, best_val_score, best_aic, shortlist


class ARIMAForecaster(Forecaster):
    """Per-container ARIMA forecaster with AIC shortlist and val MAE selection.

    Parameters
    ----------
    name : str, default 'arima'
    max_h : int, default 5
    tuning_dir : str or Path, default 'results/tuning'
    n_jobs : int, default -1
        Joblib parallelism for fitting across containers.
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
        self._best_val_score: list[float | None] = []
        self._best_aic: list[float | None] = []
        self._shortlists: list[list[tuple[int, int, int]]] = []
        self._ids: list[str] = []
        self.fallback_count_: int = 0

    def fit(
        self,
        Y_train: np.ndarray,
        ids: list[str],
        Y_full: np.ndarray | None = None,
        val_start: int | None = None,
        val_end: int | None = None,
    ) -> None:
        """Fit ARIMA orders on train, shortlist, and select by validation MAE.

        Parameters
        ----------
        Y_train : ndarray, shape [n_containers, n_train_steps]
        ids : list[str]
        Y_full : ndarray, shape [n_containers, n_steps], optional
        val_start : int, optional
        val_end : int, optional
        """
        self._ids = list(ids)
        n_c = Y_train.shape[0]

        val_origins: list[int] = []
        if Y_full is not None and val_start is not None and val_end is not None:
            val_origins = [t for t in range(val_start - 1, val_end - 5) if t >= 0]

        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_fit_one_container)(
                i,
                Y_train[i],
                Y_full[i] if Y_full is not None else None,
                val_origins,
                self.max_h,
            )
            for i in range(n_c)
        )

        # Sort by i to preserve container order
        results.sort(key=lambda x: x[0])
        self._fitted = [r[1] for r in results]
        self._chosen_order = [r[2] for r in results]
        self._best_val_score = [r[3] for r in results]
        self._best_aic = [r[4] for r in results]
        self._shortlists = [r[5] for r in results]

    def save_choices(self, target: str, run_tag: str | None = None) -> None:
        """Write per-container choices to results/tuning/<run_tag>/arima_<target>_choices.csv.

        Parameters
        ----------
        target : str
            'cpu' or 'mem'.
        run_tag : str, optional
            Run tag directory identifier.
        """
        rows = []
        for i, c_id in enumerate(self._ids):
            res = self._fitted[i]
            p, d, q = self._chosen_order[i]
            trend = _trend_for_d(d)
            aic = self._best_aic[i]
            shortlist_str = ";".join([f"({sp},{sd},{sq})" for sp, sd, sq in self._shortlists[i]])
            rows.append({
                "container_id": c_id,
                "order": f"({p},{d},{q})",
                "trend": trend,
                "val_score": self._best_val_score[i],
                "aic": aic,
                "selected_by": "val_mae",
                "shortlist": shortlist_str,
                "fallback": res is None,
                **({"run_tag": run_tag} if run_tag else {}),
            })
        dest_dir = self.tuning_dir / run_tag if run_tag else self.tuning_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            dest_dir / f"arima_{target}_choices.csv", index=False
        )

    def size_kb(self, container_idx: int = 0) -> float:
        """Return serialised size of minimal payload for one container model instance.

        Payload: order tuple (p, d, q), trend flag, and the params array.
        """
        import io
        import pickle
        if not self._fitted or container_idx >= len(self._fitted) or self._fitted[container_idx] is None:
            return 0.0
        res = self._fitted[container_idx]
        p, d, q = self._chosen_order[container_idx]
        trend = _trend_for_d(d)
        params = res.params
        payload = {
            "order": (p, d, q),
            "trend": trend,
            "params": params,
        }
        buf = io.BytesIO()
        pickle.dump(payload, buf)
        return buf.tell() / 1024.0

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
