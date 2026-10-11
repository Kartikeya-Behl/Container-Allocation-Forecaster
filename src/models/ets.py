"""
src/models/ets.py
=================
ETS (Exponential Smoothing) forecaster — per-container.

Model selection
---------------
Candidates = {simple, additive-trend, damped-additive-trend} x {Box-Cox off, on}
= 6 combinations.  The combination with the lowest AICc on TRAIN only is chosen
for each container.

Prediction at origin t
-----------------------
The fitted parameters are FIXED (no re-fitting).  The model is re-initialised
with the training-fitted parameters and run over Y[i, :t+1] using
``ETSModel(...).fit(... initial_level=...).simulate`` or the
``apply`` / re-initialise approach.  We use:
    ``res.apply(y_history, refit=False)`` -> ``forecast(max_h)``

Failures
--------
On any convergence or runtime failure the container falls back to the naive
last-value forecast, and the fallback is counted.

Tuning logs
-----------
results/tuning/ets_<target>_choices.csv
    container_id, config, aicc, fallback (bool)
"""

from __future__ import annotations

import logging
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from models.base import MODELS, Forecaster

logger = logging.getLogger(__name__)

# Candidate configurations: (error, trend, damped, use_boxcox)
_CONFIGS: list[tuple] = [
    # simple exponential smoothing
    ("add", None, False, False),
    ("add", None, False, True),
    # additive trend (Holt)
    ("add", "add", False, False),
    ("add", "add", False, True),
    # damped additive trend
    ("add", "add", True, False),
    ("add", "add", True, True),
]

_CONFIG_NAMES = [
    "simple",
    "simple_bc",
    "add_trend",
    "add_trend_bc",
    "damped_trend",
    "damped_trend_bc",
]


def _fit_ets(
    y: np.ndarray,
    error: str,
    trend: str | None,
    damped: bool,
    use_boxcox: bool,
) -> object | None:
    """Fit a single ETS model. Returns the fitted result or None on failure.

    Parameters
    ----------
    y : ndarray, shape [n_train]
    error, trend : str or None
    damped : bool
    use_boxcox : bool

    Returns
    -------
    fitted result or None
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kwargs: dict = dict(error=error, trend=trend, damped_trend=damped)
            if use_boxcox:
                kwargs["initialization_method"] = "estimated"
                kwargs["use_boxcox"] = True
            m = ETSModel(y, **kwargs)
            res = m.fit(disp=False, maxiter=200)
        return res
    except Exception:
        return None


class ETSForecaster(Forecaster):
    """Per-container ETS forecaster with AICc-based model selection on train.

    Parameters
    ----------
    name : str, default 'ets'
    max_h : int, default 5
    tuning_dir : str or Path, default 'results/tuning'
    """

    def __init__(
        self,
        name: str = "ets",
        max_h: int = 5,
        tuning_dir: str | Path = "results/tuning",
    ) -> None:
        """Initialise the ETS forecaster."""
        super().__init__(name)
        self.max_h = max_h
        self.tuning_dir = Path(tuning_dir)
        # Per-container fitted results and chosen config indices
        self._fitted: list[object | None] = []
        self._chosen_cfg: list[int] = []  # index into _CONFIGS
        self._ids: list[str] = []
        self.fallback_count_: int = 0

    def fit(self, Y_train: np.ndarray, ids: list[str]) -> None:
        """Select ETS config per container using AICc on train.

        Parameters
        ----------
        Y_train : ndarray, shape [n_containers, n_train_steps]
        ids : list[str]
        """
        self._ids = list(ids)
        self._fitted = []
        self._chosen_cfg = []
        self.fallback_count_ = 0
        log_rows: list[dict] = []

        for i, c_id in enumerate(ids):
            y = Y_train[i]
            best_res = None
            best_aicc = np.inf
            best_cfg_idx = 0

            for cfg_idx, (error, trend, damped, bc) in enumerate(_CONFIGS):
                res = _fit_ets(y, error, trend, damped, bc)
                if res is not None:
                    try:
                        aicc = res.aicc
                        if np.isfinite(aicc) and aicc < best_aicc:
                            best_aicc = aicc
                            best_res = res
                            best_cfg_idx = cfg_idx
                    except Exception:
                        pass

            self._fitted.append(best_res)
            self._chosen_cfg.append(best_cfg_idx)
            log_rows.append({
                "container_id": c_id,
                "config": _CONFIG_NAMES[best_cfg_idx],
                "aicc": best_aicc if np.isfinite(best_aicc) else None,
            })

        self._save_tuning_log(log_rows, target="__fit__")

    def _save_tuning_log(self, rows: list[dict], target: str) -> None:
        """Write tuning log CSV (called by fit and predict_origins via runner)."""
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            self.tuning_dir / f"ets_{target}_choices.csv", index=False
        )

    def save_choices(self, target: str) -> None:
        """Persist per-container choices to results/tuning/ets_<target>_choices.csv.

        Parameters
        ----------
        target : str
            'cpu' or 'mem' (used in filename).
        """
        rows = [
            {
                "container_id": self._ids[i],
                "config": _CONFIG_NAMES[self._chosen_cfg[i]],
                "aicc": None,  # already saved in fit; kept for schema compliance
                "fallback": self._fitted[i] is None,
            }
            for i in range(len(self._ids))
        ]
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            self.tuning_dir / f"ets_{target}_choices.csv", index=False
        )

    def predict_origins(
        self,
        Y: np.ndarray,
        origins: list[int],
        max_h: int = 5,
    ) -> np.ndarray:
        """Forecast at each origin using fixed fitted ETS parameters.

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
        n_c = Y.shape[0]
        n_o = len(origins)
        preds = np.full((n_c, n_o, max_h), np.nan, dtype=np.float64)
        fallbacks = 0

        for i in range(n_c):
            res = self._fitted[i]
            for j, t in enumerate(origins):
                y_hist = Y[i, : t + 1]
                forecast = self._forecast_one(res, y_hist, max_h)
                if forecast is None:
                    fallbacks += 1
                    forecast = np.full(max_h, Y[i, t], dtype=np.float64)
                preds[i, j, :] = forecast

        self.fallback_count_ += fallbacks
        return preds

    def _forecast_one(
        self, res: object | None, y_hist: np.ndarray, max_h: int
    ) -> np.ndarray | None:
        """Apply fixed ETS params to y_hist and forecast max_h steps.

        Parameters
        ----------
        res : fitted ETSResults or None
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


MODELS["ets"] = ETSForecaster
