"""
src/models/ets.py
=================
ETS (Exponential Smoothing) forecaster — per-container.

Model selection
---------------
Candidates = {simple, additive-trend, damped-additive-trend} x {Box-Cox off, on}
= 6 combinations. The combination with the lowest AICc on TRAIN only is chosen
for each container.

Prediction at origin t
-----------------------
The fitted parameters are FIXED (no re-fitting). At each origin t, the model is
instantiated over history up to t with the training-estimated smoothing parameters
(smoothing_level, smoothing_trend, damping_trend, lamda) fixed and run with
fit(optimized=False, ...) to forecast max_h steps.

Failures
--------
On any convergence or runtime failure the container falls back to the naive
last-value forecast, and the fallback is counted.

Tuning logs
-----------
results/tuning/<run_tag>/ets_<target>_choices.csv
    container_id, config, aicc, fallback (bool), run_tag
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from models.base import MODELS, Forecaster

logger = logging.getLogger(__name__)

# Candidate configurations: (trend, damped, use_boxcox)
_CONFIGS: list[tuple] = [
    (None, False, False),
    (None, False, True),
    ("add", False, False),
    ("add", False, True),
    ("add", True, False),
    ("add", True, True),
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
    trend: str | None,
    damped: bool,
    use_boxcox: bool,
) -> object | None:
    """Fit a single ExponentialSmoothing model. Return fitted result or None.

    Parameters
    ----------
    y : ndarray, shape [n_train]
    trend : str or None
    damped : bool
    use_boxcox : bool

    Returns
    -------
    fitted result or None
    """
    if use_boxcox and np.any(y <= 0):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ExponentialSmoothing(
                y,
                trend=trend,
                damped_trend=damped,
                use_boxcox=use_boxcox,
            )
            res = m.fit()
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
        self._fitted: list[object | None] = []
        self._chosen_cfg: list[int] = []
        self._best_aicc: list[float] = []
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
        """Fit all 6 ETS candidate configs on train and select by validation MAE.

        Parameters
        ----------
        Y_train : ndarray, shape [n_containers, n_train_steps]
        ids : list[str]
        Y_full : ndarray, shape [n_containers, n_steps], optional
            Full observation matrix containing validation history and targets.
        val_start : int, optional
            First step of validation split.
        val_end : int, optional
            Step index after last validation step.
        """
        self._ids = list(ids)
        self._fitted = []
        self._chosen_cfg = []
        self._best_aicc = []
        self._best_val_score = []
        self.fallback_count_ = 0

        # Determine validation origins: t in [val_start - 1, val_end - 1 - 5]
        val_origins: list[int] = []
        if Y_full is not None and val_start is not None and val_end is not None:
            val_origins = [t for t in range(val_start - 1, val_end - 5) if t >= 0]

        for i, _ in enumerate(ids):
            y = Y_train[i]
            # 1. Fit all candidates on train
            candidate_res: list[object | None] = []
            candidate_aicc: list[float] = []
            for cfg_idx, (trend, damped, bc) in enumerate(_CONFIGS):
                res = _fit_ets(y, trend, damped, bc)
                candidate_res.append(res)
                if res is not None:
                    try:
                        candidate_aicc.append(float(res.aicc))
                    except Exception:
                        candidate_aicc.append(np.inf)
                else:
                    candidate_aicc.append(np.inf)

            # 2. Select by validation forecast error (or AICc fallback if no val given)
            best_cfg_idx = 0
            best_val_score = np.inf
            best_res = None
            best_aicc = np.inf

            if val_origins and Y_full is not None:
                for cfg_idx, res in enumerate(candidate_res):
                    if res is None:
                        continue
                    errors = []
                    for t in val_origins:
                        y_hist = Y_full[i, : t + 1]
                        fc = self._forecast_one(res, cfg_idx, y_hist, self.max_h)
                        if fc is None:
                            errors.append(np.inf)
                            break
                        target_vals = Y_full[i, t + 1 : t + 1 + self.max_h]
                        errors.append(float(np.mean(np.abs(fc - target_vals))))
                    score = float(np.mean(errors)) if errors else np.inf
                    if score < best_val_score:
                        best_val_score = score
                        best_cfg_idx = cfg_idx
                        best_res = res
                        best_aicc = candidate_aicc[cfg_idx]
            else:
                for cfg_idx, res in enumerate(candidate_res):
                    if res is not None and candidate_aicc[cfg_idx] < best_aicc:
                        best_aicc = candidate_aicc[cfg_idx]
                        best_cfg_idx = cfg_idx
                        best_res = res

            self._fitted.append(best_res)
            self._chosen_cfg.append(best_cfg_idx)
            self._best_aicc.append(best_aicc)
            self._best_val_score.append(best_val_score if np.isfinite(best_val_score) else None)

    def save_choices(self, target: str, run_tag: str | None = None) -> None:
        """Persist per-container choices to results/tuning/<run_tag>/ets_<target>_choices.csv.

        Parameters
        ----------
        target : str
            'cpu' or 'mem' (used in filename).
        run_tag : str, optional
            Run tag directory identifier.
        """
        rows = [
            {
                "container_id": self._ids[i],
                "config": _CONFIG_NAMES[self._chosen_cfg[i]],
                "val_score": self._best_val_score[i],
                "aic": self._best_aicc[i] if np.isfinite(self._best_aicc[i]) else None,
                "aicc": self._best_aicc[i] if np.isfinite(self._best_aicc[i]) else None,
                "selected_by": "val_mae",
                "fallback": self._fitted[i] is None,
                **({"run_tag": run_tag} if run_tag else {}),
            }
            for i in range(len(self._ids))
        ]
        dest_dir = self.tuning_dir / run_tag if run_tag else self.tuning_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            dest_dir / f"ets_{target}_choices.csv", index=False
        )

    def size_kb(self, container_idx: int = 0) -> float:
        """Return serialised size of minimal payload for one container model instance.

        Payload: config tuple + fitted smoothing parameters and Box-Cox lambda.
        """
        import io
        import pickle
        if not self._fitted or container_idx >= len(self._fitted) or self._fitted[container_idx] is None:
            return 0.0
        res = self._fitted[container_idx]
        cfg_idx = self._chosen_cfg[container_idx]
        cfg_tuple = _CONFIGS[cfg_idx]
        kw = {
            k: v
            for k, v in res.params.items()
            if k in ["smoothing_level", "smoothing_trend", "damping_trend"]
            and not np.isnan(v)
        }
        lam = res.params.get("lamda", None) if cfg_tuple[2] else None
        payload = {
            "config": cfg_tuple,
            "params": kw,
            "lamda": lam,
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
            cfg_idx = self._chosen_cfg[i]
            for j, t in enumerate(origins):
                y_hist = Y[i, : t + 1]
                forecast = self._forecast_one(res, cfg_idx, y_hist, max_h)
                if forecast is None:
                    fallbacks += 1
                    forecast = np.full(max_h, Y[i, t], dtype=np.float64)
                preds[i, j, :] = forecast

        self.fallback_count_ += fallbacks
        return preds

    def _forecast_one(
        self,
        res: object | None,
        cfg_idx: int,
        y_hist: np.ndarray,
        max_h: int,
    ) -> np.ndarray | None:
        """Apply fixed ETS params to y_hist and forecast max_h steps.

        Parameters
        ----------
        res : fitted model result or None
        cfg_idx : int
        y_hist : ndarray, shape [t+1]
        max_h : int

        Returns
        -------
        ndarray, shape [max_h] or None on failure
        """
        if res is None:
            return None
        trend, damped, use_boxcox = _CONFIGS[cfg_idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kw = {
                    k: v
                    for k, v in res.params.items()
                    if k in ["smoothing_level", "smoothing_trend", "damping_trend"]
                    and not np.isnan(v)
                }
                lam = res.params.get("lamda", False) if use_boxcox else False
                if use_boxcox and np.any(y_hist <= 0):
                    return None
                m_hist = ExponentialSmoothing(
                    y_hist,
                    trend=trend,
                    damped_trend=damped,
                    use_boxcox=lam,
                )
                res_hist = m_hist.fit(optimized=False, **kw)
                fc = res_hist.forecast(max_h)
            return np.asarray(fc, dtype=np.float64)
        except Exception:
            return None


MODELS["ets"] = ETSForecaster
