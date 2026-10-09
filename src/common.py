"""
src/common.py
=============
Shared utilities used across all modules in the container-forecast project.

Functions
---------
set_seed(seed)          -- Fix Python, NumPy random seeds for reproducibility.
load_config(path)       -- Load a YAML config file and return as a dict.
get_logger(name)        -- Return a standard logging.Logger with consistent formatting.
save_run(result_dict, config, name)
                        -- Persist results, config, git commit hash and library
                           versions to results/<name>_<timestamp>/.

These are the ONLY helpers that may write to disk outside of an experiment script.
"""

import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix Python built-in and NumPy random seeds.

    Call this at the very top of every script and notebook before any
    stochastic operation. The seed value must match configs/default.yaml.

    Parameters
    ----------
    seed : int
        Random seed (e.g. 42 from default.yaml).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file (e.g. 'configs/default.yaml').

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a Logger with a consistent ISO-timestamp format.

    Parameters
    ----------
    name : str
        Logger name, typically __name__ of the calling module.
    level : int
        Logging level (default: logging.INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Run persistence
# ---------------------------------------------------------------------------

def _get_git_commit() -> str:
    """Return the current git commit hash, or 'no-git' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "no-git"


def _get_library_versions() -> dict[str, str]:
    """Return version strings for key project libraries."""
    libs = ["pandas", "numpy", "sklearn", "statsmodels", "xgboost", "scipy"]
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for lib in libs:
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not-installed"
    return versions


def save_run(
    result_dict: dict[str, Any],
    config: dict[str, Any],
    name: str,
    results_root: str | Path = "results",
) -> Path:
    """Save a completed experiment run to disk.

    Creates the directory::

        results/<name>_<UTC-timestamp>/

    and writes:

    * ``results.json``  -- the metrics / result dict
    * ``config.yaml``   -- the config used for this run
    * ``meta.json``     -- git commit hash, library versions, timestamp

    Parameters
    ----------
    result_dict : dict
        Metrics and any other scalar results to record.
    config : dict
        The configuration dictionary used for this run.
    name : str
        Short experiment name (e.g. 'baseline_rf', 'arima_h1').
    results_root : str | Path
        Root directory for results (default: 'results').

    Returns
    -------
    Path
        The run directory that was created.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(results_root) / f"{name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write metrics
    with (run_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(result_dict, fh, indent=2, default=str)

    # Write config
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.dump(config, fh, default_flow_style=False)

    # Write metadata
    meta = {
        "name": name,
        "timestamp_utc": timestamp,
        "git_commit": _get_git_commit(),
        "library_versions": _get_library_versions(),
    }
    with (run_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    logger = get_logger(__name__)
    logger.info("Run saved to %s", run_dir)
    return run_dir
