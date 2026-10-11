"""
tests/test_smoke.py
===================
Smoke tests that verify the project scaffold is correctly set up.

These tests do NOT test model logic — they only confirm that:
  1. Key imports work without errors.
  2. The default config loads and has the required keys.
  3. set_seed() is deterministic.
  4. save_run() creates the expected directory structure.
  5. The chronological split proportions are correct (no shuffling).

Run with:  pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

# Make src/ importable when running pytest from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from common import get_logger, load_config, save_run, set_seed
from data.split import chrono_split

# Path to the default config, relative to repo root
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# 1. Import smoke test
# ---------------------------------------------------------------------------

class TestImports:
    """Verify that all required libraries can be imported."""

    def test_pandas_import(self):
        import pandas  # noqa: F401

    def test_numpy_import(self):
        import numpy  # noqa: F401

    def test_matplotlib_import(self):
        import matplotlib  # noqa: F401

    def test_sklearn_import(self):
        import sklearn  # noqa: F401

    def test_statsmodels_import(self):
        import statsmodels  # noqa: F401

    def test_xgboost_import(self):
        import xgboost  # noqa: F401

    def test_scipy_import(self):
        import scipy  # noqa: F401

    def test_docker_import(self):
        import docker  # noqa: F401

    def test_yaml_import(self):
        import yaml  # noqa: F401

    def test_pyarrow_import(self):
        import pyarrow  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Config loading
# ---------------------------------------------------------------------------

class TestConfig:
    """Verify the default config file is valid and has required keys."""

    def test_config_file_exists(self):
        assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"

    def test_config_loads(self):
        cfg = load_config(CONFIG_PATH)
        assert isinstance(cfg, dict)

    def test_required_keys(self):
        cfg = load_config(CONFIG_PATH)
        required = ["seed", "interval_s", "split", "horizons_steps", "lags", "paths"]
        for key in required:
            assert key in cfg, f"Missing required config key: '{key}'"

    def test_split_sums_to_one(self):
        cfg = load_config(CONFIG_PATH)
        s = cfg["split"]
        total = s["train"] + s["val"] + s["test"]
        assert abs(total - 1.0) < 1e-9, f"Split proportions sum to {total}, expected 1.0"

    def test_split_proportions_correct(self):
        cfg = load_config(CONFIG_PATH)
        s = cfg["split"]
        assert s["train"] == 0.70, f"Train split should be 0.70, got {s['train']}"
        assert s["val"]   == 0.15, f"Val split should be 0.15, got {s['val']}"
        assert s["test"]  == 0.15, f"Test split should be 0.15, got {s['test']}"

    def test_seed_is_42(self):
        cfg = load_config(CONFIG_PATH)
        assert cfg["seed"] == 42

    def test_paths_keys_present(self):
        cfg = load_config(CONFIG_PATH)
        for key in ["raw", "interim", "processed", "results"]:
            assert key in cfg["paths"], f"Missing paths key: '{key}'"

    def test_config_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# 3. set_seed determinism
# ---------------------------------------------------------------------------

class TestSetSeed:
    """Verify set_seed() produces deterministic outputs."""

    def test_numpy_determinism(self):
        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b, err_msg="set_seed(42) not deterministic for numpy")

    def test_different_seeds_differ(self):
        set_seed(42)
        a = np.random.rand(10)
        set_seed(99)
        b = np.random.rand(10)
        assert not np.array_equal(a, b), "Different seeds produced identical output"


# ---------------------------------------------------------------------------
# 4. save_run structure
# ---------------------------------------------------------------------------

class TestSaveRun:
    """Verify save_run() creates the expected directory and files."""

    def test_run_dir_created(self, tmp_path):
        cfg = load_config(CONFIG_PATH)
        run_dir = save_run({"mae": 0.05, "rmse": 0.07}, cfg, "smoke_test",
                           results_root=tmp_path)
        assert run_dir.exists(), "Run directory was not created"

    def test_results_json_exists(self, tmp_path):
        cfg = load_config(CONFIG_PATH)
        run_dir = save_run({"mae": 0.05}, cfg, "smoke_test", results_root=tmp_path)
        assert (run_dir / "results.json").exists()

    def test_config_yaml_exists(self, tmp_path):
        cfg = load_config(CONFIG_PATH)
        run_dir = save_run({"mae": 0.05}, cfg, "smoke_test", results_root=tmp_path)
        assert (run_dir / "config.yaml").exists()

    def test_meta_json_exists(self, tmp_path):
        cfg = load_config(CONFIG_PATH)
        run_dir = save_run({"mae": 0.05}, cfg, "smoke_test", results_root=tmp_path)
        assert (run_dir / "meta.json").exists()

    def test_meta_has_git_commit(self, tmp_path):
        cfg = load_config(CONFIG_PATH)
        run_dir = save_run({"mae": 0.05}, cfg, "smoke_test", results_root=tmp_path)
        meta = json.loads((run_dir / "meta.json").read_text())
        assert "git_commit" in meta

    def test_meta_has_library_versions(self, tmp_path):
        cfg = load_config(CONFIG_PATH)
        run_dir = save_run({"mae": 0.05}, cfg, "smoke_test", results_root=tmp_path)
        meta = json.loads((run_dir / "meta.json").read_text())
        assert "library_versions" in meta
        assert "python" in meta["library_versions"]

    def test_results_content_preserved(self, tmp_path):
        cfg = load_config(CONFIG_PATH)
        metrics = {"mae": 0.123, "rmse": 0.456, "model": "rf"}
        run_dir = save_run(metrics, cfg, "smoke_test", results_root=tmp_path)
        loaded = json.loads((run_dir / "results.json").read_text())
        assert loaded["mae"] == 0.123
        assert loaded["rmse"] == 0.456


# ---------------------------------------------------------------------------
# 5. Chronological split correctness (no shuffle)
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    """Verify that chrono_split preserves temporal order per group without shuffling."""

    @pytest.fixture
    def multi_group_df(self):
        """Create sample DataFrame with 2 groups and 100 timestamps each."""
        rows = []
        for grp in ["container_A", "container_B"]:
            for t in range(100):
                rows.append({"container_id": grp, "timestamp": t, "val": t * 10})
        return pd.DataFrame(rows)

    def test_no_overlap_between_splits(self, multi_group_df):
        train, val, test = chrono_split(multi_group_df, group_col="container_id", time_col="timestamp")
        assert len(set(train.index) & set(val.index)) == 0
        assert len(set(train.index) & set(test.index)) == 0
        assert len(set(val.index) & set(test.index)) == 0

    def test_train_comes_first_in_time_per_group(self, multi_group_df):
        train, val, test = chrono_split(multi_group_df, group_col="container_id", time_col="timestamp")
        for grp in ["container_A", "container_B"]:
            t_train = train[train["container_id"] == grp]["timestamp"]
            t_val = val[val["container_id"] == grp]["timestamp"]
            t_test = test[test["container_id"] == grp]["timestamp"]
            assert t_train.max() < t_val.min(), f"Train max time {t_train.max()} >= Val min time {t_val.min()} for {grp}"
            assert t_val.max() < t_test.min(), f"Val max time {t_val.max()} >= Test min time {t_test.min()} for {grp}"

    def test_splits_cover_all_rows(self, multi_group_df):
        train, val, test = chrono_split(multi_group_df, group_col="container_id", time_col="timestamp")
        total_len = len(train) + len(val) + len(test)
        assert total_len == len(multi_group_df)

    def test_approximate_proportions_per_group(self, multi_group_df):
        train, val, test = chrono_split(multi_group_df, group_col="container_id", time_col="timestamp")
        for grp in ["container_A", "container_B"]:
            n_grp = 100
            assert len(train[train["container_id"] == grp]) == int(n_grp * 0.70)
            assert len(val[val["container_id"] == grp]) == int(n_grp * 0.15)
            assert len(test[test["container_id"] == grp]) == n_grp - int(n_grp * 0.70) - int(n_grp * 0.15)

    def test_single_series_split_without_group(self):
        df = pd.DataFrame({"timestamp": range(100), "val": range(100)})
        train, val, test = chrono_split(df, group_col=None, time_col="timestamp")
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert train["timestamp"].max() < val["timestamp"].min() < test["timestamp"].min()

    def test_invalid_fracs_raises(self, multi_group_df):
        with pytest.raises(ValueError):
            chrono_split(multi_group_df, group_col="container_id", time_col="timestamp", fracs=(0.5, 0.2, 0.1))

    def test_logger_returns_logger(self):
        logger = get_logger("test_logger")
        import logging
        assert isinstance(logger, logging.Logger)

