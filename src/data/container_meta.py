"""
src/data/container_meta.py
==========================
Compute per-container summary statistics and strata tiers strictly from
the training portion of the time series (AGENTS.md rules 1, 2, 14).

Columns produced:
- container_id, msname
- train_std_cpu, train_std_mem
- train_mean_cpu, train_mean_mem
- naive_mae1_cpu, naive_mae1_mem  (in-sample 1-step naive MAE on train: MASE denominator)
- mem_tier: 'flat' if train_std_mem < 0.01 else 'volatile'
- cpu_tier: 'low' / 'mid' / 'high' by terciles of train_std_cpu
- has_restart_train: bool, any 1-step relative drop > 20% in mem on train

IMPORTANT: Tiers and tags are descriptive strata for reporting and
MUST NEVER be used to exclude containers or to select anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import sys

try:
    from data.loader import load_long, split_bounds
except ImportError:
    try:
        from src.data.loader import load_long, split_bounds
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from data.loader import load_long, split_bounds


def compute_container_meta(
    df: pd.DataFrame,
    fracs: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Tuple[pd.DataFrame, dict]:
    """Compute per-container metadata and strata tiers strictly from train steps.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format container DataFrame with columns:
        container_id, msname, step, cpu, mem.
    fracs : tuple of float, default (0.70, 0.15, 0.15)
        Fractions for temporal split.

    Returns
    -------
    meta_df : pd.DataFrame
        One row per container, sorted by container_id.
    manifest : dict
        Manifest containing cutoffs, tier counts, and validation metadata.
    """
    # Determine total steps per container and split boundaries dynamically
    steps_per_c = df.groupby("container_id")["step"].nunique()
    n_steps = int(steps_per_c.iloc[0])
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = split_bounds(n_steps, fracs)

    # Filter STRICTLY to train steps only: [tr_s, tr_e)
    df_train = df[(df["step"] >= tr_s) & (df["step"] < tr_e)].copy()

    records = []
    # Ensure containers are processed in sorted order
    for c_id, grp in df_train.groupby("container_id", sort=True):
        grp_sorted = grp.sort_values("step")
        msname = grp_sorted["msname"].iloc[0]
        cpu_arr = grp_sorted["cpu"].to_numpy(dtype=np.float64)
        mem_arr = grp_sorted["mem"].to_numpy(dtype=np.float64)

        mean_cpu = float(np.mean(cpu_arr))
        std_cpu = float(np.std(cpu_arr, ddof=0))
        mean_mem = float(np.mean(mem_arr))
        std_mem = float(np.std(mem_arr, ddof=0))

        # In-sample 1-step naive forecast error on train: |y[t] - y[t-1]| for t=1..train_end-1
        naive_mae_cpu = float(np.mean(np.abs(np.diff(cpu_arr))))
        naive_mae_mem = float(np.mean(np.abs(np.diff(mem_arr))))

        # Memory restart: any 1-step relative drop > 20% in mem (where prev > 1e-6)
        prev_mem = mem_arr[:-1]
        cur_mem = mem_arr[1:]
        valid_mask = prev_mem > 1e-6
        has_restart = bool(np.any(valid_mask & ((prev_mem - cur_mem) / prev_mem > 0.20)))

        records.append({
            "container_id": c_id,
            "msname": msname,
            "train_std_cpu": std_cpu,
            "train_std_mem": std_mem,
            "train_mean_cpu": mean_cpu,
            "train_mean_mem": mean_mem,
            "naive_mae1_cpu": naive_mae_cpu,
            "naive_mae1_mem": naive_mae_mem,
            "has_restart_train": has_restart,
        })

    meta_df = pd.DataFrame(records)

    # Stratum tiers
    # 1. Memory tier: 'flat' if std < 0.01 else 'volatile'
    meta_df["mem_tier"] = np.where(meta_df["train_std_mem"] < 0.01, "flat", "volatile")

    # 2. CPU tier: terciles of train_std_cpu over all containers
    # Using qcut with 3 quantiles: 'low', 'mid', 'high'
    cpu_terciles = meta_df["train_std_cpu"].quantile([1.0 / 3.0, 2.0 / 3.0]).to_list()
    meta_df["cpu_tier"] = pd.qcut(
        meta_df["train_std_cpu"],
        q=3,
        labels=["low", "mid", "high"],
    ).astype(str)

    # Build manifest
    tier_counts = {
        "mem_tier": meta_df["mem_tier"].value_counts().to_dict(),
        "cpu_tier": meta_df["cpu_tier"].value_counts().to_dict(),
        "has_restart_train": {
            "true": int(meta_df["has_restart_train"].sum()),
            "false": int((~meta_df["has_restart_train"]).sum()),
        },
    }

    manifest = {
        "n_containers": len(meta_df),
        "n_steps_total": n_steps,
        "train_steps": tr_e - tr_s,
        "split_bounds": {
            "train": [tr_s, tr_e],
            "val": [va_s, va_e],
            "test": [te_s, te_e],
        },
        "cutoffs": {
            "mem_flat_threshold": 0.01,
            "cpu_terciles": [float(x) for x in cpu_terciles],
            "restart_drop_threshold": 0.20,
        },
        "tier_counts": tier_counts,
        "rule": (
            "Headline results are always on all 600 containers; "
            "tiers are for stratified reporting only and MUST NEVER "
            "be used to exclude containers or select subsets."
        ),
    }

    return meta_df, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build container metadata and strata tiers.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/alibaba_v2022_8h.parquet",
        help="Path to input processed parquet file",
    )
    parser.add_argument(
        "--out-parquet",
        type=str,
        default="data/processed/container_meta_8h.parquet",
        help="Path to output container meta parquet",
    )
    parser.add_argument(
        "--out-manifest",
        type=str,
        default="data/processed/container_meta_8h_manifest.json",
        help="Path to output manifest JSON",
    )
    args = parser.parse_args()

    df = load_long(args.data)
    meta_df, manifest = compute_container_meta(df)

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_parquet(out_parquet, index=False, engine="pyarrow")

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {len(meta_df)} rows to {out_parquet}")
    print(f"Saved manifest to {out_manifest}")
    print("\nTier Counts:")
    print(f"  Mem tier: {manifest['tier_counts']['mem_tier']}")
    print(f"  CPU tier: {manifest['tier_counts']['cpu_tier']}")
    print(f"  Has restart (train): {manifest['tier_counts']['has_restart_train']}")


if __name__ == "__main__":
    main()
