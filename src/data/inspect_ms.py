"""
src/data/inspect_ms.py
======================
Chunked loading and inspection of Alibaba microservices-v2022 MSResource data.
Computes completeness, sampling interval, duplicates, NaNs, distribution statistics,
and generates validation plots for 3 random complete instances (seed 42).
"""

import argparse
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def inspect_msresource(csv_paths: list[Path], plot_out: Path, seed: int = 42) -> None:
    print("=" * 70, flush=True)
    print("MSRESOURCE DATASET INSPECTION", flush=True)
    print("=" * 70, flush=True)
    print(f"Files to inspect: {[p.name for p in csv_paths]}", flush=True)

    chunk_size = 500_000
    total_rows = 0
    unique_instances = set()
    unique_msnames = set()
    nan_counts = Counter()

    ts_min = float("inf")
    ts_max = float("-inf")

    cpu_min = float("inf")
    cpu_max = float("-inf")
    cpu_sum = 0.0
    cpu_count = 0
    cpu_gt_1 = 0
    cpu_gt_100 = 0

    mem_min = float("inf")
    mem_max = float("-inf")
    mem_sum = 0.0
    mem_count = 0
    mem_gt_1 = 0
    mem_gt_100 = 0

    duplicate_rows = 0

    # Map instance -> list of timestamps
    instance_timestamps = {}

    first_chunk = True
    col_names = []
    dtypes_dict = {}

    for csv_path in csv_paths:
        print(f"\nProcessing {csv_path.name} in chunks of {chunk_size:,}...", flush=True)
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            total_rows += len(chunk)

            if first_chunk:
                col_names = list(chunk.columns)
                dtypes_dict = {col: str(chunk[col].dtype) for col in chunk.columns}
                first_chunk = False

            # NaN counts
            for col in chunk.columns:
                nan_counts[col] += int(chunk[col].isna().sum())

            # Unique counts
            if "msinstanceid" in chunk.columns:
                unique_instances.update(chunk["msinstanceid"].dropna().unique())
            if "msname" in chunk.columns:
                unique_msnames.update(chunk["msname"].dropna().unique())

            # Timestamps
            if "timestamp" in chunk.columns:
                c_ts = chunk["timestamp"].dropna()
                if len(c_ts) > 0:
                    ts_min = min(ts_min, c_ts.min())
                    ts_max = max(ts_max, c_ts.max())

            # CPU stats
            if "cpu_utilization" in chunk.columns:
                c_cpu = chunk["cpu_utilization"].dropna()
                if len(c_cpu) > 0:
                    cpu_min = min(cpu_min, c_cpu.min())
                    cpu_max = max(cpu_max, c_cpu.max())
                    cpu_sum += c_cpu.sum()
                    cpu_count += len(c_cpu)
                    cpu_gt_1 += int((c_cpu > 1.0).sum())
                    cpu_gt_100 += int((c_cpu > 100.0).sum())

            # Memory stats
            if "memory_utilization" in chunk.columns:
                c_mem = chunk["memory_utilization"].dropna()
                if len(c_mem) > 0:
                    mem_min = min(mem_min, c_mem.min())
                    mem_max = max(mem_max, c_mem.max())
                    mem_sum += c_mem.sum()
                    mem_count += len(c_mem)
                    mem_gt_1 += int((c_mem > 1.0).sum())
                    mem_gt_100 += int((c_mem > 100.0).sum())

            # Within-chunk duplicate check
            if "msinstanceid" in chunk.columns and "timestamp" in chunk.columns:
                sub = chunk[["msinstanceid", "timestamp"]].dropna()
                dup_in_chunk = int(sub.duplicated().sum())
                duplicate_rows += dup_in_chunk

                for inst, t in zip(sub["msinstanceid"], sub["timestamp"]):
                    if inst not in instance_timestamps:
                        instance_timestamps[inst] = []
                    instance_timestamps[inst].append(t)

    # 1. Columns and dtypes
    print("\n--- 1. COLUMNS & DTYPES ---", flush=True)
    for col in col_names:
        print(f"  {col}: {dtypes_dict.get(col, 'unknown')}", flush=True)

    # 2. Row count and unique counts
    print("\n--- 2. COUNTS ---", flush=True)
    print(f"Total row count:        {total_rows:,}", flush=True)
    print(f"Unique msinstanceid:    {len(unique_instances):,}", flush=True)
    print(f"Unique msname:          {len(unique_msnames):,}", flush=True)

    # 3. Timestamps
    print("\n--- 3. TIMESTAMPS ---", flush=True)
    print(f"Timestamp min:          {ts_min}", flush=True)
    print(f"Timestamp max:          {ts_max}", flush=True)
    ts_span_ms = ts_max - ts_min
    print(f"Timestamp span:         {ts_span_ms} ms ({ts_span_ms / 1000:.1f} s, {ts_span_ms / 60000:.1f} min)", flush=True)
    ts_unit = "milliseconds (ms)" if ts_min >= 10_000 else "seconds (s)"
    print(f"Inferred timestamp unit:{ts_unit}", flush=True)

    # 4. Sampling interval distribution
    print("\n--- 4. SAMPLING INTERVAL DISTRIBUTION ---", flush=True)
    diff_counter = Counter()
    complete_instances = []
    # In 1 hour at 60s intervals, expected points is 60 (or 61 if both boundaries included)
    expected_points = 60
    expected_step_ms = 60_000

    cross_chunk_dups = 0
    for inst, t_list in instance_timestamps.items():
        if len(t_list) < 2:
            continue
        sorted_t = sorted(t_list)
        diffs = np.diff(sorted_t)
        for d in diffs:
            diff_counter[int(d)] += 1
            if d == 0:
                cross_chunk_dups += 1

        # Complete gap-free hour: exactly covers 60 points with 60,000 ms step
        # Span from start to end = 59 * 60,000 = 3,540,000 ms (59 minutes)
        if len(sorted_t) >= expected_points:
            # Check if all intervals in the 60-step sequence are 60,000 ms
            if all(d == expected_step_ms for d in diffs[:expected_points - 1]):
                complete_instances.append(inst)

    duplicate_rows += cross_chunk_dups
    total_diffs = sum(diff_counter.values())
    print("Top sampling intervals:", flush=True)
    for diff_val, cnt in diff_counter.most_common(5):
        pct = (cnt / total_diffs * 100) if total_diffs > 0 else 0
        diff_s = diff_val / 1000 if ts_unit.startswith("milli") else diff_val
        print(f"  dt = {diff_val} ms ({diff_s:.1f} s): {cnt:,} occurrences ({pct:.2f}%)", flush=True)

    # 5. Completeness
    print("\n--- 5. COMPLETENESS ---", flush=True)
    print(f"Instances with complete gap-free 60-step (1-hour) series: {len(complete_instances):,}", flush=True)

    # 6. Duplicates and NaNs
    print("\n--- 6. DATA QUALITY (DUPLICATES & NANS) ---", flush=True)
    print(f"Duplicate (msinstanceid, timestamp) rows: {duplicate_rows:,}", flush=True)
    print("NaN counts per column:", flush=True)
    for col in col_names:
        print(f"  {col}: {nan_counts[col]:,} NaNs", flush=True)

    # 7. Utilization scale & statistics
    print("\n--- 7. UTILIZATION DISTRIBUTIONS ---", flush=True)
    cpu_mean = (cpu_sum / cpu_count) if cpu_count > 0 else float("nan")
    mem_mean = (mem_sum / mem_count) if mem_count > 0 else float("nan")
    print(f"CPU utilization:", flush=True)
    print(f"  Min:  {cpu_min:.6f}", flush=True)
    print(f"  Max:  {cpu_max:.6f}", flush=True)
    print(f"  Mean: {cpu_mean:.6f}", flush=True)
    print(f"  Values > 1.0:   {cpu_gt_1:,} ({(cpu_gt_1/cpu_count*100 if cpu_count else 0):.2f}%)", flush=True)
    print(f"  Values > 100.0: {cpu_gt_100:,}", flush=True)
    print(f"Memory utilization:", flush=True)
    print(f"  Min:  {mem_min:.6f}", flush=True)
    print(f"  Max:  {mem_max:.6f}", flush=True)
    print(f"  Mean: {mem_mean:.6f}", flush=True)
    print(f"  Values > 1.0:   {mem_gt_1:,} ({(mem_gt_1/mem_count*100 if mem_count else 0):.2f}%)", flush=True)
    print(f"  Values > 100.0: {mem_gt_100:,}", flush=True)

    scale_desc = "Normalized [0, 1] range (values between 0.0 and 1.0)" if (cpu_max <= 1.05 and mem_max <= 1.05) else "Percentage [0, 100] range"
    print(f"Utilization Scale Verdict: {scale_desc}", flush=True)

    # 8. Plot 3 random complete instances
    print("\n--- 8. GENERATING VALIDATION PLOT ---", flush=True)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    if not complete_instances:
        print("Warning: No complete instances found to plot!", flush=True)
        return

    rng = np.random.RandomState(seed)
    chosen_instances = rng.choice(complete_instances, size=min(3, len(complete_instances)), replace=False)
    print(f"Selected complete instances for plotting (seed {seed}): {list(chosen_instances)}", flush=True)

    # Read series for chosen instances
    chosen_data = {inst: [] for inst in chosen_instances}
    for csv_path in csv_paths:
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            sub = chunk[chunk["msinstanceid"].isin(chosen_instances)]
            for inst in chosen_instances:
                inst_sub = sub[sub["msinstanceid"] == inst]
                if len(inst_sub) > 0:
                    chosen_data[inst].append(inst_sub)

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Alibaba microservices-v2022 Sample Containers (Seed {seed})", fontsize=14)

    for i, inst in enumerate(chosen_instances):
        df_inst = pd.concat(chosen_data[inst], axis=0).sort_values("timestamp")
        t_rel_min = (df_inst["timestamp"] - df_inst["timestamp"].iloc[0]) / 60_000

        axes[i, 0].plot(t_rel_min, df_inst["cpu_utilization"], color="tab:blue", marker="o", markersize=3, label="CPU Util")
        axes[i, 0].set_ylabel("CPU Util [0, 1]")
        axes[i, 0].set_title(f"{inst} - CPU")
        axes[i, 0].grid(True, linestyle="--", alpha=0.6)
        axes[i, 0].set_ylim(-0.05, 1.05)

        axes[i, 1].plot(t_rel_min, df_inst["memory_utilization"], color="tab:orange", marker="s", markersize=3, label="Mem Util")
        axes[i, 1].set_ylabel("Mem Util [0, 1]")
        axes[i, 1].set_title(f"{inst} - Memory")
        axes[i, 1].grid(True, linestyle="--", alpha=0.6)
        axes[i, 1].set_ylim(-0.05, 1.05)

    axes[2, 0].set_xlabel("Time (minutes from hour start)")
    axes[2, 1].set_xlabel("Time (minutes from hour start)")
    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)
    plt.close()
    print(f"Saved validation plot to {plot_out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Inspect Alibaba MSResource dataset.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory containing raw data")
    parser.add_argument("--plot-out", type=str, default="results/go_nogo_plots.png", help="Path to save plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    # Deduplicate paths
    unique_csvs = {p.resolve(): p for p in raw_dir.glob("*.csv")}
    csv_files = sorted(list(unique_csvs.values()))

    if not csv_files:
        print(f"Error: No CSV files found in {raw_dir}", flush=True)
        return

    inspect_msresource(csv_files, Path(args.plot_out), seed=args.seed)


if __name__ == "__main__":
    main()
