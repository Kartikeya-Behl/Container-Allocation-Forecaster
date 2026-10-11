"""
src/data/build_dataset.py
=========================
Builds the processed Alibaba microservices-v2022 dataset for container forecasting.

Pipeline:
  a. Candidate discovery: from the FIRST 60 minutes only, find complete gap-free
     instances (60 steps). Sample candidates with seed (max 3 per msname, up to 5000).
  b. Filtered streaming: stream raw CSVs once, keeping only candidate instance rows.
  c. Deduplication: group by (msinstanceid, timestamp), taking mean of metrics.
  d. Window completeness: require exactly 240 consecutive 60 s steps, no NaNs, no gaps.
  e. Train-only variance filter: require std(cpu) > 0.005 and std(mem) > 0.001 on the
     first 168 steps (first 70% train split only). Val/test steps are never inspected.
  f. Final selection: sample n-final (default 600) instances, max 3 per msname.
  g. Export: save long parquet and manifest JSON with stage counts and parameters.
  h. Validation: verify row counts, steps, spacing, NaNs, duplicates, and ranges.

Usage:
    python src/data/build_dataset.py --raw-dir data/raw --start-hour 12 --hours 4
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Support both module and script imports
try:
    from data.loader import split_bounds
except ImportError:
    try:
        from src.data.loader import split_bounds
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from data.loader import split_bounds


def discover_csv_files(raw_dir: Path, start_hour: int, hours: int) -> list[Path]:
    """Discover raw CSV files corresponding to the requested hour range."""
    # Alibaba chunk indices: ratio is 30 mins (2 chunks per hour)
    start_idx = start_hour * 2
    end_idx = (start_hour + hours) * 2 - 1

    expected_indices = list(range(start_idx, end_idx + 1))
    files = []
    for idx in expected_indices:
        f = raw_dir / f"MSMetricsUpdate_{idx}.csv"
        if f.exists():
            files.append(f)

    # Fallback: if numbered chunks don't match, sort any .csv in raw_dir
    if not files:
        all_csvs = sorted(list({p.resolve(): p for p in raw_dir.glob("*.csv")}.values()))
        if all_csvs:
            files = all_csvs[: hours * 2]

    return files


def run_pipeline(
    csv_paths: list[Path],
    out_parquet: Path,
    manifest_path: Path,
    start_hour: int = 12,
    hours: int = 4,
    seed: int = 42,
    max_per_service: int = 3,
    n_final: int = 600,
    chunk_size: int = 200_000,
    n_steps: int | None = None,
    first_hour_steps: int = 60,
    step_interval_ms: int = 60_000,
    cpu_std_min: float = 0.005,
    mem_std_min: float = 0.001,
) -> tuple[pd.DataFrame, dict]:
    """Execute the end-to-end dataset building pipeline."""
    rng = np.random.RandomState(seed)
    if n_steps is None:
        n_steps = hours * 60
    (train_bounds, val_bounds, test_bounds) = split_bounds(n_steps)
    train_steps = train_bounds[1]

    if not csv_paths:
        raise FileNotFoundError("No raw CSV files provided or found.")

    print("=" * 70, flush=True)
    print(f"BUILDING DATASET ({len(csv_paths)} CSV files, seed={seed}, n_steps={n_steps}, train_steps={train_steps})", flush=True)
    print("=" * 70, flush=True)

    # -----------------------------------------------------------------------
    # a. Candidate discovery from FIRST 60 minutes only
    # -----------------------------------------------------------------------
    print("\n[Stage a] Scanning first 60 minutes for candidate instances...", flush=True)
    # The first 60 minutes correspond to the first 2 chunks (or first CSV if 1 hour)
    first_hour_csvs = csv_paths[:2] if len(csv_paths) >= 2 else csv_paths[:1]

    instance_first_hour_ts = {}
    instance_first_hour_svc = {}

    for csv_file in first_hour_csvs:
        print(f"  Reading {csv_file.name} for candidate screening...", flush=True)
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            sub = chunk[["msinstanceid", "msname", "timestamp"]].dropna()
            for inst, svc, t in zip(sub["msinstanceid"], sub["msname"], sub["timestamp"]):
                if inst not in instance_first_hour_ts:
                    instance_first_hour_ts[inst] = set()
                    instance_first_hour_svc[inst] = svc
                instance_first_hour_ts[inst].add(t)

    # Check gap-free first_hour_steps
    complete_first_hour_by_svc = {}
    for inst, t_set in instance_first_hour_ts.items():
        if len(t_set) >= first_hour_steps:
            sorted_t = sorted(t_set)
            # Take first 60 steps
            sub_t = sorted_t[:first_hour_steps]
            diffs = np.diff(sub_t)
            if all(d == step_interval_ms for d in diffs):
                svc = instance_first_hour_svc[inst]
                complete_first_hour_by_svc.setdefault(svc, []).append(inst)

    # Sample candidates: max max_per_service per msname, up to 5000 candidates
    candidates_list = []
    for svc in sorted(complete_first_hour_by_svc.keys()):
        insts = sorted(complete_first_hour_by_svc[svc])
        if len(insts) > max_per_service:
            chosen = rng.choice(insts, size=max_per_service, replace=False).tolist()
        else:
            chosen = insts
        candidates_list.extend(chosen)

    if len(candidates_list) > 5000:
        candidates_list = rng.choice(sorted(candidates_list), size=5000, replace=False).tolist()

    candidates_set = set(candidates_list)
    count_a = len(candidates_set)
    print(f"  Stage a complete: {count_a:,} candidate instances selected across {len(complete_first_hour_by_svc):,} services.", flush=True)

    # -----------------------------------------------------------------------
    # b. Stream ALL raw CSVs once, keeping only candidate rows
    # -----------------------------------------------------------------------
    print(f"\n[Stage b] Streaming all {len(csv_paths)} CSVs for candidate rows...", flush=True)
    cols = ["timestamp", "msname", "msinstanceid", "cpu_utilization", "memory_utilization"]
    collected_chunks = []

    for csv_file in csv_paths:
        print(f"  Streaming {csv_file.name}...", flush=True)
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size, usecols=cols):
            sub = chunk[chunk["msinstanceid"].isin(candidates_set)]
            if not sub.empty:
                collected_chunks.append(sub)

    df_candidates = pd.concat(collected_chunks, axis=0, ignore_index=True) if collected_chunks else pd.DataFrame(columns=cols)
    count_b = int(df_candidates["msinstanceid"].nunique())
    print(f"  Stage b complete: {len(df_candidates):,} rows collected for {count_b:,} unique instances.", flush=True)

    # -----------------------------------------------------------------------
    # c. Deduplicate on (msinstanceid, timestamp) by MEAN of metrics
    # -----------------------------------------------------------------------
    print("\n[Stage c] Deduplicating (msinstanceid, timestamp) by mean...", flush=True)
    df_dedup = (
        df_candidates.groupby(["msinstanceid", "timestamp"], as_index=False)
        .agg({
            "msname": "first",
            "cpu_utilization": "mean",
            "memory_utilization": "mean",
        })
    )
    count_c = int(df_dedup["msinstanceid"].nunique())
    print(f"  Stage c complete: {len(df_dedup):,} unique points for {count_c:,} instances.", flush=True)

    # -----------------------------------------------------------------------
    # d. Window completeness: exactly n_steps consecutive 60 s steps, no NaNs
    # -----------------------------------------------------------------------
    print(f"\n[Stage d] Filtering for exact {n_steps}-step window without gaps or NaNs...", flush=True)
    survivors_d = []
    grouped = df_dedup.groupby("msinstanceid")

    for inst, group in grouped:
        if len(group) != n_steps:
            continue
        if group["cpu_utilization"].isna().any() or group["memory_utilization"].isna().any():
            continue

        sorted_group = group.sort_values("timestamp")
        diffs = np.diff(sorted_group["timestamp"].values)
        if all(d == step_interval_ms for d in diffs):
            survivors_d.append(inst)

    count_d = len(survivors_d)
    print(f"  Stage d complete: {count_d:,} instances have exactly {n_steps} consecutive steps.", flush=True)

    # -----------------------------------------------------------------------
    # e. Train-only variance filter (first 70% / 168 steps only)
    # -----------------------------------------------------------------------
    print(f"\n[Stage e] Applying TRAIN-ONLY variance filter (first {train_steps} steps)...", flush=True)
    survivors_e = []
    df_survivors_d = df_dedup[df_dedup["msinstanceid"].isin(survivors_d)]
    inst_service_map = {}

    for inst, group in df_survivors_d.groupby("msinstanceid"):
        sorted_group = group.sort_values("timestamp")
        inst_service_map[inst] = sorted_group["msname"].iloc[0]

        # Use STRICTLY the first train_steps
        train_portion = sorted_group.iloc[:train_steps]
        std_cpu = float(train_portion["cpu_utilization"].std(ddof=1))
        std_mem = float(train_portion["memory_utilization"].std(ddof=1))

        if std_cpu > cpu_std_min and std_mem > mem_std_min:
            survivors_e.append(inst)

    count_e = len(survivors_e)
    print(f"  Stage e complete: {count_e:,} instances passed train variance filter (std_cpu > {cpu_std_min}, std_mem > {mem_std_min}).", flush=True)

    # -----------------------------------------------------------------------
    # f. Final selection: pick n_final from survivors, max max_per_service per msname
    # -----------------------------------------------------------------------
    print(f"\n[Stage f] Sampling final {n_final} instances (max {max_per_service} per msname)...", flush=True)
    service_to_survivors = {}
    for inst in sorted(survivors_e):
        svc = inst_service_map[inst]
        service_to_survivors.setdefault(svc, []).append(inst)

    capped_survivors = []
    for svc in sorted(service_to_survivors.keys()):
        insts = sorted(service_to_survivors[svc])
        if len(insts) > max_per_service:
            chosen = rng.choice(insts, size=max_per_service, replace=False).tolist()
        else:
            chosen = insts
        capped_survivors.extend(chosen)

    if len(capped_survivors) > n_final:
        final_instances = sorted(rng.choice(sorted(capped_survivors), size=n_final, replace=False).tolist())
    else:
        final_instances = sorted(capped_survivors)

    count_f = len(final_instances)
    print(f"  Stage f complete: {count_f:,} final instances selected across {len(set(inst_service_map[i] for i in final_instances)):,} services.", flush=True)

    # -----------------------------------------------------------------------
    # g. Save long parquet and manifest JSON
    # -----------------------------------------------------------------------
    print("\n[Stage g] Formatting and exporting final dataset...", flush=True)
    df_final_raw = df_dedup[df_dedup["msinstanceid"].isin(final_instances)].copy()

    records = []
    for inst, group in df_final_raw.groupby("msinstanceid"):
        sorted_g = group.sort_values("timestamp").reset_index(drop=True)
        for step_idx, row in sorted_g.iterrows():
            records.append({
                "container_id": row["msinstanceid"],
                "msname": row["msname"],
                "timestamp_ms": int(row["timestamp"]),
                "step": int(step_idx),
                "cpu": float(row["cpu_utilization"]),
                "mem": float(row["memory_utilization"]),
            })

    df_out = pd.DataFrame(records)
    df_out = df_out.sort_values(["container_id", "step"]).reset_index(drop=True)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_parquet, index=False, engine="pyarrow")
    print(f"  Saved long parquet to {out_parquet} ({out_parquet.stat().st_size / (1024*1024):.2f} MB)", flush=True)

    manifest = {
        "dataset": "alibaba_microservices_v2022",
        "parameters": {
            "start_hour": start_hour,
            "hours": hours,
            "seed": seed,
            "max_per_service": max_per_service,
            "n_final": n_final,
            "n_steps": n_steps,
            "train_steps": train_steps,
            "split_bounds": {
                "train": list(train_bounds),
                "val": list(val_bounds),
                "test": list(test_bounds),
            },
            "first_hour_steps": first_hour_steps,
            "step_interval_ms": step_interval_ms,
            "cpu_std_min": cpu_std_min,
            "mem_std_min": mem_std_min,
        },
        "stage_container_counts": {
            "stage_a_candidates": count_a,
            "stage_b_streamed": count_b,
            "stage_c_deduped": count_c,
            "stage_d_window_complete": count_d,
            "stage_e_train_variance_filtered": count_e,
            "stage_f_final_selected": count_f,
        },
        "total_rows": len(df_out),
        "parquet_file": str(out_parquet),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  Saved manifest JSON to {manifest_path}", flush=True)

    # -----------------------------------------------------------------------
    # h. Validation block
    # -----------------------------------------------------------------------
    validate_dataset(df_out, expected_steps=n_steps)

    return df_out, manifest


def validate_dataset(df: pd.DataFrame, expected_steps: int = 240) -> None:
    """Validate processed dataset schema, row counts, spacing, and constraints."""
    print("\n" + "=" * 70, flush=True)
    print("DATASET VALIDATION REPORT", flush=True)
    print("=" * 70, flush=True)

    n_containers = df["container_id"].nunique()
    n_rows = len(df)
    expected_rows = n_containers * expected_steps

    print(f"Number of containers:       {n_containers:,}", flush=True)
    print(f"Total row count:            {n_rows:,} (expected: {expected_rows:,})", flush=True)
    assert n_rows == expected_rows, f"Row count mismatch! {n_rows} != {expected_rows}"

    steps_per_container = df.groupby("container_id")["step"].nunique()
    assert (steps_per_container == expected_steps).all(), "Some containers do not have exactly expected unique steps!"
    print(f"Unique steps per container: {expected_steps} for all {n_containers} containers", flush=True)

    # Spacing check
    spacing_errors = 0
    for _, group in df.groupby("container_id"):
        diffs = np.diff(group.sort_values("step")["timestamp_ms"].values)
        if not (diffs == 60_000).all():
            spacing_errors += 1
    print(f"Timestamp spacing errors:   {spacing_errors} (all intervals strictly 60,000 ms)", flush=True)
    assert spacing_errors == 0, f"{spacing_errors} containers have invalid timestamp spacing!"

    nan_count = int(df.isna().sum().sum())
    print(f"Total NaN count:            {nan_count}", flush=True)
    assert nan_count == 0, f"Found {nan_count} NaNs in dataset!"

    dup_count = int(df.duplicated(subset=["container_id", "step"]).sum())
    print(f"Duplicate (id, step) count: {dup_count}", flush=True)
    assert dup_count == 0, f"Found {dup_count} duplicate rows in dataset!"

    n_services = df["msname"].nunique()
    print(f"Distinct microservices:    {n_services:,}", flush=True)

    # Max per service
    per_svc = df.groupby("msname")["container_id"].nunique()
    print(f"Max containers per service: {per_svc.max()}", flush=True)

    print("Utilization metrics:", flush=True)
    print(f"  CPU - min: {df['cpu'].min():.6f}, max: {df['cpu'].max():.6f}, mean: {df['cpu'].mean():.6f}", flush=True)
    print(f"  Mem - min: {df['mem'].min():.6f}, max: {df['mem'].max():.6f}, mean: {df['mem'].mean():.6f}", flush=True)
    print("=" * 70 + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build processed Alibaba microservices dataset.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory containing raw CSVs")
    parser.add_argument("--start-hour", type=int, default=12, help="Start hour (e.g. 12)")
    parser.add_argument("--hours", type=int, default=4, help="Number of hours to process (e.g. 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-per-service", type=int, default=3, help="Max instances per service")
    parser.add_argument("--n-final", type=int, default=600, help="Target number of final instances")
    parser.add_argument("--dataset-tag", type=str, default=None, help="Dataset tag used in output naming")
    parser.add_argument("--out", type=str, default=None, help="Output parquet path")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if args.out is not None:
        out_parquet = Path(args.out)
    elif args.dataset_tag is not None:
        out_parquet = Path(f"data/processed/{args.dataset_tag}.parquet")
    else:
        tag = f"alibaba_v2022_{args.hours}h" if args.hours != 4 else "alibaba_v2022"
        out_parquet = Path(f"data/processed/{tag}.parquet")

    manifest_path = out_parquet.parent / f"{out_parquet.stem}_manifest.json"

    csv_paths = discover_csv_files(raw_dir, args.start_hour, args.hours)
    print(f"Discovered {len(csv_paths)} CSV files for hour {args.start_hour}..{args.start_hour+args.hours}: {[p.name for p in csv_paths]}", flush=True)

    n_steps = args.hours * 60

    run_pipeline(
        csv_paths=csv_paths,
        out_parquet=out_parquet,
        manifest_path=manifest_path,
        start_hour=args.start_hour,
        hours=args.hours,
        seed=args.seed,
        max_per_service=args.max_per_service,
        n_final=args.n_final,
        n_steps=n_steps,
    )


if __name__ == "__main__":
    main()
