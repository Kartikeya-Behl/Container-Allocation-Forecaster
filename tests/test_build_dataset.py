"""
tests/test_build_dataset.py
===========================
Pytest suite for src/data/build_dataset.py using tiny synthetic datasets.
Verifies:
  1. Deduplication by mean of duplicate (msinstanceid, timestamp) rows.
  2. Gap removal (incomplete or non-consecutive series rejected).
  3. Train-only variance filtering:
     A series that is FLAT in train (steps 0..167) but oscillates wildly in
     val/test (steps 168..239) MUST BE REJECTED.
  4. Per-service container cap (max_per_service).
  5. Dataset schema, step ordering, and manifest structure.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from data.build_dataset import run_pipeline, validate_dataset


@pytest.fixture
def synthetic_data_dir(tmp_path):
    """Generate tiny synthetic CSV files simulating 4-hour (240 step) traces."""
    # We will simulate 2 CSV files (e.g. 1st half and 2nd half, or 8 files)
    # Total steps: 240. Steps 0..119 in file1, steps 120..239 in file2.
    # T0 = 43200000 ms. Step interval = 60,000 ms.
    t0 = 43_200_000
    step_ms = 60_000
    timestamps = [t0 + i * step_ms for i in range(240)]

    rng = np.random.RandomState(42)

    rows = []

    # 1. Container: "valid_svc1_c1", service: "MS_A"
    # Good variance in both train and test
    for i, t in enumerate(timestamps):
        cpu = 0.20 + 0.05 * np.sin(i / 10.0)
        mem = 0.50 + 0.02 * np.cos(i / 10.0)
        rows.append({"timestamp": t, "msname": "MS_A", "msinstanceid": "valid_svc1_c1", "cpu_utilization": cpu, "memory_utilization": mem})

    # 2. Container: "valid_svc1_c2", service: "MS_A"
    # Also valid
    for i, t in enumerate(timestamps):
        cpu = 0.30 + 0.04 * np.cos(i / 5.0)
        mem = 0.40 + 0.03 * np.sin(i / 5.0)
        rows.append({"timestamp": t, "msname": "MS_A", "msinstanceid": "valid_svc1_c2", "cpu_utilization": cpu, "memory_utilization": mem})

    # 3. Containers 3, 4, 5 for "MS_A": to test per-service cap (max 3 per service)
    for extra_id in ["valid_svc1_c3", "valid_svc1_c4", "valid_svc1_c5"]:
        for i, t in enumerate(timestamps):
            cpu = 0.15 + 0.03 * np.sin(i / 8.0)
            mem = 0.60 + 0.02 * np.cos(i / 8.0)
            rows.append({"timestamp": t, "msname": "MS_A", "msinstanceid": extra_id, "cpu_utilization": cpu, "memory_utilization": mem})

    # 4. Container: "flat_in_train_wild_in_test", service: "MS_B"
    # First 168 steps: STRICTLY FLAT (cpu=0.25, mem=0.50 -> std=0.0)
    # Last 72 steps (168..239): WILD variations (cpu 0.1 to 0.9, mem 0.2 to 0.8)
    # MUST BE REJECTED by train-only variance filter!
    for i, t in enumerate(timestamps):
        if i < 168:
            cpu = 0.25
            mem = 0.50
        else:
            cpu = 0.10 if (i % 2 == 0) else 0.90
            mem = 0.20 if (i % 2 == 0) else 0.80
        rows.append({"timestamp": t, "msname": "MS_B", "msinstanceid": "flat_in_train_wild_in_test", "cpu_utilization": cpu, "memory_utilization": mem})

    # 5. Container: "gapped_container", service: "MS_C"
    # Missing step 50
    for i, t in enumerate(timestamps):
        if i == 50:
            continue  # gap!
        cpu = 0.20 + 0.05 * np.sin(i / 10.0)
        mem = 0.50 + 0.02 * np.cos(i / 10.0)
        rows.append({"timestamp": t, "msname": "MS_C", "msinstanceid": "gapped_container", "cpu_utilization": cpu, "memory_utilization": mem})

    # 6. Container: "dup_container", service: "MS_D"
    # Valid, but has a duplicate at step 10:
    # row 1: cpu=0.20, mem=0.40; row 2: cpu=0.40, mem=0.60 -> mean must be 0.30 and 0.50
    for i, t in enumerate(timestamps):
        cpu = 0.20 + 0.05 * np.sin(i / 10.0)
        mem = 0.50 + 0.02 * np.cos(i / 10.0)
        rows.append({"timestamp": t, "msname": "MS_D", "msinstanceid": "dup_container", "cpu_utilization": cpu, "memory_utilization": mem})
        if i == 10:
            # duplicate with higher values
            rows.append({"timestamp": t, "msname": "MS_D", "msinstanceid": "dup_container", "cpu_utilization": cpu + 0.20, "memory_utilization": mem + 0.20})

    df = pd.DataFrame(rows)

    # Split into 2 CSV files (first half steps < 120, second half >= 120)
    split_t = t0 + 120 * step_ms
    df_part1 = df[df["timestamp"] < split_t].sample(frac=1.0, random_state=42)  # shuffled order in raw file
    df_part2 = df[df["timestamp"] >= split_t].sample(frac=1.0, random_state=42)

    p1 = tmp_path / "part1.csv"
    p2 = tmp_path / "part2.csv"
    df_part1.to_csv(p1, index=False)
    df_part2.to_csv(p2, index=False)

    return tmp_path, [p1, p2]


def test_build_dataset_full_pipeline(synthetic_data_dir):
    tmp_path, csv_paths = synthetic_data_dir
    out_parquet = tmp_path / "out.parquet"
    manifest_path = tmp_path / "out_manifest.json"

    df_out, manifest = run_pipeline(
        csv_paths=csv_paths,
        out_parquet=out_parquet,
        manifest_path=manifest_path,
        start_hour=12,
        hours=4,
        seed=42,
        max_per_service=3,
        n_final=10,
        n_steps=240,
        first_hour_steps=60,
    )

    # 1. Parquet exists and loads
    assert out_parquet.exists()
    assert manifest_path.exists()
    assert isinstance(df_out, pd.DataFrame)

    surviving_containers = set(df_out["container_id"].unique())

    # 2. Train-only variance filter: "flat_in_train_wild_in_test" MUST BE REJECTED
    assert "flat_in_train_wild_in_test" not in surviving_containers, (
        "CRITICAL: Container with zero train variance was not rejected!"
    )

    # 3. Gap removal: "gapped_container" MUST BE REJECTED
    assert "gapped_container" not in surviving_containers, (
        "Container with missing step was not rejected!"
    )

    # 4. Deduplication: "dup_container" must be present and deduplicated by mean
    assert "dup_container" in surviving_containers
    dup_step10 = df_out[(df_out["container_id"] == "dup_container") & (df_out["step"] == 10)]
    assert len(dup_step10) == 1
    # Original base at step 10 was: 0.20 + 0.05 * sin(1), dup was base + 0.20 -> mean is base + 0.10
    base_cpu = 0.20 + 0.05 * np.sin(10 / 10.0)
    expected_cpu = base_cpu + 0.10
    assert abs(dup_step10["cpu"].iloc[0] - expected_cpu) < 1e-5

    # 5. Per-service cap: MS_A had 5 valid containers, at most 3 must be selected
    ms_a_containers = df_out[df_out["msname"] == "MS_A"]["container_id"].unique()
    assert len(ms_a_containers) <= 3, f"Expected <= 3 containers for MS_A, got {len(ms_a_containers)}"

    # 6. Step count invariant: each surviving container must have exactly 240 steps
    for inst in surviving_containers:
        inst_df = df_out[df_out["container_id"] == inst]
        assert len(inst_df) == 240
        assert inst_df["step"].tolist() == list(range(240))

    # 7. Manifest validation
    with manifest_path.open() as f:
        m = json.load(f)
    assert m["parameters"]["seed"] == 42
    assert "stage_container_counts" in m
    assert m["total_rows"] == len(df_out)


def test_validate_dataset_raises_on_invalid_data():
    # Corrupt data with missing row
    bad_df = pd.DataFrame({
        "container_id": ["c1"] * 239,  # 239 instead of 240
        "msname": ["s1"] * 239,
        "timestamp_ms": [43200000 + i * 60000 for i in range(239)],
        "step": list(range(239)),
        "cpu": [0.2] * 239,
        "mem": [0.5] * 239,
    })
    with pytest.raises(AssertionError):
        validate_dataset(bad_df, expected_steps=240)
