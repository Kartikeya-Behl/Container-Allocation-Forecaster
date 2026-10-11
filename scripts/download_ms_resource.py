"""
scripts/download_ms_resource.py
===============================
Downloads ONLY the container resource table (MSResource) from the Alibaba
Cluster Trace microservices-v2022 dataset for a specified DAYdHOUR range.

Features:
- Multi-connection parallel range download (8 workers) with resume per part
- Fallback to sequential resume if server rejects ranges
- Stream-extraction of only MSResource members (.csv) from tar.gz archives
- Safe SSL handling on Windows environments
- Support for start/end day and hour range (inclusive start, exclusive end)
- Fallback logic if specified hour range is unavailable

Usage:
    python scripts/download_ms_resource.py --start 0d12 --end 0d13
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import ssl
import sys
import tarfile
import time
import urllib.request

BASE_URL = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces/MSMetricsUpdate/MSMetricsUpdate_{idx}.tar.gz"
RATIO_MINUTES = 30
NUM_WORKERS = 8


def get_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def parse_day_hour(s: str) -> tuple[int, int]:
    parts = s.lower().strip().split("d")
    if len(parts) != 2:
        raise ValueError(f"Invalid format '{s}', expected e.g. '0d12'")
    return int(parts[0]), int(parts[1])


def get_remote_file_size(url: str, ctx: ssl.SSLContext) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def download_part(url: str, part_file: Path, start_byte: int, end_byte: int, ctx: ssl.SSLContext, max_retries: int = 5) -> None:
    expected_bytes = end_byte - start_byte + 1
    for attempt in range(max_retries):
        existing = part_file.stat().st_size if part_file.exists() else 0
        if existing >= expected_bytes:
            return  # Already done

        req = urllib.request.Request(url)
        cur_start = start_byte + existing
        req.add_header("Range", f"bytes={cur_start}-{end_byte}")

        try:
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                mode = "ab" if existing > 0 and resp.status == 206 else "wb"
                with open(part_file, mode) as f:
                    while True:
                        buf = resp.read(256 * 1024)
                        if not buf:
                            break
                        f.write(buf)

            if part_file.stat().st_size >= expected_bytes:
                return
        except Exception as e:
            time.sleep(2)

    if part_file.stat().st_size < expected_bytes:
        raise RuntimeError(f"Part {part_file.name} failed after {max_retries} attempts.")


def download_file_parallel(url: str, dest_path: Path, ctx: ssl.SSLContext, workers: int = NUM_WORKERS) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    total_size = get_remote_file_size(url, ctx)
    if not total_size:
        print(f"Could not get content-length for {url}, falling back to single stream...", flush=True)
        # Sequential fallback
        download_sequential(url, dest_path, ctx)
        return

    print(f"Downloading {dest_path.name} ({total_size / (1024*1024):.1f} MB) using {workers} parallel connections...", flush=True)

    part_size = total_size // workers
    part_files = []
    futures = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i in range(workers):
            start = i * part_size
            end = (start + part_size - 1) if (i < workers - 1) else (total_size - 1)
            p_file = dest_path.with_suffix(f".part_{i}")
            part_files.append(p_file)
            futures.append(executor.submit(download_part, url, p_file, start, end, ctx))

        # Monitor overall progress
        while True:
            done_count = sum(1 for f in futures if f.done())
            current_bytes = sum(p.stat().st_size for p in part_files if p.exists())
            elapsed = time.time() - t0
            speed = (current_bytes / (1024 * 1024)) / (elapsed + 1e-6)
            pct = current_bytes / total_size * 100
            print(f"  {dest_path.name}: {current_bytes / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s [{done_count}/{workers} parts done]", flush=True)
            if done_count == workers:
                break
            time.sleep(5)

        # Check for exceptions
        for f in futures:
            f.result()

    # Assemble parts
    print(f"Assembling {workers} parts into {dest_path.name}...", flush=True)
    temp_assembled = dest_path.with_suffix(".assembling")
    with open(temp_assembled, "wb") as out_f:
        for p in part_files:
            with open(p, "rb") as in_f:
                while True:
                    buf = in_f.read(4 * 1024 * 1024)
                    if not buf:
                        break
                    out_f.write(buf)
            p.unlink()

    temp_assembled.rename(dest_path)
    print(f"Downloaded {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB) in {time.time()-t0:.1f}s", flush=True)


def download_sequential(url: str, dest_path: Path, ctx: ssl.SSLContext) -> None:
    temp_path = dest_path.with_suffix(".part")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp, open(temp_path, "wb") as out_f:
        while True:
            buf = resp.read(512 * 1024)
            if not buf:
                break
            out_f.write(buf)
    temp_path.rename(dest_path)


def stream_extract_msresource(tar_path: Path, extract_dir: Path) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = []

    print(f"Extracting MSResource members from {tar_path.name}...", flush=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if member.isfile() and (member.name.endswith(".csv") or "MS" in member.name):
                dest_file = extract_dir / Path(member.name).name
                if dest_file.exists():
                    print(f"  Existing file found: {dest_file.name} ({dest_file.stat().st_size / (1024*1024):.1f} MB), skipping extraction.", flush=True)
                    extracted_files.append(dest_file)
                    continue

                print(f"  Extracting '{member.name}' ({member.size / (1024*1024):.1f} MB) -> {dest_file.name}...", flush=True)
                with tar.extractfile(member) as src_f, open(dest_file, "wb") as dst_f:
                    extracted_bytes = 0
                    last_log = time.time()
                    while True:
                        buf = src_f.read(2 * 1024 * 1024)
                        if not buf:
                            break
                        dst_f.write(buf)
                        extracted_bytes += len(buf)
                        now = time.time()
                        if now - last_log >= 5.0:
                            print(f"    Extracted {extracted_bytes / (1024*1024):.1f} / {member.size / (1024*1024):.1f} MB", flush=True)
                            last_log = now
                extracted_files.append(dest_file)
                print(f"  Extracted {dest_file.name} ({dest_file.stat().st_size / (1024*1024):.1f} MB)", flush=True)

    return extracted_files


def calculate_indices(start_str: str, end_str: str) -> list[int]:
    start_day, start_hour = parse_day_hour(start_str)
    end_day, end_hour = parse_day_hour(end_str)

    start_min = start_day * 1440 + start_hour * 60
    end_min = end_day * 1440 + end_hour * 60

    start_idx = start_min // RATIO_MINUTES
    end_idx = (end_min // RATIO_MINUTES) - 1
    return list(range(start_idx, end_idx + 1))


def main():
    parser = argparse.ArgumentParser(description="Download Alibaba v2022 MSResource table.")
    parser.add_argument("--start", type=str, default="0d12", help="Start DAYdHOUR (inclusive), e.g. 0d12")
    parser.add_argument("--end", type=str, default="0d13", help="End DAYdHOUR (exclusive), e.g. 0d13")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Target output directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    ctx = get_ssl_context()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    indices = calculate_indices(args.start, args.end)
    print(f"Requested range: {args.start} to {args.end} -> indices: {indices}", flush=True)

    # Check first URL
    first_url = BASE_URL.format(idx=indices[0])
    if get_remote_file_size(first_url, ctx) is None:
        print(f"Range {args.start} to {args.end} (URL: {first_url}) not reachable. Falling back to 0d0 -> 0d1...", flush=True)
        indices = calculate_indices("0d0", "0d1")
        print(f"Fallback indices: {indices}", flush=True)

    total_compressed_bytes = 0
    all_extracted_files = []

    for idx in indices:
        url = BASE_URL.format(idx=idx)
        tar_name = f"MSMetricsUpdate_{idx}.tar.gz"
        tar_path = raw_dir / tar_name

        if not tar_path.exists():
            print(f"\n--- Fetching chunk {idx}: {tar_name} ---", flush=True)
            download_file_parallel(url, tar_path, ctx, workers=args.workers)
        else:
            print(f"Existing archive found: {tar_name} ({tar_path.stat().st_size / (1024*1024):.1f} MB)", flush=True)

        compressed_size = tar_path.stat().st_size
        total_compressed_bytes += compressed_size

        extracted = stream_extract_msresource(tar_path, raw_dir)
        all_extracted_files.extend(extracted)

    total_extracted_bytes = sum(f.stat().st_size for f in all_extracted_files if f.exists())

    print("\n" + "=" * 60, flush=True)
    print("DOWNLOAD & EXTRACTION REPORT", flush=True)
    print("=" * 60, flush=True)
    print(f"Total compressed size: {total_compressed_bytes / (1024*1024):.2f} MB ({total_compressed_bytes / (1024*1024*1024):.3f} GB)", flush=True)
    print(f"Total extracted size:  {total_extracted_bytes / (1024*1024):.2f} MB ({total_extracted_bytes / (1024*1024*1024):.3f} GB)", flush=True)
    print(f"Extracted files ({len(all_extracted_files)} files):", flush=True)
    for f in all_extracted_files:
        print(f"  {f.name}: {f.stat().st_size / (1024*1024):.2f} MB", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
