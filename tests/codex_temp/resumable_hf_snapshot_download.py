from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, constants, snapshot_download


@dataclass
class BlobProgress:
    blob_bytes: int
    incomplete_bytes: int
    incomplete_count: int
    latest_write_time: float | None


def repo_cache_dir_name(repo_id: str, repo_type: str) -> str:
    if repo_type == "model":
        prefix = "models"
    elif repo_type == "dataset":
        prefix = "datasets"
    elif repo_type == "space":
        prefix = "spaces"
    else:
        raise ValueError(f"Unsupported repo_type: {repo_type}")
    return f"{prefix}--{repo_id.replace('/', '--')}"


def force_endpoint(endpoint: str) -> None:
    normalized = endpoint.rstrip("/")
    os.environ["HF_ENDPOINT"] = normalized
    constants.ENDPOINT = normalized
    constants.HUGGINGFACE_CO_URL_TEMPLATE = (
        normalized + "/{repo_id}/resolve/{revision}/{filename}"
    )


def iter_blob_files(repo_root: Path) -> Iterable[Path]:
    blob_dir = repo_root / "blobs"
    if not blob_dir.exists():
        return []
    return (path for path in blob_dir.iterdir() if path.is_file())


def collect_blob_progress(cache_dir: Path, repo_id: str, repo_type: str) -> BlobProgress:
    repo_root = cache_dir / repo_cache_dir_name(repo_id, repo_type)
    blob_bytes = 0
    incomplete_bytes = 0
    incomplete_count = 0
    latest_write_time = None

    for path in iter_blob_files(repo_root):
        size = path.stat().st_size
        blob_bytes += size
        if path.name.endswith(".incomplete"):
            incomplete_bytes += size
            incomplete_count += 1
        write_time = path.stat().st_mtime
        if latest_write_time is None or write_time > latest_write_time:
            latest_write_time = write_time

    return BlobProgress(
        blob_bytes=blob_bytes,
        incomplete_bytes=incomplete_bytes,
        incomplete_count=incomplete_count,
        latest_write_time=latest_write_time,
    )


def fetch_expected_bytes(endpoint: str, repo_id: str) -> int | None:
    try:
        force_endpoint(endpoint)
        api = HfApi(endpoint=endpoint)
        info = api.model_info(repo_id, files_metadata=True)
    except Exception:
        return None

    total = 0
    for sibling in info.siblings:
        size = getattr(sibling, "size", None)
        if size:
            total += int(size)
    return total or None


def format_gb(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def terminate_process_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_child(args: argparse.Namespace) -> int:
    force_endpoint(args.endpoint)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        cache_dir=str(args.cache_dir),
        max_workers=args.max_workers,
        resume_download=True,
        local_files_only=False,
        etag_timeout=args.etag_timeout,
        endpoint=args.endpoint,
    )
    return 0


def run_parent(args: argparse.Namespace) -> int:
    expected_bytes = fetch_expected_bytes(args.endpoint, args.repo_id)
    print(
        f"[parent] repo={args.repo_id} cache_dir={args.cache_dir} "
        f"expected={format_gb(expected_bytes)}"
    )

    for attempt in range(1, args.max_attempts + 1):
        before = collect_blob_progress(args.cache_dir, args.repo_id, args.repo_type)
        print(
            f"[parent] attempt={attempt}/{args.max_attempts} "
            f"blob={format_gb(before.blob_bytes)} incomplete={before.incomplete_count}"
        )
        child_cmd = [
            sys.executable,
            "-u",
            __file__,
            "--child",
            "--repo-id",
            args.repo_id,
            "--repo-type",
            args.repo_type,
            "--cache-dir",
            str(args.cache_dir),
            "--endpoint",
            args.endpoint,
            "--max-workers",
            str(args.max_workers),
            "--etag-timeout",
            str(args.etag_timeout),
        ]
        child = subprocess.Popen(child_cmd)
        last_blob_bytes = before.blob_bytes
        last_progress_time = time.monotonic()
        last_log_time = 0.0

        while True:
            exit_code = child.poll()
            now = time.monotonic()
            progress = collect_blob_progress(args.cache_dir, args.repo_id, args.repo_type)

            if progress.blob_bytes > last_blob_bytes:
                delta = progress.blob_bytes - last_blob_bytes
                print(
                    f"[parent] progress +{delta / (1024 ** 2):.2f} MB "
                    f"total={format_gb(progress.blob_bytes)}"
                )
                last_blob_bytes = progress.blob_bytes
                last_progress_time = now

            if now - last_log_time >= args.status_interval:
                latest_write = (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(progress.latest_write_time))
                    if progress.latest_write_time
                    else "none"
                )
                expected = ""
                if expected_bytes:
                    ratio = progress.blob_bytes / expected_bytes
                    expected = f" ratio={ratio:.2%}"
                print(
                    f"[parent] heartbeat blob={format_gb(progress.blob_bytes)} "
                    f"incomplete={progress.incomplete_count} "
                    f"incomplete_bytes={format_gb(progress.incomplete_bytes)} "
                    f"latest_write={latest_write}{expected}"
                )
                last_log_time = now

            if exit_code is not None:
                if exit_code == 0:
                    final_progress = collect_blob_progress(args.cache_dir, args.repo_id, args.repo_type)
                    print(
                        f"[parent] success blob={format_gb(final_progress.blob_bytes)} "
                        f"incomplete={final_progress.incomplete_count}"
                    )
                    return 0
                print(f"[parent] child_exit={exit_code}, retrying after cooldown")
                break

            if now - last_progress_time >= args.stall_timeout:
                print(
                    f"[parent] stall_detected timeout={args.stall_timeout}s "
                    f"blob={format_gb(progress.blob_bytes)}"
                )
                terminate_process_tree(child.pid)
                child.wait(timeout=30)
                break

            time.sleep(args.poll_interval)

        if attempt < args.max_attempts:
            time.sleep(args.retry_cooldown)

    print("[parent] exhausted retries without finishing download")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--etag-timeout", type=float, default=30.0)
    parser.add_argument("--stall-timeout", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--status-interval", type=float, default=30.0)
    parser.add_argument("--retry-cooldown", type=float, default=15.0)
    parser.add_argument("--max-attempts", type=int, default=8)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
