from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "codex_temp" / "resumable_hf_snapshot_download.py"
)
SPEC = importlib.util.spec_from_file_location("resumable_hf_snapshot_download", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_repo_cache_dir_name_for_models() -> None:
    assert MODULE.repo_cache_dir_name("zai-org/GLM-4-9B-Chat-HF", "model") == (
        "models--zai-org--GLM-4-9B-Chat-HF"
    )


def test_repo_cache_dir_name_for_datasets() -> None:
    assert MODULE.repo_cache_dir_name("wikitext/wikitext-103-v1", "dataset") == (
        "datasets--wikitext--wikitext-103-v1"
    )


def test_collect_blob_progress_counts_incomplete_files(tmp_path: Path) -> None:
    repo_root = tmp_path / "models--org--repo"
    blob_dir = repo_root / "blobs"
    blob_dir.mkdir(parents=True)
    (blob_dir / "done.bin").write_bytes(b"12345")
    (blob_dir / "part.bin.incomplete").write_bytes(b"123456789")

    progress = MODULE.collect_blob_progress(tmp_path, "org/repo", "model")

    assert progress.blob_bytes == 14
    assert progress.incomplete_bytes == 9
    assert progress.incomplete_count == 1
    assert progress.latest_write_time is not None
