#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "google/gemma-4-E2B-it"
CACHE_DIR = Path(r"D:\develop\model\hub")
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage506_download_gemma4_hf_20260404"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_hf_endpoint() -> str | None:
    value = os.environ.get("HF_ENDPOINT", "").strip()
    if not value:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        return value
    fixed = f"https://{value}"
    os.environ["HF_ENDPOINT"] = fixed
    return fixed


NORMALIZED_HF_ENDPOINT = normalize_hf_endpoint()

from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor


def find_snapshot_root(cache_dir: Path) -> str | None:
    target_root = cache_dir / "models--google--gemma-4-E2B-it"
    if not target_root.exists():
        return None
    snapshots = target_root / "snapshots"
    if not snapshots.exists():
        return str(target_root)
    children = [p for p in snapshots.iterdir() if p.is_dir()]
    if not children:
        return str(target_root)
    children.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(children[0])


def snapshot_download_with_retry(repo_id: str, cache_dir: Path, *, retries: int = 3) -> str:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir),
                max_workers=1,
                resume_download=True,
            )
        except Exception as exc:
            last_error = exc
            print(f"[stage506] 下载失败，第 {attempt}/{retries} 次重试：{type(exc).__name__}: {exc}")
            if attempt == retries:
                raise
            time.sleep(3)
    raise RuntimeError(f"下载失败: {last_error}")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    normalized_hf_endpoint = NORMALIZED_HF_ENDPOINT

    snapshot_root = snapshot_download_with_retry(MODEL_ID, CACHE_DIR, retries=3)

    processor_status = "not_attempted"
    model_status = "not_attempted"
    processor_class = None
    model_class = None
    load_error = None

    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
        processor_status = "ok"
        processor_class = processor.__class__.__name__
    except Exception as exc:
        processor_status = "failed"
        load_error = f"processor: {type(exc).__name__}: {exc}"

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            cache_dir=str(CACHE_DIR),
            low_cpu_mem_usage=True,
        )
        model_status = "ok"
        model_class = model.__class__.__name__
    except Exception as exc:
        model_status = "failed"
        if load_error:
            load_error = load_error + " | " + f"model: {type(exc).__name__}: {exc}"
        else:
            load_error = f"model: {type(exc).__name__}: {exc}"

    summary = {
        "stage": "stage506_download_gemma4_hf",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": MODEL_ID,
        "cache_dir": str(CACHE_DIR),
        "normalized_hf_endpoint": normalized_hf_endpoint,
        "snapshot_root": str(snapshot_root),
        "processor_status": processor_status,
        "model_status": model_status,
        "processor_class": processor_class,
        "model_class": model_class,
        "load_error": load_error,
        "elapsed_seconds": round(time.time() - started, 3),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "REPORT.md").write_text(
        "\n".join(
            [
                "# stage506 Gemma4 HF 下载结果",
                "",
                f"- 模型：`{MODEL_ID}`",
                f"- 缓存目录：`{CACHE_DIR}`",
                f"- 规范化后的 HF_ENDPOINT：`{normalized_hf_endpoint}`",
                f"- 快照目录：`{snapshot_root}`",
                f"- Processor 状态：`{processor_status}`",
                f"- Model 状态：`{model_status}`",
                f"- Processor 类：`{processor_class}`",
                f"- Model 类：`{model_class}`",
                f"- 加载错误：`{load_error}`",
                f"- 耗时：`{summary['elapsed_seconds']}` 秒",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
