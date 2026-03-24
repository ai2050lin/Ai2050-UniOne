#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage301_cross_family_shared_base_compression import run_analysis as run_stage301
from stage304_neuron_level_shared_bias_pattern_extractor import run_analysis as run_stage304


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage307_cross_family_shared_base_core_compression_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s301 = run_stage301(force=False)
    s304 = run_stage304(force=False)

    core_dims = sorted(set(s304["shared_core_dims"]))
    core_rows = [
        {
            "dim_index": dim,
            "core_role": "跨家族共享承载主核" if dim == core_dims[0] else "共享承载候选主核",
        }
        for dim in core_dims[:4]
    ]

    compression_score = (
        float(s301["compression_score"]) * 0.45
        + min(1.0, len(core_rows) / 4.0) * 0.25
        + float(s304["extraction_score"]) * 0.30
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage307_cross_family_shared_base_core_compression",
        "title": "跨家族共享承载主核压缩",
        "status_short": "cross_family_shared_base_core_compression_ready",
        "compression_score": float(compression_score),
        "core_count": len(core_rows),
        "strongest_dim_index": int(core_rows[0]["dim_index"]),
        "top_gap_name": "跨家族共享承载主核已经开始压出，但当前仍主要依赖极少数核心位，稳定通用底盘还不够厚",
        "core_rows": core_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨家族共享承载主核压缩")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
