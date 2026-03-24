#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage298_shared_base_position_role_card import run_analysis as run_stage298


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_STAGE291 = PROJECT_ROOT / "tests" / "codex_temp" / "stage291_shared_base_position_map_20260324" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage301_cross_family_shared_base_compression_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s298 = run_stage298(force=False)
    s291 = load_json(INPUT_STAGE291)

    rows = s298["position_rows"]
    cross_family_like = [row for row in rows if row["family_hit_count"] >= 2 and row["base_load"] >= 0.16]
    compressed_count = min(4, len(cross_family_like))
    compression_score = (
        float(s298["role_score"]) * 0.45
        + float(s291["cross_class_shared_count"]) / max(1.0, float(s291["base_candidate_count"])) * 0.25
        + min(1.0, compressed_count / 4.0) * 0.30
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage301_cross_family_shared_base_compression",
        "title": "跨家族共享承载位压缩",
        "status_short": "cross_family_shared_base_compression_ready",
        "compression_score": float(compression_score),
        "compressed_count": compressed_count,
        "cross_family_like_count": len(cross_family_like),
        "top_gap_name": "跨家族共享承载位已经出现，但数量仍偏少，距离真正通用底盘主核还有压缩空间",
        "position_rows": cross_family_like[:8],
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
    parser = argparse.ArgumentParser(description="跨家族共享承载位压缩")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
