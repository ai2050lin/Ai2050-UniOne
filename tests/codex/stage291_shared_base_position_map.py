#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_STAGE245 = PROJECT_ROOT / "tests" / "codex_temp" / "stage245_large_scale_noun_shared_delta_tensor_20260324" / "summary.json"
INPUT_STAGE251 = PROJECT_ROOT / "tests" / "codex_temp" / "stage251_delta_position_role_map_20260324" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage291_shared_base_position_map_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s245 = load_json(INPUT_STAGE245)
    s251 = load_json(INPUT_STAGE251)

    role_rows = s251["role_rows"]
    base_candidates = [
        row for row in role_rows
        if row["base_load"] >= 0.12 and row["mean_delta_load"] <= 0.31
    ]
    strong_shared = [row for row in base_candidates if row["role_name"] == "跨类共享触发"]
    fruit_support = [row for row in base_candidates if row["role_name"] == "水果内部差分" and row["base_load"] >= 0.12]

    base_score = (
        s245["shared_base_mean"] * 0.45
        + s245["effective_mean"] * 0.25
        + min(1.0, len(base_candidates) / 12.0) * 0.15
        + min(1.0, len(strong_shared) / 6.0) * 0.15
    )

    top_rows = sorted(base_candidates, key=lambda row: (-row["base_load"], row["mean_delta_load"], row["dim_index"]))[:16]
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage291_shared_base_position_map",
        "title": "共享基底位置图",
        "status_short": "shared_base_position_map_ready",
        "shared_base_score": float(base_score),
        "shared_base_mean": float(s245["shared_base_mean"]),
        "effective_mean": float(s245["effective_mean"]),
        "base_candidate_count": len(base_candidates),
        "cross_class_shared_count": len(strong_shared),
        "fruit_support_count": len(fruit_support),
        "top_gap_name": "共享基底更像落在一批中高基底负载、低到中等差分负载的位置上，而不是平均铺在所有参数位",
        "position_rows": [
            {
                "dim_index": row["dim_index"],
                "role_name": row["role_name"],
                "base_load": row["base_load"],
                "mean_delta_load": row["mean_delta_load"],
                "family_hit_count": row["family_hit_count"],
            }
            for row in top_rows
        ],
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
    parser = argparse.ArgumentParser(description="共享基底位置图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
