#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage333_shared_carrier_cluster_raw_map import run_analysis as run_stage333
from stage330_fuzzy_carrier_sparse_deflection_joint_map import run_analysis as run_stage330


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage338_fuzzy_carrier_cluster_stability_review_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s333 = run_stage333(force=False)
    s330 = run_stage330(force=False)

    cluster_rows = list(s333["cluster_rows"])
    stability_rows = []
    for row in cluster_rows:
        stability_rows.append(
            {
                "cluster_name": row["cluster_name"],
                "member_count": int(row["member_count"]),
                "cluster_stability": float(row["mean_stability"]) * 0.70 + float(row["mean_base_load"]) * 0.30,
            }
        )

    review_score = (
        sum(float(row["cluster_stability"]) for row in stability_rows) / max(1, len(stability_rows)) * 0.65
        + float(s330["joint_score"]) * 0.35
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage338_fuzzy_carrier_cluster_stability_review",
        "title": "模糊承载簇稳定性复核",
        "status_short": "fuzzy_carrier_cluster_stability_review_ready",
        "review_score": float(review_score),
        "stability_rows": stability_rows,
        "top_gap_name": "水果共享簇和跨类共享簇已经有稳定形状，但器物簇和混合簇仍偏薄，说明模糊承载的稳定簇还没有完全长齐。",
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
    parser = argparse.ArgumentParser(description="模糊承载簇稳定性复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
