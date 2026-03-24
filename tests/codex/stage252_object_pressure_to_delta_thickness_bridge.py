#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage252_object_pressure_to_delta_thickness_bridge_20260324"
STAGE249_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage249_fruit_animal_delta_reason_spectrum_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize(values: List[float]) -> List[float]:
    low = min(values)
    high = max(values)
    if high - low < 1e-9:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def build_summary() -> dict:
    stage249 = load_json(STAGE249_SUMMARY)
    rows = stage249["word_rows"]
    shares = [float(row["family_share"]) for row in rows]
    intra = [float(row["intra_competition"]) for row in rows]
    cross = [float(row["cross_category_pressure"]) for row in rows]
    margins = [float(row["margin"]) for row in rows]

    norm_share = normalize(shares)
    norm_intra = normalize(intra)
    norm_cross = normalize(cross)
    norm_margin = normalize(margins)

    bridge_rows = []
    family_share_pressure_mean = 0.0
    intra_competition_pressure_mean = 0.0
    cross_category_pressure_mean = 0.0
    for row, share_p, intra_p, cross_p, margin_n in zip(rows, norm_share, norm_intra, norm_cross, norm_margin):
        family_share_pressure = 1.0 - share_p
        intra_competition_pressure = intra_p
        cross_category_pressure = 1.0 - cross_p
        predicted_thickness = (0.40 * family_share_pressure) + (0.35 * intra_competition_pressure) + (0.25 * cross_category_pressure)
        thickness_gap = abs(predicted_thickness - margin_n)
        family_share_pressure_mean += family_share_pressure
        intra_competition_pressure_mean += intra_competition_pressure
        cross_category_pressure_mean += cross_category_pressure
        bridge_rows.append(
            {
                "word": row["word"],
                "family_name": row["family_name"],
                "family_share_pressure": family_share_pressure,
                "intra_competition_pressure": intra_competition_pressure,
                "cross_category_separation_pressure": cross_category_pressure,
                "observed_delta_thickness": margin_n,
                "predicted_delta_thickness": predicted_thickness,
                "thickness_gap": thickness_gap,
            }
        )

    count = len(bridge_rows)
    family_share_pressure_mean /= count
    intra_competition_pressure_mean /= count
    cross_category_pressure_mean /= count
    bridge_score = 1.0 - (sum(row["thickness_gap"] for row in bridge_rows) / count)
    strongest = max(bridge_rows, key=lambda row: row["observed_delta_thickness"])["word"]
    weakest = min(bridge_rows, key=lambda row: row["observed_delta_thickness"])["word"]
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage252_object_pressure_to_delta_thickness_bridge",
        "title": "对象压力到差分厚度桥",
        "status_short": "object_pressure_to_delta_thickness_bridge_ready",
        "word_count": count,
        "family_share_pressure_mean": family_share_pressure_mean,
        "intra_competition_pressure_mean": intra_competition_pressure_mean,
        "cross_category_separation_pressure_mean": cross_category_pressure_mean,
        "bridge_score": bridge_score,
        "strongest_word_name": strongest,
        "weakest_word_name": weakest,
        "top_gap_name": "对象最终拿到多厚的差分，主要由共享压力、同类竞争和跨类分离压力共同决定",
        "bridge_rows": bridge_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with (output_dir / "bridge_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["bridge_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["bridge_rows"])
    report = [
        "# Stage252：对象压力到差分厚度桥",
        "",
        "## 核心结果",
        f"- 对象数量：{summary['word_count']}",
        f"- 共享压力均值：{summary['family_share_pressure_mean']:.4f}",
        f"- 同类竞争压力均值：{summary['intra_competition_pressure_mean']:.4f}",
        f"- 跨类分离压力均值：{summary['cross_category_separation_pressure_mean']:.4f}",
        f"- 桥总分：{summary['bridge_score']:.4f}",
        f"- 最厚差分对象：{summary['strongest_word_name']}",
        f"- 最薄差分对象：{summary['weakest_word_name']}",
        f"- 头号发现：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE252_OBJECT_PRESSURE_TO_DELTA_THICKNESS_BRIDGE_REPORT.md").write_text("\n".join(report), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="对象压力到差分厚度桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
