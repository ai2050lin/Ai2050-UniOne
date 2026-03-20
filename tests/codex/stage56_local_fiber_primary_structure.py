from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_local_fiber_primary_structure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_local_fiber_primary_structure_summary() -> dict:
    strengthened = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_local_differential_fiber_strengthening_20260320" / "summary.json"
    )
    cross_asset = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_concept_chart_cross_asset_validation_20260320" / "summary.json"
    )
    concept = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_encoding_formation_20260320" / "summary.json")

    shm = strengthened["headline_metrics"]
    hca = cross_asset["headline_metrics"]
    chm = concept["headline_metrics"]

    fiber_structure_gain = shm["mean_strengthened_local_fiber"] * (
        1.0 + hca["chart_separation_support"] + hca["cross_asset_support_v2"]
    )
    apple_local_structure = shm["apple_strengthened_local_margin"] * (
        1.0 + chm["family_anchor_strength"] + chm["apple_local_offset_norm"]
    )
    local_primary_structure = fiber_structure_gain + apple_local_structure

    return {
        "headline_metrics": {
            "fiber_structure_gain": fiber_structure_gain,
            "apple_local_structure": apple_local_structure,
            "local_primary_structure": local_primary_structure,
        },
        "structure_equation": {
            "gain_term": "G_fiber = mean_strengthened_local_fiber * (1 + chart_separation_support + cross_asset_support_v2)",
            "apple_term": "L_fiber = apple_strengthened_local_margin * (1 + family_anchor_strength + apple_local_offset_norm)",
            "margin_term": "M_fiber_primary = G_fiber + L_fiber",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 局部差分纤维主结构化报告",
        "",
        f"- fiber_structure_gain: {hm['fiber_structure_gain']:.6f}",
        f"- apple_local_structure: {hm['apple_local_structure']:.6f}",
        f"- local_primary_structure: {hm['local_primary_structure']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_local_fiber_primary_structure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
