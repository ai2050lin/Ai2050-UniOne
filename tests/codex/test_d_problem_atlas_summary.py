#!/usr/bin/env python
"""
Aggregate task block D evidence into one frontend-friendly payload.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate D problem atlas payload")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/d_problem_atlas_summary_20260309.json")
    args = ap.parse_args()

    bridge = load_json(ROOT / "tests" / "codex_temp" / "gpt2_qwen3_deepseek7b_highdim_grounding_bridge_20260309.json")
    dual = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_dual_store_scan_20260309.json")
    residual = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_consolidation_law_scan_20260309.json")
    bayes = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_bayesian_consolidation_scan_20260309.json")
    learned = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_learned_controller_scan_20260309.json")
    two_phase = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_two_phase_consolidation_scan_20260309.json")
    three_phase = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_three_phase_consolidation_scan_20260309.json")
    base_offset = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_base_offset_consolidation_scan_20260309.json")
    offset_stabilization = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_offset_stabilization_scan_20260309.json")
    multistage = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_multistage_stabilization_scan_20260309.json")
    multimodal = load_json(ROOT / "tests" / "codex_temp" / "continuous_multimodal_grounding_proto_20260309.json")

    model_rows = []
    for model_name, row in bridge["models"].items():
        systems = row["bridge"]["systems"]
        gains = row["bridge"]["gains_vs_direct"]
        model_rows.append(
            {
                "model": model_name,
                "direct_grounding": float(systems["direct"]["grounding_score"]),
                "raw_shared_grounding": float(systems["raw_shared"]["grounding_score"]),
                "geometry_grounding": float(systems["geometry_dual_store"]["grounding_score"]),
                "raw_shared_novel_gain": float(gains["raw_shared"]["novel_concept_gain"]),
                "raw_shared_retention_gain": float(gains["raw_shared"]["retention_concept_gain"]),
                "geometry_novel_gain": float(gains["geometry_dual_store"]["novel_concept_gain"]),
                "geometry_retention_gain": float(gains["geometry_dual_store"]["retention_concept_gain"]),
                "geometry_overall_gain": float(gains["geometry_dual_store"]["overall_concept_gain"]),
            }
        )

    summary = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "focus": "D_problem_atlas",
        },
        "models": model_rows,
        "scan": {
            "feasible_count": int(dual["feasible_count"]),
            "best_overall": dual["best_overall"],
            "top_candidates": dual["top_candidates"],
        },
        "residual_gate": {
            "dual_positive_count": int(residual["dual_positive_count"]),
            "full_positive_count": int(residual["full_positive_count"]),
            "best_dual_positive": residual["best_dual_positive"],
            "top_dual_positive": residual["top_dual_positive"],
            "top_overall": residual["top_overall"],
        },
        "bayes_consolidation": {
            "dual_positive_count": int(bayes["dual_positive_count"]),
            "full_positive_count": int(bayes["full_positive_count"]),
            "best_dual_positive": bayes["best_dual_positive"],
            "top_dual_positive": bayes["top_dual_positive"],
            "top_overall": bayes["top_overall"],
        },
        "learned_controller": {
            "dual_positive_count": int(learned["dual_positive_count"]),
            "full_positive_count": int(learned["full_positive_count"]),
            "best_dual_positive": learned["best_dual_positive"],
            "best_overall": learned["best_overall"],
            "top_overall": learned["top_overall"],
        },
        "two_phase_consolidation": {
            "dual_positive_count": int(two_phase["dual_positive_count"]),
            "full_positive_count": int(two_phase["full_positive_count"]),
            "best_dual_positive": two_phase["best_dual_positive"],
            "best_overall": two_phase["best_overall"],
            "top_overall": two_phase["top_overall"],
        },
        "three_phase_consolidation": {
            "dual_positive_count": int(three_phase["dual_positive_count"]),
            "full_positive_count": int(three_phase["full_positive_count"]),
            "best_dual_positive": three_phase["best_dual_positive"],
            "best_overall": three_phase["best_overall"],
            "top_overall": three_phase["top_overall"],
        },
        "base_offset_consolidation": {
            "dual_positive_count": int(base_offset["dual_positive_count"]),
            "full_positive_count": int(base_offset["full_positive_count"]),
            "best_dual_positive": base_offset["best_dual_positive"],
            "best_full_positive": base_offset["best_full_positive"],
            "top_overall": base_offset["top_overall"],
        },
        "offset_stabilization": {
            "dual_positive_count": int(offset_stabilization["dual_positive_count"]),
            "full_positive_count": int(offset_stabilization["full_positive_count"]),
            "best_dual_positive": offset_stabilization["best_dual_positive"],
            "best_full_positive": offset_stabilization["best_full_positive"],
            "top_overall": offset_stabilization["top_overall"],
        },
        "multistage_stabilization": {
            "dual_positive_count": int(multistage["dual_positive_count"]),
            "full_positive_count": int(multistage["full_positive_count"]),
            "best_dual_positive": multistage["best_dual_positive"],
            "best_full_positive": multistage["best_full_positive"],
            "top_overall": multistage["top_overall"],
        },
        "multimodal_proto": {
            "systems": multimodal["systems"],
            "gains_vs_direct": multimodal["gains_vs_direct"],
        },
        "global_summary": {
            "all_models_fail_novel_and_retention": bool(
                all(not row["bridge"]["hypotheses"]["geometry_beats_direct_on_novel_and_retention"] for row in bridge["models"].values())
            ),
            "residual_dual_positive_exists": bool(residual["hypotheses"]["H1_dual_positive_region_exists"]),
            "bayes_dual_positive_exists": bool(bayes["hypotheses"]["H1_dual_positive_region_exists"]),
            "base_offset_dual_positive_exists": bool(base_offset["hypotheses"]["H1_dual_positive_region_exists"]),
            "base_offset_best_overall_gain": float(base_offset["top_overall"][0]["overall_gain"]),
            "base_offset_best_novel_gain": float(base_offset["top_overall"][0]["novel_gain"]),
            "base_offset_best_retention_gain": float(base_offset["top_overall"][0]["retention_gain"]),
            "offset_stabilization_dual_positive_exists": bool(offset_stabilization["hypotheses"]["H1_dual_positive_region_exists"]),
            "offset_stabilization_best_overall_gain": float(offset_stabilization["top_overall"][0]["overall_gain"]),
            "multistage_best_overall_gain": float(multistage["top_overall"][0]["overall_gain"]),
            "multistage_best_novel_gain": float(multistage["top_overall"][0]["novel_gain"]),
            "multistage_best_retention_gain": float(multistage["top_overall"][0]["retention_gain"]),
            "learned_controller_best_overall_gain": float(learned["best_overall"]["overall_gain"]),
            "two_phase_best_overall_gain": float(two_phase["best_overall"]["overall_gain"]),
            "three_phase_best_overall_gain": float(three_phase["best_overall"]["overall_gain"]),
            "multimodal_grounding_gain": float(multimodal["gains_vs_direct"]["grounding_score_gain"]),
            "multimodal_consistency_gain": float(multimodal["gains_vs_direct"]["crossmodal_consistency_gain"]),
            "best_overall_gain_across_methods": float(
                max(
                    residual["top_overall"][0]["overall_gain"],
                    bayes["top_overall"][0]["overall_gain"],
                    learned["best_overall"]["overall_gain"],
                    two_phase["best_overall"]["overall_gain"],
                    three_phase["best_overall"]["overall_gain"],
                    base_offset["top_overall"][0]["overall_gain"],
                    offset_stabilization["top_overall"][0]["overall_gain"],
                    multistage["top_overall"][0]["overall_gain"],
                )
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["global_summary"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
