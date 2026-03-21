from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage57_learning_rule_dual_candidate_review_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_learning_rule_bounded_law_comparison import build_learning_rule_bounded_law_comparison_summary
from stage56_local_generative_law_emergence import build_local_generative_law_emergence_summary
from stage56_native_variable_candidate_mapping import build_native_variable_candidate_mapping_summary


def build_learning_rule_dual_candidate_review_summary() -> dict:
    comparison = build_learning_rule_bounded_law_comparison_summary()
    local = build_local_generative_law_emergence_summary()["headline_metrics"]
    native = build_native_variable_candidate_mapping_summary()["headline_metrics"]

    candidates = comparison["candidate_laws"]
    reviewed = {}
    for mode in ("sqrt", "log"):
        law = candidates[mode]
        structure_anchor_score = max(
            0.0,
            min(
                1.0,
                0.45 * (1.0 - law["domination_penalty"])
                + 0.30 * local["patch_coherence"]
                + 0.25 * native["native_mapping_completeness"],
            ),
        )
        local_law_compatibility = max(
            0.0,
            min(
                1.0,
                0.40 * law["stability"]
                + 0.35 * local["derivability_score"]
                + 0.25 * local["route_separation"],
            ),
        )
        interpretability = 0.84 if mode == "log" else 0.74
        overall_readiness = max(
            0.0,
            min(
                1.0,
                0.32 * law["readiness"]
                + 0.24 * structure_anchor_score
                + 0.24 * local_law_compatibility
                + 0.20 * interpretability,
            ),
        )
        reviewed[mode] = {
            "bounded_ratio": law["bounded_ratio"],
            "domination_penalty": law["domination_penalty"],
            "structure_anchor_score": structure_anchor_score,
            "local_law_compatibility": local_law_compatibility,
            "interpretability": interpretability,
            "overall_readiness": overall_readiness,
        }

    best_name, best_metrics = max(reviewed.items(), key=lambda item: item[1]["overall_readiness"])
    other_name = "log" if best_name == "sqrt" else "sqrt"
    readiness_margin = best_metrics["overall_readiness"] - reviewed[other_name]["overall_readiness"]

    return {
        "headline_metrics": {
            "best_candidate_name": best_name,
            "best_candidate_overall_readiness": best_metrics["overall_readiness"],
            "best_candidate_bounded_ratio": best_metrics["bounded_ratio"],
            "best_candidate_domination_penalty": best_metrics["domination_penalty"],
            "best_candidate_structure_anchor_score": best_metrics["structure_anchor_score"],
            "best_candidate_local_law_compatibility": best_metrics["local_law_compatibility"],
            "readiness_margin": readiness_margin,
        },
        "candidate_review": reviewed,
        "project_readout": {
            "summary": "Dual-candidate review keeps only sqrt and log bounded learning updates, then scores them by bounded stability, structure anchoring, local-law compatibility, and interpretability.",
            "next_question": "Use the selected candidate together with fiber reinforcement and context grounding before turning it into the next minimum falsifiable kernel.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage57 Learning Rule Dual Candidate Review",
        "",
        f"- best_candidate_name: {hm['best_candidate_name']}",
        f"- best_candidate_overall_readiness: {hm['best_candidate_overall_readiness']:.6f}",
        f"- best_candidate_bounded_ratio: {hm['best_candidate_bounded_ratio']:.6f}",
        f"- best_candidate_domination_penalty: {hm['best_candidate_domination_penalty']:.6f}",
        f"- best_candidate_structure_anchor_score: {hm['best_candidate_structure_anchor_score']:.6f}",
        f"- best_candidate_local_law_compatibility: {hm['best_candidate_local_law_compatibility']:.6f}",
        f"- readiness_margin: {hm['readiness_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_learning_rule_dual_candidate_review_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
