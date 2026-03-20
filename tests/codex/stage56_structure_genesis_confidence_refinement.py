from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_structure_genesis_confidence_refinement_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_structure_genesis_confidence_refinement_summary() -> dict:
    structure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_genesis_direct_measure_v2_20260320" / "summary.json"
    )
    origin = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_origin_direct_refinement_20260320" / "summary.json"
    )
    terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_terminal_closure_20260320" / "summary.json"
    )

    hs = structure["headline_metrics"]
    ho = origin["headline_metrics"]
    ht = terminal["headline_metrics"]

    branching_refined_v2 = hs["structure_branching_direct"] * (1.0 + ho["origin_stability_v2"])
    binding_refined_v2 = hs["closure_binding_direct"] * (1.0 + ho["origin_stability_v2"])
    feedback_refined_v2 = hs["feedback_stability_direct"] * (1.0 + ho["origin_stability_v2"])
    structure_genesis_margin_v3 = branching_refined_v2 + binding_refined_v2 + feedback_refined_v2
    structure_direct_confidence_v3 = structure_genesis_margin_v3 / (1.0 + 0.25 * ht["terminal_closure_margin_v3"])

    return {
        "headline_metrics": {
            "branching_refined_v2": branching_refined_v2,
            "binding_refined_v2": binding_refined_v2,
            "feedback_refined_v2": feedback_refined_v2,
            "structure_genesis_margin_v3": structure_genesis_margin_v3,
            "structure_direct_confidence_v3": structure_direct_confidence_v3,
        },
        "confidence_equation": {
            "branch_term": "S_branch_v2 = S_branch * (1 + S_origin_v2)",
            "bind_term": "S_bind_v2 = S_bind * (1 + S_origin_v2)",
            "feedback_term": "S_fb_v2 = S_fb * (1 + S_origin_v2)",
            "margin_term": "M_struct_v3 = S_branch_v2 + S_bind_v2 + S_fb_v2",
            "confidence_term": "C_struct_v3 = M_struct_v3 / (1 + 0.25 * Tc_margin)",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 结构生成置信度强化报告",
        "",
        f"- branching_refined_v2: {hm['branching_refined_v2']:.6f}",
        f"- binding_refined_v2: {hm['binding_refined_v2']:.6f}",
        f"- feedback_refined_v2: {hm['feedback_refined_v2']:.6f}",
        f"- structure_genesis_margin_v3: {hm['structure_genesis_margin_v3']:.6f}",
        f"- structure_direct_confidence_v3: {hm['structure_direct_confidence_v3']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_structure_genesis_confidence_refinement_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
