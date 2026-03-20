from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_structure_genesis_confidence_stabilization_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_structure_genesis_confidence_stabilization_summary() -> dict:
    confidence = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_genesis_confidence_refinement_20260320" / "summary.json"
    )
    origin = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_origin_direct_refinement_20260320" / "summary.json"
    )
    terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_terminal_closure_20260320" / "summary.json"
    )

    hc = confidence["headline_metrics"]
    ho = origin["headline_metrics"]
    ht = terminal["headline_metrics"]

    stabilized_branching = hc["branching_refined_v2"]
    stabilized_binding = hc["binding_refined_v2"] * (1.0 + ho["origin_stability_v2"])
    stabilized_feedback = hc["feedback_refined_v2"] * (1.0 + hc["structure_direct_confidence_v3"])
    stabilized_margin = stabilized_branching + stabilized_binding + stabilized_feedback
    stabilized_confidence = stabilized_margin / (1.0 + 0.1 * ht["terminal_closure_margin_v3"])

    return {
        "headline_metrics": {
            "stabilized_branching": stabilized_branching,
            "stabilized_binding": stabilized_binding,
            "stabilized_feedback": stabilized_feedback,
            "stabilized_margin": stabilized_margin,
            "stabilized_confidence": stabilized_confidence,
        },
        "stabilization_equation": {
            "branch_term": "S_branch_v3 = S_branch_v2",
            "bind_term": "S_bind_v3 = S_bind_v2 * (1 + S_origin_v2)",
            "feedback_term": "S_fb_v3 = S_fb_v2 * (1 + C_struct_v3)",
            "margin_term": "M_struct_v4 = S_branch_v3 + S_bind_v3 + S_fb_v3",
            "confidence_term": "C_struct_v4 = M_struct_v4 / (1 + 0.1 * Tc_margin)",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 结构生成置信度稳定化报告",
        "",
        f"- stabilized_branching: {hm['stabilized_branching']:.6f}",
        f"- stabilized_binding: {hm['stabilized_binding']:.6f}",
        f"- stabilized_feedback: {hm['stabilized_feedback']:.6f}",
        f"- stabilized_margin: {hm['stabilized_margin']:.6f}",
        f"- stabilized_confidence: {hm['stabilized_confidence']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_structure_genesis_confidence_stabilization_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
