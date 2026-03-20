from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_structure_coupling_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_structure_coupling_summary() -> dict:
    feature_layer = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_definition_20260320" / "summary.json"
    )
    circuit_v8 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v8_20260320" / "summary.json"
    )
    canonical = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_canonical_closure_20260320" / "summary.json"
    )

    hf = feature_layer["headline_metrics"]
    hc = circuit_v8["headline_metrics"]
    hl = canonical["headline_metrics"]

    feature_to_circuit = hf["feature_layer_core"] * (1.0 + hc["direct_binding_v8"])
    feature_to_structure = hf["feature_layer_core"] * (1.0 + hc["direct_attractor_v8"])
    structure_feedback = hl["canonical_structure"] / max(hf["feature_layer_core"], 1e-9)
    coupling_margin = feature_to_circuit + feature_to_structure - hc["direct_gate_v8"]

    return {
        "headline_metrics": {
            "feature_to_circuit": feature_to_circuit,
            "feature_to_structure": feature_to_structure,
            "structure_feedback": structure_feedback,
            "coupling_margin": coupling_margin,
        },
        "coupling_equation": {
            "circuit_term": "C_fc = F_core * (1 + direct_binding_v8)",
            "structure_term": "C_fs = F_core * (1 + direct_attractor_v8)",
            "feedback_term": "S_fb = canonical_structure / F_core",
            "margin_term": "M_fs = C_fc + C_fs - direct_gate_v8",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征层与网络结构耦合报告",
        "",
        f"- feature_to_circuit: {hm['feature_to_circuit']:.6f}",
        f"- feature_to_structure: {hm['feature_to_structure']:.6f}",
        f"- structure_feedback: {hm['structure_feedback']:.6f}",
        f"- coupling_margin: {hm['coupling_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_structure_coupling_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
