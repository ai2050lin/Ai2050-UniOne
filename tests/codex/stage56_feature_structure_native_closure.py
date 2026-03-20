from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_structure_native_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_structure_native_closure_summary() -> dict:
    feature_direct = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_native_direct_measure_20260320" / "summary.json"
    )
    circuit_v8 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v8_20260320" / "summary.json"
    )
    canonical = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_canonical_closure_20260320" / "summary.json"
    )

    hf = feature_direct["headline_metrics"]
    hc = circuit_v8["headline_metrics"]
    hl = canonical["headline_metrics"]

    closure_circuit_link = hf["feature_direct_core_v3"] * (1.0 + hc["direct_binding_v8"])
    closure_structure_link = hf["feature_direct_core_v3"] * (1.0 + hc["direct_attractor_v8"])
    closure_feedback = (hl["canonical_structure"] + hl["canonical_feature"]) / max(hf["feature_direct_core_v3"], 1e-9)
    native_closure_margin = closure_circuit_link + closure_structure_link + closure_feedback - hc["direct_gate_v8"]

    return {
        "headline_metrics": {
            "closure_circuit_link": closure_circuit_link,
            "closure_structure_link": closure_structure_link,
            "closure_feedback": closure_feedback,
            "native_closure_margin": native_closure_margin,
        },
        "native_closure_equation": {
            "circuit_term": "Cl_fc = F_direct_v3 * (1 + direct_binding_v8)",
            "structure_term": "Cl_fs = F_direct_v3 * (1 + direct_attractor_v8)",
            "feedback_term": "Cl_fb = (canonical_structure + canonical_feature) / F_direct_v3",
            "margin_term": "Cl_margin = Cl_fc + Cl_fs + Cl_fb - direct_gate_v8",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征到结构原生闭合报告",
        "",
        f"- closure_circuit_link: {hm['closure_circuit_link']:.6f}",
        f"- closure_structure_link: {hm['closure_structure_link']:.6f}",
        f"- closure_feedback: {hm['closure_feedback']:.6f}",
        f"- native_closure_margin: {hm['native_closure_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_structure_native_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
