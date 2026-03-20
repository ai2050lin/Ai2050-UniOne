from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_structure_native_coupling_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_structure_native_coupling_summary() -> dict:
    native_feature = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_nativeization_20260320" / "summary.json"
    )
    circuit_v8 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v8_20260320" / "summary.json"
    )
    canonical = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_canonical_closure_20260320" / "summary.json"
    )

    hf = native_feature["headline_metrics"]
    hc = circuit_v8["headline_metrics"]
    hl = canonical["headline_metrics"]

    native_circuit_link = hf["feature_native_core_v2"] * (1.0 + hc["direct_binding_v8"]) / (1.0 + hc["direct_gate_v8"])
    native_structure_link = hf["feature_native_core_v2"] * (1.0 + hc["direct_attractor_v8"]) / (1.0 + hc["direct_gate_v8"])
    native_feedback = hl["canonical_structure"] / max(hf["feature_native_core_v2"], 1e-9)
    native_coupling_margin = native_circuit_link + native_structure_link - hc["direct_gate_v8"]

    return {
        "headline_metrics": {
            "native_circuit_link": native_circuit_link,
            "native_structure_link": native_structure_link,
            "native_feedback": native_feedback,
            "native_coupling_margin": native_coupling_margin,
        },
        "native_coupling_equation": {
            "circuit_term": "Cn_fc = F_native_v2 * (1 + direct_binding_v8) / (1 + direct_gate_v8)",
            "structure_term": "Cn_fs = F_native_v2 * (1 + direct_attractor_v8) / (1 + direct_gate_v8)",
            "feedback_term": "Sn_fb = canonical_structure / F_native_v2",
            "margin_term": "Mn_fs = Cn_fc + Cn_fs - direct_gate_v8",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征层到结构原生耦合报告",
        "",
        f"- native_circuit_link: {hm['native_circuit_link']:.6f}",
        f"- native_structure_link: {hm['native_structure_link']:.6f}",
        f"- native_feedback: {hm['native_feedback']:.6f}",
        f"- native_coupling_margin: {hm['native_coupling_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_structure_native_coupling_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
