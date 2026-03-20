from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_circuit_native_variable_refinement_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_circuit_native_variable_refinement_summary() -> dict:
    dyn = _load_json(ROOT / "tests" / "codex_temp" / "stage56_circuit_dynamics_bridge_v2_20260320" / "summary.json")
    native = _load_json(ROOT / "tests" / "codex_temp" / "stage56_spike_feature_native_variables_20260320" / "summary.json")
    v7 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v7_20260320" / "summary.json")

    hd = dyn["headline_metrics"]
    hn = native["headline_metrics"]
    hv7 = v7["headline_metrics"]

    native_binding = hd["recurrent_binding"] * (1.0 + hn["native_selectivity"])
    native_gate = hd["competitive_gate"] / (1.0 + hn["native_feature"])
    native_attractor = hd["attractor_loading"] * (1.0 + hv7["stability_term_v7"] / (1.0 + hv7["structure_term_v7"]))
    circuit_native_margin = native_binding + native_attractor - native_gate

    return {
        "headline_metrics": {
            "native_binding": native_binding,
            "native_gate": native_gate,
            "native_attractor": native_attractor,
            "circuit_native_margin": circuit_native_margin,
        },
        "native_equation": {
            "binding_term": "B_native = recurrent_binding * (1 + native_selectivity)",
            "gate_term": "G_native = competitive_gate / (1 + native_feature)",
            "attractor_term": "A_native = attractor_loading * (1 + stability_term_v7 / (1 + structure_term_v7))",
            "margin_term": "M_circuit_native = B_native + A_native - G_native",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 回路级原生变量强化报告",
        "",
        f"- native_binding: {hm['native_binding']:.6f}",
        f"- native_gate: {hm['native_gate']:.6f}",
        f"- native_attractor: {hm['native_attractor']:.6f}",
        f"- circuit_native_margin: {hm['circuit_native_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_circuit_native_variable_refinement_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
