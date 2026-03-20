from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_circuit_direct_refinement_v2_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_circuit_direct_refinement_v2_summary() -> dict:
    direct = _load_json(ROOT / "tests" / "codex_temp" / "stage56_circuit_native_direct_measure_20260320" / "summary.json")
    circuit_native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_variable_refinement_20260320" / "summary.json"
    )
    learning = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_pulse_to_network_learning_dynamics_20260320" / "summary.json"
    )

    hd = direct["headline_metrics"]
    hc = circuit_native["headline_metrics"]
    hl = learning["headline_metrics"]

    direct_binding_v2 = hd["direct_binding_measure"] + 0.1 * hl["learning_feature"]
    direct_gate_v2 = hd["direct_gate_measure"] / (1.0 + 0.1 * hc["native_binding"])
    direct_attractor_v2 = hd["direct_attractor_measure"] + 0.1 * hl["learning_structure"]
    direct_margin_v2 = direct_binding_v2 + direct_attractor_v2 - direct_gate_v2

    return {
        "headline_metrics": {
            "direct_binding_v2": direct_binding_v2,
            "direct_gate_v2": direct_gate_v2,
            "direct_attractor_v2": direct_attractor_v2,
            "direct_margin_v2": direct_margin_v2,
        },
        "refinement_equation": {
            "binding_term": "B_direct_v2 = direct_binding_measure + 0.1 * learning_feature",
            "gate_term": "G_direct_v2 = direct_gate_measure / (1 + 0.1 * native_binding)",
            "attractor_term": "A_direct_v2 = direct_attractor_measure + 0.1 * learning_structure",
            "margin_term": "M_direct_v2 = B_direct_v2 + A_direct_v2 - G_direct_v2",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 回路级直测强化第二版报告",
        "",
        f"- direct_binding_v2: {hm['direct_binding_v2']:.6f}",
        f"- direct_gate_v2: {hm['direct_gate_v2']:.6f}",
        f"- direct_attractor_v2: {hm['direct_attractor_v2']:.6f}",
        f"- direct_margin_v2: {hm['direct_margin_v2']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_circuit_direct_refinement_v2_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
