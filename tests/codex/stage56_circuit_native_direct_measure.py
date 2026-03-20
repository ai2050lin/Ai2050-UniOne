from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_circuit_native_direct_measure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_circuit_native_direct_measure_summary() -> dict:
    native = _load_json(ROOT / "tests" / "codex_temp" / "stage56_spike_feature_native_variables_20260320" / "summary.json")
    circuit = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_variable_refinement_20260320" / "summary.json"
    )
    dynamics = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_pulse_to_network_continuous_dynamics_20260320" / "summary.json"
    )

    hn = native["headline_metrics"]
    hc = circuit["headline_metrics"]
    hd = dynamics["headline_metrics"]

    direct_binding_measure = hc["native_binding"] * (1.0 + hd["d_feature"] / (1.0 + hd["d_seed"]))
    direct_gate_measure = hc["native_gate"] * (1.0 + hn["native_inhibition"])
    direct_attractor_measure = hc["native_attractor"] * (1.0 + hd["d_structure"] / (1.0 + hd["d_global"]))
    direct_circuit_margin = direct_binding_measure + direct_attractor_measure - direct_gate_measure

    return {
        "headline_metrics": {
            "direct_binding_measure": direct_binding_measure,
            "direct_gate_measure": direct_gate_measure,
            "direct_attractor_measure": direct_attractor_measure,
            "direct_circuit_margin": direct_circuit_margin,
        },
        "direct_measure_equation": {
            "binding_term": "B_direct = native_binding * (1 + dFeature/dt / (1 + dSeed/dt))",
            "gate_term": "G_direct = native_gate * (1 + native_inhibition)",
            "attractor_term": "A_direct = native_attractor * (1 + dStructure/dt / (1 + dGlobal/dt))",
            "margin_term": "M_direct = B_direct + A_direct - G_direct",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 回路级原生量直测报告",
        "",
        f"- direct_binding_measure: {hm['direct_binding_measure']:.6f}",
        f"- direct_gate_measure: {hm['direct_gate_measure']:.6f}",
        f"- direct_attractor_measure: {hm['direct_attractor_measure']:.6f}",
        f"- direct_circuit_margin: {hm['direct_circuit_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_circuit_native_direct_measure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
