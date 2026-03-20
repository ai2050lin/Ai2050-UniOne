from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_circuit_dynamics_bridge_v2_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_circuit_dynamics_bridge_v2_summary() -> dict:
    circuit_v3 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_circuit_bridge_v3_20260320" / "summary.json")
    growth = _load_json(ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_network_growth_20260320" / "summary.json")
    neuro = _load_json(ROOT / "tests" / "codex_temp" / "stage56_continuous_neurodynamics_bridge_20260320" / "summary.json")

    hc = circuit_v3["headline_metrics"]
    hg = growth["headline_metrics"]
    hn = neuro["headline_metrics"]

    recurrent_binding = hc["bind_balanced"] + hn["dS_dt"]
    competitive_gate = hg["structure_pressure"] + hn["dB_dt"]
    attractor_loading = hg["global_steady_drive"] / (1.0 + competitive_gate)
    circuit_dynamic_margin = recurrent_binding + attractor_loading - competitive_gate

    return {
        "headline_metrics": {
            "recurrent_binding": recurrent_binding,
            "competitive_gate": competitive_gate,
            "attractor_loading": attractor_loading,
            "circuit_dynamic_margin": circuit_dynamic_margin,
        },
        "dynamic_equation": {
            "binding_term": "B_rec = bind_balanced + dS_dt",
            "gate_term": "G_comp = structure_pressure + dB_dt",
            "attractor_term": "A_load = global_steady_drive / (1 + G_comp)",
            "margin_term": "M_dyn = B_rec + A_load - G_comp",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 回路级动力学桥接第二版报告",
        "",
        f"- recurrent_binding: {hm['recurrent_binding']:.6f}",
        f"- competitive_gate: {hm['competitive_gate']:.6f}",
        f"- attractor_loading: {hm['attractor_loading']:.6f}",
        f"- circuit_dynamic_margin: {hm['circuit_dynamic_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_circuit_dynamics_bridge_v2_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
