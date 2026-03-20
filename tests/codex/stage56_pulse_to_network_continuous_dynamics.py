from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_pulse_to_network_continuous_dynamics_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_pulse_to_network_continuous_dynamics_summary() -> dict:
    native = _load_json(ROOT / "tests" / "codex_temp" / "stage56_spike_feature_native_variables_20260320" / "summary.json")
    feature = _load_json(ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_primary_structure_20260320" / "summary.json")
    circuit = _load_json(ROOT / "tests" / "codex_temp" / "stage56_circuit_native_variable_refinement_20260320" / "summary.json")

    hn = native["headline_metrics"]
    hf = feature["headline_metrics"]
    hc = circuit["headline_metrics"]

    d_seed = hn["native_seed"] - hn["native_inhibition"]
    d_feature = hf["feature_structure_support"] - hn["native_selectivity"]
    d_structure = hc["native_attractor"] - hc["native_gate"]
    d_global = d_seed + d_feature + d_structure

    return {
        "headline_metrics": {
            "d_seed": d_seed,
            "d_feature": d_feature,
            "d_structure": d_structure,
            "d_global": d_global,
        },
        "continuous_equation": {
            "seed_ode": "dSeed/dt = native_seed - native_inhibition",
            "feature_ode": "dFeature/dt = feature_structure_support - native_selectivity",
            "structure_ode": "dStructure/dt = native_attractor - native_gate",
            "global_ode": "dGlobal/dt = dSeed/dt + dFeature/dt + dStructure/dt",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 脉冲到网络连续动力学报告",
        "",
        f"- d_seed: {hm['d_seed']:.6f}",
        f"- d_feature: {hm['d_feature']:.6f}",
        f"- d_structure: {hm['d_structure']:.6f}",
        f"- d_global: {hm['d_global']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_pulse_to_network_continuous_dynamics_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
