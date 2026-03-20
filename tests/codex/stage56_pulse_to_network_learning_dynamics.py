from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_pulse_to_network_learning_dynamics_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_pulse_to_network_learning_dynamics_summary() -> dict:
    continuous = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_pulse_to_network_continuous_dynamics_20260320" / "summary.json"
    )
    direct = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_direct_measure_20260320" / "summary.json"
    )
    feature = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_primary_structure_20260320" / "summary.json"
    )

    hc = continuous["headline_metrics"]
    hd = direct["headline_metrics"]
    hf = feature["headline_metrics"]

    learning_seed = hc["d_seed"] / (1.0 + hd["direct_gate_measure"])
    learning_feature = hc["d_feature"] + hf["feature_structure_support"]
    learning_structure = hc["d_structure"] + hd["direct_attractor_measure"]
    learning_global = learning_seed + learning_feature + learning_structure

    return {
        "headline_metrics": {
            "learning_seed": learning_seed,
            "learning_feature": learning_feature,
            "learning_structure": learning_structure,
            "learning_global": learning_global,
        },
        "learning_equation": {
            "seed_term": "L_seed = dSeed/dt / (1 + direct_gate_measure)",
            "feature_term": "L_feature = dFeature/dt + feature_structure_support",
            "structure_term": "L_structure = dStructure/dt + direct_attractor_measure",
            "global_term": "L_global = L_seed + L_feature + L_structure",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 脉冲到网络连续学习动力学报告",
        "",
        f"- learning_seed: {hm['learning_seed']:.6f}",
        f"- learning_feature: {hm['learning_feature']:.6f}",
        f"- learning_structure: {hm['learning_structure']:.6f}",
        f"- learning_global: {hm['learning_global']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_pulse_to_network_learning_dynamics_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
