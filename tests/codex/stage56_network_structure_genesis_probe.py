from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_network_structure_genesis_probe_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_network_structure_genesis_probe_summary() -> dict:
    chain = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_feature_network_chain_20260320" / "summary.json"
    )
    neuron = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_native_direct_closure_20260320" / "summary.json"
    )

    hc = chain["headline_metrics"]
    hn = neuron["headline_metrics"]

    feature_to_structure_gain = hc["network_growth_signal"] / (1.0 + hc["feature_lock_signal"])
    circuit_binding_gain = hc["circuit_closure_signal"] / (1.0 + hc["neuron_seed_signal"] + hc["feature_selection_signal"])
    feedback_retention = hc["steady_feedback_signal"] * (1.0 + hn["neuron_closure_confidence"])
    genesis_margin = feature_to_structure_gain + circuit_binding_gain + feedback_retention

    return {
        "headline_metrics": {
            "feature_to_structure_gain": feature_to_structure_gain,
            "circuit_binding_gain": circuit_binding_gain,
            "feedback_retention": feedback_retention,
            "genesis_margin": genesis_margin,
        },
        "genesis_equation": {
            "growth_term": "G_fs = N_struct_growth / (1 + N_lock)",
            "binding_term": "G_cb = N_circuit / (1 + N_seed + N_feat)",
            "feedback_term": "G_fb = N_feedback * (1 + C_neuron)",
            "margin_term": "M_genesis = G_fs + G_cb + G_fb",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征到网络结构生成链报告",
        "",
        f"- feature_to_structure_gain: {hm['feature_to_structure_gain']:.6f}",
        f"- circuit_binding_gain: {hm['circuit_binding_gain']:.6f}",
        f"- feedback_retention: {hm['feedback_retention']:.6f}",
        f"- genesis_margin: {hm['genesis_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_network_structure_genesis_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
