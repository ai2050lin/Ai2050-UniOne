from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_neuron_feature_network_chain_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_neuron_feature_network_chain_summary() -> dict:
    feature_terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_terminal_direct_20260320" / "summary.json"
    )
    structure_terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_terminal_closure_20260320" / "summary.json"
    )
    equal_level = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_equal_level_closure_20260320" / "summary.json"
    )

    hf = feature_terminal["headline_metrics"]
    hs = structure_terminal["headline_metrics"]
    he = equal_level["headline_metrics"]

    neuron_seed_signal = hf["direct_basis_v5"]
    feature_selection_signal = hf["direct_selectivity_v5"]
    feature_lock_signal = hf["direct_lock_v5"]
    network_growth_signal = hs["terminal_structure_closure"]
    circuit_closure_signal = hs["terminal_circuit_closure"]
    steady_feedback_signal = hs["terminal_feedback_closure"]
    chain_margin = (
        neuron_seed_signal
        + feature_selection_signal
        + feature_lock_signal
        + network_growth_signal
        + circuit_closure_signal
        + steady_feedback_signal
    ) / (1.0 + (1.0 - he["equalization_confidence"]))

    return {
        "headline_metrics": {
            "neuron_seed_signal": neuron_seed_signal,
            "feature_selection_signal": feature_selection_signal,
            "feature_lock_signal": feature_lock_signal,
            "network_growth_signal": network_growth_signal,
            "circuit_closure_signal": circuit_closure_signal,
            "steady_feedback_signal": steady_feedback_signal,
            "chain_margin": chain_margin,
        },
        "chain_equation": {
            "seed_term": "N_seed = F_basis_v5",
            "selection_term": "N_feat = F_sel_v5",
            "lock_term": "N_lock = F_lock_v5",
            "growth_term": "N_struct = Tc_fs + Tc_fc + Tc_fb",
            "margin_term": "M_chain = (N_seed + N_feat + N_lock + N_struct) / (1 + (1 - C_equal))",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 神经元到特征再到网络结构链报告",
        "",
        f"- neuron_seed_signal: {hm['neuron_seed_signal']:.6f}",
        f"- feature_selection_signal: {hm['feature_selection_signal']:.6f}",
        f"- feature_lock_signal: {hm['feature_lock_signal']:.6f}",
        f"- network_growth_signal: {hm['network_growth_signal']:.6f}",
        f"- circuit_closure_signal: {hm['circuit_closure_signal']:.6f}",
        f"- steady_feedback_signal: {hm['steady_feedback_signal']:.6f}",
        f"- chain_margin: {hm['chain_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_neuron_feature_network_chain_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
