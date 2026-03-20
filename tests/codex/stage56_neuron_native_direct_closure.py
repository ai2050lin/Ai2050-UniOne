from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_neuron_native_direct_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_neuron_native_direct_closure_summary() -> dict:
    feature_terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_terminal_direct_20260320" / "summary.json"
    )
    v26 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v26_20260320" / "summary.json"
    )

    hf = feature_terminal["headline_metrics"]
    hv = v26["headline_metrics"]

    neuron_seed_direct = hf["direct_basis_v5"]
    neuron_select_direct = hf["direct_selectivity_v5"]
    neuron_lock_direct = hf["direct_lock_v5"]
    neuron_native_core = neuron_seed_direct + neuron_select_direct + neuron_lock_direct
    neuron_feature_ratio = neuron_lock_direct / (1.0 + neuron_seed_direct + neuron_select_direct)
    neuron_closure_confidence = neuron_native_core / (1.0 + hv["feature_term_v26"])

    return {
        "headline_metrics": {
            "neuron_seed_direct": neuron_seed_direct,
            "neuron_select_direct": neuron_select_direct,
            "neuron_lock_direct": neuron_lock_direct,
            "neuron_native_core": neuron_native_core,
            "neuron_feature_ratio": neuron_feature_ratio,
            "neuron_closure_confidence": neuron_closure_confidence,
        },
        "closure_equation": {
            "seed_term": "N_seed_direct = F_basis_v5",
            "select_term": "N_select_direct = F_sel_v5",
            "lock_term": "N_lock_direct = F_lock_v5",
            "core_term": "N_core = N_seed_direct + N_select_direct + N_lock_direct",
            "confidence_term": "C_neuron = N_core / (1 + K_f_v26)",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 神经元近直测收口报告",
        "",
        f"- neuron_seed_direct: {hm['neuron_seed_direct']:.6f}",
        f"- neuron_select_direct: {hm['neuron_select_direct']:.6f}",
        f"- neuron_lock_direct: {hm['neuron_lock_direct']:.6f}",
        f"- neuron_native_core: {hm['neuron_native_core']:.6f}",
        f"- neuron_feature_ratio: {hm['neuron_feature_ratio']:.6f}",
        f"- neuron_closure_confidence: {hm['neuron_closure_confidence']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_neuron_native_direct_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
