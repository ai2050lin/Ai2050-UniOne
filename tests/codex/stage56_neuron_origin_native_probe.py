from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_neuron_origin_native_probe_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_neuron_origin_native_probe_summary() -> dict:
    feature_terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_terminal_direct_20260320" / "summary.json"
    )
    neuron_direct = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_native_direct_closure_20260320" / "summary.json"
    )

    hf = feature_terminal["headline_metrics"]
    hn = neuron_direct["headline_metrics"]

    pulse_source_strength = hf["direct_basis_v5"] + 0.5 * hf["direct_selectivity_v5"]
    selectivity_focus = hf["direct_selectivity_v5"] / (1.0 + hf["direct_basis_v5"])
    lock_retention = hf["direct_lock_v5"] / (1.0 + hf["direct_selectivity_v5"])
    neuron_origin_core = pulse_source_strength + selectivity_focus + lock_retention
    neuron_origin_confidence = neuron_origin_core / (1.0 + hn["neuron_native_core"])

    return {
        "headline_metrics": {
            "pulse_source_strength": pulse_source_strength,
            "selectivity_focus": selectivity_focus,
            "lock_retention": lock_retention,
            "neuron_origin_core": neuron_origin_core,
            "neuron_origin_confidence": neuron_origin_confidence,
        },
        "origin_equation": {
            "source_term": "N_source = F_basis_v5 + 0.5 * F_sel_v5",
            "focus_term": "N_focus = F_sel_v5 / (1 + F_basis_v5)",
            "retention_term": "N_retention = F_lock_v5 / (1 + F_sel_v5)",
            "core_term": "N_origin = N_source + N_focus + N_retention",
            "confidence_term": "C_origin = N_origin / (1 + N_core)",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 神经元起点原生探针报告",
        "",
        f"- pulse_source_strength: {hm['pulse_source_strength']:.6f}",
        f"- selectivity_focus: {hm['selectivity_focus']:.6f}",
        f"- lock_retention: {hm['lock_retention']:.6f}",
        f"- neuron_origin_core: {hm['neuron_origin_core']:.6f}",
        f"- neuron_origin_confidence: {hm['neuron_origin_confidence']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_neuron_origin_native_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
