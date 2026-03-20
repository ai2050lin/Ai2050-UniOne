from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v7_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v7_summary() -> dict:
    v6 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_spike_closed_form_v6_20260320" / "summary.json")
    feature_bal = _load_json(ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_balance_refinement_20260320" / "summary.json")
    native = _load_json(ROOT / "tests" / "codex_temp" / "stage56_spike_feature_native_variables_20260320" / "summary.json")
    circuit_dyn = _load_json(ROOT / "tests" / "codex_temp" / "stage56_circuit_dynamics_bridge_v2_20260320" / "summary.json")

    hv6 = v6["headline_metrics"]
    hfb = feature_bal["headline_metrics"]
    hn = native["headline_metrics"]
    hcd = circuit_dyn["headline_metrics"]

    seed_feature_term_v7 = hv6["seed_core_v6"] + hfb["feature_balance_margin"] + hn["native_feature"]
    structure_term_v7 = hv6["structure_core_v6"] + hcd["circuit_dynamic_margin"] + hn["native_selectivity"]
    stability_term_v7 = hv6["steady_core_v6"] + hcd["attractor_loading"]
    pressure_term_v7 = hv6["pressure_core_v6"] + hcd["competitive_gate"]
    encoding_margin_v7 = seed_feature_term_v7 + structure_term_v7 + stability_term_v7 - pressure_term_v7

    return {
        "headline_metrics": {
            "seed_feature_term_v7": seed_feature_term_v7,
            "structure_term_v7": structure_term_v7,
            "stability_term_v7": stability_term_v7,
            "pressure_term_v7": pressure_term_v7,
            "encoding_margin_v7": encoding_margin_v7,
        },
        "closed_form_equation": {
            "seed_feature_term": "K_sf_v7 = seed_core_v6 + feature_balance_margin + native_feature",
            "structure_term": "K_st_v7 = structure_core_v6 + circuit_dynamic_margin + native_selectivity",
            "stability_term": "K_ss_v7 = steady_core_v6 + attractor_loading",
            "pressure_term": "P_v7 = pressure_core_v6 + competitive_gate",
            "margin_term": "M_encoding_v7 = K_sf_v7 + K_st_v7 + K_ss_v7 - P_v7",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第七版报告",
        "",
        f"- seed_feature_term_v7: {hm['seed_feature_term_v7']:.6f}",
        f"- structure_term_v7: {hm['structure_term_v7']:.6f}",
        f"- stability_term_v7: {hm['stability_term_v7']:.6f}",
        f"- pressure_term_v7: {hm['pressure_term_v7']:.6f}",
        f"- encoding_margin_v7: {hm['encoding_margin_v7']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v7_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
