from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v8_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v8_summary() -> dict:
    v7 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v7_20260320" / "summary.json")
    feature = _load_json(ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_primary_structure_20260320" / "summary.json")
    circuit = _load_json(ROOT / "tests" / "codex_temp" / "stage56_circuit_native_variable_refinement_20260320" / "summary.json")
    dynamics = _load_json(ROOT / "tests" / "codex_temp" / "stage56_pulse_to_network_continuous_dynamics_20260320" / "summary.json")

    hv7 = v7["headline_metrics"]
    hf = feature["headline_metrics"]
    hc = circuit["headline_metrics"]
    hd = dynamics["headline_metrics"]

    seed_feature_term_v8 = hv7["seed_feature_term_v7"] + hf["feature_primary_margin"]
    structure_term_v8 = hv7["structure_term_v7"] + hc["circuit_native_margin"]
    stability_term_v8 = hv7["stability_term_v7"] + hd["d_global"]
    pressure_term_v8 = hv7["pressure_term_v7"] + hc["native_gate"]
    encoding_margin_v8 = seed_feature_term_v8 + structure_term_v8 + stability_term_v8 - pressure_term_v8

    return {
        "headline_metrics": {
            "seed_feature_term_v8": seed_feature_term_v8,
            "structure_term_v8": structure_term_v8,
            "stability_term_v8": stability_term_v8,
            "pressure_term_v8": pressure_term_v8,
            "encoding_margin_v8": encoding_margin_v8,
        },
        "closed_form_equation": {
            "seed_feature_term": "K_sf_v8 = seed_feature_term_v7 + feature_primary_margin",
            "structure_term": "K_st_v8 = structure_term_v7 + circuit_native_margin",
            "stability_term": "K_ss_v8 = stability_term_v7 + d_global",
            "pressure_term": "P_v8 = pressure_term_v7 + native_gate",
            "margin_term": "M_encoding_v8 = K_sf_v8 + K_st_v8 + K_ss_v8 - P_v8",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第八版报告",
        "",
        f"- seed_feature_term_v8: {hm['seed_feature_term_v8']:.6f}",
        f"- structure_term_v8: {hm['structure_term_v8']:.6f}",
        f"- stability_term_v8: {hm['stability_term_v8']:.6f}",
        f"- pressure_term_v8: {hm['pressure_term_v8']:.6f}",
        f"- encoding_margin_v8: {hm['encoding_margin_v8']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v8_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
