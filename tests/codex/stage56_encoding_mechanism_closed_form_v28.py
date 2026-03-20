from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v28_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v28_summary() -> dict:
    v27 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v27_20260320" / "summary.json"
    )
    neuron_origin = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_origin_native_probe_20260320" / "summary.json"
    )
    structure_direct = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_genesis_direct_measure_v2_20260320" / "summary.json"
    )

    hv = v27["headline_metrics"]
    hn = neuron_origin["headline_metrics"]
    hs = structure_direct["headline_metrics"]

    feature_term_v28 = hv["feature_term_v27"] + hn["neuron_origin_core"]
    structure_term_v28 = hv["structure_term_v27"] + hs["structure_genesis_direct_core"]
    learning_term_v28 = (
        hv["learning_term_v27"]
        + hs["feedback_stability_direct"]
        + hs["structure_genesis_direct_core"] * hn["neuron_origin_confidence"]
    )
    pressure_term_v28 = hv["pressure_term_v27"] + (1.0 - hs["structure_direct_confidence"])
    encoding_margin_v28 = feature_term_v28 + structure_term_v28 + learning_term_v28 - pressure_term_v28

    return {
        "headline_metrics": {
            "feature_term_v28": feature_term_v28,
            "structure_term_v28": structure_term_v28,
            "learning_term_v28": learning_term_v28,
            "pressure_term_v28": pressure_term_v28,
            "encoding_margin_v28": encoding_margin_v28,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v28 = K_f_v27 + N_origin",
            "structure_term": "K_s_v28 = K_s_v27 + S_core",
            "learning_term": "K_l_v28 = K_l_v27 + S_fb + S_core * C_origin",
            "pressure_term": "P_v28 = P_v27 + (1 - C_struct)",
            "margin_term": "M_encoding_v28 = K_f_v28 + K_s_v28 + K_l_v28 - P_v28",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第二十八版报告",
        "",
        f"- feature_term_v28: {hm['feature_term_v28']:.6f}",
        f"- structure_term_v28: {hm['structure_term_v28']:.6f}",
        f"- learning_term_v28: {hm['learning_term_v28']:.6f}",
        f"- pressure_term_v28: {hm['pressure_term_v28']:.6f}",
        f"- encoding_margin_v28: {hm['encoding_margin_v28']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v28_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
