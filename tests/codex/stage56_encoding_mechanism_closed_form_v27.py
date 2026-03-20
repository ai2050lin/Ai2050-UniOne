from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v27_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v27_summary() -> dict:
    v26 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v26_20260320" / "summary.json"
    )
    neuron = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_native_direct_closure_20260320" / "summary.json"
    )
    genesis = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_network_structure_genesis_probe_20260320" / "summary.json"
    )

    hv = v26["headline_metrics"]
    hn = neuron["headline_metrics"]
    hg = genesis["headline_metrics"]

    feature_term_v27 = hv["feature_term_v26"] + hn["neuron_native_core"]
    structure_term_v27 = hv["structure_term_v26"] + hg["genesis_margin"]
    learning_term_v27 = hv["learning_term_v26"] + hg["feedback_retention"] + hg["genesis_margin"] * hn["neuron_closure_confidence"]
    pressure_term_v27 = hv["pressure_term_v26"] + (1.0 - hn["neuron_closure_confidence"])
    encoding_margin_v27 = feature_term_v27 + structure_term_v27 + learning_term_v27 - pressure_term_v27

    return {
        "headline_metrics": {
            "feature_term_v27": feature_term_v27,
            "structure_term_v27": structure_term_v27,
            "learning_term_v27": learning_term_v27,
            "pressure_term_v27": pressure_term_v27,
            "encoding_margin_v27": encoding_margin_v27,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v27 = K_f_v26 + N_core",
            "structure_term": "K_s_v27 = K_s_v26 + M_genesis",
            "learning_term": "K_l_v27 = K_l_v26 + G_fb + M_genesis * C_neuron",
            "pressure_term": "P_v27 = P_v26 + (1 - C_neuron)",
            "margin_term": "M_encoding_v27 = K_f_v27 + K_s_v27 + K_l_v27 - P_v27",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第二十七版报告",
        "",
        f"- feature_term_v27: {hm['feature_term_v27']:.6f}",
        f"- structure_term_v27: {hm['structure_term_v27']:.6f}",
        f"- learning_term_v27: {hm['learning_term_v27']:.6f}",
        f"- pressure_term_v27: {hm['pressure_term_v27']:.6f}",
        f"- encoding_margin_v27: {hm['encoding_margin_v27']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v27_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
