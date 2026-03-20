from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v10_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v10_summary() -> dict:
    v9 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v9_20260320" / "summary.json")
    dominance = _load_json(ROOT / "tests" / "codex_temp" / "stage56_feature_primary_dominance_20260320" / "summary.json")
    circuit_v2 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_circuit_direct_refinement_v2_20260320" / "summary.json")
    terminal = _load_json(ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_form_20260320" / "summary.json")

    hv9 = v9["headline_metrics"]
    hd = dominance["headline_metrics"]
    hc = circuit_v2["headline_metrics"]
    ht = terminal["headline_metrics"]

    feature_term_v10 = hv9["feature_term_v9"] + hd["dominance_margin"]
    structure_term_v10 = hv9["structure_term_v9"] + hc["direct_margin_v2"]
    learning_term_v10 = hv9["learning_term_v9"] + ht["terminal_global"]
    pressure_term_v10 = hv9["pressure_term_v9"] + hc["direct_gate_v2"]
    encoding_margin_v10 = feature_term_v10 + structure_term_v10 + learning_term_v10 - pressure_term_v10

    return {
        "headline_metrics": {
            "feature_term_v10": feature_term_v10,
            "structure_term_v10": structure_term_v10,
            "learning_term_v10": learning_term_v10,
            "pressure_term_v10": pressure_term_v10,
            "encoding_margin_v10": encoding_margin_v10,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v10 = feature_term_v9 + dominance_margin",
            "structure_term": "K_s_v10 = structure_term_v9 + direct_margin_v2",
            "learning_term": "K_l_v10 = learning_term_v9 + terminal_global",
            "pressure_term": "P_v10 = pressure_term_v9 + direct_gate_v2",
            "margin_term": "M_encoding_v10 = K_f_v10 + K_s_v10 + K_l_v10 - P_v10",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第十版报告",
        "",
        f"- feature_term_v10: {hm['feature_term_v10']:.6f}",
        f"- structure_term_v10: {hm['structure_term_v10']:.6f}",
        f"- learning_term_v10: {hm['learning_term_v10']:.6f}",
        f"- pressure_term_v10: {hm['pressure_term_v10']:.6f}",
        f"- encoding_margin_v10: {hm['encoding_margin_v10']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v10_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
