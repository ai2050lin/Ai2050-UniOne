from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v14_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v14_summary() -> dict:
    v13 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v13_20260320" / "summary.json")
    irreversibility = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_irreversibility_20260320" / "summary.json"
    )
    circuit_v6 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v6_20260320" / "summary.json"
    )
    learning_irrev = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_irreversible_20260320" / "summary.json"
    )

    hv = v13["headline_metrics"]
    hi = irreversibility["headline_metrics"]
    hc = circuit_v6["headline_metrics"]
    hl = learning_irrev["headline_metrics"]

    feature_term_v14 = hv["feature_term_v13"] + hi["irreversible_margin"]
    structure_term_v14 = hv["structure_term_v13"] + hc["direct_margin_v6"]
    learning_term_v14 = hv["learning_term_v13"] + hl["irreversible_global"]
    pressure_term_v14 = hv["pressure_term_v13"] + hc["direct_gate_v6"]
    encoding_margin_v14 = feature_term_v14 + structure_term_v14 + learning_term_v14 - pressure_term_v14

    return {
        "headline_metrics": {
            "feature_term_v14": feature_term_v14,
            "structure_term_v14": structure_term_v14,
            "learning_term_v14": learning_term_v14,
            "pressure_term_v14": pressure_term_v14,
            "encoding_margin_v14": encoding_margin_v14,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v14 = feature_term_v13 + irreversible_margin",
            "structure_term": "K_s_v14 = structure_term_v13 + direct_margin_v6",
            "learning_term": "K_l_v14 = learning_term_v13 + irreversible_global",
            "pressure_term": "P_v14 = pressure_term_v13 + direct_gate_v6",
            "margin_term": "M_encoding_v14 = K_f_v14 + K_s_v14 + K_l_v14 - P_v14",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第十四版报告",
        "",
        f"- feature_term_v14: {hm['feature_term_v14']:.6f}",
        f"- structure_term_v14: {hm['structure_term_v14']:.6f}",
        f"- learning_term_v14: {hm['learning_term_v14']:.6f}",
        f"- pressure_term_v14: {hm['pressure_term_v14']:.6f}",
        f"- encoding_margin_v14: {hm['encoding_margin_v14']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v14_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
