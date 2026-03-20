from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v11_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v11_summary() -> dict:
    v10 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v10_20260320" / "summary.json")
    reinforce = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_reinforcement_20260320" / "summary.json"
    )
    circuit_v3 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_terminal_measure_20260320" / "summary.json"
    )
    closure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_closure_20260320" / "summary.json"
    )

    hv = v10["headline_metrics"]
    hr = reinforce["headline_metrics"]
    hc = circuit_v3["headline_metrics"]
    hl = closure["headline_metrics"]

    feature_term_v11 = hv["feature_term_v10"] + hr["reinforced_margin"]
    structure_term_v11 = hv["structure_term_v10"] + hc["direct_margin_v3"]
    learning_term_v11 = hv["learning_term_v10"] + hl["closure_global"]
    pressure_term_v11 = hv["pressure_term_v10"] + hc["direct_gate_v3"]
    encoding_margin_v11 = feature_term_v11 + structure_term_v11 + learning_term_v11 - pressure_term_v11

    return {
        "headline_metrics": {
            "feature_term_v11": feature_term_v11,
            "structure_term_v11": structure_term_v11,
            "learning_term_v11": learning_term_v11,
            "pressure_term_v11": pressure_term_v11,
            "encoding_margin_v11": encoding_margin_v11,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v11 = feature_term_v10 + reinforced_margin",
            "structure_term": "K_s_v11 = structure_term_v10 + direct_margin_v3",
            "learning_term": "K_l_v11 = learning_term_v10 + closure_global",
            "pressure_term": "P_v11 = pressure_term_v10 + direct_gate_v3",
            "margin_term": "M_encoding_v11 = K_f_v11 + K_s_v11 + K_l_v11 - P_v11",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第十一版报告",
        "",
        f"- feature_term_v11: {hm['feature_term_v11']:.6f}",
        f"- structure_term_v11: {hm['structure_term_v11']:.6f}",
        f"- learning_term_v11: {hm['learning_term_v11']:.6f}",
        f"- pressure_term_v11: {hm['pressure_term_v11']:.6f}",
        f"- encoding_margin_v11: {hm['encoding_margin_v11']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v11_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
