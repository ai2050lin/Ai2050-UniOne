from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v12_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v12_summary() -> dict:
    v11 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v11_20260320" / "summary.json")
    final_feature = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_finalization_20260320" / "summary.json"
    )
    circuit_v4 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_direct_closure_v4_20260320" / "summary.json"
    )
    learning_final = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_final_20260320" / "summary.json"
    )

    hv = v11["headline_metrics"]
    hf = final_feature["headline_metrics"]
    hc = circuit_v4["headline_metrics"]
    hl = learning_final["headline_metrics"]

    feature_term_v12 = hv["feature_term_v11"] + hf["final_margin"]
    structure_term_v12 = hv["structure_term_v11"] + hc["direct_margin_v4"]
    learning_term_v12 = hv["learning_term_v11"] + hl["final_global"]
    pressure_term_v12 = hv["pressure_term_v11"] + hc["direct_gate_v4"]
    encoding_margin_v12 = feature_term_v12 + structure_term_v12 + learning_term_v12 - pressure_term_v12

    return {
        "headline_metrics": {
            "feature_term_v12": feature_term_v12,
            "structure_term_v12": structure_term_v12,
            "learning_term_v12": learning_term_v12,
            "pressure_term_v12": pressure_term_v12,
            "encoding_margin_v12": encoding_margin_v12,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v12 = feature_term_v11 + final_margin",
            "structure_term": "K_s_v12 = structure_term_v11 + direct_margin_v4",
            "learning_term": "K_l_v12 = learning_term_v11 + final_global",
            "pressure_term": "P_v12 = pressure_term_v11 + direct_gate_v4",
            "margin_term": "M_encoding_v12 = K_f_v12 + K_s_v12 + K_l_v12 - P_v12",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第十二版报告",
        "",
        f"- feature_term_v12: {hm['feature_term_v12']:.6f}",
        f"- structure_term_v12: {hm['structure_term_v12']:.6f}",
        f"- learning_term_v12: {hm['learning_term_v12']:.6f}",
        f"- pressure_term_v12: {hm['pressure_term_v12']:.6f}",
        f"- encoding_margin_v12: {hm['encoding_margin_v12']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v12_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
