from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v15_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v15_summary() -> dict:
    v14 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v14_20260320" / "summary.json")
    feature_lock = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_irreversible_lock_20260320" / "summary.json"
    )
    circuit_v7 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v7_20260320" / "summary.json"
    )
    closure_v2 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_final_closure_20260320" / "summary.json"
    )

    hv = v14["headline_metrics"]
    hf = feature_lock["headline_metrics"]
    hc = circuit_v7["headline_metrics"]
    hl = closure_v2["headline_metrics"]

    feature_term_v15 = hv["feature_term_v14"] + hf["lock_margin"]
    structure_term_v15 = hv["structure_term_v14"] + hc["direct_margin_v7"]
    learning_term_v15 = hv["learning_term_v14"] + hl["closure_global_v2"]
    pressure_term_v15 = hv["pressure_term_v14"] + hc["direct_gate_v7"]
    encoding_margin_v15 = feature_term_v15 + structure_term_v15 + learning_term_v15 - pressure_term_v15

    return {
        "headline_metrics": {
            "feature_term_v15": feature_term_v15,
            "structure_term_v15": structure_term_v15,
            "learning_term_v15": learning_term_v15,
            "pressure_term_v15": pressure_term_v15,
            "encoding_margin_v15": encoding_margin_v15,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v15 = feature_term_v14 + lock_margin",
            "structure_term": "K_s_v15 = structure_term_v14 + direct_margin_v7",
            "learning_term": "K_l_v15 = learning_term_v14 + closure_global_v2",
            "pressure_term": "P_v15 = pressure_term_v14 + direct_gate_v7",
            "margin_term": "M_encoding_v15 = K_f_v15 + K_s_v15 + K_l_v15 - P_v15",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第十五版报告",
        "",
        f"- feature_term_v15: {hm['feature_term_v15']:.6f}",
        f"- structure_term_v15: {hm['structure_term_v15']:.6f}",
        f"- learning_term_v15: {hm['learning_term_v15']:.6f}",
        f"- pressure_term_v15: {hm['pressure_term_v15']:.6f}",
        f"- encoding_margin_v15: {hm['encoding_margin_v15']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v15_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
