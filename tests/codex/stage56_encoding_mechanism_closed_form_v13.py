from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v13_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v13_summary() -> dict:
    v12 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v12_20260320" / "summary.json")
    feature_lock = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_locking_20260320" / "summary.json"
    )
    circuit_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_direct_terminal_lock_20260320" / "summary.json"
    )
    learning_lock = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_lock_20260320" / "summary.json"
    )

    hv = v12["headline_metrics"]
    hf = feature_lock["headline_metrics"]
    hc = circuit_v5["headline_metrics"]
    hl = learning_lock["headline_metrics"]

    feature_term_v13 = hv["feature_term_v12"] + hf["locking_margin"]
    structure_term_v13 = hv["structure_term_v12"] + hc["direct_margin_v5"]
    learning_term_v13 = hv["learning_term_v12"] + hl["locked_global"]
    pressure_term_v13 = hv["pressure_term_v12"] + hc["direct_gate_v5"]
    encoding_margin_v13 = feature_term_v13 + structure_term_v13 + learning_term_v13 - pressure_term_v13

    return {
        "headline_metrics": {
            "feature_term_v13": feature_term_v13,
            "structure_term_v13": structure_term_v13,
            "learning_term_v13": learning_term_v13,
            "pressure_term_v13": pressure_term_v13,
            "encoding_margin_v13": encoding_margin_v13,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v13 = feature_term_v12 + locking_margin",
            "structure_term": "K_s_v13 = structure_term_v12 + direct_margin_v5",
            "learning_term": "K_l_v13 = learning_term_v12 + locked_global",
            "pressure_term": "P_v13 = pressure_term_v12 + direct_gate_v5",
            "margin_term": "M_encoding_v13 = K_f_v13 + K_s_v13 + K_l_v13 - P_v13",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第十三版报告",
        "",
        f"- feature_term_v13: {hm['feature_term_v13']:.6f}",
        f"- structure_term_v13: {hm['structure_term_v13']:.6f}",
        f"- learning_term_v13: {hm['learning_term_v13']:.6f}",
        f"- pressure_term_v13: {hm['pressure_term_v13']:.6f}",
        f"- encoding_margin_v13: {hm['encoding_margin_v13']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v13_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
