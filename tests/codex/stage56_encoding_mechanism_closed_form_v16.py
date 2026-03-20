from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v16_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v16_summary() -> dict:
    v15 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v15_20260320" / "summary.json")
    absolute_lock = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_absolute_lock_20260320" / "summary.json"
    )
    circuit_v8 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v8_20260320" / "summary.json"
    )
    canonical = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_canonical_closure_20260320" / "summary.json"
    )

    hv = v15["headline_metrics"]
    ha = absolute_lock["headline_metrics"]
    hc = circuit_v8["headline_metrics"]
    hl = canonical["headline_metrics"]

    feature_term_v16 = hv["feature_term_v15"] + ha["absolute_margin"]
    structure_term_v16 = hv["structure_term_v15"] + hc["direct_margin_v8"]
    learning_term_v16 = hv["learning_term_v15"] + hl["canonical_global"]
    pressure_term_v16 = hv["pressure_term_v15"] + hc["direct_gate_v8"]
    encoding_margin_v16 = feature_term_v16 + structure_term_v16 + learning_term_v16 - pressure_term_v16

    return {
        "headline_metrics": {
            "feature_term_v16": feature_term_v16,
            "structure_term_v16": structure_term_v16,
            "learning_term_v16": learning_term_v16,
            "pressure_term_v16": pressure_term_v16,
            "encoding_margin_v16": encoding_margin_v16,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v16 = feature_term_v15 + absolute_margin",
            "structure_term": "K_s_v16 = structure_term_v15 + direct_margin_v8",
            "learning_term": "K_l_v16 = learning_term_v15 + canonical_global",
            "pressure_term": "P_v16 = pressure_term_v15 + direct_gate_v8",
            "margin_term": "M_encoding_v16 = K_f_v16 + K_s_v16 + K_l_v16 - P_v16",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第十六版报告",
        "",
        f"- feature_term_v16: {hm['feature_term_v16']:.6f}",
        f"- structure_term_v16: {hm['structure_term_v16']:.6f}",
        f"- learning_term_v16: {hm['learning_term_v16']:.6f}",
        f"- pressure_term_v16: {hm['pressure_term_v16']:.6f}",
        f"- encoding_margin_v16: {hm['encoding_margin_v16']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v16_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
