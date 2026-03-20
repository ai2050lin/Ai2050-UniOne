from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v23_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v23_summary() -> dict:
    v22 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v22_20260320" / "summary.json"
    )
    balance = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_balance_normalization_20260320" / "summary.json"
    )

    hv = v22["headline_metrics"]
    hb = balance["headline_metrics"]

    feature_term_v23 = hb["balanced_feature"]
    structure_term_v23 = hb["balanced_structure"]
    learning_term_v23 = hv["learning_term_v22"] * (1.0 + hb["balance_gain"])
    pressure_term_v23 = hv["pressure_term_v22"] + hb["residual_gap"] / max(hv["structure_term_v22"], 1e-9)
    encoding_margin_v23 = feature_term_v23 + structure_term_v23 + learning_term_v23 - pressure_term_v23

    return {
        "headline_metrics": {
            "feature_term_v23": feature_term_v23,
            "structure_term_v23": structure_term_v23,
            "learning_term_v23": learning_term_v23,
            "pressure_term_v23": pressure_term_v23,
            "encoding_margin_v23": encoding_margin_v23,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v23 = balanced_feature",
            "structure_term": "K_s_v23 = balanced_structure",
            "learning_term": "K_l_v23 = learning_term_v22 * (1 + balance_gain)",
            "pressure_term": "P_v23 = pressure_term_v22 + residual_gap / structure_term_v22",
            "margin_term": "M_encoding_v23 = K_f_v23 + K_s_v23 + K_l_v23 - P_v23",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第二十三版报告",
        "",
        f"- feature_term_v23: {hm['feature_term_v23']:.6f}",
        f"- structure_term_v23: {hm['structure_term_v23']:.6f}",
        f"- learning_term_v23: {hm['learning_term_v23']:.6f}",
        f"- pressure_term_v23: {hm['pressure_term_v23']:.6f}",
        f"- encoding_margin_v23: {hm['encoding_margin_v23']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v23_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
