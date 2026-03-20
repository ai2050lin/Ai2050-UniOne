from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v24_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v24_summary() -> dict:
    v23 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v23_20260320" / "summary.json"
    )
    native_balance = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_native_balance_20260320" / "summary.json"
    )

    hv = v23["headline_metrics"]
    hb = native_balance["headline_metrics"]

    feature_term_v24 = hb["native_balanced_feature_v2"]
    structure_term_v24 = hb["native_balanced_structure_v2"]
    learning_term_v24 = hv["learning_term_v23"] * (1.0 + hb["native_balance_ratio_v2"])
    pressure_term_v24 = hv["pressure_term_v23"] + hb["native_balance_gap_v2"] / max(hv["structure_term_v23"], 1e-9)
    encoding_margin_v24 = feature_term_v24 + structure_term_v24 + learning_term_v24 - pressure_term_v24

    return {
        "headline_metrics": {
            "feature_term_v24": feature_term_v24,
            "structure_term_v24": structure_term_v24,
            "learning_term_v24": learning_term_v24,
            "pressure_term_v24": pressure_term_v24,
            "encoding_margin_v24": encoding_margin_v24,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v24 = native_balanced_feature_v2",
            "structure_term": "K_s_v24 = native_balanced_structure_v2",
            "learning_term": "K_l_v24 = learning_term_v23 * (1 + native_balance_ratio_v2)",
            "pressure_term": "P_v24 = pressure_term_v23 + native_balance_gap_v2 / structure_term_v23",
            "margin_term": "M_encoding_v24 = K_f_v24 + K_s_v24 + K_l_v24 - P_v24",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第二十四版报告",
        "",
        f"- feature_term_v24: {hm['feature_term_v24']:.6f}",
        f"- structure_term_v24: {hm['structure_term_v24']:.6f}",
        f"- learning_term_v24: {hm['learning_term_v24']:.6f}",
        f"- pressure_term_v24: {hm['pressure_term_v24']:.6f}",
        f"- encoding_margin_v24: {hm['encoding_margin_v24']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v24_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
