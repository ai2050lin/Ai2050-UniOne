from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v38_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v38_summary() -> dict:
    v37 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v37_20260320" / "summary.json"
    )
    stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )
    keep = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_high_retention_cross_version_validation_20260320" / "summary.json"
    )

    hv = v37["headline_metrics"]
    hs = stable["headline_metrics"]
    hk = keep["headline_metrics"]

    feature_term_v38 = hv["feature_term_v37"] + hv["feature_term_v37"] * hs["feature_growth_stable"]
    structure_term_v38 = hv["structure_term_v37"] + hv["structure_term_v37"] * hs["structure_growth_stable"]
    learning_term_v38 = hv["learning_term_v37"] + hv["learning_term_v37"] * hs["cross_version_stability_stable"] + hk["cross_keep_core"] * 1000.0
    pressure_term_v38 = max(0.0, hv["pressure_term_v37"] - hs["stability_gain"] - hk["cross_keep_floor"] + hk["cross_keep_margin"])
    encoding_margin_v38 = feature_term_v38 + structure_term_v38 + learning_term_v38 - pressure_term_v38

    return {
        "headline_metrics": {
            "feature_term_v38": feature_term_v38,
            "structure_term_v38": structure_term_v38,
            "learning_term_v38": learning_term_v38,
            "pressure_term_v38": pressure_term_v38,
            "encoding_margin_v38": encoding_margin_v38,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v38 = K_f_v37 + K_f_v37 * G_f_star",
            "structure_term": "K_s_v38 = K_s_v37 + K_s_v37 * G_s_star",
            "learning_term": "K_l_v38 = K_l_v37 + K_l_v37 * S_cross_star + K_cross * 1000",
            "pressure_term": "P_v38 = P_v37 - Delta_cross + C_margin",
            "margin_term": "M_encoding_v38 = K_f_v38 + K_s_v38 + K_l_v38 - P_v38",
        },
        "project_readout": {
            "summary": "第三十八版主核开始把跨版本稳定强化和高留核跨版本验证一起并回主式，目标是让主核不仅更强，而且更能跨版本保持。",
            "next_question": "下一步要继续检验 v38 是否能把 rollback risk 压到更低，并把跨版本稳定推进到更高区间。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第三十八版报告",
        "",
        f"- feature_term_v38: {hm['feature_term_v38']:.6f}",
        f"- structure_term_v38: {hm['structure_term_v38']:.6f}",
        f"- learning_term_v38: {hm['learning_term_v38']:.6f}",
        f"- pressure_term_v38: {hm['pressure_term_v38']:.6f}",
        f"- encoding_margin_v38: {hm['encoding_margin_v38']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v38_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
