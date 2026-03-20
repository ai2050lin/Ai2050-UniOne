from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v36_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v36_summary() -> dict:
    v35 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v35_20260320" / "summary.json"
    )
    stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_stability_strengthening_20260320" / "summary.json"
    )
    high = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_icspb_unification_high_closure_20260320" / "summary.json"
    )

    hv = v35["headline_metrics"]
    hs = stable["headline_metrics"]
    hh = high["headline_metrics"]

    feature_term_v36 = hv["feature_term_v35"] + hv["feature_term_v35"] * hh["remap_closure_high"] * hs["readout_retention_stable"]
    structure_term_v36 = hv["structure_term_v35"] + hv["structure_term_v35"] * hh["object_unification_high"] * hs["update_retention_stable"]
    learning_term_v36 = hv["learning_term_v35"] + hs["stability_lift"] * 1200.0 + hh["high_closure_gain"] * hv["learning_term_v35"]
    pressure_term_v36 = max(
        0.0,
        hv["pressure_term_v35"]
        - hs["stability_lift"]
        - hh["high_closure_gain"]
        - 0.5 * hs["channel_compaction"],
    )
    encoding_margin_v36 = feature_term_v36 + structure_term_v36 + learning_term_v36 - pressure_term_v36

    return {
        "headline_metrics": {
            "feature_term_v36": feature_term_v36,
            "structure_term_v36": structure_term_v36,
            "learning_term_v36": learning_term_v36,
            "pressure_term_v36": pressure_term_v36,
            "encoding_margin_v36": encoding_margin_v36,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v36 = K_f_v35 + K_f_v35 * C_unify_high * R_keep_star",
            "structure_term": "K_s_v36 = K_s_v35 + K_s_v35 * U_object_high * U_keep_star",
            "learning_term": "K_l_v36 = K_l_v35 + Delta_keep_star + K_l_v35 * Delta_high",
            "pressure_term": "P_v36 = P_v35 - Delta_stability - Delta_high - 0.5 * C_compact",
            "margin_term": "M_encoding_v36 = K_f_v36 + K_s_v36 + K_l_v36 - P_v36",
        },
        "project_readout": {
            "summary": "第三十六版主核开始把高闭合统一核和高留核稳定核同时并进主式，目标不再只是中等统一，而是让弱通道和主闭合核一起抬高。",
            "next_question": "下一步要验证 v36 的高闭合和高留核是否能跨版本保持，而不是在 v35 到 v36 之间局部跃升后再次回落。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第三十六版报告",
        "",
        f"- feature_term_v36: {hm['feature_term_v36']:.6f}",
        f"- structure_term_v36: {hm['structure_term_v36']:.6f}",
        f"- learning_term_v36: {hm['learning_term_v36']:.6f}",
        f"- pressure_term_v36: {hm['pressure_term_v36']:.6f}",
        f"- encoding_margin_v36: {hm['encoding_margin_v36']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v36_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
