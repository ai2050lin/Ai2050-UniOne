from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v34_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v34_summary() -> dict:
    v33 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v33_20260320" / "summary.json"
    )
    unify = _load_json(ROOT / "tests" / "codex_temp" / "stage56_icspb_unification_closure_20260320" / "summary.json")
    retention = _load_json(ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_retention_20260320" / "summary.json")

    hv = v33["headline_metrics"]
    hu = unify["headline_metrics"]
    hr = retention["headline_metrics"]

    feature_term_v34 = hv["feature_term_v33"] + hv["feature_term_v33"] * hu["remap_closure_core"] * hr["readout_retention"]
    structure_term_v34 = hv["structure_term_v33"] + hv["structure_term_v33"] * hu["object_unification_strength"] * hr["update_retention"]
    learning_term_v34 = hv["learning_term_v33"] + hr["retention_margin"] * hu["transport_unification_strength"]
    pressure_term_v34 = hv["pressure_term_v33"] + hu["support_gap_reduced"] + (1.0 - hr["transport_kernel_stability"])
    encoding_margin_v34 = feature_term_v34 + structure_term_v34 + learning_term_v34 - pressure_term_v34

    return {
        "headline_metrics": {
            "feature_term_v34": feature_term_v34,
            "structure_term_v34": structure_term_v34,
            "learning_term_v34": learning_term_v34,
            "pressure_term_v34": pressure_term_v34,
            "encoding_margin_v34": encoding_margin_v34,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v34 = K_f_v33 + K_f_v33 * C_unify * R_keep",
            "structure_term": "K_s_v34 = K_s_v33 + K_s_v33 * U_object * U_keep",
            "learning_term": "K_l_v34 = K_l_v33 + M_keep * U_transport",
            "pressure_term": "P_v34 = P_v33 + G_unify + (1 - K_keep)",
            "margin_term": "M_encoding_v34 = K_f_v34 + K_s_v34 + K_l_v34 - P_v34",
        },
        "project_readout": {
            "summary": "第 34 版不再只是把旧对象并回主核，而是开始考察它们能否在统一收口和留核稳定性两条线上同时成立。也就是说，v34 更像“统一主核的稳定版”，不是简单叠加版。",
            "next_question": "下一步要进一步压低 unified gap，并把留核稳定性从单次版本提升到跨版本稳定性。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第三十四版报告",
        "",
        f"- feature_term_v34: {hm['feature_term_v34']:.6f}",
        f"- structure_term_v34: {hm['structure_term_v34']:.6f}",
        f"- learning_term_v34: {hm['learning_term_v34']:.6f}",
        f"- pressure_term_v34: {hm['pressure_term_v34']:.6f}",
        f"- encoding_margin_v34: {hm['encoding_margin_v34']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v34_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
