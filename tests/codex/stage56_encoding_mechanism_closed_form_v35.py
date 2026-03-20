from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v35_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v35_summary() -> dict:
    v34 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v34_20260320" / "summary.json"
    )
    retention_plus = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_retention_reinforcement_20260320" / "summary.json"
    )
    unify_plus = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_icspb_unification_reinforcement_20260320" / "summary.json"
    )

    hv = v34["headline_metrics"]
    hr = retention_plus["headline_metrics"]
    hu = unify_plus["headline_metrics"]

    feature_term_v35 = hv["feature_term_v34"] + hv["feature_term_v34"] * hu["remap_closure_reinforced"] * hr["readout_retention_reinforced"]
    structure_term_v35 = hv["structure_term_v34"] + hv["structure_term_v34"] * hu["object_unification_reinforced"] * hr["update_retention_reinforced"]
    learning_term_v35 = hv["learning_term_v34"] + hr["retention_recovery_margin"] * 1000.0 + hu["unification_gain"] * hv["learning_term_v34"]
    pressure_term_v35 = max(
        0.0,
        hv["pressure_term_v34"]
        - hr["admissible_update_lift"]
        - hr["retention_recovery_margin"]
        - hu["unification_gain"],
    )
    encoding_margin_v35 = feature_term_v35 + structure_term_v35 + learning_term_v35 - pressure_term_v35

    return {
        "headline_metrics": {
            "feature_term_v35": feature_term_v35,
            "structure_term_v35": structure_term_v35,
            "learning_term_v35": learning_term_v35,
            "pressure_term_v35": pressure_term_v35,
            "encoding_margin_v35": encoding_margin_v35,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v35 = K_f_v34 + K_f_v34 * C_unify_plus * R_keep_plus",
            "structure_term": "K_s_v35 = K_s_v34 + K_s_v34 * U_object_plus * U_keep_plus",
            "learning_term": "K_l_v35 = K_l_v34 + Delta_keep + K_l_v34 * Delta_unify",
            "pressure_term": "P_v35 = P_v34 + G_unify_plus + (1 - K_keep_plus) - Delta_update",
            "margin_term": "M_encoding_v35 = K_f_v35 + K_s_v35 + K_l_v35 - P_v35",
        },
        "project_readout": {
            "summary": "第三十五版主核不再只看旧对象能否并回，而是开始同时要求统一收口更强、留核率更高、尤其是 admissible update 在结构层不再轻易脱核。",
            "next_question": "下一步要验证这些强化后的对象，能否跨更多版本长期留在主核里，而不是随一次强化后再次回落。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第三十五版报告",
        "",
        f"- feature_term_v35: {hm['feature_term_v35']:.6f}",
        f"- structure_term_v35: {hm['structure_term_v35']:.6f}",
        f"- learning_term_v35: {hm['learning_term_v35']:.6f}",
        f"- pressure_term_v35: {hm['pressure_term_v35']:.6f}",
        f"- encoding_margin_v35: {hm['encoding_margin_v35']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v35_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
