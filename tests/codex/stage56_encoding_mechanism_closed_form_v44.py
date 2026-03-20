from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v44_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v44_summary() -> dict:
    v43 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v43_20260320" / "summary.json"
    )
    train = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_rule_20260320" / "summary.json"
    )
    proto = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_prototype_network_readiness_20260320" / "summary.json"
    )
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )

    hv = v43["headline_metrics"]
    ht = train["headline_metrics"]
    hp = proto["headline_metrics"]
    hc = cross["headline_metrics"]

    feature_term_v44 = hv["feature_term_v43"] + hv["feature_term_v43"] * hp["language_stack_readiness"] * 0.02
    structure_term_v44 = hv["structure_term_v43"] + hv["structure_term_v43"] * ht["terminal_update_strength"] * 0.02
    learning_term_v44 = (
        hv["learning_term_v43"]
        + hv["learning_term_v43"] * hp["prototype_network_readiness"]
        + ht["training_terminal_readiness"] * 1000.0
    )
    pressure_term_v44 = max(
        0.0,
        hv["pressure_term_v43"]
        + hp["agi_delivery_gap"]
        + ht["terminal_training_gap"]
        - hc["stability_gain"],
    )
    encoding_margin_v44 = feature_term_v44 + structure_term_v44 + learning_term_v44 - pressure_term_v44

    return {
        "headline_metrics": {
            "feature_term_v44": feature_term_v44,
            "structure_term_v44": structure_term_v44,
            "learning_term_v44": learning_term_v44,
            "pressure_term_v44": pressure_term_v44,
            "encoding_margin_v44": encoding_margin_v44,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v44 = K_f_v43 + K_f_v43 * R_lang_stack * 0.02",
            "structure_term": "K_s_v44 = K_s_v43 + K_s_v43 * U_term * 0.02",
            "learning_term": "K_l_v44 = K_l_v43 + K_l_v43 * R_proto + R_train * 1000",
            "pressure_term": "P_v44 = P_v43 + G_agi + G_train - Delta_stability_star",
            "margin_term": "M_encoding_v44 = K_f_v44 + K_s_v44 + K_l_v44 - P_v44",
        },
        "project_readout": {
            "summary": "第四十四版主核把训练终式和原型网络就绪度一起并回主式，开始让主核更接近可施工的网络设计判断。",
            "next_question": "下一步要把这组对象推进到真实小型原型网络里，确认主式不只是在理论上可写，而是真的能指导训练。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第四十四版报告",
        "",
        f"- feature_term_v44: {hm['feature_term_v44']:.6f}",
        f"- structure_term_v44: {hm['structure_term_v44']:.6f}",
        f"- learning_term_v44: {hm['learning_term_v44']:.6f}",
        f"- pressure_term_v44: {hm['pressure_term_v44']:.6f}",
        f"- encoding_margin_v44: {hm['encoding_margin_v44']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v44_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
