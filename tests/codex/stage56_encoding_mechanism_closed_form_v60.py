from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v60_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v60_summary() -> dict:
    v59 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v59_20260321" / "summary.json"
    )
    plasticity = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_boost_20260321" / "summary.json"
    )
    bridge_v4 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v4_20260321" / "summary.json"
    )

    hv = v59["headline_metrics"]
    hp = plasticity["headline_metrics"]
    hb = bridge_v4["headline_metrics"]

    feature_term_v60 = (
        hv["feature_term_v59"]
        + hv["feature_term_v59"] * hp["shared_guard_after_boost"] * 0.003
        + hv["feature_term_v59"] * hp["injected_plasticity_gain"] * 0.002
    )
    structure_term_v60 = (
        hv["structure_term_v59"]
        + hv["structure_term_v59"] * hp["structural_plasticity_balance"] * 0.008
        + hv["structure_term_v59"] * hb["structure_rule_alignment_v4"] * 0.005
    )
    learning_term_v60 = (
        hv["learning_term_v59"]
        + hv["learning_term_v59"] * hb["topology_training_readiness_v4"]
        + hp["plasticity_boost_margin"] * 1000.0
    )
    pressure_term_v60 = max(
        0.0,
        hv["pressure_term_v59"] + hb["topology_training_gap_v4"] + max(0.0, 1.0 - hp["retention_after_boost"]),
    )
    encoding_margin_v60 = feature_term_v60 + structure_term_v60 + learning_term_v60 - pressure_term_v60

    return {
        "headline_metrics": {
            "feature_term_v60": feature_term_v60,
            "structure_term_v60": structure_term_v60,
            "learning_term_v60": learning_term_v60,
            "pressure_term_v60": pressure_term_v60,
            "encoding_margin_v60": encoding_margin_v60,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v60 = K_f_v59 + K_f_v59 * H_boost * 0.003 + K_f_v59 * P_inject * 0.002",
            "structure_term": "K_s_v60 = K_s_v59 + K_s_v59 * S_boost * 0.008 + K_s_v59 * B_struct_v4 * 0.005",
            "learning_term": "K_l_v60 = K_l_v59 + K_l_v59 * R_train_v4 + M_plasticity * 1000",
            "pressure_term": "P_v60 = P_v59 + G_train_v4 + (1 - R_boost)",
            "margin_term": "M_encoding_v60 = K_f_v60 + K_s_v60 + K_l_v60 - P_v60",
        },
        "project_readout": {
            "summary": "第六十版主核把长时间尺度可塑性增强和训练终式第四桥一起并回主式，使主核第一次同时显式容纳结构保持与持续注入新知识两条目标。",
            "next_question": "下一步要继续把塑性增益从中等区推高，否则系统会长期稳定但增量学习能力不足。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第六十版报告",
        "",
        f"- feature_term_v60: {hm['feature_term_v60']:.6f}",
        f"- structure_term_v60: {hm['structure_term_v60']:.6f}",
        f"- learning_term_v60: {hm['learning_term_v60']:.6f}",
        f"- pressure_term_v60: {hm['pressure_term_v60']:.6f}",
        f"- encoding_margin_v60: {hm['encoding_margin_v60']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v60_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
