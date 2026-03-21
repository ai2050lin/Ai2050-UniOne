from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v77_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v77_summary() -> dict:
    v76 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v76_20260321" / "summary.json"
    )
    persistence = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_anti_attenuation_persistence_20260321" / "summary.json"
    )
    brain_v15 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v15_20260321" / "summary.json"
    )
    bridge_v21 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v21_20260321" / "summary.json"
    )

    hv = v76["headline_metrics"]
    hp = persistence["headline_metrics"]
    hb = brain_v15["headline_metrics"]
    ht = bridge_v21["headline_metrics"]

    feature_term_v77 = (
        hv["feature_term_v76"]
        + hv["feature_term_v76"] * hp["persistence_score"] * 0.004
        + hv["feature_term_v76"] * ht["plasticity_rule_alignment_v21"] * 0.001
        + hv["feature_term_v76"] * hb["direct_feature_measure_v15"] * 0.001
    )
    structure_term_v77 = (
        hv["structure_term_v76"]
        + hv["structure_term_v76"] * hp["persistence_structure"] * 0.007
        + hv["structure_term_v76"] * ht["structure_rule_alignment_v21"] * 0.004
        + hv["structure_term_v76"] * hb["direct_structure_measure_v15"] * 0.002
    )
    learning_term_v77 = (
        hv["learning_term_v76"]
        + hv["learning_term_v76"] * ht["topology_training_readiness_v21"]
        + hp["persistence_margin"] * 1000.0
        + hp["persistence_score"] * 1000.0
        + hb["direct_brain_measure_v15"] * 1000.0
    )
    pressure_term_v77 = max(
        0.0,
        hv["pressure_term_v76"]
        + ht["topology_training_gap_v21"]
        + hp["persistence_penalty"]
        + (1.0 - hp["persistence_route"]) * 0.2,
    )
    encoding_margin_v77 = feature_term_v77 + structure_term_v77 + learning_term_v77 - pressure_term_v77

    return {
        "headline_metrics": {
            "feature_term_v77": feature_term_v77,
            "structure_term_v77": structure_term_v77,
            "learning_term_v77": learning_term_v77,
            "pressure_term_v77": pressure_term_v77,
            "encoding_margin_v77": encoding_margin_v77,
        },
        "closed_form_equation_v77": {
            "feature_term": "K_f_v77 = K_f_v76 + K_f_v76 * S_persist_score * 0.004 + K_f_v76 * B_plastic_v21 * 0.001 + K_f_v76 * D_feature_v15 * 0.001",
            "structure_term": "K_s_v77 = K_s_v76 + K_s_v76 * S_persist * 0.007 + K_s_v76 * B_struct_v21 * 0.004 + K_s_v76 * D_structure_v15 * 0.002",
            "learning_term": "K_l_v77 = K_l_v76 + K_l_v76 * R_train_v21 + M_persist * 1000 + S_persist_score * 1000 + M_brain_direct_v15 * 1000",
            "pressure_term": "P_v77 = P_v76 + G_train_v21 + P_persist + 0.2 * (1 - R_persist)",
            "margin_term": "M_encoding_v77 = K_f_v77 + K_s_v77 + K_l_v77 - P_v77",
        },
        "project_readout": {
            "summary": "第七十七版主核开始把反衰减持续性、脑编码第十五版直测链和训练终式第二十一桥一起并回主核，使主核开始直接回答补偿式回升能否持续站住。",
            "next_question": "下一步要把这条主核推进到更大的可训练系统里，检验持续回升是否会稳定成真正的系统级突破。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第七十七版报告",
        "",
        f"- feature_term_v77: {hm['feature_term_v77']:.6f}",
        f"- structure_term_v77: {hm['structure_term_v77']:.6f}",
        f"- learning_term_v77: {hm['learning_term_v77']:.6f}",
        f"- pressure_term_v77: {hm['pressure_term_v77']:.6f}",
        f"- encoding_margin_v77: {hm['encoding_margin_v77']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v77_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
