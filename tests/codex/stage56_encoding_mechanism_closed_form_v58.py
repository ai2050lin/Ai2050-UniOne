from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v58_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v58_summary() -> dict:
    v57 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v57_20260321" / "summary.json"
    )
    topo_proto = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_trainable_prototype_20260321" / "summary.json"
    )
    train_bridge = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v2_20260321" / "summary.json"
    )

    hv = v57["headline_metrics"]
    hp = topo_proto["headline_metrics"]
    hb = train_bridge["headline_metrics"]

    feature_term_v58 = (
        hv["feature_term_v57"]
        + hv["feature_term_v57"] * hp["path_reuse_score"] * 0.005
        + hv["feature_term_v57"] * hp["local_transport_score"] * 0.005
    )
    structure_term_v58 = (
        hv["structure_term_v57"]
        + hv["structure_term_v57"] * hp["structural_persistence"] * 0.01
        + hv["structure_term_v57"] * hb["topology_rule_alignment"] * 0.005
    )
    learning_term_v58 = (
        hv["learning_term_v57"]
        + hv["learning_term_v57"] * hb["terminal_bridge_readiness"]
        + hp["topology_trainable_margin"] * 1000.0
    )
    pressure_term_v58 = max(
        0.0,
        hv["pressure_term_v57"] + hb["terminal_bridge_gap"] + max(0.0, 1.0 - hp["structural_persistence"]),
    )
    encoding_margin_v58 = feature_term_v58 + structure_term_v58 + learning_term_v58 - pressure_term_v58

    return {
        "headline_metrics": {
            "feature_term_v58": feature_term_v58,
            "structure_term_v58": structure_term_v58,
            "learning_term_v58": learning_term_v58,
            "pressure_term_v58": pressure_term_v58,
            "encoding_margin_v58": encoding_margin_v58,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v58 = K_f_v57 + K_f_v57 * R_topo * 0.005 + K_f_v57 * T_topo * 0.005",
            "structure_term": "K_s_v58 = K_s_v57 + K_s_v57 * S_topo * 0.01 + K_s_v57 * B_topo * 0.005",
            "learning_term": "K_l_v58 = K_l_v57 + K_l_v57 * R_train_v2 + M_topo_proto * 1000",
            "pressure_term": "P_v58 = P_v57 + G_train_v2 + (1 - S_topo)",
            "margin_term": "M_encoding_v58 = K_f_v58 + K_s_v58 + K_l_v58 - P_v58",
        },
        "project_readout": {
            "summary": "第五十八版主核把三维拓扑可训练原型和训练终式第二桥一起并回主式，开始从框架总整理推进到更接近施工的训练链。",
            "next_question": "下一步最关键的是把这个训练链推进到更长时间尺度即时学习里，看结构保持是否会在规模化更新下塌缩。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第五十八版报告",
        "",
        f"- feature_term_v58: {hm['feature_term_v58']:.6f}",
        f"- structure_term_v58: {hm['structure_term_v58']:.6f}",
        f"- learning_term_v58: {hm['learning_term_v58']:.6f}",
        f"- pressure_term_v58: {hm['pressure_term_v58']:.6f}",
        f"- encoding_margin_v58: {hm['encoding_margin_v58']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v58_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
