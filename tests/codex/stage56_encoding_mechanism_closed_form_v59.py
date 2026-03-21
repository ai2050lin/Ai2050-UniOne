from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v59_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v59_summary() -> dict:
    v58 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v58_20260321" / "summary.json"
    )
    topo_long = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_prototype_20260321" / "summary.json"
    )
    bridge_v3 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v3_20260321" / "summary.json"
    )

    hv = v58["headline_metrics"]
    ht = topo_long["headline_metrics"]
    hb = bridge_v3["headline_metrics"]

    feature_term_v59 = hv["feature_term_v58"] + hv["feature_term_v58"] * ht["topo_long_shared_survival"] * 0.003
    structure_term_v59 = (
        hv["structure_term_v58"]
        + hv["structure_term_v58"] * ht["topo_long_structural_survival"] * 0.01
        + hv["structure_term_v58"] * hb["structure_guard_strength_v3"] * 0.005
    )
    learning_term_v59 = (
        hv["learning_term_v58"]
        + hv["learning_term_v58"] * hb["topology_bridge_readiness_v3"]
        + ht["topo_long_margin"] * 1000.0
    )
    pressure_term_v59 = max(
        0.0,
        hv["pressure_term_v58"] + hb["topology_bridge_gap_v3"] + max(0.0, 1.0 - ht["topo_long_structural_survival"]),
    )
    encoding_margin_v59 = feature_term_v59 + structure_term_v59 + learning_term_v59 - pressure_term_v59

    return {
        "headline_metrics": {
            "feature_term_v59": feature_term_v59,
            "structure_term_v59": structure_term_v59,
            "learning_term_v59": learning_term_v59,
            "pressure_term_v59": pressure_term_v59,
            "encoding_margin_v59": encoding_margin_v59,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v59 = K_f_v58 + K_f_v58 * H_topo_long * 0.003",
            "structure_term": "K_s_v59 = K_s_v58 + K_s_v58 * S_topo_long * 0.01 + K_s_v58 * B_guard_v3 * 0.005",
            "learning_term": "K_l_v59 = K_l_v58 + K_l_v58 * R_train_v3 + M_topo_long * 1000",
            "pressure_term": "P_v59 = P_v58 + G_train_v3 + (1 - S_topo_long)",
            "margin_term": "M_encoding_v59 = K_f_v59 + K_s_v59 + K_l_v59 - P_v59",
        },
        "project_readout": {
            "summary": "第五十九版主核把长时间尺度三维脉冲原型和训练终式第三桥并回主式，开始更直接反映规模化在线更新下的结构保持能力。",
            "next_question": "下一步要继续抬高结构生存率，否则训练链虽然能工作，但会在规模化动态更新下长期偏脆。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第五十九版报告",
        "",
        f"- feature_term_v59: {hm['feature_term_v59']:.6f}",
        f"- structure_term_v59: {hm['structure_term_v59']:.6f}",
        f"- learning_term_v59: {hm['learning_term_v59']:.6f}",
        f"- pressure_term_v59: {hm['pressure_term_v59']:.6f}",
        f"- encoding_margin_v59: {hm['encoding_margin_v59']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v59_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
