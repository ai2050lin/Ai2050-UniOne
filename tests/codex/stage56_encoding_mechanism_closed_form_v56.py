from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v56_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v56_summary() -> dict:
    v55 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v55_20260321" / "summary.json"
    )
    topo = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_topology_efficiency_20260321" / "summary.json"
    )

    hv = v55["headline_metrics"]
    ht = topo["headline_metrics"]

    feature_term_v56 = hv["feature_term_v55"] + hv["feature_term_v55"] * ht["path_superposition_capacity"] * 0.01
    structure_term_v56 = hv["structure_term_v55"] + hv["structure_term_v55"] * ht["topology_grid_efficiency"] * 0.01
    learning_term_v56 = hv["learning_term_v55"] + hv["learning_term_v55"] * ht["online_stability_coupling"] + ht["topology_encoding_margin"] * 1000.0
    pressure_term_v56 = max(
        0.0,
        hv["pressure_term_v55"]
        + (1.0 - ht["minimal_transport_efficiency"])
        + (1.0 - ht["global_steady_coupling"]),
    )
    encoding_margin_v56 = feature_term_v56 + structure_term_v56 + learning_term_v56 - pressure_term_v56

    return {
        "headline_metrics": {
            "feature_term_v56": feature_term_v56,
            "structure_term_v56": structure_term_v56,
            "learning_term_v56": learning_term_v56,
            "pressure_term_v56": pressure_term_v56,
            "encoding_margin_v56": encoding_margin_v56,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v56 = K_f_v55 + K_f_v55 * P_super * 0.01",
            "structure_term": "K_s_v56 = K_s_v55 + K_s_v55 * G_3d * 0.01",
            "learning_term": "K_l_v56 = K_l_v55 + K_l_v55 * C_online + M_topology * 1000",
            "pressure_term": "P_v56 = P_v55 + (1 - T_min) + (1 - C_global)",
            "margin_term": "M_encoding_v56 = K_f_v56 + K_s_v56 + K_l_v56 - P_v56",
        },
        "project_readout": {
            "summary": "第五十六版主核把最小传送量原理、三维拓扑网格效率和路径叠加编码一起并回主式，开始表达为什么脉冲网络有机会同时兼顾即时学习与全局稳态。",
            "next_question": "下一步要把这套三维拓扑主线落进可训练脉冲原型，否则它仍然只是高层结构解释。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第五十六版报告",
        "",
        f"- feature_term_v56: {hm['feature_term_v56']:.6f}",
        f"- structure_term_v56: {hm['structure_term_v56']:.6f}",
        f"- learning_term_v56: {hm['learning_term_v56']:.6f}",
        f"- pressure_term_v56: {hm['pressure_term_v56']:.6f}",
        f"- encoding_margin_v56: {hm['encoding_margin_v56']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v56_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
