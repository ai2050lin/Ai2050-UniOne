from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_project_framework_synthesis_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_project_framework_synthesis_summary() -> dict:
    language = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_total_analysis_20260320" / "summary.json"
    )
    lang_system = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_system_principles_20260320" / "summary.json"
    )
    brain = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v3_20260321" / "summary.json"
    )
    topo_mech = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_3d_topology_encoding_mechanism_20260321" / "summary.json"
    )
    topo_scale = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_3d_topology_scaling_analysis_20260321" / "summary.json"
    )
    horizon = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_long_horizon_stability_20260321" / "summary.json"
    )
    train = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_rule_20260320" / "summary.json"
    )

    hl = language["headline_metrics"]
    hs = lang_system["headline_metrics"]
    hb = brain["headline_metrics"]
    hm = topo_mech["headline_metrics"]
    hz = topo_scale["headline_metrics"]
    hh = horizon["headline_metrics"]
    ht = train["headline_metrics"]

    language_anchor = _clip01((hl["language_principle_completion"] + hl["language_feature_resolution"] + hs["language_entry_core"]) / 3.0)
    language_to_brain_bridge = _clip01((hl["language_structure_resolution"] + hb["direct_feature_measure_v3"] + hb["direct_route_measure_v3"]) / 3.0)
    brain_to_topology_bridge = _clip01((hb["direct_brain_measure_v3"] + hm["three_d_encoding_margin"] / 5.0 + hz["scale_transport_retention"]) / 3.0)
    topology_to_training_bridge = _clip01((hz["scale_ready_score"] + hh["long_horizon_retention"] + ht["training_terminal_readiness"]) / 3.0)
    framework_synthesis_margin = (
        language_anchor
        + language_to_brain_bridge
        + brain_to_topology_bridge
        + topology_to_training_bridge
    )
    critical_bottleneck = max(
        1.0 - language_anchor,
        1.0 - language_to_brain_bridge,
        1.0 - brain_to_topology_bridge,
        1.0 - topology_to_training_bridge,
    )

    return {
        "headline_metrics": {
            "language_anchor": language_anchor,
            "language_to_brain_bridge": language_to_brain_bridge,
            "brain_to_topology_bridge": brain_to_topology_bridge,
            "topology_to_training_bridge": topology_to_training_bridge,
            "framework_synthesis_margin": framework_synthesis_margin,
            "critical_bottleneck": critical_bottleneck,
        },
        "framework_equation": {
            "language_term": "F_lang_anchor = mean(C_lang_principle, R_lang_feature, E_lang_entry)",
            "bridge_term_1": "B_lang_brain = mean(R_lang_structure, D_feature_v3, D_route_v3)",
            "bridge_term_2": "B_brain_topology = mean(D_brain_v3, M_3d_encode / 5, S_trans)",
            "bridge_term_3": "B_topology_train = mean(M_scale, R_h_long, R_train_terminal)",
            "system_term": "M_framework = F_lang_anchor + B_lang_brain + B_brain_topology + B_topology_train",
        },
        "project_readout": {
            "summary": "当前最合理的项目框架已经从语言入口、脑编码逆向分析、三维拓扑编码机制，一路连到训练终式。真正要通过语言系统破解脑编码机制，关键不是继续只补语言细节，而是把语言骨架持续翻译成脑编码链、拓扑链和训练链。",
            "next_question": "下一步最该做的是把三维拓扑和动态学习一起放进可训练脉冲原型，验证这条框架链是不是不仅能解释，而且能施工。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 项目框架总整理报告",
        "",
        f"- language_anchor: {hm['language_anchor']:.6f}",
        f"- language_to_brain_bridge: {hm['language_to_brain_bridge']:.6f}",
        f"- brain_to_topology_bridge: {hm['brain_to_topology_bridge']:.6f}",
        f"- topology_to_training_bridge: {hm['topology_to_training_bridge']:.6f}",
        f"- framework_synthesis_margin: {hm['framework_synthesis_margin']:.6f}",
        f"- critical_bottleneck: {hm['critical_bottleneck']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_project_framework_synthesis_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
