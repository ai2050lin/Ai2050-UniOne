#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage548_topology_field_neuron_algorithm_20260405"
)

STAGE540_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage540_invariant_recheck_20260405"
    / "summary.json"
)
STAGE541_FIELD_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage541_field_control_lever_20260404"
    / "summary.json"
)
STAGE541_BINDING_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage541_binding_invariant_recheck_20260405"
    / "summary.json"
)
STAGE526_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage526_language_band_circuit_dynamics_20260404"
    / "summary.json"
)
STAGE530_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage530_four_model_wordclass_bridge_typology_20260404"
    / "summary.json"
)
STAGE547_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage547_puzzle_synthesis_20260405_012406.json"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()

    stage540 = load_json(STAGE540_PATH)
    stage541_field = load_json(STAGE541_FIELD_PATH)
    stage541_binding = load_json(STAGE541_BINDING_PATH)
    stage526 = load_json(STAGE526_PATH)
    stage530 = load_json(STAGE530_PATH)
    stage547 = load_json(STAGE547_PATH)

    topology_r = float(stage540["corrected_distance_metrics"]["topology_pearson_r"])
    field_count = int(stage541_field["field_count"])
    point_count = int(stage541_field["point_count"])
    efficiency_rho = float(stage541_binding["efficiency_rank_consistency"]["spearman_like_rho"])
    causal_rho = float(stage541_binding["causal_rank_consistency"]["spearman_like_rho"])
    exact_layer_match_rate = float(stage540["layer_non_invariance"]["match_rate"])

    field_concentrations = []
    for model_key in ("qwen3_summary", "deepseek7b_summary"):
        for binding_type in ("attribute", "relation", "grammar", "association"):
            row = stage541_field[model_key][binding_type]
            field_concentrations.append(float(row["avg_topk_concentration"]["topk100"]))
    avg_top100_concentration = sum(field_concentrations) / max(len(field_concentrations), 1)

    model_band_rows = []
    for row in stage526["model_rows"]:
        model_band_rows.append(
            {
                "model_key": row["model_key"],
                "route_band": row["route_band"],
                "cross_task_core_band": row["cross_task_core_band"],
                "early_functions": row["early_functions"],
                "middle_functions": row["middle_functions"],
                "late_functions": row["late_functions"],
            }
        )

    algorithm_steps = [
        {
            "step_id": 1,
            "name": "topology_recovery",
            "name_zh": "拓扑恢复",
            "goal": "先恢复概念或词类之间的稳定相对几何关系，而不是先盯单个神经元。",
            "inputs": ["distance_matrix", "family_labels", "task_labels"],
            "outputs": ["topology_order", "family_backbone_candidates"],
        },
        {
            "step_id": 2,
            "name": "field_estimation",
            "name_zh": "场统计恢复",
            "goal": "比较绑定前后、切换前后、路由前后的隐藏状态统计，判断控制是场式还是点式。",
            "inputs": ["hidden_state_deltas", "entropy", "sparsity", "topk_concentration"],
            "outputs": ["field_axes", "field_vs_point_verdict", "candidate_bands"],
        },
        {
            "step_id": 3,
            "name": "band_localization",
            "name_zh": "层带定位",
            "goal": "寻找拓扑变化最大且场差异最大的宽带功能区，而不是只找单层峰值。",
            "inputs": ["topology_order", "field_axes", "early_middle_late_bands"],
            "outputs": ["route_bands", "binding_bands", "late_readout_bands"],
        },
        {
            "step_id": 4,
            "name": "field_to_neuron_projection",
            "name_zh": "场到神经元投影",
            "goal": "将关键场方向投影回注意力头、神经元、残差方向，得到候选结构部件。",
            "inputs": ["field_axes", "attention_outputs", "mlp_activations", "residual_stream"],
            "outputs": ["backbone_units", "adapter_units", "bridge_units", "switch_units"],
        },
        {
            "step_id": 5,
            "name": "mixed_causal_search",
            "name_zh": "混合因果搜索",
            "goal": "在注意力头、神经元、残差方向的联合空间中搜索最小可解释子回路。",
            "inputs": ["candidate_units", "candidate_heads", "candidate_residual_dirs"],
            "outputs": ["minimal_causal_subcircuits", "control_safe_subcircuits"],
        },
        {
            "step_id": 6,
            "name": "strong_control_validation",
            "name_zh": "强控制复核",
            "goal": "确认打掉的是特异编码结构，而不是整体语言能力。",
            "inputs": ["target_tasks", "control_tasks", "heldout_tasks"],
            "outputs": ["validated_neuron_level_structure"],
        },
    ]

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage548_topology_field_neuron_algorithm",
        "title": "拓扑不变量与场控制杆联合下的神经元级结构算法原型",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": {
            "stage540": str(STAGE540_PATH),
            "stage541_field": str(STAGE541_FIELD_PATH),
            "stage541_binding": str(STAGE541_BINDING_PATH),
            "stage526": str(STAGE526_PATH),
            "stage530": str(STAGE530_PATH),
            "stage547": str(STAGE547_PATH),
        },
        "invariant_assessment": {
            "binding_field_control_lever": {
                "credible": True,
                "field_count": field_count,
                "point_count": point_count,
                "avg_top100_concentration": avg_top100_concentration,
                "reason": "绑定差异主要以场式分布承载，而不是由少数点状神经元单独承载。",
            },
            "topology_order": {
                "credible": True,
                "pearson_r": topology_r,
                "reason": "跨家族相对拓扑排序高度一致，说明稳定的相对几何关系存在。",
            },
            "binding_efficiency_ranking": {
                "credible": True,
                "rho": efficiency_rho,
            },
            "binding_causal_ranking": {
                "credible": False,
                "rho": causal_rho,
                "reason": "最强因果层在不同模型中仍不稳定，不能当成强不变量。",
            },
            "exact_layer_position": {
                "credible": False,
                "match_rate": exact_layer_match_rate,
                "reason": "精确层号不是不变量，只能使用早中晚层带口径。",
            },
        },
        "model_band_rows": model_band_rows,
        "wordclass_bridge_snapshot": stage530["core_answer"],
        "puzzle_snapshot": stage547["puzzle_summary"],
        "algorithm_steps": algorithm_steps,
        "core_answer": (
            "当前最可信的组合不是“找前几个最高分神经元”，而是“先恢复拓扑，再恢复绑定场，再把场投影回神经元、注意力头和残差方向”。"
            "也就是说，神经元级编码结构应被看成‘拓扑骨架 + 场式控制 + 小型尖锐控制杆’的联合系统。"
        ),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# stage548 拓扑不变量与场控制杆联合算法原型",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
        "## 不变量可信度",
        f"- 绑定场控制杆：可信，FIELD/POINT = {field_count}/{point_count}，平均 top100 集中度 = {avg_top100_concentration:.6f}",
        f"- 拓扑排序：可信，Pearson r = {topology_r:.6f}",
        f"- 绑定效率排序：可信，rho = {efficiency_rho:.6f}",
        f"- 绑定因果排序：暂不够强，rho = {causal_rho:.6f}",
        f"- 精确层号：不可信，match_rate = {exact_layer_match_rate:.6f}",
        "",
        "## 算法步骤",
    ]
    for step in algorithm_steps:
        report_lines.append(f"{step['step_id']}. {step['name_zh']}：{step['goal']}")
    report_lines.extend(
        [
            "",
            "## 统一理解",
            "- 拓扑不变量负责告诉我们哪些编码彼此更近。",
            "- 场控制杆不变量负责告诉我们绑定控制不是少数点状神经元单独承载。",
            "- 层带图谱负责告诉我们搜索应优先落在哪些宽带功能区。",
            "- 最小因果回路负责告诉我们真正可打中的控制杆组合是什么。",
        ]
    )
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
