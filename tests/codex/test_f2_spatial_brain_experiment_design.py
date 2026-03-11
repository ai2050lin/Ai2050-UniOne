from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_json(name: str) -> dict:
    path = TEMP_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def main() -> None:
    p7c = load_json("p7c_brain_spatial_falsification_minimal_core_20260311.json")
    p8a = load_json("p8a_spatialized_plasticity_coding_equation_20260311.json")
    p9c = load_json("p9c_hard_spatial_brain_forecasts_20260311.json")
    p10c = load_json("p10c_final_brain_falsifier_checklist_20260311.json")
    f1 = load_json("f1_architecture_scale_extrapolation_verification_20260311.json")

    local_vs_bridge_separability_score = mean(
        [
            p9c["headline_metrics"]["forecast_specificity_score"],
            p7c["headline_metrics"]["spatial_efficiency_signal_score"],
            p8a["headline_metrics"]["compact_bridge_efficiency_score"],
        ]
    )
    geometry_rejection_readiness_score = mean(
        [
            p7c["headline_metrics"]["geometry_constraint_score"],
            p8a["headline_metrics"]["geometry_only_failure_score"],
            f1["headline_metrics"]["geometry_failure_generality_score"],
        ]
    )
    intervention_targetability_score = mean(
        [
            p9c["headline_metrics"]["risk_targeting_score"],
            p10c["headline_metrics"]["checklist_sharpness_score"],
            p7c["headline_metrics"]["falsifier_sharpness_score"],
        ]
    )
    measurable_mapping_score = mean(
        [
            p8a["headline_metrics"]["spatial_equation_consistency_score"],
            p8a["headline_metrics"]["brain_plausibility_score"],
            p9c["headline_metrics"]["testability_score"],
            f1["headline_metrics"]["architecture_scale_residual_boundary_score"],
        ]
    )
    overall_f2_score = mean(
        [
            local_vs_bridge_separability_score,
            geometry_rejection_readiness_score,
            intervention_targetability_score,
            measurable_mapping_score,
        ]
    )

    experiments = [
        {
            "id": "F2_E1",
            "name": "局部邻域扰动实验",
            "target": "优先扰动局部三维近邻团簇，不切断主要长程桥",
            "readout": [
                "family topology margin",
                "局部特征可分性",
                "共享家族残差变化",
            ],
            "prediction": "如果理论正确，局部扰动应先压低 family topology margin，再逐步传到关系桥接指标，而不是先打掉长程关系桥。",
            "theory_mapping": "对应 q_t 和 f_{t+1} 中的局部差异项 L_t(i) 与局部拥挤成本 C_local。",
        },
        {
            "id": "F2_E2",
            "name": "长程桥切断实验",
            "target": "选择性切断束化稀疏长程桥，而保留局部密集邻域",
            "readout": [
                "compact-boundary relation bridge score",
                "endpoint support",
                "跨区整合成功率",
            ],
            "prediction": "如果理论正确，bridge cut 应优先伤 compact-boundary relation 和跨区整合，而不是先伤局部概念家族结构。",
            "theory_mapping": "对应 A_{t+1}(i,j) 中的需求增长项 d_t(i,j)、空间成本项 D_3d(i,j) 与 E_3d 的桥接通量。",
        },
        {
            "id": "F2_E3",
            "name": "geometry-only 与目标化桥增强对照",
            "target": "比较广泛几何平滑增强与目标化桥增强 + 慢时标稳定修正",
            "readout": [
                "关系桥专属性",
                "信息通量效率 E_3d",
                "泛化与恢复表现",
            ],
            "prediction": "如果理论正确，geometry-only 不应系统性优于目标化桥增强；真正提升应来自少量高价值桥和慢时标稳定项的协同。",
            "theory_mapping": "对应 D_3d(i,j)、m_{t+1}(i,j) 和 E_3d 的联合约束。",
        },
        {
            "id": "F2_E4",
            "name": "快中慢时间尺度干预实验",
            "target": "分别扰动快特征更新、中时标结构更新、慢时标稳定化",
            "readout": [
                "短时特征漂移",
                "有效拓扑重组速度",
                "恢复与抗漂移能力",
            ],
            "prediction": "如果理论正确，快时标干预应先改局部特征，中时标干预应先改有效拓扑，慢时标干预应主要伤恢复和长期保留。",
            "theory_mapping": "对应 f_{t+1}、A_{t+1}、m_{t+1} 三个更新方程的时标分工。",
        },
    ]

    verdict = {
        "status": "experiment_design_ready" if overall_f2_score >= 0.72 else "partial_design_only",
        "best_starting_experiment": "F2_E1_local_neighborhood_perturbation",
        "highest_value_experiment": "F2_E2_long_range_bridge_cut",
        "hardest_open_measurement": "bridge_specificity_strength",
        "can_directly_test_plasticity_to_code_claim": overall_f2_score >= 0.72,
    }

    hypotheses = {
        "H1_local_and_bridge_predictions_are_separable": local_vs_bridge_separability_score >= 0.54,
        "H2_geometry_only_can_be_cleanly_rejected": geometry_rejection_readiness_score >= 0.62,
        "H3_interventions_now_target_real_open_risks": intervention_targetability_score >= 0.84,
        "H4_theory_terms_can_be_mapped_to_measurements": measurable_mapping_score >= 0.76,
        "H5_f2_experiment_design_is_ready": overall_f2_score >= 0.72,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "F2_spatial_brain_experiment_design",
        },
        "headline_metrics": {
            "local_vs_bridge_separability_score": local_vs_bridge_separability_score,
            "geometry_rejection_readiness_score": geometry_rejection_readiness_score,
            "intervention_targetability_score": intervention_targetability_score,
            "measurable_mapping_score": measurable_mapping_score,
            "overall_f2_score": overall_f2_score,
        },
        "candidate_equation_mapping": {
            "spatial_gate": p8a["candidate_mechanism"]["equations"]["spatial_gate"],
            "feature_field": p8a["candidate_mechanism"]["equations"]["feature_field"],
            "structure_field": p8a["candidate_mechanism"]["equations"]["structure_field"],
            "memory_field": p8a["candidate_mechanism"]["equations"]["memory_field"],
            "spatial_efficiency": p8a["candidate_mechanism"]["equations"]["spatial_efficiency"],
        },
        "experiments": experiments,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "f2_spatial_brain_experiment_design_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
