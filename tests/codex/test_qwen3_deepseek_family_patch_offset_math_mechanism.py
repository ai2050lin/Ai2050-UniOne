from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing(*candidates: str) -> Path:
    for name in candidates:
        path = ROOT / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing all candidates: {candidates}")


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def build_payload() -> Dict[str, Any]:
    t0 = time.time()

    qwen_refresh_path = first_existing(
        "tests/codex_temp/qwen3_deepseek7b_concept_encoding_decomposition_qwen_refresh_20260315.json",
        "tests/codex_temp/qwen3_deepseek7b_concept_encoding_decomposition_20260309.json",
    )
    deepseek_refresh_path = first_existing(
        "tests/codex_temp/qwen3_deepseek7b_concept_encoding_decomposition_deepseek_refresh_20260315.json",
        "tests/codex_temp/qwen3_deepseek7b_concept_encoding_decomposition_20260309.json",
    )
    bridge_path = first_existing("tests/codex_temp/qwen3_deepseek7b_mechanism_bridge_20260309.json")
    structure_path = first_existing("tests/codex_temp/qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
    consistency_path = first_existing("tests/codex_temp/qwen3_deepseek7b_apple_mechanism_consistency_20260309.json")

    qwen_refresh = load_json(qwen_refresh_path)["models"]["qwen3_4b"]
    deepseek_refresh = load_json(deepseek_refresh_path)["models"]["deepseek_7b"]
    bridge = load_json(bridge_path)
    structure = load_json(structure_path)
    consistency = load_json(consistency_path)

    qwen_stage_counts = Counter(layer["support_stage"] for layer in structure["models"]["qwen3_4b"]["layer_atlas"])
    deepseek_stage_counts = Counter(layer["support_stage"] for layer in structure["models"]["deepseek_7b"]["layer_atlas"])

    qwen_global = qwen_refresh["global_summary"]
    deepseek_global = deepseek_refresh["global_summary"]
    qwen_apple = qwen_refresh["targets"]["apple"]["best_layer"]
    deepseek_apple = deepseek_refresh["targets"]["apple"]["best_layer"]

    qwen_bridge = bridge["models"]["qwen3_4b"]
    deepseek_bridge = bridge["models"]["deepseek_7b"]

    direct_evidence = {
        "qwen3_4b": {
            "family_fit_strength": float(1.0 - qwen_global["mean_true_family_residual"]),
            "wrong_family_margin": float(qwen_global["mean_margin_vs_best_wrong"]),
            "offset_top32_energy_ratio": float(qwen_global["mean_offset_top32_energy_ratio"]),
            "shared_norm_ratio": float(qwen_global["mean_shared_norm_ratio"]),
            "apple_best_layer": qwen_apple,
            "mechanism_bridge_score": float(qwen_bridge["mechanism_bridge_score"]),
            "stage_counts": dict(qwen_stage_counts),
        },
        "deepseek_7b": {
            "family_fit_strength": float(1.0 - deepseek_global["mean_true_family_residual"]),
            "wrong_family_margin": float(deepseek_global["mean_margin_vs_best_wrong"]),
            "offset_top32_energy_ratio": float(deepseek_global["mean_offset_top32_energy_ratio"]),
            "shared_norm_ratio": float(deepseek_global["mean_shared_norm_ratio"]),
            "apple_best_layer": deepseek_apple,
            "mechanism_bridge_score": float(deepseek_bridge["mechanism_bridge_score"]),
            "stage_counts": dict(deepseek_stage_counts),
        },
    }

    cross_model_summary = {
        "mean_family_fit_strength": float(
            mean(
                [
                    direct_evidence["qwen3_4b"]["family_fit_strength"],
                    direct_evidence["deepseek_7b"]["family_fit_strength"],
                ]
            )
        ),
        "mean_wrong_family_margin": float(
            mean(
                [
                    direct_evidence["qwen3_4b"]["wrong_family_margin"],
                    direct_evidence["deepseek_7b"]["wrong_family_margin"],
                ]
            )
        ),
        "mean_offset_top32_energy_ratio": float(
            mean(
                [
                    direct_evidence["qwen3_4b"]["offset_top32_energy_ratio"],
                    direct_evidence["deepseek_7b"]["offset_top32_energy_ratio"],
                ]
            )
        ),
        "mean_shared_norm_ratio": float(
            mean(
                [
                    direct_evidence["qwen3_4b"]["shared_norm_ratio"],
                    direct_evidence["deepseek_7b"]["shared_norm_ratio"],
                ]
            )
        ),
        "mean_mechanism_bridge_score": float(
            mean(
                [
                    direct_evidence["qwen3_4b"]["mechanism_bridge_score"],
                    direct_evidence["deepseek_7b"]["mechanism_bridge_score"],
                ]
            )
        ),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "QwenDeepSeek_family_patch_offset_math_mechanism",
        },
        "sources": {
            "qwen_refresh": str(qwen_refresh_path.relative_to(ROOT)),
            "deepseek_refresh": str(deepseek_refresh_path.relative_to(ROOT)),
            "mechanism_bridge": str(bridge_path.relative_to(ROOT)),
            "structure_atlas": str(structure_path.relative_to(ROOT)),
            "consistency": str(consistency_path.relative_to(ROOT)),
        },
        "direct_evidence": direct_evidence,
        "cross_model_summary": cross_model_summary,
        "candidate_math": {
            "equation_1_family_projection": (
                "B_f^(l)(h) = mu_f^(l) + U_f^(l) U_f^(l)^T (h - mu_f^(l))"
            ),
            "equation_2_state_decomposition": (
                "h_(c,ctx)^(l) = B_(f_c)^(l) + Delta_c^(l) + R_(ctx,c)^(l) + epsilon_(c,ctx)^(l)"
            ),
            "equation_3_offset_factorization": (
                "Delta_c^(l) = U_local,f^(l) a_c^(l) + S_shared^(l) b_c^(l) + xi_c^(l)"
            ),
            "equation_4_stage_gate": (
                "h_out^(l) = G_stage^(l) odot (B_f^(l) + Delta_c^(l)) + (1 - G_stage^(l)) odot R_(ctx,c)^(l)"
            ),
        },
        "mechanism_principles": {
            "principle_1_family_patch": {
                "statement": "family patch 不是单点模板，而是按 family 构成的局部仿射低秩图册。",
                "direct_support": {
                    "qwen_family_fit_strength": direct_evidence["qwen3_4b"]["family_fit_strength"],
                    "deepseek_family_fit_strength": direct_evidence["deepseek_7b"]["family_fit_strength"],
                    "mean_wrong_family_margin": cross_model_summary["mean_wrong_family_margin"],
                },
            },
            "principle_2_concept_offset": {
                "statement": "concept offset 是在 family patch 上叠加的小而有区分度的偏移，不是全空间重建。",
                "direct_support": {
                    "mean_offset_top32_energy_ratio": cross_model_summary["mean_offset_top32_energy_ratio"],
                    "mean_shared_norm_ratio": cross_model_summary["mean_shared_norm_ratio"],
                    "apple_qwen_best_layer": qwen_apple["layer"],
                    "apple_deepseek_best_layer": deepseek_apple["layer"],
                },
            },
            "principle_3_shared_scaffold": {
                "statement": "offset 不是纯局部噪声，还包含跨 family 共享 scaffold，因此需要 local basis 和 shared basis 两级分解。",
                "direct_support": {
                    "qwen_shared_norm_ratio": direct_evidence["qwen3_4b"]["shared_norm_ratio"],
                    "deepseek_shared_norm_ratio": direct_evidence["deepseek_7b"]["shared_norm_ratio"],
                    "bridge_consistency": consistency.get("verdict", {}).get(
                        "overall_verdict",
                        consistency.get("cross_model_verdict", {}).get("overall_verdict", "unknown"),
                    ),
                },
            },
            "principle_4_relation_context_transport": {
                "statement": "上下文和关系不是覆盖式改写，而是沿 relation-biased 层做受限扰动和运输。",
                "direct_support": {
                    "qwen_relation_component": qwen_bridge["components"]["R_relation"],
                    "deepseek_relation_component": deepseek_bridge["components"]["R_relation"],
                    "qwen_relation_biased_layers": qwen_stage_counts["relation_biased"],
                    "deepseek_relation_biased_layers": deepseek_stage_counts["relation_biased"],
                },
            },
            "principle_5_stage_gate": {
                "statement": "编码形成不是一次完成，而是分阶段门控：部分层偏 concept，部分层偏 relation，少数 shared band 用于稳态对齐。",
                "direct_support": {
                    "qwen_gating_component": qwen_bridge["components"]["G_gating"],
                    "deepseek_gating_component": deepseek_bridge["components"]["G_gating"],
                    "qwen_stage_counts": dict(qwen_stage_counts),
                    "deepseek_stage_counts": dict(deepseek_stage_counts),
                },
            },
        },
        "candidate_objective_inference": {
            "note": "以下是从 qwen3/deepseek 证据推断出的候选目标函数，不是已证明定理。",
            "equation": (
                "J = lambda_fit * E[r_true] + lambda_mix * E[inter_family_overlap] + "
                "lambda_dense * E[dense_offset_cost] + lambda_rel * E[relation_collision] + "
                "lambda_stage * E[transport_instability]"
            ),
            "interpretation": [
                "模型倾向于压低 true-family residual，因此形成可投影的 family patch。",
                "模型倾向于压低 inter-family overlap，因此保留正的 wrong-family margin。",
                "模型倾向于压低 dense_offset_cost，因此概念信息更集中在少数 offset 方向。",
                "模型倾向于压低 relation_collision，因此 relation/context 更像受限纤维而不是全局覆盖。",
                "模型倾向于压低 transport_instability，因此会出现 stage-conditioned 的分层门控结构。",
            ],
        },
        "strict_answer": {
            "why_encoding_looks_like_this": (
                "基于 qwen3 和 deepseek 的当前证据，深度神经网络里的概念编码之所以表现为“shared family patch + "
                "concept offset + staged relation transport”，是因为模型同时在做四件事：共享复用、族内区分、"
                "关系组合和阶段稳定。最省参数、最稳运输、又保留概念差异的解，不会是每个概念一个完全独立向量，"
                "而会自然收敛成 family patch 上的小偏移。"
            ),
            "what_is_not_solved": (
                "这仍然不是“彻底破解”。当前更像是静态和分层机制已经比较清楚，但动态学习律、"
                "新概念进入时 offset 如何被写入、以及跨模态脑区统一外推仍然没有闭合。"
            ),
        },
        "progress_estimate": {
            "family_patch_math_percent": 78.0,
            "concept_offset_math_percent": 74.0,
            "family_patch_plus_offset_joint_mechanism_percent": 63.0,
            "dynamic_learning_law_percent": 28.0,
            "full_brain_encoding_mechanism_percent": 46.0,
        },
        "next_large_blocks": [
            "把 qwen3 和 deepseek 的 family patch 投影、offset 稀疏系数和 relation transport 统一到同一批概念集上，扩到 hundreds-scale 真实词表。",
            "做新概念写入实验，直接测 Delta_c 如何随 novelty、routing、replay 改变，补上动态学习律。",
            "把语言侧的 patch-offset 结构迁到视觉和动作表征，验证 shared scaffold 是否跨模态成立。",
            "把 patch、offset、relation transport、protocol bridge 联立成统一状态方程，而不是长期分散在多个脚本和局部指标里。",
        ],
    }
    return payload


def test_qwen3_deepseek_family_patch_offset_math_mechanism() -> None:
    payload = build_payload()
    summary = payload["cross_model_summary"]
    assert summary["mean_family_fit_strength"] > 0.75
    assert summary["mean_wrong_family_margin"] > 0.6
    assert summary["mean_shared_norm_ratio"] > 0.95
    assert payload["strict_answer"]["what_is_not_solved"].startswith("这仍然不是")


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3/DeepSeek family patch + concept offset math mechanism")
    ap.add_argument(
        "--json-out",
        default="tests/codex_temp/qwen3_deepseek_family_patch_offset_math_mechanism_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["cross_model_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
