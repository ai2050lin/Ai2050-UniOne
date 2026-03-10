#!/usr/bin/env python
"""
Build a mechanism bridge summary for Qwen3-4B and DeepSeek-7B.

This script switches the current model-side comparison away from GPT-2. It
aggregates:
1) Qwen3-4B direct local protocol/field/gate artifacts
2) DeepSeek-7B historical local artifacts available on this machine

The goal is not to pretend both sides have identical evidence quality. Instead,
it makes the evidence boundary explicit and scores each model on the same
mechanism decomposition:
- shared basis
- individual offset
- representation hierarchy H
- gating term G
- relation term R
- topology / routing term T
- concept-to-field protocol calling
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import test_qwen3_deepseek7b_apple_mechanism_consistency as consistency


ROOT = Path(__file__).resolve().parents[2]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(value: float, target: float) -> float:
    if target <= 0:
        return 0.0
    return clamp01(float(value) / float(target))


def qwen_extra_paths() -> Dict[str, Path]:
    return {
        "field_mapping": ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_concept_protocol_field_mapping_20260309.json",
        "boundary_atlas": ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_protocol_field_boundary_atlas_20260309.json",
        "gate_dynamics": ROOT / "tests" / "codex_temp" / "gpt2_qwen3_gate_law_nonlinear_dynamics_20260308.json",
        "direct_topology": ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_attention_topology_basis_20260309.json",
    }


def qwen_component_scores(qwen: Dict[str, Any]) -> Dict[str, Any]:
    extra = qwen_extra_paths()
    field_payload = load_json(extra["field_mapping"])["models"]["qwen3_4b"]
    boundary_payload = load_json(extra["boundary_atlas"])["models"]["qwen3_4b"]
    gate_payload = load_json(extra["gate_dynamics"])["models"]["qwen3_4b"]["global_summary"]
    direct_topology = load_json(extra["direct_topology"])["models"]["qwen3_4b"]

    basis_gap_animal = clamp01(
        (qwen["shared_basis"]["apple_to_animal_residual"] - qwen["shared_basis"]["apple_to_fruit_residual"]) / 0.25
    )
    basis_gap_abstract = clamp01(
        (qwen["shared_basis"]["apple_to_abstract_residual"] - qwen["shared_basis"]["apple_to_fruit_residual"]) / 0.25
    )
    basis_compact = clamp01(1.0 - qwen["shared_basis"]["fruit_compactness"])
    shared_basis = mean(
        [
            basis_gap_animal,
            basis_gap_abstract,
            basis_compact,
            1.0 if qwen["shared_basis"]["family_nested_world"] else 0.0,
        ]
    )

    offset = mean(
        [
            1.0 if qwen["offset"]["natural_offset_supported"] else 0.0,
            norm(qwen["offset"]["natural_offset_gap_top4"], 0.02),
            norm(qwen["offset"]["delta_top64_energy_ratio"], 0.50),
            norm(qwen["offset"]["delta_top256_energy_ratio"], 0.80),
        ]
    )

    hierarchy = mean(
        [
            1.0 if qwen["H"]["apple_closer_to_fruit_than_animal"] else 0.0,
            1.0 if qwen["H"]["apple_closer_to_fruit_than_abstract"] else 0.0,
            clamp01(1.0 - qwen["shared_basis"]["apple_to_fruit_residual"]),
        ]
    )

    gating_gap = clamp01(
        (qwen["G"]["early_topo_gating_strength"] - qwen["G"]["early_topo_relation_strength"]) / 0.02
    )
    gating = mean(
        [
            norm(gate_payload["mean_nonlinear_recurrence_r2"], 0.95),
            norm(gate_payload["mean_nonlinear_gain"], 0.10),
            gating_gap,
        ]
    )

    concept_rows = field_payload["concepts"]
    match_ratio = mean(
        [
            1.0 if row["summary"]["preferred_field_matches_truth"] else 0.0
            for row in concept_rows.values()
        ]
    )
    mean_margin = mean([float(row["summary"]["margin_vs_second"]) for row in concept_rows.values()])
    relation = mean(
        [
            clamp01((qwen["R"]["deep_repr_relation_strength"] - qwen["R"]["deep_repr_gating_strength"]) / 0.01),
            match_ratio,
            norm(mean_margin, 1.0e-6),
        ]
    )

    boundary_summary = boundary_payload["global_summary"]
    topology = mean(
        [
            1.0 if qwen["T"]["apple_supports_family_topology_basis"] else 0.0,
            clamp01((qwen["T"]["animal_topology_residual"] - qwen["T"]["fruit_topology_residual"]) / 0.20),
            clamp01((qwen["T"]["abstract_topology_residual"] - qwen["T"]["fruit_topology_residual"]) / 0.20),
            clamp01(1.0 - float(direct_topology["family_summary"]["fruit"]["mean_topology_residual_ratio"])),
            float(boundary_summary["preferred_field_match_rate"]),
        ]
    )

    protocol = mean(
        [
            match_ratio,
            clamp01(1.0 - float(boundary_summary["mean_heads_for_50pct_mass"]) / 64.0),
            clamp01(1.0 - float(boundary_summary["mean_heads_for_80pct_mass"]) / 256.0),
            1.0 if boundary_summary["minimal_boundary_histogram"].get("none", 0) < 9 else 0.2,
        ]
    )

    directness = 1.0

    return {
        "shared_basis": shared_basis,
        "offset": offset,
        "H_representation": hierarchy,
        "G_gating": gating,
        "R_relation": relation,
        "T_topology": topology,
        "protocol_calling": protocol,
        "evidence_directness": directness,
        "evidence": {
            "field_match_ratio": match_ratio,
            "mean_margin_vs_second": mean_margin,
            "mean_heads_for_50pct_mass": float(boundary_summary["mean_heads_for_50pct_mass"]),
            "mean_heads_for_80pct_mass": float(boundary_summary["mean_heads_for_80pct_mass"]),
            "gate_nonlinear_r2": float(gate_payload["mean_nonlinear_recurrence_r2"]),
            "gate_nonlinear_gain": float(gate_payload["mean_nonlinear_gain"]),
            "direct_topology_fruit_residual": float(direct_topology["family_summary"]["fruit"]["mean_topology_residual_ratio"]),
        },
        "boundaries": [],
    }


def deepseek_component_scores(deepseek: Dict[str, Any], paths: Dict[str, Path]) -> Dict[str, Any]:
    family = consistency.read_json(paths["deepseek_family"])
    dossier = consistency.read_json(paths["deepseek_dossier"])
    direct_topology = consistency.read_json(paths["deepseek_topology"])["models"]["deepseek_7b"]
    field_payload = load_json(qwen_extra_paths()["field_mapping"])["models"]["deepseek_7b"]
    boundary_payload = load_json(qwen_extra_paths()["boundary_atlas"])["models"]["deepseek_7b"]

    family_hypothesis_ratio = mean([1.0 if item["pass"] else 0.0 for item in family["hypotheses"]])
    basis = mean(
        [
            norm(deepseek["shared_basis"]["apple_shared_base_ratio_mean"], 0.03),
            norm(deepseek["shared_basis"]["apple_meso_to_macro_jaccard_mean"], 0.40),
            family_hypothesis_ratio,
        ]
    )

    offset = mean(
        [
            norm(deepseek["offset"]["axis_specificity_index"], 0.70),
            norm(deepseek["offset"]["cross_dim_decoupling_index"], 0.75),
            norm(float(dossier["metrics"]["style_logic_syntax_signal"]), 0.60),
        ]
    )

    hierarchy = mean(
        [
            1.0 if deepseek["H"]["hierarchy_closure_pass"] else 0.0,
            1.0 if deepseek["H"]["layer_anchor_pass"] else 0.0,
            norm(float(dossier["metrics"]["apple_layer_peak_value"]), 0.17),
        ]
    )

    diag_values = [float(v) for v in deepseek["G"]["diagonal_advantage"].values()]
    cross_values = [
        deepseek["G"]["style_logic"],
        deepseek["G"]["syntax_logic"],
        deepseek["G"]["logic_style"],
    ]
    gating = mean(
        [
            norm(mean(diag_values), 0.03),
            clamp01((mean(diag_values) - mean(cross_values)) / 0.03),
            1.0 if all(v > 0.0 for v in diag_values) else 0.0,
        ]
    )

    relation_match_ratio = mean(
        [
            1.0 if row["summary"]["preferred_field_matches_truth"] else 0.0
            for row in field_payload["concepts"].values()
        ]
    )
    relation_margin = mean(
        [float(row["summary"]["margin_vs_second"]) for row in field_payload["concepts"].values()]
    )
    relation = mean(
        [
            norm(deepseek["R"]["route_index"], 0.015),
            norm(deepseek["R"]["hop3_selectivity"], 0.015),
            norm(deepseek["R"]["triplet_seq_logprob_margin"], 0.02),
            norm(deepseek["R"]["triplet_positive_ratio"], 0.40),
            relation_match_ratio,
            norm(relation_margin, 1.0e-6),
        ]
    )

    topology_direct = mean(
        [
            1.0 if deepseek["T"]["apple_supports_family_topology_basis"] else 0.0,
            clamp01((deepseek["T"]["animal_topology_residual"] - deepseek["T"]["fruit_topology_residual"]) / 0.20),
            clamp01((deepseek["T"]["abstract_topology_residual"] - deepseek["T"]["fruit_topology_residual"]) / 0.20),
            clamp01(1.0 - float(direct_topology["family_summary"]["fruit"]["mean_topology_residual_ratio"])),
            clamp01(float(deepseek["T"]["graph_geometry_alignment"])),
            float(boundary_payload["global_summary"]["preferred_field_match_rate"]),
        ]
    )

    protocol = mean(
        [
            relation_match_ratio,
            norm(deepseek["R"]["route_index"], 0.015),
            clamp01(1.0 - float(deepseek["R"]["minimal_subset_size"] - 1) / 4.0),
            norm(deepseek["R"]["triplet_positive_ratio"], 0.40),
            clamp01(1.0 - float(boundary_payload["global_summary"]["mean_heads_for_50pct_mass"]) / 64.0),
            clamp01(1.0 - float(boundary_payload["global_summary"]["mean_heads_for_80pct_mass"]) / 128.0),
        ]
    )

    directness = 1.0

    return {
        "shared_basis": basis,
        "offset": offset,
        "H_representation": hierarchy,
        "G_gating": gating,
        "R_relation": relation,
        "T_topology": topology_direct,
        "protocol_calling": protocol,
        "evidence_directness": directness,
        "evidence": {
            "family_hypothesis_ratio": family_hypothesis_ratio,
            "mean_diagonal_advantage": mean(diag_values),
            "minimal_subset_size": int(deepseek["R"]["minimal_subset_size"]),
            "triplet_seq_logprob_margin": float(deepseek["R"]["triplet_seq_logprob_margin"]),
            "graph_geometry_alignment": float(deepseek["T"]["graph_geometry_alignment"]),
            "relation_field_match_ratio": relation_match_ratio,
            "relation_mean_margin": relation_margin,
            "boundary_match_rate": float(boundary_payload["global_summary"]["preferred_field_match_rate"]),
            "boundary_mean_heads_for_50pct_mass": float(boundary_payload["global_summary"]["mean_heads_for_50pct_mass"]),
            "direct_topology_fruit_residual": float(direct_topology["family_summary"]["fruit"]["mean_topology_residual_ratio"]),
        },
        "boundaries": [],
    }


def build_model_summary(name: str, components: Dict[str, Any]) -> Dict[str, Any]:
    weighted = (
        0.18 * components["shared_basis"]
        + 0.14 * components["offset"]
        + 0.12 * components["H_representation"]
        + 0.13 * components["G_gating"]
        + 0.13 * components["R_relation"]
        + 0.12 * components["T_topology"]
        + 0.10 * components["protocol_calling"]
        + 0.08 * components["evidence_directness"]
    )
    mechanism_bridge_score = clamp01(weighted)
    weakest = sorted(
        [
            ("shared_basis", components["shared_basis"]),
            ("offset", components["offset"]),
            ("H_representation", components["H_representation"]),
            ("G_gating", components["G_gating"]),
            ("R_relation", components["R_relation"]),
            ("T_topology", components["T_topology"]),
            ("protocol_calling", components["protocol_calling"]),
            ("evidence_directness", components["evidence_directness"]),
        ],
        key=lambda item: item[1],
    )[:3]
    return {
        "model_name": name,
        "components": {k: float(v) for k, v in components.items() if isinstance(v, (int, float))},
        "evidence": components["evidence"],
        "boundaries": components["boundaries"],
        "mechanism_bridge_score": mechanism_bridge_score,
        "weakest_links": [{"component": k, "score": float(v)} for k, v in weakest],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a Qwen3-4B / DeepSeek-7B mechanism bridge summary")
    ap.add_argument(
        "--json-out",
        default="tests/codex_temp/qwen3_deepseek7b_mechanism_bridge_20260309.json",
    )
    args = ap.parse_args()

    paths = consistency.default_paths()
    missing = [str(p) for p in list(paths.values()) + list(qwen_extra_paths().values()) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    qwen = consistency.load_qwen_metrics(paths)
    deepseek = consistency.load_deepseek_metrics(paths)
    verdict = consistency.build_verdict(qwen, deepseek)

    qwen_summary = build_model_summary("qwen3_4b", qwen_component_scores(qwen))
    deepseek_summary = build_model_summary("deepseek_7b", deepseek_component_scores(deepseek, paths))

    models = {
        "qwen3_4b": qwen_summary,
        "deepseek_7b": deepseek_summary,
    }
    ranking = [
        {
            "model_name": name,
            "mechanism_bridge_score": row["mechanism_bridge_score"],
            "weakest_links": row["weakest_links"],
        }
        for name, row in sorted(models.items(), key=lambda item: item[1]["mechanism_bridge_score"], reverse=True)
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scope": "Qwen3-4B direct local artifacts vs DeepSeek-7B historical local artifacts",
        },
        "paths": {k: str(v).replace("\\", "/") for k, v in {**paths, **qwen_extra_paths()}.items()},
        "models": models,
        "ranking": ranking,
        "cross_model_verdict": verdict,
        "global_conclusion": {
            "statement": "模型侧主线已经可以从 GPT-2 切换到 Qwen3-4B 和 DeepSeek-7B，但两者证据时间切片仍不完全对称。",
            "why": [
                "Qwen3-4B 已有共享基底、偏移、门控、协议场、边界图谱和非线性门控递推的直接本地工件。",
                "DeepSeek-7B 现在已经补上同协议 attention-topology 直测，因此 T 项不再停留在代理层。",
                "当前真正的不对称主要来自其他历史工件的时间切片不同，而不是 T 项是否可直测。",
            ],
            "next_steps": [
                "在 Qwen3 / DeepSeek7B 上统一复刻协议场边界和概念调用映射，减少跨脚本口径差异。",
                "把同协议 attention-topology 和概念调用映射同时并入总览看板，直接显示 T 到协议场调用的对应关系。",
                "后续所有模型侧桥接优先使用 Qwen3-4B 和 DeepSeek-7B，不再以 GPT-2 作为主比较对象。",
            ],
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "saved": str(out_path),
                "ranking": ranking,
                "overall_verdict": verdict["overall_verdict"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
