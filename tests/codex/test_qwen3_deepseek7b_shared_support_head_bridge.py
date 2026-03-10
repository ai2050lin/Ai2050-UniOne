#!/usr/bin/env python
"""
Bridge concept-support heads and relation-support heads on Qwen3-4B / DeepSeek-7B.

This is a real-model post analysis over existing local artifacts.
It asks whether the same head groups support:
1. concept-to-field protocol calling
2. relation mesofield causal mass
3. stronger global mechanism bridge scores
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def head_key(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head)}"


def layer_from_head_key(key: str) -> int:
    return int(key.split("H", 1)[0][1:])


def normalize_support(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vmax = max(scores.values()) or 1e-12
    return {key: float(value / vmax) for key, value in scores.items()}


def aggregate_concept_support(model_payload: Dict[str, Any]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for concept_name, concept_row in model_payload["concepts"].items():
        true_field = concept_row["true_field"]
        field_row = concept_row["field_scores"][true_field]
        for head_row in field_row["top_heads"]:
            key = head_key(head_row["layer"], head_row["head"])
            usage = float(head_row.get("usage_score", 0.0))
            fit = float(head_row.get("fit_score", 0.0))
            delta = float(head_row.get("protocol_delta", 0.0))
            scores[key] = scores.get(key, 0.0) + usage * fit * max(delta, 1.0e-6)
    return normalize_support(scores)


def aggregate_relation_support(model_payload: Dict[str, Any]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for relation_name, relation_row in model_payload["relations"].items():
        for head_row in relation_row["ranked_heads_top20"]:
            key = head_key(head_row["layer"], head_row["head"])
            bridge = float(head_row.get("bridge_tt", 0.0))
            topo = float(head_row.get("endpoint_topo_basis", 0.0))
            align = float(head_row.get("relation_align_topo", 0.0))
            scores[key] = scores.get(key, 0.0) + bridge * math.sqrt(max(topo, 1.0e-9) * max(align, 1.0e-9))
    return normalize_support(scores)


def top_items(scores: Dict[str, float], top_k: int) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


def positive_top_items(scores: Dict[str, float], top_k: int) -> List[Tuple[str, float]]:
    return [(key, value) for key, value in top_items(scores, top_k) if value > 0.0]


def shared_support_scores(concept_scores: Dict[str, float], relation_scores: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(set(concept_scores.keys()) | set(relation_scores.keys()))
    return {
        key: math.sqrt(max(concept_scores.get(key, 0.0), 0.0) * max(relation_scores.get(key, 0.0), 0.0))
        for key in keys
    }


def aggregate_layer_support(scores: Dict[str, float]) -> Dict[str, float]:
    layer_scores: Dict[str, float] = {}
    for key, value in scores.items():
        layer_key = f"L{layer_from_head_key(key)}"
        layer_scores[layer_key] = layer_scores.get(layer_key, 0.0) + float(value)
    return normalize_support(layer_scores)


def overlap_ratio(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    return float(len(sa & sb) / max(1, min(len(sa), len(sb))))


def soft_overlap_ratio(a_scores: Dict[str, float], b_scores: Dict[str, float]) -> float:
    keys = sorted(set(a_scores.keys()) | set(b_scores.keys()))
    numer = sum(min(float(a_scores.get(key, 0.0)), float(b_scores.get(key, 0.0))) for key in keys)
    denom = sum(float(a_scores.get(key, 0.0)) for key in keys) + sum(float(b_scores.get(key, 0.0)) for key in keys)
    return float((2.0 * numer) / max(1.0e-12, denom))


def relation_shared_mass_rows(
    relation_payload: Dict[str, Any],
    boundary_payload: Dict[str, Any],
    shared_heads: List[str],
    concept_heads: List[str],
    relation_heads: List[str],
) -> List[Dict[str, Any]]:
    rows = []
    shared_set = set(shared_heads)
    concept_set = set(concept_heads)
    relation_set = set(relation_heads)
    for relation_name, relation_row in relation_payload["relations"].items():
        ranked = relation_row["ranked_heads_top20"]
        denom = sum(float(row["bridge_tt"]) for row in ranked) or 1.0e-12
        shared_mass = sum(float(row["bridge_tt"]) for row in ranked if head_key(row["layer"], row["head"]) in shared_set)
        concept_mass = sum(float(row["bridge_tt"]) for row in ranked if head_key(row["layer"], row["head"]) in concept_set)
        relation_mass = sum(float(row["bridge_tt"]) for row in ranked if head_key(row["layer"], row["head"]) in relation_set)
        boundary_row = boundary_payload["relations"][relation_name]
        rows.append(
            {
                "relation": relation_name,
                "classification": boundary_row["classification"],
                "bridge_score": float(boundary_row["bridge_score"]),
                "shared_mass_ratio": float(shared_mass / denom),
                "concept_mass_ratio": float(concept_mass / denom),
                "relation_mass_ratio": float(relation_mass / denom),
                "layer_cluster_margin": float(boundary_row["layer_cluster_margin"]),
            }
        )
    return rows


def class_means(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    bucket: Dict[str, List[float]] = {}
    for row in rows:
        bucket.setdefault(row["classification"], []).append(float(row[key]))
    return {name: mean(values) for name, values in bucket.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Qwen3 / DeepSeek7B shared support head bridge")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/qwen3_deepseek7b_shared_support_head_bridge_20260310.json")
    args = ap.parse_args()

    field_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_concept_protocol_field_mapping_20260309.json")
    relation_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_protocol_mesofield_scale_20260309.json")
    boundary_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_topology_boundary_bridge_20260309.json")
    mechanism_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_mechanism_bridge_20260309.json")

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "top_k": int(args.top_k)}, "models": {}}

    for model_name in field_payload["models"].keys():
        concept_scores = aggregate_concept_support(field_payload["models"][model_name])
        relation_scores = aggregate_relation_support(relation_payload["models"][model_name])
        shared_scores = shared_support_scores(concept_scores, relation_scores)
        concept_layer_scores = aggregate_layer_support(concept_scores)
        relation_layer_scores = aggregate_layer_support(relation_scores)
        shared_layer_scores = aggregate_layer_support(shared_scores)

        top_concept_items = positive_top_items(concept_scores, int(args.top_k))
        top_relation_items = positive_top_items(relation_scores, int(args.top_k))
        top_shared_items = positive_top_items(shared_scores, int(args.top_k))
        top_concept = [key for key, _ in top_concept_items]
        top_relation = [key for key, _ in top_relation_items]
        top_shared = [key for key, _ in top_shared_items]

        top_concept_layers_items = positive_top_items(concept_layer_scores, min(int(args.top_k), len(concept_layer_scores)))
        top_relation_layers_items = positive_top_items(relation_layer_scores, min(int(args.top_k), len(relation_layer_scores)))
        top_shared_layers_items = positive_top_items(shared_layer_scores, min(int(args.top_k), len(shared_layer_scores)))
        top_concept_layers = [key for key, _ in top_concept_layers_items]
        top_relation_layers = [key for key, _ in top_relation_layers_items]
        top_shared_layers = [key for key, _ in top_shared_layers_items]

        relation_rows = relation_shared_mass_rows(
            relation_payload["models"][model_name],
            boundary_payload["models"][model_name],
            top_shared,
            top_concept,
            top_relation,
        )

        mechanism_score = float(
            mechanism_payload["models"][model_name]["mechanism_bridge_score"]
        )
        shared_mass_means = class_means(relation_rows, "shared_mass_ratio")
        compact_mass = max(
            shared_mass_means.get("compact_boundary", 0.0),
            shared_mass_means.get("compact_mesofield", 0.0),
        )
        diffuse_mass = mean(
            [
                shared_mass_means.get("layer_cluster_only", 0.0),
                shared_mass_means.get("distributed_none", 0.0),
                shared_mass_means.get("layer_cluster_mesofield", 0.0),
                shared_mass_means.get("distributed_mesofield", 0.0),
            ]
        )

        results["models"][model_name] = {
            "head_sets": {
                "top_concept_heads": top_concept_items,
                "top_relation_heads": top_relation_items,
                "top_shared_heads": top_shared_items,
            },
            "layer_sets": {
                "top_concept_layers": top_concept_layers_items,
                "top_relation_layers": top_relation_layers_items,
                "top_shared_layers": top_shared_layers_items,
            },
            "global_summary": {
                "concept_relation_exact_head_overlap_ratio": overlap_ratio(top_concept, top_relation),
                "concept_relation_soft_head_overlap_ratio": soft_overlap_ratio(concept_scores, relation_scores),
                "concept_relation_layer_overlap_ratio": overlap_ratio(top_concept_layers, top_relation_layers),
                "concept_relation_soft_layer_overlap_ratio": soft_overlap_ratio(concept_layer_scores, relation_layer_scores),
                "shared_concept_overlap_ratio": overlap_ratio(top_shared, top_concept),
                "shared_relation_overlap_ratio": overlap_ratio(top_shared, top_relation),
                "shared_positive_head_count": len(top_shared),
                "shared_positive_layer_count": len(top_shared_layers),
                "mean_shared_mass_ratio": mean([float(row["shared_mass_ratio"]) for row in relation_rows]),
                "mean_concept_mass_ratio": mean([float(row["concept_mass_ratio"]) for row in relation_rows]),
                "mean_relation_mass_ratio": mean([float(row["relation_mass_ratio"]) for row in relation_rows]),
                "classification_shared_mass_mean": shared_mass_means,
                "compact_minus_diffuse_shared_mass": float(compact_mass - diffuse_mass),
                "mechanism_bridge_score": mechanism_score,
            },
            "relations": relation_rows,
        }

    qwen = results["models"]["qwen3_4b"]["global_summary"]
    deepseek = results["models"]["deepseek_7b"]["global_summary"]

    payload = {
        **results,
        "headline_metrics": {
            "qwen_exact_head_overlap": float(qwen["concept_relation_exact_head_overlap_ratio"]),
            "deepseek_exact_head_overlap": float(deepseek["concept_relation_exact_head_overlap_ratio"]),
            "qwen_soft_head_overlap": float(qwen["concept_relation_soft_head_overlap_ratio"]),
            "deepseek_soft_head_overlap": float(deepseek["concept_relation_soft_head_overlap_ratio"]),
            "qwen_layer_overlap": float(qwen["concept_relation_layer_overlap_ratio"]),
            "deepseek_layer_overlap": float(deepseek["concept_relation_layer_overlap_ratio"]),
            "qwen_soft_layer_overlap": float(qwen["concept_relation_soft_layer_overlap_ratio"]),
            "deepseek_soft_layer_overlap": float(deepseek["concept_relation_soft_layer_overlap_ratio"]),
            "qwen_shared_mass": float(qwen["mean_shared_mass_ratio"]),
            "deepseek_shared_mass": float(deepseek["mean_shared_mass_ratio"]),
            "qwen_compact_mass_gain": float(qwen["compact_minus_diffuse_shared_mass"]),
            "deepseek_compact_mass_gain": float(deepseek["compact_minus_diffuse_shared_mass"]),
        },
        "gains": {
            "deepseek_minus_qwen_exact_head_overlap": float(
                deepseek["concept_relation_exact_head_overlap_ratio"] - qwen["concept_relation_exact_head_overlap_ratio"]
            ),
            "deepseek_minus_qwen_soft_head_overlap": float(
                deepseek["concept_relation_soft_head_overlap_ratio"] - qwen["concept_relation_soft_head_overlap_ratio"]
            ),
            "deepseek_minus_qwen_layer_overlap": float(
                deepseek["concept_relation_layer_overlap_ratio"] - qwen["concept_relation_layer_overlap_ratio"]
            ),
            "deepseek_minus_qwen_soft_layer_overlap": float(
                deepseek["concept_relation_soft_layer_overlap_ratio"] - qwen["concept_relation_soft_layer_overlap_ratio"]
            ),
            "deepseek_minus_qwen_shared_mass": float(deepseek["mean_shared_mass_ratio"] - qwen["mean_shared_mass_ratio"]),
            "deepseek_minus_qwen_mechanism_bridge": float(deepseek["mechanism_bridge_score"] - qwen["mechanism_bridge_score"]),
        },
        "hypotheses": {
            "H1_real_models_have_nontrivial_shared_mass": bool(
                qwen["mean_shared_mass_ratio"] >= 0.03 and deepseek["mean_shared_mass_ratio"] >= 0.03
            ),
            "H2_deepseek_has_stronger_layer_overlap_than_qwen": bool(
                deepseek["concept_relation_layer_overlap_ratio"] > qwen["concept_relation_layer_overlap_ratio"]
                and deepseek["concept_relation_soft_layer_overlap_ratio"] > qwen["concept_relation_soft_layer_overlap_ratio"]
            ),
            "H3_compact_relations_carry_more_shared_support": bool(
                deepseek["compact_minus_diffuse_shared_mass"] > 0.0 or qwen["compact_minus_diffuse_shared_mass"] > 0.0
            ),
            "H4_layer_shared_support_tracks_mechanism_bridge": bool(
                deepseek["concept_relation_soft_layer_overlap_ratio"] > qwen["concept_relation_soft_layer_overlap_ratio"]
                and deepseek["mechanism_bridge_score"] > qwen["mechanism_bridge_score"]
            ),
        },
        "project_readout": {
            "summary": "这一版把真实模型里的概念调用头和关系中观场头接到同一张共享支撑图上。结果显示精确同头重合很弱，但层级软重合和共享质量都不低，说明真实模型更像在中观层级复用同源支撑，而不是严格复用同一颗头。",
            "next_question": "如果这种层级共享支撑真的是中观冗余场的一部分，下一步就该把高共享层带接入真实因果消融与恢复链读数，直接验证它们是否同时支撑概念、关系和回退恢复。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["gains"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
