#!/usr/bin/env python
"""
Bridge relation boundary classes with relation-support-family topology structure.

This joins:
- relation topology atlas
- relation mesofield scan
- relation boundary atlas

Goal:
- explain why some relations become compact while others remain distributed
  in terms of endpoint-family topology support and head concentration proxies
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

import test_qwen3_deepseek7b_relation_protocol_mesofield_scale as relation_scale


ROOT = Path(__file__).resolve().parents[2]


def mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def top_share(rows: List[Dict[str, Any]], top_k: int) -> float:
    values = [float(row["bridge_tt"]) for row in rows]
    denom = sum(values) or 1e-12
    return float(sum(values[:top_k]) / denom)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build relation topology-boundary bridge for Qwen3 and DeepSeek7B")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_relation_topology_boundary_bridge_20260309.json",
    )
    args = ap.parse_args()

    topo_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_topology_atlas_20260309.json")
    meso_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_protocol_mesofield_scale_20260309.json")
    boundary_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_boundary_atlas_20260309.json")

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    specs = relation_scale.relation_specs()

    for model_name in topo_payload["models"].keys():
        topo_model = topo_payload["models"][model_name]
        meso_model = meso_payload["models"][model_name]
        boundary_model = boundary_payload["models"][model_name]

        relations = {}
        class_rows: Dict[str, List[float]] = {}
        for relation_name, spec in specs.items():
            endpoint_families = sorted(set(spec["endpoint_families"].values()))
            endpoint_words = sorted(spec["endpoint_families"].keys())

            family_true_residual = mean(
                [
                    float(topo_model["family_summary"][family]["mean_topology_residual_ratio"])
                    for family in endpoint_families
                ]
            )
            family_entropy = mean(
                [
                    float(topo_model["family_summary"][family]["mean_last_token_entropy"])
                    for family in endpoint_families
                ]
            )
            endpoint_margin = mean(
                [
                    float(topo_model["concepts"][word]["summary"]["margin_vs_best_wrong"])
                    for word in endpoint_words
                ]
            )
            endpoint_support = mean(
                [
                    1.0 if topo_model["concepts"][word]["summary"]["supports_family_topology_basis"] else 0.0
                    for word in endpoint_words
                ]
            )

            meso_row = meso_model["relations"][relation_name]
            boundary_row = boundary_model["relations"][relation_name]
            concentration_top4 = top_share(meso_row["ranked_heads_top20"], 4)
            concentration_top8 = top_share(meso_row["ranked_heads_top20"], 8)
            classification = boundary_row["classification"]

            topology_compactness = max(0.0, 1.0 - family_true_residual)
            bridge_score = mean(
                [
                    topology_compactness,
                    endpoint_margin,
                    endpoint_support,
                    concentration_top4,
                    max(0.0, float(meso_row["mesofield_summary"]["layer_cluster_margin"])),
                ]
            )

            relations[relation_name] = {
                "classification": classification,
                "endpoint_families": endpoint_families,
                "endpoint_words": endpoint_words,
                "topology_compactness": float(topology_compactness),
                "endpoint_margin_mean": float(endpoint_margin),
                "endpoint_support_rate": float(endpoint_support),
                "family_true_residual_mean": float(family_true_residual),
                "family_entropy_mean": float(family_entropy),
                "top4_bridge_share_in_top20": float(concentration_top4),
                "top8_bridge_share_in_top20": float(concentration_top8),
                "layer_cluster_margin": float(meso_row["mesofield_summary"]["layer_cluster_margin"]),
                "minimal_stronger_than_control_k": meso_row["mesofield_summary"]["minimal_stronger_than_control_k"],
                "bridge_score": float(bridge_score),
            }
            class_rows.setdefault(classification, []).append(float(bridge_score))

        results["models"][model_name] = {
            "relations": relations,
            "global_summary": {
                "classification_bridge_mean": {
                    cls: mean(values) for cls, values in class_rows.items()
                },
                "top_relation_order": [
                    {
                        "relation": name,
                        "bridge_score": float(row["bridge_score"]),
                        "classification": row["classification"],
                    }
                    for name, row in sorted(relations.items(), key=lambda item: item[1]["bridge_score"], reverse=True)
                ],
            },
        }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for model_name, row in results["models"].items():
        print(f"[summary] {model_name} bridge_mean={row['global_summary']['classification_bridge_mean']}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
