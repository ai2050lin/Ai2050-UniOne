#!/usr/bin/env python
"""
Condense the relation meso-field k-scan into a boundary atlas for Qwen3 and DeepSeek-7B.

This is a post-analysis script over the existing relation protocol mesofield
scan. It answers:
- which relation families have a compact top-k boundary?
- which only appear at layer-cluster scale?
- which still look boundary-free?
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def load_mesoscan() -> dict:
    path = Path("tests/codex_temp/qwen3_deepseek7b_relation_protocol_mesofield_scale_20260309.json")
    return json.loads(path.read_text(encoding="utf-8"))


def classify_relation(entry: dict) -> str:
    summary = entry.get("mesofield_summary", {})
    min_pos = summary.get("minimal_positive_margin_k")
    min_ctrl = summary.get("minimal_stronger_than_control_k")
    layer_margin = float(summary.get("layer_cluster_margin", 0.0))
    if min_ctrl is not None:
        return "compact_boundary"
    if min_pos is not None and layer_margin > 0:
        return "mixed_boundary"
    if min_pos is None and layer_margin > 0:
        return "layer_cluster_only"
    return "distributed_none"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build relation boundary atlas from meso-field scan")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_relation_boundary_atlas_20260309.json",
    )
    args = ap.parse_args()

    source = load_mesoscan()
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_json": "tests/codex_temp/qwen3_deepseek7b_relation_protocol_mesofield_scale_20260309.json",
        },
        "models": {},
    }

    for model_name, row in source["models"].items():
        relations = {}
        class_hist = {}
        for relation_name, entry in row["relations"].items():
            summary = entry["mesofield_summary"]
            relation_class = classify_relation(entry)
            class_hist[relation_class] = class_hist.get(relation_class, 0) + 1
            relations[relation_name] = {
                "minimal_positive_margin_k": summary.get("minimal_positive_margin_k"),
                "minimal_stronger_than_control_k": summary.get("minimal_stronger_than_control_k"),
                "best_k_by_causal_margin": summary.get("best_k_by_causal_margin"),
                "layer_cluster_margin": summary.get("layer_cluster_margin"),
                "classification": relation_class,
            }

        results["models"][model_name] = {
            "global_summary": {
                "k_values": row["global_summary"].get("k_values", []),
                "classification_histogram": class_hist,
                "mean_causal_margin_by_k": row["global_summary"].get("mean_causal_margin_by_k", {}),
            },
            "relations": relations,
        }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for model_name, row in results["models"].items():
        print(f"[summary] {model_name} classes={row['global_summary']['classification_histogram']}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
