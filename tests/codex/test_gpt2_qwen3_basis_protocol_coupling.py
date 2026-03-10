"""
Coupling test for the hypothesis that:

1. larger concept-specific offsets require stronger topology reordering to enter
   relation/protocol space
2. more stable shared-basis projections support more stable protocol entry

This script reuses existing GPT-2 / Qwen3 result artifacts and computes a
lightweight first-pass coupling analysis without re-running the models.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: List[float]) -> float:
    return float(np.mean(np.array(xs, dtype=np.float64))) if xs else 0.0


def corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def argmax_idx(xs: List[float]) -> int:
    return int(np.argmax(np.array(xs, dtype=np.float64))) if xs else -1


def normalized_entropy(xs: List[float]) -> float:
    arr = np.array(xs, dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 1e-12 or len(arr) <= 1:
        return 0.0
    p = arr / total
    p = p[p > 0]
    h = float(-np.sum(p * np.log(p)))
    return h / math.log(len(arr))


def topk_mean(xs: List[float], k: int) -> float:
    if not xs:
        return 0.0
    arr = np.array(xs, dtype=np.float64)
    idx = np.argsort(arr)[::-1][:k]
    return float(np.mean(arr[idx]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze basis/protocol coupling from existing GPT-2/Qwen3 artifacts")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_basis_protocol_coupling_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    path_sig = load_json(ROOT / "tests" / "codex_temp" / "gpt2_qwen3_concept_path_signature_20260308.json")
    protocol_map = load_json(ROOT / "tests" / "codex_temp" / "gpt2_qwen3_concept_protocol_field_mapping_20260308.json")

    out: Dict[str, Any] = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": 0.0,
            "source_files": [
                "gpt2_qwen3_concept_path_signature_20260308.json",
                "gpt2_qwen3_concept_protocol_field_mapping_20260308.json",
            ],
        },
        "models": {},
    }

    pooled_basis_strength: List[float] = []
    pooled_protocol_stability: List[float] = []

    for model_name, model_payload in path_sig["models"].items():
        sig_concepts = model_payload["concepts"]
        map_concepts = protocol_map["models"][model_name]["concepts"]

        layerwise_offset: List[float] = []
        layerwise_protocol_reorder: List[float] = []
        layerwise_gating_topo: List[float] = []
        layerwise_relation_topo: List[float] = []

        concept_rows = []
        concept_basis_strengths: List[float] = []
        concept_protocol_stabilities: List[float] = []

        for concept_name, concept_payload in sig_concepts.items():
            d_repr = [float(x) for x in concept_payload["D_repr_by_layer"]]
            g_topo = [float(x) for x in concept_payload["G_topo_by_layer"]]
            r_topo = [float(x) for x in concept_payload["R_topo_by_layer"]]
            b_repr = [float(x) for x in concept_payload["B_repr_by_layer"]]

            protocol_pressure = [g + r for g, r in zip(g_topo, r_topo)]
            layerwise_offset.extend(d_repr)
            layerwise_protocol_reorder.extend(protocol_pressure)
            layerwise_gating_topo.extend(g_topo)
            layerwise_relation_topo.extend(r_topo)

            true_field = map_concepts[concept_name]["true_field"]
            true_field_payload = map_concepts[concept_name]["field_scores"][true_field]
            layer_usage = [float(x) for x in true_field_payload["layer_usage_by_layer"]]
            top_heads = true_field_payload["top_heads"]
            protocol_deltas = [float(h["protocol_delta"]) for h in top_heads[:5]]

            basis_strength = topk_mean(b_repr, 4)
            basis_stability = 1.0 - normalized_entropy(b_repr)
            protocol_stability = 1.0 - normalized_entropy(layer_usage)
            protocol_delta_mean = mean(protocol_deltas)

            concept_basis_strengths.append(basis_strength)
            concept_protocol_stabilities.append(protocol_stability)
            pooled_basis_strength.append(basis_strength)
            pooled_protocol_stability.append(protocol_stability)

            concept_rows.append(
                {
                    "concept": concept_name,
                    "family": concept_payload["family"],
                    "offset_protocol_corr_by_layer": corr(d_repr, protocol_pressure),
                    "offset_gating_corr_by_layer": corr(d_repr, g_topo),
                    "offset_relation_corr_by_layer": corr(d_repr, r_topo),
                    "peak_offset_layer": argmax_idx(d_repr),
                    "peak_protocol_layer": argmax_idx(protocol_pressure),
                    "peak_layer_gap": abs(argmax_idx(d_repr) - argmax_idx(protocol_pressure)),
                    "basis_strength_top4_mean": basis_strength,
                    "basis_stability_inverse_entropy": basis_stability,
                    "protocol_stability_inverse_entropy": protocol_stability,
                    "protocol_delta_top5_mean": protocol_delta_mean,
                    "true_field": true_field,
                }
            )

        model_out = {
            "layerwise_coupling": {
                "offset_vs_protocol_reorder_corr": corr(layerwise_offset, layerwise_protocol_reorder),
                "offset_vs_gating_topo_corr": corr(layerwise_offset, layerwise_gating_topo),
                "offset_vs_relation_topo_corr": corr(layerwise_offset, layerwise_relation_topo),
                "mean_offset": mean(layerwise_offset),
                "mean_protocol_reorder": mean(layerwise_protocol_reorder),
            },
            "conceptwise_coupling": {
                "basis_strength_vs_protocol_stability_corr": corr(
                    concept_basis_strengths, concept_protocol_stabilities
                ),
                "sample_size": len(concept_basis_strengths),
                "note": "Small-sample conceptwise check over apple/cat/truth only; use as weak evidence.",
            },
            "concepts": concept_rows,
        }
        out["models"][model_name] = model_out

    out["global_summary"] = {
        "basis_strength_vs_protocol_stability_corr_pooled": corr(
            pooled_basis_strength, pooled_protocol_stability
        ),
        "pooled_sample_size": len(pooled_basis_strength),
        "interpretation": [
            "Positive offset-vs-protocol correlation supports the idea that larger concept-specific deviation requires more topology reordering.",
            "Positive basis-vs-stability correlation supports the idea that stable shared-basis anchoring helps protocol entry stabilize.",
            "The conceptwise basis/protocol result is weak-evidence only because the current artifact has three anchor concepts per model.",
        ],
    }
    out["meta"]["runtime_sec"] = float(time.time() - t0)

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
