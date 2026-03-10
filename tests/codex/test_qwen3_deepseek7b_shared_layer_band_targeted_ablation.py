#!/usr/bin/env python
"""
Run a lightweight real-model targeted ablation over high-shared layer bands.

Goal:
- ablate the top shared layer band on Qwen3 / DeepSeek7B
- compare against low-support control layers
- verify whether Qwen3 is concept-first fragile
- verify whether DeepSeek is relation-first fragile
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from test_qwen3_deepseek7b_protocol_field_boundary_atlas import (
    HeadGroupAblator,
    compute_field_scores,
    default_model_specs,
    family_words,
    load_model,
)
from test_qwen3_deepseek7b_relation_protocol_mesofield_scale import (
    base_prompts,
    prompt_len,
    relation_score,
    relation_specs,
    run_model,
)


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def head_key(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head)}"


def layer_key(layer: int) -> str:
    return f"L{int(layer)}"


def normalize_support(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vmax = max(scores.values()) or 1.0e-12
    return {key: float(value / vmax) for key, value in scores.items()}


def aggregate_concept_layer_scores(model_payload: Dict[str, Any]) -> Dict[int, float]:
    layer_scores: Dict[str, float] = {}
    for concept_row in model_payload["concepts"].values():
        true_field = concept_row["true_field"]
        for head_row in concept_row["field_scores"][true_field]["top_heads"]:
            key = layer_key(int(head_row["layer"]))
            usage = float(head_row["usage_score"])
            fit = float(head_row["fit_selectivity"])
            delta = float(head_row["protocol_delta"])
            layer_scores[key] = layer_scores.get(key, 0.0) + usage * fit * max(delta, 1.0e-6)
    norm = normalize_support(layer_scores)
    return {int(key[1:]): float(value) for key, value in norm.items()}


def aggregate_relation_layer_scores(model_payload: Dict[str, Any]) -> Dict[int, float]:
    layer_scores: Dict[str, float] = {}
    for relation_row in model_payload["relations"].values():
        for head_row in relation_row["ranked_heads_top20"]:
            key = layer_key(int(head_row["layer"]))
            bridge = float(head_row["bridge_tt"])
            topo = float(head_row["endpoint_topo_basis"])
            align = float(head_row["relation_align_topo"])
            layer_scores[key] = layer_scores.get(key, 0.0) + bridge * math.sqrt(max(topo, 1.0e-9) * max(align, 1.0e-9))
    norm = normalize_support(layer_scores)
    return {int(key[1:]): float(value) for key, value in norm.items()}


def aggregate_shared_layer_scores(concept_layers: Dict[int, float], relation_layers: Dict[int, float], n_layers: int) -> Dict[int, float]:
    shared = {}
    for layer in range(n_layers):
        shared[layer] = math.sqrt(max(concept_layers.get(layer, 0.0), 0.0) * max(relation_layers.get(layer, 0.0), 0.0))
    return shared


def pick_layer_groups(shared_scores: Dict[int, float], top_k: int, control_k: int) -> Tuple[List[int], List[int]]:
    ranked = sorted(shared_scores.items(), key=lambda item: item[1], reverse=True)
    target_layers = [layer for layer, score in ranked if score > 0.0][:top_k]
    target_set = set(target_layers)
    control_candidates = [layer for layer, _score in sorted(shared_scores.items(), key=lambda item: item[1]) if layer not in target_set]
    control_layers = control_candidates[:control_k]
    return target_layers, control_layers


def build_layer_group(layers: Sequence[int], n_heads: int) -> List[Tuple[int, int]]:
    return [(int(layer), int(head)) for layer in layers for head in range(n_heads)]


def build_fixed_fit_selectivity(concept_row: Dict[str, Any], fields: Sequence[str], n_layers: int, n_heads: int) -> Dict[str, Dict[Tuple[int, int], float]]:
    fixed_fit_selectivity: Dict[str, Dict[Tuple[int, int], float]] = {}
    field_scores = concept_row["field_scores"]
    for field in fields:
        field_map = {
            (int(row["layer"]), int(row["head"])): float(row["fit_selectivity"])
            for row in field_scores[field]["top_heads"]
        }
        for layer in range(n_layers):
            for head in range(n_heads):
                field_map.setdefault((layer, head), 0.0)
        fixed_fit_selectivity[field] = field_map
    return fixed_fit_selectivity


def field_margin(scores: Dict[str, float], true_field: str) -> float:
    return float(scores[true_field] - max(value for field, value in scores.items() if field != true_field))


def summarize_causal_margin(baseline: float, targeted: float, control: float) -> Dict[str, float]:
    eps = 1.0e-12
    return {
        "target_collapse_ratio": float(max(0.0, (baseline - targeted) / (abs(baseline) + eps))),
        "control_collapse_ratio": float(max(0.0, (baseline - control) / (abs(baseline) + eps))),
        "causal_margin": float((control - targeted) / (abs(baseline) + eps)),
    }


def last_token_layer_vectors(out, target_len: int) -> Dict[int, np.ndarray]:
    topo = {}
    for li, attn in enumerate(out.attentions):
        arr = attn[0].detach().float().cpu().numpy().astype(np.float32)
        last_row = arr[:, -1, :]
        pad = target_len - last_row.shape[1]
        if pad > 0:
            last_row = np.pad(last_row, ((0, 0), (0, pad)), mode="constant")
        topo[li] = last_row.reshape(-1).astype(np.float32)
    return topo


def mean_stack(rows: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32)


def compute_word_topology(model, tok, ablator: HeadGroupAblator | None, words: Sequence[str], group: Sequence[Tuple[int, int]] | None) -> Dict[str, Dict[int, np.ndarray]]:
    if ablator is not None:
        if group:
            ablator.set_active_group(group)
        else:
            ablator.clear()
    target_len = max(prompt_len(tok, text) for word in words for text in base_prompts(word))
    word_topo = {}
    for word in words:
        rows = {}
        for text in base_prompts(word):
            out = run_model(model, tok, text)
            topo = last_token_layer_vectors(out, target_len)
            for layer, vec in topo.items():
                rows.setdefault(layer, []).append(vec)
        word_topo[word] = {layer: mean_stack(vs) for layer, vs in rows.items()}
    return word_topo


def compute_pair_bridge_peak(relation_name: str, word_topo: Dict[str, Dict[int, np.ndarray]]) -> Dict[str, Any]:
    spec = relation_specs()[relation_name]
    (a, b), (c, d) = spec["pairs"]
    n_layers = len(next(iter(word_topo.values())))
    per_layer = []
    for layer in range(n_layers):
        vec1 = word_topo[b][layer] - word_topo[a][layer]
        vec2 = word_topo[d][layer] - word_topo[c][layer]
        _err, _cos, score = relation_score(vec1, vec2)
        per_layer.append(float(score))
    peak_layer = int(np.argmax(per_layer))
    return {
        "pair_bridge_by_layer": per_layer,
        "peak_layer": peak_layer,
        "peak_pair_bridge": float(per_layer[peak_layer]),
    }


def top_probe_names(rows: List[Dict[str, Any]], key: str, top_k: int) -> List[str]:
    ranked = sorted(rows, key=lambda row: float(row[key]), reverse=True)
    return [str(row["concept"] if "concept" in row else row["relation"]) for row in ranked[:top_k]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run targeted ablation on shared layer bands for Qwen3 / DeepSeek7B")
    ap.add_argument("--top-layers", type=int, default=2)
    ap.add_argument("--probe-concepts", type=int, default=2)
    ap.add_argument("--probe-relations", type=int, default=2)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_shared_layer_band_targeted_ablation_20260310.json",
    )
    args = ap.parse_args()

    concept_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_protocol_field_boundary_atlas_20260309.json")
    relation_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_protocol_mesofield_scale_20260309.json")
    orientation_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_layer_band_causal_orientation_20260310.json")

    model_specs = dict(default_model_specs())
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "top_layers": int(args.top_layers),
            "probe_concepts": int(args.probe_concepts),
            "probe_relations": int(args.probe_relations),
        },
        "models": {},
    }

    for model_name, model_path in model_specs.items():
        model, tok = load_model(model_path, "bfloat16", True)
        n_layers = int(getattr(model.config, "num_hidden_layers"))
        n_heads = int(getattr(model.config, "num_attention_heads"))

        concept_layer_scores = aggregate_concept_layer_scores(concept_payload["models"][model_name])
        relation_layer_scores = aggregate_relation_layer_scores(relation_payload["models"][model_name])
        shared_layer_scores = aggregate_shared_layer_scores(concept_layer_scores, relation_layer_scores, n_layers)
        target_layers = [
            int(row["layer"])
            for row in orientation_payload["models"][model_name]["shared_layers"][: int(args.top_layers)]
        ]
        _unused_target_layers, control_layers = pick_layer_groups(shared_layer_scores, int(args.top_layers), int(args.top_layers))
        control_layers = [int(layer) for layer in control_layers if int(layer) not in set(target_layers)][: int(args.top_layers)]
        target_group = build_layer_group(target_layers, n_heads)
        control_group = build_layer_group(control_layers, n_heads)

        concept_probe_names = top_probe_names(orientation_payload["models"][model_name]["concepts"], "shared_layer_hit_ratio", int(args.probe_concepts))
        relation_probe_names = top_probe_names(orientation_payload["models"][model_name]["relations"], "shared_layer_hit_ratio", int(args.probe_relations))
        ablator = HeadGroupAblator(model)
        fields = list(family_words().keys())

        concept_rows = []
        relation_rows = []
        try:
            for concept_name in concept_probe_names:
                concept_row = concept_payload["models"][model_name]["concepts"][concept_name]
                fixed_fit_selectivity = build_fixed_fit_selectivity(concept_row, fields, n_layers, n_heads)
                baseline_scores = compute_field_scores(model, tok, ablator, concept_name, fields, None, fixed_fit_selectivity)
                target_scores = compute_field_scores(model, tok, ablator, concept_name, fields, target_group, fixed_fit_selectivity)
                control_scores = compute_field_scores(model, tok, ablator, concept_name, fields, control_group, fixed_fit_selectivity)
                baseline_margin = field_margin(baseline_scores, concept_row["true_field"])
                target_margin = field_margin(target_scores, concept_row["true_field"])
                control_margin = field_margin(control_scores, concept_row["true_field"])
                summary = summarize_causal_margin(baseline_margin, target_margin, control_margin)
                concept_rows.append(
                    {
                        "concept": concept_name,
                        "true_field": concept_row["true_field"],
                        "baseline_margin": float(baseline_margin),
                        "target_margin": float(target_margin),
                        "control_margin": float(control_margin),
                        **summary,
                    }
                )

            for relation_name in relation_probe_names:
                spec = relation_specs()[relation_name]
                pair_words = sorted({word for pair in spec["pairs"] for word in pair})
                baseline_topo = compute_word_topology(model, tok, ablator, pair_words, None)
                target_topo = compute_word_topology(model, tok, ablator, pair_words, target_group)
                control_topo = compute_word_topology(model, tok, ablator, pair_words, control_group)
                baseline_bridge = compute_pair_bridge_peak(relation_name, baseline_topo)
                target_bridge = compute_pair_bridge_peak(relation_name, target_topo)
                control_bridge = compute_pair_bridge_peak(relation_name, control_topo)
                summary = summarize_causal_margin(
                    float(baseline_bridge["peak_pair_bridge"]),
                    float(target_bridge["peak_pair_bridge"]),
                    float(control_bridge["peak_pair_bridge"]),
                )
                relation_rows.append(
                    {
                        "relation": relation_name,
                        "baseline_peak_pair_bridge": float(baseline_bridge["peak_pair_bridge"]),
                        "target_peak_pair_bridge": float(target_bridge["peak_pair_bridge"]),
                        "control_peak_pair_bridge": float(control_bridge["peak_pair_bridge"]),
                        **summary,
                    }
                )
        finally:
            ablator.close()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        concept_mean_margin = mean([float(row["causal_margin"]) for row in concept_rows])
        relation_mean_margin = mean([float(row["causal_margin"]) for row in relation_rows])
        actual_orientation = float(relation_mean_margin - concept_mean_margin)
        predicted_orientation = float(orientation_payload["models"][model_name]["global_summary"]["shared_layer_orientation"])

        results["models"][model_name] = {
            "target_layers": [{"layer": int(layer), "shared_score": float(shared_layer_scores[layer])} for layer in target_layers],
            "control_layers": [{"layer": int(layer), "shared_score": float(shared_layer_scores[layer])} for layer in control_layers],
            "probe_concepts": concept_rows,
            "probe_relations": relation_rows,
            "global_summary": {
                "mean_concept_causal_margin": concept_mean_margin,
                "mean_relation_causal_margin": relation_mean_margin,
                "actual_targeted_orientation": actual_orientation,
                "actual_orientation_label": "relation_led" if actual_orientation >= 0.05 else "concept_led" if actual_orientation <= -0.05 else "balanced",
                "predicted_orientation": predicted_orientation,
                "predicted_orientation_label": orientation_payload["models"][model_name]["global_summary"]["orientation_label"],
            },
        }

    qwen = results["models"]["qwen3_4b"]["global_summary"]
    deepseek = results["models"]["deepseek_7b"]["global_summary"]

    payload = {
        **results,
        "headline_metrics": {
            "qwen_concept_causal_margin": float(qwen["mean_concept_causal_margin"]),
            "qwen_relation_causal_margin": float(qwen["mean_relation_causal_margin"]),
            "deepseek_concept_causal_margin": float(deepseek["mean_concept_causal_margin"]),
            "deepseek_relation_causal_margin": float(deepseek["mean_relation_causal_margin"]),
            "qwen_actual_orientation": float(qwen["actual_targeted_orientation"]),
            "deepseek_actual_orientation": float(deepseek["actual_targeted_orientation"]),
        },
        "gains": {
            "qwen_concept_over_relation_margin": float(qwen["mean_concept_causal_margin"] - qwen["mean_relation_causal_margin"]),
            "deepseek_relation_over_concept_margin": float(deepseek["mean_relation_causal_margin"] - deepseek["mean_concept_causal_margin"]),
            "deepseek_minus_qwen_actual_orientation": float(deepseek["actual_targeted_orientation"] - qwen["actual_targeted_orientation"]),
        },
        "hypotheses": {
            "H1_qwen_targeted_ablation_hits_concepts_more": bool(qwen["mean_concept_causal_margin"] > qwen["mean_relation_causal_margin"]),
            "H2_deepseek_targeted_ablation_hits_relations_more": bool(deepseek["mean_relation_causal_margin"] > deepseek["mean_concept_causal_margin"]),
            "H3_targeted_ablation_beats_control_on_both_models": bool(
                qwen["mean_concept_causal_margin"] > 0.0
                and qwen["mean_relation_causal_margin"] > 0.0
                and deepseek["mean_concept_causal_margin"] > 0.0
                and deepseek["mean_relation_causal_margin"] > 0.0
            ),
            "H4_actual_orientation_matches_prediction": bool(
                qwen["actual_orientation_label"] == qwen["predicted_orientation_label"]
                and deepseek["actual_orientation_label"] == deepseek["predicted_orientation_label"]
            ),
        },
        "project_readout": {
            "summary": "这一版不再只做后验取向判断，而是直接对真实模型的高共享层带做定向消融。结果如果稳定，就能把“共享层带偏概念还是偏关系”从统计取向推进成真实干预证据。",
            "next_question": "如果定向消融的实际伤害方向和预测取向一致，下一步就该把这些高共享层带接回最小控制桥和在线回退链，验证它们是否对应真实脆弱区。",
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
