#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from stage122_adverb_context_route_shift_probe import build_case_bundle, find_subsequence, ids_for_word
from stage124_noun_neuron_basic_probe import OUTPUT_DIR as STAGE124_OUTPUT_DIR
from wordclass_neuron_basic_probe_lib import clamp01, load_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage128_noun_static_route_bridge_20260323"
ROUTE_LAYER_INDEX = 3
NOUN_LAYER_INDEX = 11
TOP_NOUN_NEURONS = 16
TOP_ROUTE_NEURONS = 16


def l2_normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.linalg.norm(vec).clamp_min(1e-8)


def load_stage124_summary() -> Dict[str, object]:
    summary_path = STAGE124_OUTPUT_DIR / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8-sig"))


def select_l11_noun_neurons(summary: Dict[str, object]) -> List[Dict[str, object]]:
    rows = [row for row in summary["top_general_neurons"] if int(row["layer_index"]) == NOUN_LAYER_INDEX]
    if len(rows) < TOP_NOUN_NEURONS:
        extra = [row for row in summary["top_general_neurons"] if int(row["layer_index"]) != NOUN_LAYER_INDEX]
        rows.extend(extra)
    return rows[:TOP_NOUN_NEURONS]


def build_noun_output_directions(model, noun_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    c_proj = model.transformer.h[NOUN_LAYER_INDEX].mlp.c_proj.weight.detach().cpu().float()
    directions = []
    for row in noun_rows:
        neuron_idx = int(row["neuron_index"])
        direction = l2_normalize(c_proj[neuron_idx])
        directions.append(
            {
                "layer_index": NOUN_LAYER_INDEX,
                "neuron_index": neuron_idx,
                "general_rule_score": float(row["general_rule_score"]),
                "direction": direction,
            }
        )
    return directions


def capture_route_layer_activations(model, tokenizer) -> Dict[str, object]:
    layer_buffer: List[torch.Tensor | None] = [None]

    def hook(_module, _inputs, output):
        layer_buffer[0] = output.detach().cpu()

    handle = model.transformer.h[ROUTE_LAYER_INDEX].mlp.act.register_forward_hook(hook)
    neuron_diff_sum = None
    positive_count = None
    case_rows = []

    try:
        for case in build_case_bundle():
            prompts = [case["base_prompt"], case["adverb_prompt"], case["adjective_prompt"]]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            with torch.inference_mode():
                model(**encoded, use_cache=False, return_dict=True)

            layer_tensor = layer_buffer[0]
            if layer_tensor is None:
                raise RuntimeError("未捕获到 L3 选路层神经元激活")

            input_ids = encoded["input_ids"]
            verb_ids = ids_for_word(tokenizer, case["verb"])
            verb_positions = []
            for prompt_idx in range(3):
                seq = input_ids[prompt_idx].tolist()
                if tokenizer.pad_token_id in seq:
                    seq = seq[: seq.index(tokenizer.pad_token_id)]
                verb_pos = find_subsequence(seq, verb_ids)
                if verb_pos is None:
                    raise RuntimeError(f"未定位到动词位置: {case}")
                verb_positions.append(verb_pos)

            base_vec = layer_tensor[0, verb_positions[0], :].float()
            adverb_vec = layer_tensor[1, verb_positions[1], :].float()
            adjective_vec = layer_tensor[2, verb_positions[2], :].float()
            diff_vec = adverb_vec - adjective_vec

            if neuron_diff_sum is None:
                neuron_diff_sum = torch.zeros_like(diff_vec, dtype=torch.float64)
                positive_count = torch.zeros_like(diff_vec, dtype=torch.float64)
            neuron_diff_sum += diff_vec.to(torch.float64)
            positive_count += (diff_vec > 0).to(torch.float64)
            case_rows.append(diff_vec.to(torch.float64))
    finally:
        handle.remove()

    diff_mean = neuron_diff_sum / max(1, len(case_rows))
    positive_rate = positive_count / max(1, len(case_rows))
    score = diff_mean.clamp_min(0.0) * (0.5 + 0.5 * positive_rate)
    top_vals, top_idx = torch.topk(score, k=TOP_ROUTE_NEURONS)
    c_fc = model.transformer.h[ROUTE_LAYER_INDEX].mlp.c_fc.weight.detach().cpu().float()
    route_rows = []
    for rank_idx, neuron_idx in enumerate(top_idx.tolist()):
        route_rows.append(
            {
                "layer_index": ROUTE_LAYER_INDEX,
                "neuron_index": int(neuron_idx),
                "route_neuron_score": float(top_vals[rank_idx].item()),
                "route_diff_mean": float(diff_mean[neuron_idx].item()),
                "route_positive_rate": float(positive_rate[neuron_idx].item()),
                "direction": l2_normalize(c_fc[:, neuron_idx]),
            }
        )
    return {
        "case_count": len(case_rows),
        "route_rows": route_rows,
        "route_diff_mean_vector": diff_mean,
    }


def build_bridge_summary(noun_rows: Sequence[Dict[str, object]], route_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    noun_matrix = torch.stack([row["direction"] for row in noun_rows], dim=0)
    route_matrix = torch.stack([row["direction"] for row in route_rows], dim=0)
    sim = noun_matrix @ route_matrix.T

    pair_rows = []
    for noun_idx, noun_row in enumerate(noun_rows):
        for route_idx, route_row in enumerate(route_rows):
            pair_rows.append(
                {
                    "noun_layer_index": noun_row["layer_index"],
                    "noun_neuron_index": noun_row["neuron_index"],
                    "noun_general_rule_score": noun_row["general_rule_score"],
                    "route_layer_index": route_row["layer_index"],
                    "route_neuron_index": route_row["neuron_index"],
                    "route_neuron_score": route_row["route_neuron_score"],
                    "cosine_alignment": float(sim[noun_idx, route_idx].item()),
                }
            )

    strongest_pairs = sorted(pair_rows, key=lambda row: row["cosine_alignment"], reverse=True)[:20]
    noun_best = sim.max(dim=1).values
    route_best = sim.max(dim=0).values
    bridge_alignment_mean = float(noun_best.mean().item())
    route_alignment_mean = float(route_best.mean().item())
    positive_bridge_rate = float((noun_best > 0.08).to(torch.float64).mean().item())
    bridge_score = (
        0.45 * clamp01(bridge_alignment_mean / 0.18)
        + 0.25 * clamp01(route_alignment_mean / 0.18)
        + 0.30 * positive_bridge_rate
    )
    return {
        "noun_selected_count": len(noun_rows),
        "route_selected_count": len(route_rows),
        "bridge_alignment_mean": bridge_alignment_mean,
        "route_alignment_mean": route_alignment_mean,
        "positive_bridge_rate": positive_bridge_rate,
        "noun_static_route_bridge_score": float(bridge_score),
        "strongest_bridge_pairs": strongest_pairs,
    }


def sanitize_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    clean_rows = []
    for row in rows:
        clean_row = dict(row)
        clean_row.pop("direction", None)
        clean_rows.append(clean_row)
    return clean_rows


def build_summary(
    stage124_summary: Dict[str, object],
    route_info: Dict[str, object],
    bridge_summary: Dict[str, object],
    noun_rows: Sequence[Dict[str, object]],
    route_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage128_noun_static_route_bridge",
        "title": "Noun 静态层到选路动态层桥",
        "status_short": "gpt2_noun_static_route_bridge_ready",
        "model_name": "gpt2",
        "model_path": str(load_model.__globals__["MODEL_PATH"]),
        "source_stage_static": "stage124_noun_neuron_basic_probe",
        "source_stage_route": "stage122_adverb_context_route_shift_probe",
        "noun_layer_index": NOUN_LAYER_INDEX,
        "route_layer_index": ROUTE_LAYER_INDEX,
        "source_case_count": route_info["case_count"],
        "stage124_dominant_general_layer_index": stage124_summary["dominant_general_layer_index"],
        **bridge_summary,
        "selected_noun_neurons": sanitize_rows(noun_rows),
        "selected_route_neurons": sanitize_rows(route_rows),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage128: Noun 静态层到选路动态层桥",
        "",
        "## 核心结果",
        f"- 名词静态层: L{summary['noun_layer_index']}",
        f"- 选路动态层: L{summary['route_layer_index']}",
        f"- 选中名词神经元数: {summary['noun_selected_count']}",
        f"- 选中选路神经元数: {summary['route_selected_count']}",
        f"- 几何桥平均对齐: {summary['bridge_alignment_mean']:.4f}",
        f"- 路由侧平均对齐: {summary['route_alignment_mean']:.4f}",
        f"- 正向桥接比率: {summary['positive_bridge_rate']:.4f}",
        f"- 静态到选路桥分数: {summary['noun_static_route_bridge_score']:.4f}",
        "",
        "## 最强桥接对",
    ]
    for row in summary["strongest_bridge_pairs"][:12]:
        lines.append(
            "- "
            f"L{row['noun_layer_index']} N{row['noun_neuron_index']} -> "
            f"L{row['route_layer_index']} N{row['route_neuron_index']}: "
            f"align={row['cosine_alignment']:.4f}"
        )
    lines.extend(
        [
            "",
            "## 理论提示",
            "- 这不是前向因果回传，而是同一残差空间中的共享几何桥。",
            "- 如果 L11 名词神经元输出方向与 L3 选路敏感神经元输入方向稳定对齐，说明静态名词聚合与早层选路链并非完全分离。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "STAGE128_NOUN_STATIC_ROUTE_BRIDGE_REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")
    (output_dir / "strongest_bridge_pairs.json").write_text(
        json.dumps(summary["strongest_bridge_pairs"], ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR) -> Dict[str, object]:
    stage124_summary = load_stage124_summary()
    noun_candidates = select_l11_noun_neurons(stage124_summary)
    model, tokenizer = load_model()
    noun_rows = build_noun_output_directions(model, noun_candidates)
    route_info = capture_route_layer_activations(model, tokenizer)
    route_rows = route_info["route_rows"]
    bridge_summary = build_bridge_summary(noun_rows, route_rows)
    summary = build_summary(stage124_summary, route_info, bridge_summary, noun_rows, route_rows)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Noun 静态层到选路动态层桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage128 输出目录")
    args = parser.parse_args()

    summary = run_analysis(output_dir=Path(args.output_dir))
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "noun_static_route_bridge_score": summary["noun_static_route_bridge_score"],
                "bridge_alignment_mean": summary["bridge_alignment_mean"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
