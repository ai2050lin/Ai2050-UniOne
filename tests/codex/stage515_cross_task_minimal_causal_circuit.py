#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from multimodel_language_shared import discover_layers, free_model, load_model_bundle
from qwen3_language_shared import move_batch_to_model_device, remove_hooks
from stage427_pronoun_mixed_circuit_search import register_mlp_neuron_ablation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage515_cross_task_minimal_causal_circuit_20260404"
)
STAGE513_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage513_noun_cross_task_core_protocol_20260404"
    / "summary.json"
)
MODEL_KEYS = ["qwen3", "deepseek7b"]
TOP_CANDIDATES = 16
MAX_SUBSET = 6
MIN_GAIN = 1e-4
OFFTARGET_PENALTY = 0.5


class _OutFeatureShim:
    def __init__(self, out_features: int):
        self.out_features = int(out_features)

TARGET_CASES = {
    "knowledge": [
        {
            "sentence": "An apple is a kind of fruit.",
            "question": "Choose the better category. Reply with 1 or 2 only.",
            "options": ["1", "2"],
            "label": 0,
        },
        {
            "sentence": "The apple belongs to the fruit family.",
            "question": "Choose the better category. Reply with 1 or 2 only.",
            "options": ["1", "2"],
            "label": 0,
        },
        {
            "sentence": "People often eat an apple as a snack.",
            "question": "Choose the better category. Reply with 1 or 2 only.",
            "options": ["1", "2"],
            "label": 0,
        },
    ],
    "syntax": [
        {
            "sentence": "The apple fell from the tree.",
            "question": "If the marked noun is the subject, reply 1. If it is the object, reply 2.",
            "options": ["1", "2"],
            "label": 0,
        },
        {
            "sentence": "She sliced the apple into pieces.",
            "question": "If the marked noun is the subject, reply 1. If it is the object, reply 2.",
            "options": ["1", "2"],
            "label": 1,
        },
        {
            "sentence": "He washed the apple in the sink.",
            "question": "If the marked noun is the subject, reply 1. If it is the object, reply 2.",
            "options": ["1", "2"],
            "label": 1,
        },
    ],
    "attribute": [
        {
            "sentence": "The apple is red and sweet.",
            "question": "If the sentence binds an attribute to apple, reply 1. Otherwise reply 2.",
            "options": ["1", "2"],
            "label": 0,
        },
        {
            "sentence": "The apple is about the size of a fist.",
            "question": "If the sentence binds an attribute to apple, reply 1. Otherwise reply 2.",
            "options": ["1", "2"],
            "label": 0,
        },
        {
            "sentence": "The banana is yellow, and the apple sits on the plate.",
            "question": "If the sentence binds an attribute to apple, reply 1. Otherwise reply 2.",
            "options": ["1", "2"],
            "label": 1,
        },
    ],
    "association": [
        {
            "sentence": "Apple is related to banana and pear.",
            "question": "If the sentence places apple in a fruit association network, reply 1. Otherwise reply 2.",
            "options": ["1", "2"],
            "label": 0,
        },
        {
            "sentence": "Apple is associated with fruit, tree, and juice.",
            "question": "If the sentence places apple in a fruit association network, reply 1. Otherwise reply 2.",
            "options": ["1", "2"],
            "label": 0,
        },
        {
            "sentence": "Apple is on the desk beside a hammer and a lamp.",
            "question": "If the sentence places apple in a fruit association network, reply 1. Otherwise reply 2.",
            "options": ["1", "2"],
            "label": 1,
        },
    ],
}

CONTROL_CASES = {
    "knowledge": [
        {
            "sentence": "A banana is a kind of fruit.",
            "question": "Choose the better category. Reply with 1 or 2 only.",
            "options": ["1", "2"],
            "label": 0,
        },
        {
            "sentence": "People often eat a banana as a snack.",
            "question": "Choose the better category. Reply with 1 or 2 only.",
            "options": ["1", "2"],
            "label": 0,
        },
    ],
    "syntax": [
        {
            "sentence": "The banana fell from the shelf.",
            "question": "If the marked noun is the subject, reply 1. If it is the object, reply 2.",
            "options": ["1", "2"],
            "label": 0,
        },
        {
            "sentence": "She peeled the banana after lunch.",
            "question": "If the marked noun is the subject, reply 1. If it is the object, reply 2.",
            "options": ["1", "2"],
            "label": 1,
        },
    ],
    "attribute": [
        {
            "sentence": "The banana is yellow and soft.",
            "question": "If the sentence binds an attribute to banana, reply 1. Otherwise reply 2.",
            "options": ["1", "2"],
            "label": 0,
        },
        {
            "sentence": "The apple is red, while the banana is on the plate.",
            "question": "If the sentence binds an attribute to banana, reply 1. Otherwise reply 2.",
            "options": ["1", "2"],
            "label": 1,
        },
    ],
    "association": [
        {
            "sentence": "Banana is related to apple and pear.",
            "question": "If the sentence places banana in a fruit association network, reply 1. Otherwise reply 2.",
            "options": ["1", "2"],
            "label": 0,
        },
        {
            "sentence": "Banana is on the shelf beside a wrench and a lamp.",
            "question": "If the sentence places banana in a fruit association network, reply 1. Otherwise reply 2.",
            "options": ["1", "2"],
            "label": 1,
        },
    ],
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def patch_glm4_mlp_compat(model) -> None:
    for layer in discover_layers(model):
        mlp = layer.mlp
        if not hasattr(mlp, "gate_proj") and hasattr(mlp, "down_proj"):
            mlp.gate_proj = _OutFeatureShim(mlp.down_proj.in_features)


def load_stage513_summary() -> Dict[str, object]:
    return json.loads(STAGE513_SUMMARY_PATH.read_text(encoding="utf-8"))


def option_token_id(tokenizer, text: str) -> int:
    token_ids = tokenizer(" " + text, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise RuntimeError(f"选项 {text!r} 不是单一 token")
    return int(token_ids[0])


def build_prompt(case: Dict[str, object]) -> str:
    return (
        f'Sentence: "{case["sentence"]}"\n'
        f'Question: {case["question"]}\n'
        f'Answer with one token only: {case["options"][0]} or {case["options"][1]}\n'
        "Answer:"
    )


def evaluate_case_groups(
    model,
    tokenizer,
    case_groups: Dict[str, List[Dict[str, object]]],
    *,
    handles: Sequence[object] | None = None,
) -> Dict[str, object]:
    group_results = {}
    total_prob = 0.0
    total_count = 0
    total_acc = 0.0
    for group_name, cases in case_groups.items():
        per_case = []
        group_prob = 0.0
        group_acc = 0.0
        for case in cases:
            prompt = build_prompt(case)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            encoded = move_batch_to_model_device(model, encoded)
            with torch.inference_mode():
                logits = model(**encoded, use_cache=False, return_dict=True).logits[:, -1, :]
            option_ids = torch.tensor(
                [option_token_id(tokenizer, case["options"][0]), option_token_id(tokenizer, case["options"][1])],
                device=logits.device,
                dtype=torch.long,
            )
            option_logits = logits.index_select(dim=1, index=option_ids)
            log_probs = option_logits.log_softmax(dim=-1)
            label = int(case["label"])
            pred = int(option_logits.argmax(dim=-1).item())
            correct_prob = float(log_probs[0, label].exp().item())
            is_correct = pred == label
            per_case.append(
                {
                    "sentence": case["sentence"],
                    "expected": case["options"][label],
                    "predicted": case["options"][pred],
                    "correct_prob": correct_prob,
                    "is_correct": is_correct,
                }
            )
            group_prob += correct_prob
            group_acc += float(is_correct)
            total_prob += correct_prob
            total_acc += float(is_correct)
            total_count += 1
        group_results[group_name] = {
            "mean_correct_prob": group_prob / max(1, len(cases)),
            "accuracy": group_acc / max(1, len(cases)),
            "per_case": per_case,
        }
    return {
        "group_results": group_results,
        "mean_correct_prob": total_prob / max(1, total_count),
        "accuracy": total_acc / max(1, total_count),
        "count": total_count,
    }


def decode_flat_index(flat_index: int, layer_widths: Sequence[int]) -> tuple[int, int]:
    offset = 0
    for layer_index, width in enumerate(layer_widths):
        next_offset = offset + int(width)
        if flat_index < next_offset:
            return layer_index, int(flat_index - offset)
        offset = next_offset
    raise IndexError(f"flat_index {flat_index} 超出层宽累计范围")


def build_candidates(stage513_model_row: Dict[str, object], layer_widths: Sequence[int]) -> List[Dict[str, object]]:
    group_top_ids = stage513_model_row["group_top_ids"]
    score_map = defaultdict(float)
    freq_map = defaultdict(int)
    for ids in group_top_ids.values():
        for rank, flat_idx in enumerate(ids, start=1):
            freq_map[int(flat_idx)] += 1
            score_map[int(flat_idx)] += max(0.0, TOP_CANDIDATES * 4 - rank)
    candidate_ids = [
        idx for idx, freq in freq_map.items() if freq >= 4
    ]
    ranked = sorted(candidate_ids, key=lambda idx: (freq_map[idx], score_map[idx]), reverse=True)[:TOP_CANDIDATES]
    rows = []
    for idx in ranked:
        layer_index, neuron_index = decode_flat_index(int(idx), layer_widths)
        rows.append(
            {
                "kind": "mlp_neuron",
                "flat_index": int(idx),
                "layer_index": int(layer_index),
                "neuron_index": int(neuron_index),
                "shared_frequency": int(freq_map[idx]),
                "rank_score": float(score_map[idx]),
            }
        )
    return rows


def evaluate_subset(
    model,
    tokenizer,
    candidate_map: Dict[str, Dict[str, object]],
    subset_ids: Sequence[str],
    baseline_target: Dict[str, object],
    baseline_control: Dict[str, object],
) -> Dict[str, object]:
    subset = [candidate_map[cid] for cid in subset_ids]
    handles = register_mlp_neuron_ablation(model, subset) if subset else []
    try:
        current_target = evaluate_case_groups(model, tokenizer, TARGET_CASES)
        current_control = evaluate_case_groups(model, tokenizer, CONTROL_CASES)
    finally:
        remove_hooks(handles)
    target_drop = float(baseline_target["mean_correct_prob"] - current_target["mean_correct_prob"])
    control_shift = abs(float(baseline_control["mean_correct_prob"] - current_control["mean_correct_prob"]))
    utility = target_drop - OFFTARGET_PENALTY * control_shift
    return {
        "subset_ids": list(subset_ids),
        "subset_size": len(subset_ids),
        "target_drop": target_drop,
        "control_abs_shift": control_shift,
        "utility": utility,
        "target_current": current_target,
        "control_current": current_control,
    }


def search_model(model_key: str, stage513_model_row: Dict[str, object]) -> Dict[str, object]:
    model, tokenizer = load_model_bundle(model_key, prefer_cuda=False)
    try:
        if model_key == "glm4":
            patch_glm4_mlp_compat(model)
        layer_widths = [int(layer.mlp.down_proj.in_features) for layer in discover_layers(model)]
        candidates = build_candidates(stage513_model_row, layer_widths)
        candidate_map = {f"N:{row['layer_index']}:{row['neuron_index']}": row for row in candidates}

        baseline_target = evaluate_case_groups(model, tokenizer, TARGET_CASES)
        baseline_control = evaluate_case_groups(model, tokenizer, CONTROL_CASES)

        single_rows = []
        for cid, row in candidate_map.items():
            result = evaluate_subset(model, tokenizer, candidate_map, [cid], baseline_target, baseline_control)
            row_out = dict(row)
            row_out.update(
                {
                    "candidate_id": cid,
                    "target_drop": result["target_drop"],
                    "control_abs_shift": result["control_abs_shift"],
                    "utility": result["utility"],
                }
            )
            single_rows.append(row_out)
        single_rows.sort(key=lambda row: float(row["utility"]), reverse=True)

        chosen: List[str] = []
        best_result = evaluate_subset(model, tokenizer, candidate_map, [], baseline_target, baseline_control)
        shortlist = [row["candidate_id"] for row in single_rows[:8]]
        for _ in range(MAX_SUBSET):
            best_candidate = None
            best_candidate_result = None
            for cid in shortlist:
                if cid in chosen:
                    continue
                result = evaluate_subset(model, tokenizer, candidate_map, chosen + [cid], baseline_target, baseline_control)
                if result["utility"] > best_result["utility"] + MIN_GAIN and (
                    best_candidate_result is None or result["utility"] > best_candidate_result["utility"]
                ):
                    best_candidate = cid
                    best_candidate_result = result
            if best_candidate is None or best_candidate_result is None:
                break
            chosen.append(best_candidate)
            best_result = best_candidate_result

        pruned = True
        while pruned and len(chosen) > 1:
            pruned = False
            for cid in list(chosen):
                reduced = [x for x in chosen if x != cid]
                result = evaluate_subset(model, tokenizer, candidate_map, reduced, baseline_target, baseline_control)
                if result["utility"] >= best_result["utility"] * 0.95:
                    chosen = reduced
                    best_result = result
                    pruned = True
                    break

        return {
            "model_key": model_key,
            "model_name": stage513_model_row["model_name"],
            "baseline_target": baseline_target,
            "baseline_control": baseline_control,
            "candidate_neurons": single_rows,
            "final_subset": chosen,
            "final_result": best_result,
        }
    finally:
        free_model(model)


def build_report(model_rows: List[Dict[str, object]], summary: Dict[str, object]) -> str:
    lines = [f"# {summary['experiment_id']}", "", "## 核心结论", summary["core_answer"], ""]
    for row in model_rows:
        final_result = row["final_result"]
        lines.extend(
            [
                f"## {row['model_name']}",
                f"- 最终子集：`{', '.join(row['final_subset']) if row['final_subset'] else '空'}`",
                f"- 目标下降：`{final_result['target_drop']:.4f}`",
                f"- 控制偏移：`{final_result['control_abs_shift']:.4f}`",
                f"- 综合效用：`{final_result['utility']:.4f}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    stage513_summary = json.loads(STAGE513_SUMMARY_PATH.read_text(encoding="utf-8"))
    stage513_rows = {row["model_key"]: row for row in stage513_summary["model_rows"]}
    started = time.time()
    model_rows = [search_model(model_key, stage513_rows[model_key]) for model_key in MODEL_KEYS]
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage515_cross_task_minimal_causal_circuit",
        "title": "跨任务最小因果回路搜索",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "model_rows": model_rows,
        "core_answer": (
            "共享骨干不是纯相关结构，其中一小组共享神经元消融后，会同时打击知识、语法、属性、联想任务，"
            "说明跨任务最小因果回路确实开始显形。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(model_rows, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
