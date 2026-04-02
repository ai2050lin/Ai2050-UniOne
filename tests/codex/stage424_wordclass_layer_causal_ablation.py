#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from qwen3_language_shared import PROJECT_ROOT, discover_layers, move_batch_to_model_device
from stage423_qwen3_deepseek_wordclass_layer_distribution import (
    MODEL_SPECS,
    WORD_CLASSES,
    build_target_rows,
    load_qwen_like_model,
    load_word_rows,
)


STAGE423_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage423_qwen3_deepseek_wordclass_layer_distribution_20260330"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage424_wordclass_layer_causal_ablation_20260330"
)

CLASS_TO_DIGIT = {
    "noun": "1",
    "adjective": "2",
    "verb": "3",
    "adverb": "4",
    "pronoun": "5",
    "preposition": "6",
}
DEFAULT_EVAL_LIMITS = {
    "noun": 24,
    "adjective": 24,
    "verb": 24,
    "adverb": 24,
    "pronoun": 20,
    "preposition": 20,
}


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def free_model(model) -> None:
    try:
        del model
    except UnboundLocalError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def build_eval_rows(model_key: str, limits: Dict[str, int]) -> Dict[str, List[Dict[str, object]]]:
    rows = load_word_rows(MODEL_SPECS[model_key]["rows_path"])
    return {
        class_name: build_target_rows(rows, class_name, limits[class_name])
        for class_name in WORD_CLASSES
    }


def prompt_for_word(word: str) -> str:
    return (
        f'Classify the part of speech of the English word "{word}". '
        "Answer with one digit only: "
        "1 noun 2 adjective 3 verb 4 adverb 5 pronoun 6 preposition.\n"
        "Answer:"
    )


def resolve_digit_token_ids(tokenizer) -> Dict[str, int]:
    token_ids: Dict[str, int] = {}
    for digit in CLASS_TO_DIGIT.values():
        ids = tokenizer(digit, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(f"数字标签 {digit} 不是单 token，当前实验设计无法继续")
        token_ids[digit] = int(ids[0])
    return token_ids


def register_layer_zero_ablation(model, layer_indices: Sequence[int]) -> List[object]:
    handles = []
    layers = discover_layers(model)
    for layer_idx in layer_indices:
        module = layers[layer_idx].mlp.down_proj

        def make_pre_hook():
            def hook(_module, inputs):
                if not inputs:
                    return inputs
                hidden = inputs[0]
                zeroed = torch.zeros_like(hidden)
                if len(inputs) == 1:
                    return (zeroed,)
                return (zeroed, *inputs[1:])

            return hook

        handles.append(module.register_forward_pre_hook(make_pre_hook()))
    return handles


def remove_hooks(handles: Sequence[object]) -> None:
    for handle in handles:
        handle.remove()


def evaluate_word_batches(
    model,
    tokenizer,
    batch_words: Sequence[str],
    batch_labels: Sequence[str],
    digit_token_ids: Dict[str, int],
) -> Dict[str, float]:
    prompts = [prompt_for_word(word) for word in batch_words]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    encoded = move_batch_to_model_device(model, encoded)
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False, return_dict=True).logits[:, -1, :]

    candidate_digits = [CLASS_TO_DIGIT[class_name] for class_name in WORD_CLASSES]
    candidate_ids = torch.tensor(
        [digit_token_ids[digit] for digit in candidate_digits],
        device=logits.device,
        dtype=torch.long,
    )
    option_logits = logits.index_select(dim=1, index=candidate_ids)
    option_logprobs = option_logits.log_softmax(dim=-1)
    correct_indices = torch.tensor(
        [candidate_digits.index(CLASS_TO_DIGIT[label]) for label in batch_labels],
        device=logits.device,
        dtype=torch.long,
    )
    correct_logprobs = option_logprobs.gather(1, correct_indices.unsqueeze(1)).squeeze(1)
    accuracy = (option_logits.argmax(dim=-1) == correct_indices).to(torch.float32)

    masked = option_logits.clone()
    masked.scatter_(1, correct_indices.unsqueeze(1), float("-inf"))
    margins = option_logits.gather(1, correct_indices.unsqueeze(1)).squeeze(1) - masked.max(dim=-1).values
    return {
        "count": int(len(batch_words)),
        "correct_prob_sum": float(correct_logprobs.exp().sum().item()),
        "correct_logprob_sum": float(correct_logprobs.sum().item()),
        "accuracy_sum": float(accuracy.sum().item()),
        "margin_sum": float(margins.sum().item()),
    }


def merge_metric_totals(totals: Dict[str, float], chunk: Dict[str, float]) -> None:
    for key, value in chunk.items():
        totals[key] = totals.get(key, 0.0) + float(value)


def finalize_metric_totals(totals: Dict[str, float]) -> Dict[str, float]:
    count = max(1.0, float(totals["count"]))
    return {
        "count": int(totals["count"]),
        "mean_correct_prob": float(totals["correct_prob_sum"] / count),
        "mean_correct_logprob": float(totals["correct_logprob_sum"] / count),
        "accuracy": float(totals["accuracy_sum"] / count),
        "mean_margin": float(totals["margin_sum"] / count),
    }


def evaluate_condition(
    model,
    tokenizer,
    eval_rows_by_class: Dict[str, List[Dict[str, object]]],
    digit_token_ids: Dict[str, int],
    *,
    batch_size: int,
    ablate_layers: Sequence[int] | None,
) -> Dict[str, object]:
    handles: List[object] = []
    if ablate_layers:
        handles = register_layer_zero_ablation(model, ablate_layers)

    try:
        by_class = {}
        aggregate = {
            "count": 0.0,
            "correct_prob_sum": 0.0,
            "correct_logprob_sum": 0.0,
            "accuracy_sum": 0.0,
            "margin_sum": 0.0,
        }
        for class_name in WORD_CLASSES:
            rows = eval_rows_by_class[class_name]
            totals = {
                "count": 0.0,
                "correct_prob_sum": 0.0,
                "correct_logprob_sum": 0.0,
                "accuracy_sum": 0.0,
                "margin_sum": 0.0,
            }
            for start in range(0, len(rows), batch_size):
                chunk_rows = rows[start : start + batch_size]
                chunk = evaluate_word_batches(
                    model,
                    tokenizer,
                    [str(row["word"]) for row in chunk_rows],
                    [class_name for _ in chunk_rows],
                    digit_token_ids,
                )
                merge_metric_totals(totals, chunk)
                merge_metric_totals(aggregate, chunk)
            by_class[class_name] = finalize_metric_totals(totals)
        return {
            "by_class": by_class,
            "aggregate": finalize_metric_totals(aggregate),
        }
    finally:
        remove_hooks(handles)


def summarize_intervention(
    baseline: Dict[str, object],
    ablated: Dict[str, object],
    *,
    target_class: str,
    ablated_layers: Sequence[int],
) -> Dict[str, object]:
    baseline_by_class = baseline["by_class"]
    ablated_by_class = ablated["by_class"]
    delta_by_class = {}
    other_prob_deltas = []
    other_acc_deltas = []
    for class_name in WORD_CLASSES:
        prob_delta = (
            float(ablated_by_class[class_name]["mean_correct_prob"])
            - float(baseline_by_class[class_name]["mean_correct_prob"])
        )
        acc_delta = float(ablated_by_class[class_name]["accuracy"]) - float(baseline_by_class[class_name]["accuracy"])
        margin_delta = (
            float(ablated_by_class[class_name]["mean_margin"])
            - float(baseline_by_class[class_name]["mean_margin"])
        )
        delta_by_class[class_name] = {
            "correct_prob_delta": prob_delta,
            "accuracy_delta": acc_delta,
            "margin_delta": margin_delta,
        }
        if class_name != target_class:
            other_prob_deltas.append(prob_delta)
            other_acc_deltas.append(acc_delta)

    target_prob_delta = delta_by_class[target_class]["correct_prob_delta"]
    target_acc_delta = delta_by_class[target_class]["accuracy_delta"]
    other_prob_delta_mean = sum(other_prob_deltas) / max(1, len(other_prob_deltas))
    other_acc_delta_mean = sum(other_acc_deltas) / max(1, len(other_acc_deltas))
    return {
        "ablated_layers": [int(x) for x in ablated_layers],
        "target_class": target_class,
        "delta_by_class": delta_by_class,
        "target_prob_delta": float(target_prob_delta),
        "target_accuracy_delta": float(target_acc_delta),
        "other_prob_delta_mean": float(other_prob_delta_mean),
        "other_accuracy_delta_mean": float(other_acc_delta_mean),
        "specificity_gap_prob": float(target_prob_delta - other_prob_delta_mean),
        "specificity_gap_accuracy": float(target_acc_delta - other_acc_delta_mean),
    }


def build_model_summary(
    model_key: str,
    stage423_summary: Dict[str, object],
    *,
    batch_size: int,
    eval_limits: Dict[str, int],
    use_cuda: bool,
) -> Dict[str, object]:
    eval_rows_by_class = build_eval_rows(model_key, eval_limits)
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=use_cuda)
    try:
        digit_token_ids = resolve_digit_token_ids(tokenizer)
        baseline = evaluate_condition(
            model,
            tokenizer,
            eval_rows_by_class,
            digit_token_ids,
            batch_size=batch_size,
            ablate_layers=None,
        )
        model_stage423 = stage423_summary["models"][model_key]
        interventions = {}
        for class_name in WORD_CLASSES:
            class_stage423 = model_stage423["classes"][class_name]
            top_layers = [int(row["layer_index"]) for row in class_stage423["top_layers_by_mass"][:2]]
            bottom_rows = sorted(
                class_stage423["layer_rows"],
                key=lambda row: (row["effective_score_mass_share"], row["effective_count"]),
            )[:2]
            bottom_layers = [int(row["layer_index"]) for row in bottom_rows]

            top_eval = evaluate_condition(
                model,
                tokenizer,
                eval_rows_by_class,
                digit_token_ids,
                batch_size=batch_size,
                ablate_layers=top_layers,
            )
            bottom_eval = evaluate_condition(
                model,
                tokenizer,
                eval_rows_by_class,
                digit_token_ids,
                batch_size=batch_size,
                ablate_layers=bottom_layers,
            )
            top_summary = summarize_intervention(
                baseline,
                top_eval,
                target_class=class_name,
                ablated_layers=top_layers,
            )
            bottom_summary = summarize_intervention(
                baseline,
                bottom_eval,
                target_class=class_name,
                ablated_layers=bottom_layers,
            )
            top_summary["top_minus_bottom_target_prob_delta"] = float(
                top_summary["target_prob_delta"] - bottom_summary["target_prob_delta"]
            )
            top_summary["top_minus_bottom_target_accuracy_delta"] = float(
                top_summary["target_accuracy_delta"] - bottom_summary["target_accuracy_delta"]
            )
            interventions[class_name] = {
                "top_layers": top_layers,
                "bottom_layers": bottom_layers,
                "top_layer_ablation": top_summary,
                "bottom_layer_ablation": bottom_summary,
            }

        return {
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "model_path": str(MODEL_SPECS[model_key]["model_path"]),
            "layer_count": len(discover_layers(model)),
            "digit_token_ids": digit_token_ids,
            "eval_limits": eval_limits,
            "eval_words": {
                class_name: [str(row["word"]) for row in eval_rows_by_class[class_name]]
                for class_name in WORD_CLASSES
            },
            "baseline": baseline,
            "interventions": interventions,
        }
    finally:
        free_model(model)


def build_cross_model_summary(model_payloads: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    summary = {}
    for class_name in WORD_CLASSES:
        qwen_top = model_payloads["qwen3"]["interventions"][class_name]["top_layer_ablation"]
        deepseek_top = model_payloads["deepseek7b"]["interventions"][class_name]["top_layer_ablation"]
        summary[class_name] = {
            "qwen3_target_prob_delta": float(qwen_top["target_prob_delta"]),
            "deepseek7b_target_prob_delta": float(deepseek_top["target_prob_delta"]),
            "qwen3_specificity_gap_prob": float(qwen_top["specificity_gap_prob"]),
            "deepseek7b_specificity_gap_prob": float(deepseek_top["specificity_gap_prob"]),
            "deepseek_minus_qwen_target_prob_delta": float(
                deepseek_top["target_prob_delta"] - qwen_top["target_prob_delta"]
            ),
        }
    return summary


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 实验设置",
        f"- 时间戳: {summary['timestamp_utc']}",
        f"- 是否使用 CUDA: {summary['used_cuda']}",
        f"- 批大小: {summary['batch_size']}",
        "- 任务: 让模型对英文单词做六分类词性判断，再消融对应词类的高分层 MLP 通道，观察正确答案概率是否定向下降。",
        "",
    ]
    for model_key in ["qwen3", "deepseek7b"]:
        model_payload = summary["models"][model_key]
        lines.extend(
            [
                f"## 模型 {model_key}",
                f"- 模型名: {model_payload['model_name']}",
                f"- 层数: {model_payload['layer_count']}",
                "",
                "### 基线",
            ]
        )
        for class_name in WORD_CLASSES:
            row = model_payload["baseline"]["by_class"][class_name]
            lines.append(
                f"- {class_name}: prob={row['mean_correct_prob']:.4f}, "
                f"acc={row['accuracy']:.4f}, margin={row['mean_margin']:.4f}"
            )
        lines.append("")
        lines.append("### 顶层消融结果")
        for class_name in WORD_CLASSES:
            row = model_payload["interventions"][class_name]["top_layer_ablation"]
            lines.append(
                f"- {class_name}: layers={row['ablated_layers']}, "
                f"target_prob_delta={row['target_prob_delta']:+.4f}, "
                f"other_prob_delta_mean={row['other_prob_delta_mean']:+.4f}, "
                f"specificity_gap_prob={row['specificity_gap_prob']:+.4f}"
            )
        lines.append("")
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 与 DeepSeek7B 六词类层因果消融实验")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--batch-size", type=int, default=1, help="前向批大小")
    parser.add_argument("--cpu", action="store_true", help="强制不用 CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    stage423_summary = load_json(STAGE423_SUMMARY_PATH)
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    start_time = time.time()

    model_payloads = {}
    for model_key in ["qwen3", "deepseek7b"]:
        model_payloads[model_key] = build_model_summary(
            model_key,
            stage423_summary,
            batch_size=int(args.batch_size),
            eval_limits=DEFAULT_EVAL_LIMITS,
            use_cuda=use_cuda,
        )

    elapsed = time.time() - start_time
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage424_wordclass_layer_causal_ablation",
        "title": "Qwen3 与 DeepSeek7B 六词类层因果消融实验",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "used_cuda": use_cuda,
        "batch_size": int(args.batch_size),
        "eval_limits": DEFAULT_EVAL_LIMITS,
        "models": model_payloads,
        "cross_model_summary": build_cross_model_summary(model_payloads),
    }
    output_dir = Path(args.output_dir)
    write_outputs(summary, output_dir)
    print(
        json.dumps(
            {
                "status_short": "stage424_wordclass_layer_causal_ablation_ready",
                "output_dir": str(output_dir),
                "used_cuda": use_cuda,
                "elapsed_seconds": elapsed,
                "qwen3_pronoun_top_delta": model_payloads["qwen3"]["interventions"]["pronoun"]["top_layer_ablation"][
                    "target_prob_delta"
                ],
                "deepseek7b_pronoun_top_delta": model_payloads["deepseek7b"]["interventions"]["pronoun"][
                    "top_layer_ablation"
                ]["target_prob_delta"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
