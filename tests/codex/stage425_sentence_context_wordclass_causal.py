#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from qwen3_language_shared import PROJECT_ROOT, discover_layers, move_batch_to_model_device
from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, WORD_CLASSES, load_qwen_like_model


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
    / "stage425_sentence_context_wordclass_causal_20260330"
)

CLASS_TO_DIGIT = {
    "noun": "1",
    "adjective": "2",
    "verb": "3",
    "adverb": "4",
    "pronoun": "5",
    "preposition": "6",
}

SENTENCE_CASES: Dict[str, List[Dict[str, str]]] = {
    "noun": [
        {"word": "dolphin", "sentence": "The [dolphin] swam beside the research boat at dawn."},
        {"word": "sailor", "sentence": "The [sailor] checked the ropes before the storm arrived."},
        {"word": "bartender", "sentence": "The [bartender] polished the glasses after closing time."},
        {"word": "kayak", "sentence": "The red [kayak] drifted toward the quiet shore."},
        {"word": "glacier", "sentence": "The vast [glacier] cracked loudly in the afternoon sun."},
        {"word": "yacht", "sentence": "The white [yacht] disappeared beyond the harbor wall."},
        {"word": "pineapple", "sentence": "The ripe [pineapple] rested on the kitchen counter."},
        {"word": "shark", "sentence": "The young [shark] circled the reef before sunset."},
        {"word": "lawyer", "sentence": "The [lawyer] reviewed the contract before the meeting."},
        {"word": "whale", "sentence": "The [whale] surfaced near the ice and sprayed water."},
        {"word": "burger", "sentence": "The hot [burger] cooled slowly on the plate."},
        {"word": "canoe", "sentence": "The old [canoe] scraped against the rocks in the stream."},
    ],
    "adjective": [
        {"word": "yellow", "sentence": "The [yellow] lantern glowed through the fog."},
        {"word": "bright", "sentence": "The [bright] screen lit the entire room."},
        {"word": "dark", "sentence": "The [dark] tunnel echoed with distant steps."},
        {"word": "purple", "sentence": "The [purple] scarf matched her winter coat."},
        {"word": "green", "sentence": "The [green] field stretched toward the hills."},
        {"word": "thick", "sentence": "A [thick] blanket covered the wooden bench."},
        {"word": "small", "sentence": "The [small] drawer held the missing key."},
        {"word": "cold", "sentence": "The [cold] wind pushed against the windows."},
        {"word": "huge", "sentence": "A [huge] shadow moved across the wall."},
        {"word": "warm", "sentence": "The [warm] bread filled the house with a sweet smell."},
        {"word": "dirty", "sentence": "The [dirty] boots stood near the door."},
        {"word": "thin", "sentence": "A [thin] line of smoke rose above the fire."},
    ],
    "verb": [
        {"word": "create", "sentence": "They will [create] a new map before lunch."},
        {"word": "write", "sentence": "Please [write] the answer on the top line."},
        {"word": "take", "sentence": "We should [take] the earlier train tomorrow."},
        {"word": "make", "sentence": "Can you [make] a copy of this page?"}, 
        {"word": "give", "sentence": "Please [give] the key to the guard."},
        {"word": "learn", "sentence": "Children [learn] patterns faster than adults in this game."},
        {"word": "choose", "sentence": "You must [choose] one route before we leave."},
        {"word": "explain", "sentence": "The teacher will [explain] the rule once more."},
        {"word": "bring", "sentence": "Could you [bring] the notebook to my desk?"},
        {"word": "remember", "sentence": "Try to [remember] the code after reading it."},
        {"word": "solve", "sentence": "Engineers [solve] the failure before sunrise."},
        {"word": "apply", "sentence": "Students [apply] the formula to each case."},
    ],
    "adverb": [
        {"word": "quickly", "sentence": "She [quickly] solved the final puzzle."},
        {"word": "usually", "sentence": "The lights [usually] turn on at sunset."},
        {"word": "mostly", "sentence": "The lake is [mostly] calm in early spring."},
        {"word": "rarely", "sentence": "They [rarely] cancel the morning flight."},
        {"word": "often", "sentence": "We [often] meet near the library steps."},
        {"word": "probably", "sentence": "He will [probably] arrive after the rain stops."},
        {"word": "nearly", "sentence": "The runner [nearly] missed the narrow gate."},
        {"word": "partly", "sentence": "The road was [partly] blocked by snow."},
        {"word": "almost", "sentence": "The battery is [almost] empty now."},
        {"word": "really", "sentence": "She [really] likes the old piano."},
        {"word": "slowly", "sentence": "The door [slowly] opened without a sound."},
        {"word": "clearly", "sentence": "The witness [clearly] saw the license number."},
    ],
    "pronoun": [
        {"word": "they", "sentence": "[They] opened the gate after the bell rang."},
        {"word": "we", "sentence": "[We] finished the report before dinner."},
        {"word": "you", "sentence": "[You] left the umbrella by the stairs."},
        {"word": "he", "sentence": "[He] carried the boxes into the hallway."},
        {"word": "she", "sentence": "[She] repaired the clock with steady hands."},
        {"word": "it", "sentence": "[It] rolled under the table during the game."},
        {"word": "this", "sentence": "[This] belongs on the top shelf."},
        {"word": "that", "sentence": "[That] explains the sudden change in tone."},
        {"word": "these", "sentence": "[These] need more time in the oven."},
        {"word": "who", "sentence": "[Who] called the office so late at night?"},
        {"word": "which", "sentence": "[Which] fits the lock on the blue cabinet?"},
        {"word": "whose", "sentence": "[Whose] was found near the fountain?"},
    ],
    "preposition": [
        {"word": "under", "sentence": "The cat slept [under] the table all morning."},
        {"word": "with", "sentence": "She arrived [with] her younger brother at noon."},
        {"word": "between", "sentence": "The cabin stood [between] two dark pine trees."},
        {"word": "during", "sentence": "The crowd remained quiet [during] the ceremony."},
        {"word": "after", "sentence": "We walked home [after] the concert ended."},
        {"word": "through", "sentence": "Sunlight passed [through] the narrow window."},
        {"word": "into", "sentence": "The fox slipped [into] the empty barn."},
        {"word": "before", "sentence": "Check the oil [before] the long drive begins."},
        {"word": "within", "sentence": "The answer was hidden [within] the final paragraph."},
        {"word": "about", "sentence": "They argued [about] the budget for an hour."},
        {"word": "from", "sentence": "A cold draft came [from] the broken vent."},
        {"word": "without", "sentence": "He left [without] his notebook by mistake."},
    ],
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


def prompt_for_case(sentence: str, word: str) -> str:
    plain_sentence = sentence.replace("[", "").replace("]", "")
    return (
        f'Sentence: "{plain_sentence}"\n'
        f'The marked word is "{word}". '
        "Classify its part of speech in this sentence. "
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


def evaluate_case_batch(
    model,
    tokenizer,
    batch_cases: Sequence[Dict[str, str]],
    batch_labels: Sequence[str],
    digit_token_ids: Dict[str, int],
) -> Dict[str, float]:
    prompts = [prompt_for_case(case["sentence"], case["word"]) for case in batch_cases]
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
        "count": float(len(batch_cases)),
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
    sentence_cases: Dict[str, List[Dict[str, str]]],
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
            totals = {
                "count": 0.0,
                "correct_prob_sum": 0.0,
                "correct_logprob_sum": 0.0,
                "accuracy_sum": 0.0,
                "margin_sum": 0.0,
            }
            cases = sentence_cases[class_name]
            for start in range(0, len(cases), batch_size):
                chunk_cases = cases[start : start + batch_size]
                chunk = evaluate_case_batch(
                    model,
                    tokenizer,
                    chunk_cases,
                    [class_name for _ in chunk_cases],
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
    use_cuda: bool,
) -> Dict[str, object]:
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=use_cuda)
    try:
        digit_token_ids = resolve_digit_token_ids(tokenizer)
        baseline = evaluate_condition(
            model,
            tokenizer,
            SENTENCE_CASES,
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
                SENTENCE_CASES,
                digit_token_ids,
                batch_size=batch_size,
                ablate_layers=top_layers,
            )
            bottom_eval = evaluate_condition(
                model,
                tokenizer,
                SENTENCE_CASES,
                digit_token_ids,
                batch_size=batch_size,
                ablate_layers=bottom_layers,
            )
            interventions[class_name] = {
                "top_layers": top_layers,
                "bottom_layers": bottom_layers,
                "top_layer_ablation": summarize_intervention(
                    baseline,
                    top_eval,
                    target_class=class_name,
                    ablated_layers=top_layers,
                ),
                "bottom_layer_ablation": summarize_intervention(
                    baseline,
                    bottom_eval,
                    target_class=class_name,
                    ablated_layers=bottom_layers,
                ),
            }
        return {
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "model_path": str(MODEL_SPECS[model_key]["model_path"]),
            "layer_count": len(discover_layers(model)),
            "digit_token_ids": digit_token_ids,
            "sentence_cases": SENTENCE_CASES,
            "baseline": baseline,
            "interventions": interventions,
        }
    finally:
        free_model(model)


def build_cross_model_summary(model_payloads: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    summary = {}
    for class_name in WORD_CLASSES:
        qwen = model_payloads["qwen3"]["interventions"][class_name]["top_layer_ablation"]
        deepseek = model_payloads["deepseek7b"]["interventions"][class_name]["top_layer_ablation"]
        summary[class_name] = {
            "qwen3_target_prob_delta": float(qwen["target_prob_delta"]),
            "deepseek7b_target_prob_delta": float(deepseek["target_prob_delta"]),
            "qwen3_specificity_gap_prob": float(qwen["specificity_gap_prob"]),
            "deepseek7b_specificity_gap_prob": float(deepseek["specificity_gap_prob"]),
            "deepseek_minus_qwen_target_prob_delta": float(
                deepseek["target_prob_delta"] - qwen["target_prob_delta"]
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
        "- 任务: 在句子中标记目标词，让模型依据上下文判断其词性，再消融 stage423 的顶部层与底部层，比较正确答案概率变化。",
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
            base = model_payload["baseline"]["by_class"][class_name]
            lines.append(
                f"- {class_name}: prob={base['mean_correct_prob']:.4f}, "
                f"acc={base['accuracy']:.4f}, margin={base['mean_margin']:.4f}"
            )
        lines.append("")
        lines.append("### 顶部层消融")
        for class_name in WORD_CLASSES:
            top = model_payload["interventions"][class_name]["top_layer_ablation"]
            lines.append(
                f"- {class_name}: layers={top['ablated_layers']}, "
                f"target_prob_delta={top['target_prob_delta']:+.4f}, "
                f"other_prob_delta_mean={top['other_prob_delta_mean']:+.4f}, "
                f"specificity_gap_prob={top['specificity_gap_prob']:+.4f}"
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
    parser = argparse.ArgumentParser(description="Qwen3 与 DeepSeek7B 句子上下文词性因果实验")
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
            use_cuda=use_cuda,
        )

    elapsed = time.time() - start_time
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage425_sentence_context_wordclass_causal",
        "title": "Qwen3 与 DeepSeek7B 句子上下文词性因果实验",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "used_cuda": use_cuda,
        "batch_size": int(args.batch_size),
        "models": model_payloads,
        "cross_model_summary": build_cross_model_summary(model_payloads),
    }
    output_dir = Path(args.output_dir)
    write_outputs(summary, output_dir)
    print(
        json.dumps(
            {
                "status_short": "stage425_sentence_context_wordclass_causal_ready",
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
