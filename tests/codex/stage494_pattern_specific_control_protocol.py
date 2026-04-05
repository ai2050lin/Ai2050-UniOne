#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

from qwen3_language_shared import remove_hooks
from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage492_chinese_pattern_route_atlas import (
    ensure_dir,
    free_model,
    mean_target_prob,
    register_mixed_ablation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE493_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage493_chinese_language_master_atlas_20260403"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage494_pattern_specific_control_protocol_20260403"
)

CONTROL_CHARS: Dict[str, List[str]] = {
    "apple": ["菜", "子", "园"],
    "grape": ["果", "子", "园"],
    "butterfly": ["花", "虫", "子"],
    "tomorrow": ["地", "人", "时"],
    "evening": ["下", "边", "面"],
    "several_times": ["个", "回", "天"],
    "although": ["后", "者", "且"],
    "finally": ["后", "者", "里"],
    "for_example": ["后", "者", "今"],
}
STRONG_BASELINE_THRESHOLD = 0.01
SPECIFIC_SUPPORT_RATIO = 2.0
SPECIFIC_SUPPORT_MARGIN = 0.005


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"{text!r} 不是单一词元")
    return int(ids[0])


def parse_candidate_id(candidate_id: str) -> Dict[str, int | str]:
    parts = candidate_id.split(":")
    if len(parts) != 3:
        raise RuntimeError(f"无法解析 candidate_id: {candidate_id}")
    kind_tag, layer_str, unit_str = parts
    if kind_tag == "H":
        return {
            "kind": "attention_head",
            "layer_index": int(layer_str),
            "head_index": int(unit_str),
        }
    return {
        "kind": "mlp_neuron",
        "layer_index": int(layer_str),
        "neuron_index": int(unit_str),
    }


def evaluate_pattern_specificity(model, tokenizer, pattern_row: dict) -> dict:
    pattern_key = str(pattern_row["pattern_key"])
    target_char = str(pattern_row["target_char"])
    control_chars = CONTROL_CHARS[pattern_key]
    prefixes = [row["prefix"] for row in pattern_row["selected_contexts"]]
    target_id = token_id(tokenizer, target_char)
    control_ids = {char: token_id(tokenizer, char) for char in control_chars}
    subset = [parse_candidate_id(candidate_id) for candidate_id in pattern_row["mixed_route_subset"]["subset_ids"]]

    baseline_target = mean_target_prob(model, tokenizer, prefixes, target_id)
    baseline_controls = {
        char: mean_target_prob(model, tokenizer, prefixes, control_id)
        for char, control_id in control_ids.items()
    }
    handles = register_mixed_ablation(model, subset)
    try:
        current_target = mean_target_prob(model, tokenizer, prefixes, target_id)
        current_controls = {
            char: mean_target_prob(model, tokenizer, prefixes, control_id)
            for char, control_id in control_ids.items()
        }
    finally:
        remove_hooks(handles)

    target_drop = float(baseline_target["mean_target_prob"] - current_target["mean_target_prob"])
    control_deltas = {
        char: float(current_controls[char]["mean_target_prob"] - baseline_controls[char]["mean_target_prob"])
        for char in control_chars
    }
    control_abs_shift = float(statistics.mean(abs(value) for value in control_deltas.values()))
    specificity_margin = float(target_drop - control_abs_shift)
    specificity_ratio = float(target_drop / max(control_abs_shift, 1e-8))
    supported = bool(
        target_drop >= SPECIFIC_SUPPORT_MARGIN
        and specificity_ratio >= SPECIFIC_SUPPORT_RATIO
    )
    return {
        "pattern_key": pattern_key,
        "family": pattern_row["family"],
        "target_char": target_char,
        "control_chars": control_chars,
        "baseline_target_prob": float(baseline_target["mean_target_prob"]),
        "current_target_prob": float(current_target["mean_target_prob"]),
        "target_drop": target_drop,
        "control_prob_deltas": control_deltas,
        "mean_control_abs_shift": control_abs_shift,
        "specificity_margin": specificity_margin,
        "specificity_ratio": specificity_ratio,
        "supported": supported,
        "subset_ids": list(pattern_row["mixed_route_subset"]["subset_ids"]),
        "prefixes": prefixes,
    }


def build_report(summary: dict) -> str:
    lines = ["# stage494 强控制特异性因果协议", ""]
    lines.append("## 总结")
    lines.append("")
    for model_key, model_row in summary["models"].items():
        lines.append(f"### {model_key}")
        lines.append("")
        lines.append(
            f"- 强基线模式数：`{model_row['aggregate']['strong_pattern_count']}`，"
            f"特异性支持数：`{model_row['aggregate']['supported_count']}`，"
            f"支持率：`{model_row['aggregate']['support_rate']:.4f}`。"
        )
        lines.append(
            f"- 平均目标下降：`{model_row['aggregate']['mean_target_drop']:.4f}`，"
            f"平均控制绝对偏移：`{model_row['aggregate']['mean_control_abs_shift']:.4f}`。"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证中文模式路线是否对目标后字具有特异性")
    parser.add_argument("--prefer-cuda", action="store_true", help="优先使用 CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    atlas = load_json(STAGE493_SUMMARY_PATH)
    summary = {
        "stage": "stage494_pattern_specific_control_protocol",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prefer_cuda": bool(args.prefer_cuda),
        "models": {},
    }
    for model_key in ["qwen3", "deepseek7b"]:
        model = None
        tokenizer = None
        try:
            model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=args.prefer_cuda)
            pattern_rows = atlas["models"][model_key]["patterns"]
            strong_patterns = [
                row
                for row in pattern_rows
                if row.get("status") == "ok"
                and row["pattern_key"] in CONTROL_CHARS
                and float(row["baseline_mean_target_prob"]) >= STRONG_BASELINE_THRESHOLD
                and row["mixed_route_subset"]["subset_ids"]
            ]
            specificity_rows = [evaluate_pattern_specificity(model, tokenizer, row) for row in strong_patterns]
            supported_count = sum(1 for row in specificity_rows if row["supported"])
            aggregate = {
                "strong_pattern_count": len(strong_patterns),
                "supported_count": supported_count,
                "support_rate": float(supported_count / max(1, len(strong_patterns))),
                "mean_target_drop": float(statistics.mean(row["target_drop"] for row in specificity_rows)) if specificity_rows else 0.0,
                "mean_control_abs_shift": float(statistics.mean(row["mean_control_abs_shift"] for row in specificity_rows)) if specificity_rows else 0.0,
                "mean_specificity_margin": float(statistics.mean(row["specificity_margin"] for row in specificity_rows)) if specificity_rows else 0.0,
                "mean_specificity_ratio": float(statistics.mean(row["specificity_ratio"] for row in specificity_rows)) if specificity_rows else 0.0,
            }
            summary["models"][model_key] = {
                "used_cuda": True if args.prefer_cuda else False,
                "rows": specificity_rows,
                "aggregate": aggregate,
            }
        finally:
            if model is not None:
                free_model(model)
    summary["elapsed_seconds"] = float(time.time() - started)
    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "REPORT.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_report(summary), encoding="utf-8")
    print(f"summary written to {summary_path}")
    print(f"report written to {report_path}")


if __name__ == "__main__":
    main()
