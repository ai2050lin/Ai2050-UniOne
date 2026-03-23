#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage122: adverb（副词）-context（上下文）-route（选路）偏移探针。

目标：
1. 在真实 GPT-2 前向中测试副词插入是否会系统性改变动词位置的选路代理量。
2. 用 adjective（形容词）修饰名词作为控制组，区分“普通多加一个修饰词”和“副词选路效应”。
3. 为 q / b / g（条件门控场 / 上下文偏置 / 门控路由）动态研究提供第一份上下文证据。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage119_gpt2_embedding_full_vocab_scan import MODEL_PATH
from stage119_gpt2_embedding_full_vocab_scan import OUTPUT_DIR as STAGE119_OUTPUT_DIR
from stage119_gpt2_embedding_full_vocab_scan import load_embedding_weight
from stage121_adverb_gate_bridge_probe import ensure_stage119_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage122_adverb_context_route_shift_probe_20260323"

ADVERBS = ["actually", "eventually", "simply", "always", "probably", "therefore", "also", "finally"]
VERBS = ["solve", "build", "change", "explain", "compare", "move"]
TEMPLATES = [
    {
        "base": "They will {verb} the system today.",
        "adverb": "They will {adverb} {verb} the system today.",
        "adjective": "They will {verb} the stable system today.",
        "noun_word": "system",
        "adjective_word": "stable",
    },
    {
        "base": "We can {verb} the model now.",
        "adverb": "We can {adverb} {verb} the model now.",
        "adjective": "We can {verb} the formal model now.",
        "noun_word": "model",
        "adjective_word": "formal",
    },
    {
        "base": "Researchers might {verb} the process soon.",
        "adverb": "Researchers might {adverb} {verb} the process soon.",
        "adjective": "Researchers might {verb} the complex process soon.",
        "noun_word": "process",
        "adjective_word": "complex",
    },
]


def load_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
        attn_implementation="eager",
    )
    model.eval()
    model.config.output_attentions = True
    return model, tokenizer


def l2_normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.linalg.norm(vec).clamp_min(1e-8)


def build_route_prototypes(rows: Sequence[Dict[str, object]], embed_weight: torch.Tensor) -> Dict[str, torch.Tensor]:
    def centroid(predicate) -> torch.Tensor:
        indices = [int(row["token_id"]) for row in rows if predicate(row)]
        if not indices:
            raise RuntimeError("构建 route prototype（选路原型）失败：样本为空")
        return l2_normalize(embed_weight[indices].float().mean(dim=0))

    return {
        "verb": centroid(
            lambda row: row["lexical_type"] == "verb"
            and row["band"] == "macro"
            and row["group"] == "macro_action"
            and float(row["lexical_type_score"]) >= 0.55
        ),
        "function": centroid(
            lambda row: row["lexical_type"] == "function"
            and row["band"] == "macro"
            and float(row["lexical_type_score"]) >= 0.55
        ),
        "noun": centroid(
            lambda row: row["lexical_type"] == "noun"
            and row["band"] == "meso"
            and float(row["lexical_type_score"]) >= 0.55
        ),
        "adjective": centroid(
            lambda row: row["lexical_type"] == "adjective"
            and row["band"] == "micro"
            and float(row["lexical_type_score"]) >= 0.55
        ),
    }


def route_score(vec: torch.Tensor, prototypes: Dict[str, torch.Tensor]) -> float:
    unit = l2_normalize(vec.float())
    route = (torch.dot(unit, prototypes["verb"]) + torch.dot(unit, prototypes["function"])) / 2.0
    content = (torch.dot(unit, prototypes["noun"]) + torch.dot(unit, prototypes["adjective"])) / 2.0
    return float((route - content).item())


def ids_for_word(tokenizer: AutoTokenizer, word: str) -> List[int]:
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if ids:
        return list(ids)
    return list(tokenizer.encode(word, add_special_tokens=False))


def find_subsequence(sequence: Sequence[int], target: Sequence[int]) -> int | None:
    if not target:
        return None
    for idx in range(len(sequence) - len(target) + 1):
        if list(sequence[idx : idx + len(target)]) == list(target):
            return idx
    return None


def build_case_bundle() -> List[Dict[str, str]]:
    cases = []
    for template in TEMPLATES:
        for verb in VERBS:
            for adverb in ADVERBS:
                cases.append(
                    {
                        "verb": verb,
                        "adverb": adverb,
                        "adjective": template["adjective_word"],
                        "noun_word": template["noun_word"],
                        "base_prompt": template["base"].format(verb=verb),
                        "adverb_prompt": template["adverb"].format(adverb=adverb, verb=verb),
                        "adjective_prompt": template["adjective"].format(verb=verb),
                    }
                )
    return cases


def build_prompt_metrics(
    outputs,
    tokenizer,
    input_ids: torch.Tensor,
    prompt_idx: int,
    prompt_kind: str,
    prototypes: Dict[str, torch.Tensor],
    verb_ids: Sequence[int],
    adverb_ids: Sequence[int],
    adjective_ids: Sequence[int],
) -> Dict[str, object]:
    seq = input_ids[prompt_idx].tolist()
    if tokenizer.pad_token_id in seq:
        seq = seq[: seq.index(tokenizer.pad_token_id)]

    verb_pos = find_subsequence(seq, verb_ids)
    if verb_pos is None:
        raise RuntimeError("未定位到 verb（动词）位置")

    modifier_pos = None
    if prompt_kind == "adverb":
        modifier_pos = find_subsequence(seq, adverb_ids)
    elif prompt_kind == "adjective":
        modifier_pos = find_subsequence(seq, adjective_ids)

    verb_route_by_layer: List[float] = []
    last_route_by_layer: List[float] = []
    modifier_attention_by_layer: List[float] = []
    last_pos = len(seq) - 1

    for layer_idx, hidden_state in enumerate(outputs.hidden_states[1:]):
        verb_route_by_layer.append(route_score(hidden_state[prompt_idx, verb_pos, :], prototypes))
        last_route_by_layer.append(route_score(hidden_state[prompt_idx, last_pos, :], prototypes))
        if modifier_pos is not None:
            attn = outputs.attentions[layer_idx][prompt_idx]
            verb_attn = float(attn[:, verb_pos, modifier_pos].mean().item())
            last_attn = float(attn[:, last_pos, modifier_pos].mean().item())
            modifier_attention_by_layer.append((verb_attn + last_attn) / 2.0)

    return {
        "kind": prompt_kind,
        "verb_route_by_layer": [float(x) for x in verb_route_by_layer],
        "last_route_by_layer": [float(x) for x in last_route_by_layer],
        "modifier_attention_by_layer": [float(x) for x in modifier_attention_by_layer],
        "verb_route_mean": float(sum(verb_route_by_layer) / len(verb_route_by_layer)),
        "last_route_mean": float(sum(last_route_by_layer) / len(last_route_by_layer)),
        "modifier_attention_mean": float(sum(modifier_attention_by_layer) / max(1, len(modifier_attention_by_layer))),
    }


def analyze_case(
    model,
    tokenizer,
    prototypes: Dict[str, torch.Tensor],
    case: Dict[str, str],
) -> Dict[str, object]:
    prompts = [case["base_prompt"], case["adverb_prompt"], case["adjective_prompt"]]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    with torch.inference_mode():
        outputs = model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    input_ids = encoded["input_ids"]
    verb_ids = ids_for_word(tokenizer, case["verb"])
    adverb_ids = ids_for_word(tokenizer, case["adverb"])
    adjective_ids = ids_for_word(tokenizer, case["adjective"])

    prompt_metrics = [
        build_prompt_metrics(
            outputs=outputs,
            tokenizer=tokenizer,
            input_ids=input_ids,
            prompt_idx=prompt_idx,
            prompt_kind=prompt_kind,
            prototypes=prototypes,
            verb_ids=verb_ids,
            adverb_ids=adverb_ids,
            adjective_ids=adjective_ids,
        )
        for prompt_idx, prompt_kind in enumerate(["base", "adverb", "adjective"])
    ]

    base_metrics, adverb_metrics, adjective_metrics = prompt_metrics

    verb_route_delta_by_layer = [
        float(adv - base)
        for adv, base in zip(adverb_metrics["verb_route_by_layer"], base_metrics["verb_route_by_layer"])
    ]
    adjective_verb_delta_by_layer = [
        float(adj - base)
        for adj, base in zip(adjective_metrics["verb_route_by_layer"], base_metrics["verb_route_by_layer"])
    ]
    verb_route_advantage_by_layer = [
        float(adv - adj)
        for adv, adj in zip(adverb_metrics["verb_route_by_layer"], adjective_metrics["verb_route_by_layer"])
    ]
    last_route_advantage_by_layer = [
        float(adv - adj)
        for adv, adj in zip(adverb_metrics["last_route_by_layer"], adjective_metrics["last_route_by_layer"])
    ]
    modifier_attention_advantage_by_layer = [
        float(adv - adj)
        for adv, adj in zip(
            adverb_metrics["modifier_attention_by_layer"],
            adjective_metrics["modifier_attention_by_layer"],
        )
    ]

    adverb_verb_peak = max(verb_route_delta_by_layer)
    adjective_verb_peak = max(adjective_verb_delta_by_layer)
    adverb_last_peak = max(
        adv - base
        for adv, base in zip(adverb_metrics["last_route_by_layer"], base_metrics["last_route_by_layer"])
    )
    adjective_last_peak = max(
        adj - base
        for adj, base in zip(adjective_metrics["last_route_by_layer"], base_metrics["last_route_by_layer"])
    )
    peak_layer_index = max(
        range(len(verb_route_advantage_by_layer)),
        key=lambda idx: verb_route_advantage_by_layer[idx],
    )

    return {
        "verb": case["verb"],
        "adverb": case["adverb"],
        "adjective": case["adjective"],
        "noun_word": case["noun_word"],
        "base_prompt": case["base_prompt"],
        "adverb_prompt": case["adverb_prompt"],
        "adjective_prompt": case["adjective_prompt"],
        "adverb_verb_route_delta": adverb_metrics["verb_route_mean"] - base_metrics["verb_route_mean"],
        "adjective_verb_route_delta": adjective_metrics["verb_route_mean"] - base_metrics["verb_route_mean"],
        "adverb_last_route_delta": adverb_metrics["last_route_mean"] - base_metrics["last_route_mean"],
        "adjective_last_route_delta": adjective_metrics["last_route_mean"] - base_metrics["last_route_mean"],
        "verb_route_advantage": adverb_metrics["verb_route_mean"] - adjective_metrics["verb_route_mean"],
        "last_route_advantage": adverb_metrics["last_route_mean"] - adjective_metrics["last_route_mean"],
        "adverb_modifier_attention_mean": adverb_metrics["modifier_attention_mean"],
        "adjective_modifier_attention_mean": adjective_metrics["modifier_attention_mean"],
        "modifier_attention_advantage": (
            adverb_metrics["modifier_attention_mean"] - adjective_metrics["modifier_attention_mean"]
        ),
        "verb_route_delta_by_layer": verb_route_delta_by_layer,
        "adjective_verb_delta_by_layer": adjective_verb_delta_by_layer,
        "verb_route_advantage_by_layer": verb_route_advantage_by_layer,
        "last_route_advantage_by_layer": last_route_advantage_by_layer,
        "modifier_attention_advantage_by_layer": modifier_attention_advantage_by_layer,
        "adverb_verb_route_peak_delta": float(adverb_verb_peak),
        "adjective_verb_route_peak_delta": float(adjective_verb_peak),
        "verb_route_peak_advantage": float(adverb_verb_peak - adjective_verb_peak),
        "adverb_last_route_peak_delta": float(adverb_last_peak),
        "adjective_last_route_peak_delta": float(adjective_last_peak),
        "last_route_peak_advantage": float(adverb_last_peak - adjective_last_peak),
        "peak_layer_index": int(peak_layer_index),
    }


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_summary(case_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    count = max(1, len(case_rows))
    adv_verb_mean = sum(float(row["adverb_verb_route_delta"]) for row in case_rows) / count
    adj_verb_mean = sum(float(row["adjective_verb_route_delta"]) for row in case_rows) / count
    adv_last_mean = sum(float(row["adverb_last_route_delta"]) for row in case_rows) / count
    adj_last_mean = sum(float(row["adjective_last_route_delta"]) for row in case_rows) / count
    adv_peak_mean = sum(float(row["adverb_verb_route_peak_delta"]) for row in case_rows) / count
    adj_peak_mean = sum(float(row["adjective_verb_route_peak_delta"]) for row in case_rows) / count
    attn_adv_mean = sum(float(row["adverb_modifier_attention_mean"]) for row in case_rows) / count
    attn_adj_mean = sum(float(row["adjective_modifier_attention_mean"]) for row in case_rows) / count
    route_advantage = adv_verb_mean - adj_verb_mean
    last_advantage = adv_last_mean - adj_last_mean
    peak_advantage = adv_peak_mean - adj_peak_mean
    attention_advantage = attn_adv_mean - attn_adj_mean
    positive_rate = sum(1 for row in case_rows if float(row["verb_route_advantage"]) > 0.0) / count
    positive_peak_rate = sum(1 for row in case_rows if float(row["verb_route_peak_advantage"]) > 0.0) / count

    dynamic_score = (
        0.30 * clamp01(route_advantage / 0.004)
        + 0.25 * clamp01(peak_advantage / 0.008)
        + 0.25 * positive_peak_rate
        + 0.20 * clamp01(attention_advantage / 0.06)
    )

    best_cases = sorted(case_rows, key=lambda row: float(row["verb_route_peak_advantage"]), reverse=True)[:20]
    strongest_attention_cases = sorted(
        case_rows,
        key=lambda row: float(row["modifier_attention_advantage"]),
        reverse=True,
    )[:20]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage122_adverb_context_route_shift_probe",
        "title": "Adverb 上下文选路偏移探针",
        "status_short": "gpt2_adverb_context_route_shift_ready",
        "model_name": "gpt2",
        "model_path": str(MODEL_PATH),
        "case_count": len(case_rows),
        "adverb_verb_route_delta_mean": float(adv_verb_mean),
        "adjective_verb_route_delta_mean": float(adj_verb_mean),
        "verb_route_advantage_mean": float(route_advantage),
        "adverb_last_route_delta_mean": float(adv_last_mean),
        "adjective_last_route_delta_mean": float(adj_last_mean),
        "last_route_advantage_mean": float(last_advantage),
        "adverb_verb_route_peak_delta_mean": float(adv_peak_mean),
        "adjective_verb_route_peak_delta_mean": float(adj_peak_mean),
        "verb_route_peak_advantage_mean": float(peak_advantage),
        "adverb_modifier_attention_mean": float(attn_adv_mean),
        "adjective_modifier_attention_mean": float(attn_adj_mean),
        "modifier_attention_advantage_mean": float(attention_advantage),
        "positive_route_shift_case_rate": float(positive_rate),
        "positive_peak_route_shift_case_rate": float(positive_peak_rate),
        "adverb_context_route_shift_score": float(dynamic_score),
        "best_route_shift_cases": best_cases,
        "strongest_attention_cases": strongest_attention_cases,
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage122: Adverb 上下文选路偏移探针",
        "",
        "## 核心结果",
        f"- 样本对数量: {summary['case_count']}",
        f"- adverb（副词）动词位选路偏移均值: {summary['adverb_verb_route_delta_mean']:.6f}",
        f"- adjective（形容词）控制组动词位选路偏移均值: {summary['adjective_verb_route_delta_mean']:.6f}",
        f"- 动词位优势: {summary['verb_route_advantage_mean']:.6f}",
        f"- 动词位峰值优势: {summary['verb_route_peak_advantage_mean']:.6f}",
        f"- 修饰词注意力优势: {summary['modifier_attention_advantage_mean']:.6f}",
        f"- 正向案例比率: {summary['positive_route_shift_case_rate']:.4f}",
        f"- 峰值正向案例比率: {summary['positive_peak_route_shift_case_rate']:.4f}",
        f"- 动态选路偏移分数: {summary['adverb_context_route_shift_score']:.4f}",
        "",
        "## 解读",
        "- 如果副词组在动词位的 route（选路）偏移高于形容词控制组，说明副词并不只是普通修饰词。",
        "- 如果修饰词注意力优势也为正，说明模型确实在动态处理中吸收了副词位置信号。",
        "",
        "## 最强案例",
    ]

    for row in summary["best_route_shift_cases"][:12]:
        lines.append(
            "- "
            f"{row['adverb']} / {row['verb']}: "
            f"peak_adv={row['verb_route_peak_advantage']:.6f}, "
            f"attn_adv={row['modifier_attention_advantage']:.6f}"
        )

    lines.extend(
        [
            "",
            "## 理论提示",
            "- 这不是最终的 q / b / g 动态闭合，但已经给出第一份“副词插入会改动上下文选路代理量”的前向证据。",
            "- 下一步应把这一效应推进到层级定位与词位定位，看看偏移主要发生在早层、中层还是后层。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    summary: Dict[str, object],
    case_rows: Sequence[Dict[str, object]],
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "STAGE122_ADVERB_CONTEXT_ROUTE_SHIFT_PROBE_REPORT.md"
    cases_path = output_dir / "best_route_shift_cases.json"
    trace_path = output_dir / "layer_trace_rows.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report_path.write_text(build_report(summary), encoding="utf-8-sig")
    cases_path.write_text(
        json.dumps(summary["best_route_shift_cases"], ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    trace_rows = [
        {
            "verb": row["verb"],
            "adverb": row["adverb"],
            "adjective": row["adjective"],
            "noun_word": row["noun_word"],
            "base_prompt": row["base_prompt"],
            "adverb_prompt": row["adverb_prompt"],
            "adjective_prompt": row["adjective_prompt"],
            "verb_route_delta_by_layer": row["verb_route_delta_by_layer"],
            "adjective_verb_delta_by_layer": row["adjective_verb_delta_by_layer"],
            "verb_route_advantage_by_layer": row["verb_route_advantage_by_layer"],
            "last_route_advantage_by_layer": row["last_route_advantage_by_layer"],
            "modifier_attention_advantage_by_layer": row["modifier_attention_advantage_by_layer"],
            "peak_layer_index": row["peak_layer_index"],
            "verb_route_peak_advantage": row["verb_route_peak_advantage"],
            "modifier_attention_advantage": row["modifier_attention_advantage"],
        }
        for row in case_rows
    ]
    trace_path.write_text(json.dumps(trace_rows, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    return {
        "summary": summary_path,
        "report": report_path,
        "best_cases": cases_path,
        "layer_traces": trace_path,
    }


def run_analysis(
    *,
    input_dir: Path = STAGE119_OUTPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, object]:
    _stage119_summary, rows = ensure_stage119_rows(input_dir)
    embed_weight = load_embedding_weight()
    prototypes = build_route_prototypes(rows, embed_weight)
    model, tokenizer = load_model()

    cases = build_case_bundle()
    case_rows = [analyze_case(model, tokenizer, prototypes, case) for case in cases]
    summary = build_summary(case_rows)
    write_outputs(summary, case_rows, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Adverb 上下文选路偏移探针")
    parser.add_argument("--input-dir", default=str(STAGE119_OUTPUT_DIR), help="Stage119 输出目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage122 输出目录")
    args = parser.parse_args()

    summary = run_analysis(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "adverb_context_route_shift_score": summary["adverb_context_route_shift_score"],
                "verb_route_advantage_mean": summary["verb_route_advantage_mean"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
