#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage442_binding_mixed_subcircuit_search import (
    BINDING_TASKS,
    OUTPUT_DIR as STAGE442_OUTPUT_DIR,
    STAGE435_SUMMARY_PATH,
    build_candidate_rows,
    evaluate_groups,
    evaluate_subset,
    flush_cuda,
    free_model,
    load_json,
    resolve_digit_token_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage444_qwen3_binding_failure_boundary_20260402"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report_lines = [
        f"# {summary['experiment_id']}",
        "",
        f"- model_key: {summary['model_key']}",
        f"- model_name: {summary['model_name']}",
        f"- used_cuda: {summary['used_cuda']}",
        f"- batch_size: {summary['batch_size']}",
        f"- max_heads: {summary['max_heads']}",
        f"- max_neurons: {summary['max_neurons']}",
        "",
    ]
    for family_row in summary["family_results"]:
        report_lines.append(f"## {family_row['family_name']}")
        report_lines.append(f"- baseline_search_ok: {family_row['baseline_search_ok']}")
        if family_row.get("baseline_search_error"):
            report_lines.append(f"- baseline_search_error: {family_row['baseline_search_error']}")
        report_lines.append(f"- baseline_heldout_ok: {family_row['baseline_heldout_ok']}")
        if family_row.get("baseline_heldout_error"):
            report_lines.append(f"- baseline_heldout_error: {family_row['baseline_heldout_error']}")
        report_lines.append(f"- head_probe: {family_row['head_probe']}")
        report_lines.append(f"- neuron_probe: {family_row['neuron_probe']}")
        report_lines.append("")
    (output_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8-sig")


def probe_candidates(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    search_groups: Dict[str, List[Dict[str, object]]],
    heldout_cases: Sequence[Dict[str, object]],
    baseline_search: Dict[str, object],
    baseline_heldout: Dict[str, float],
    candidates: Sequence[Dict[str, object]],
    *,
    batch_size: int,
) -> Dict[str, object]:
    success_ids: List[str] = []
    first_error = None
    for row in candidates:
        if row["kind"] == "attention_head":
            candidate_label = f"H:{row['layer_index']}:{row['head_index']}"
        else:
            candidate_label = f"N:{row['layer_index']}:{row['neuron_index']}"
        try:
            evaluate_subset(
                model,
                tokenizer,
                digit_token_ids,
                search_groups,
                heldout_cases,
                baseline_search,
                baseline_heldout,
                [row],
                batch_size=batch_size,
            )
            flush_cuda()
            success_ids.append(candidate_label)
        except Exception as exc:  # noqa: BLE001
            flush_cuda()
            first_error = {"candidate": candidate_label, "error": str(exc)}
            break
    return {
        "tested_count": len(candidates),
        "success_count": len(success_ids),
        "success_ids": success_ids,
        "first_error": first_error,
    }


def analyze_family_boundary(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    model_row: Dict[str, object],
    family_name: str,
    *,
    batch_size: int,
    max_heads: int,
    max_neurons: int,
) -> Dict[str, object]:
    task = BINDING_TASKS[family_name]
    search_groups = {key: list(value) for key, value in task["search_cases"].items()}
    heldout_cases = list(task["heldout_binding_cases"])
    out: Dict[str, object] = {
        "family_name": family_name,
        "baseline_search_ok": False,
        "baseline_heldout_ok": False,
        "baseline_search_error": None,
        "baseline_heldout_error": None,
        "head_probe": None,
        "neuron_probe": None,
    }

    baseline_search = None
    baseline_heldout = None
    try:
        baseline_search = evaluate_groups(model, tokenizer, digit_token_ids, search_groups, batch_size=batch_size)
        flush_cuda()
        out["baseline_search_ok"] = True
    except Exception as exc:  # noqa: BLE001
        flush_cuda()
        out["baseline_search_error"] = str(exc)
        return out

    try:
        baseline_heldout = evaluate_groups(
            model,
            tokenizer,
            digit_token_ids,
            {"binding": heldout_cases},
            batch_size=batch_size,
        )["by_group"]["binding"]
        flush_cuda()
        out["baseline_heldout_ok"] = True
    except Exception as exc:  # noqa: BLE001
        flush_cuda()
        out["baseline_heldout_error"] = str(exc)
        return out

    raw_candidates = build_candidate_rows(model, model_row, family_name)
    head_candidates = [row for row in raw_candidates if row["kind"] == "attention_head"][:max_heads]
    neuron_candidates = [row for row in raw_candidates if row["kind"] == "mlp_neuron"][:max_neurons]
    out["head_probe"] = probe_candidates(
        model,
        tokenizer,
        digit_token_ids,
        search_groups,
        heldout_cases,
        baseline_search,
        baseline_heldout,
        head_candidates,
        batch_size=batch_size,
    )
    out["neuron_probe"] = probe_candidates(
        model,
        tokenizer,
        digit_token_ids,
        search_groups,
        heldout_cases,
        baseline_search,
        baseline_heldout,
        neuron_candidates,
        batch_size=batch_size,
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 绑定搜索失败边界诊断")
    parser.add_argument("--model-key", default="qwen3", help="模型键")
    parser.add_argument("--families", default="color,taste,size", help="逗号分隔的家族名")
    parser.add_argument("--batch-size", type=int, default=1, help="前向批大小")
    parser.add_argument("--max-heads", type=int, default=6, help="每个家族最多探测多少个头")
    parser.add_argument("--max-neurons", type=int, default=6, help="每个家族最多探测多少个神经元")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefer_cuda = (not args.cpu) and torch.cuda.is_available()
    family_names = [item.strip() for item in args.families.split(",") if item.strip()]
    stage435 = load_json(STAGE435_SUMMARY_PATH)
    model_row = next(row for row in stage435["model_results"] if row["model_key"] == args.model_key)

    model = None
    start_time = time.time()
    try:
        model, tokenizer = load_qwen_like_model(MODEL_SPECS[args.model_key]["model_path"], prefer_cuda=prefer_cuda)
        digit_token_ids = resolve_digit_token_ids(tokenizer)
        family_results = []
        for family_name in family_names:
            family_results.append(
                analyze_family_boundary(
                    model,
                    tokenizer,
                    digit_token_ids,
                    model_row,
                    family_name,
                    batch_size=args.batch_size,
                    max_heads=args.max_heads,
                    max_neurons=args.max_neurons,
                )
            )
        summary = {
            "schema_version": "agi_research_result.v1",
            "experiment_id": "stage444_qwen3_binding_failure_boundary",
            "title": "Qwen3 绑定搜索失败边界诊断",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": time.time() - start_time,
            "used_cuda": bool(prefer_cuda),
            "batch_size": int(args.batch_size),
            "max_heads": int(args.max_heads),
            "max_neurons": int(args.max_neurons),
            "model_key": args.model_key,
            "model_name": MODEL_SPECS[args.model_key]["model_name"],
            "stage442_reference_dir": str(STAGE442_OUTPUT_DIR),
            "family_results": family_results,
        }
        write_outputs(summary, Path(args.output_dir))
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
