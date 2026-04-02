#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage442_binding_mixed_subcircuit_search import (
    OUTPUT_DIR as STAGE442_OUTPUT_DIR,
    STAGE435_SUMMARY_PATH,
    analyze_family,
    free_model,
    load_json,
    resolve_digit_token_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage443_binding_family_split_probe_20260402"


def run_family(model_key: str, family_name: str, *, batch_size: int, prefer_cuda: bool) -> Dict[str, object]:
    stage435 = load_json(STAGE435_SUMMARY_PATH)
    model_row = next(row for row in stage435["model_results"] if row["model_key"] == model_key)
    model = None
    try:
        model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=prefer_cuda)
        digit_token_ids = resolve_digit_token_ids(tokenizer)
        result = analyze_family(model, tokenizer, digit_token_ids, model_row, family_name, batch_size=batch_size)
        return {
            "family_name": family_name,
            "ok": True,
            "result": result,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "family_name": family_name,
            "ok": False,
            "error": str(exc),
        }
    finally:
        if model is not None:
            free_model(model)


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        f"- model_key: {summary['model_key']}",
        f"- model_name: {summary['model_name']}",
        f"- used_cuda: {summary['used_cuda']}",
        f"- batch_size: {summary['batch_size']}",
        "",
    ]
    for family_row in summary["family_results"]:
        lines.append(f"## {family_row['family_name']}")
        if not family_row["ok"]:
            lines.extend([f"- ok: False", f"- error: {family_row['error']}", ""])
            continue
        result = family_row["result"]
        effect = result["search_state"]["pruned_result"]["effect"]
        lines.extend(
            [
                "- ok: True",
                f"- candidate_probe: {result['candidate_probe']}",
                f"- subset: {result['search_state']['pruned_subset_ids']}",
                f"- binding_drop: {effect['binding_drop']:.6f}",
                f"- heldout_binding_drop: {effect['heldout_binding_drop']:.6f}",
                f"- utility: {effect['utility']:.6f}",
                f"- mixed_support: {result['mixed_support']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绑定分家族独立探针")
    parser.add_argument("--model-key", default="deepseek7b", help="模型键")
    parser.add_argument("--families", default="color,taste,size", help="逗号分隔的家族名")
    parser.add_argument("--batch-size", type=int, default=1, help="前向批大小")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefer_cuda = (not args.cpu) and torch.cuda.is_available()
    family_names = [item.strip() for item in args.families.split(",") if item.strip()]
    start_time = time.time()
    family_results: List[Dict[str, object]] = []
    for family_name in family_names:
        family_results.append(run_family(args.model_key, family_name, batch_size=args.batch_size, prefer_cuda=prefer_cuda))
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage443_binding_family_split_probe",
        "title": "绑定分家族独立探针",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - start_time,
        "used_cuda": bool(prefer_cuda),
        "batch_size": int(args.batch_size),
        "model_key": args.model_key,
        "model_name": MODEL_SPECS[args.model_key]["model_name"],
        "stage442_reference_dir": str(STAGE442_OUTPUT_DIR),
        "family_results": family_results,
    }
    write_outputs(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
