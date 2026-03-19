from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from stage56_token_trajectory_equation import (  # noqa: E402
    AXES,
    FullTokenGateCollector,
    aggregate_case_rows,
    build_trace_delta,
    load_model,
    run_prompt_trace,
    select_representative_cases,
    summarize_case_axis,
)

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def natural_prompt_variants(term: str) -> List[Dict[str, str]]:
    return [
        {
            "axis": "control",
            "variant": "natural_plain",
            "prompt": (
                f"At a market stall, someone pointed to {term} and asked what kind of thing it was. "
                f"The answer was that {term} belongs to the category of"
            ),
        },
        {
            "axis": "style",
            "variant": "natural_chat",
            "prompt": (
                f"In a casual conversation, someone pointed to {term} and asked what kind of thing it was. "
                f"The reply was that {term} belongs to the category of"
            ),
        },
        {
            "axis": "logic",
            "variant": "natural_reasoned",
            "prompt": (
                f"Because {term} is one member of a broader class, we can say that {term} belongs to the category of"
            ),
        },
        {
            "axis": "syntax",
            "variant": "natural_structured",
            "prompt": (
                f"The category to which {term} belongs is"
            ),
        },
    ]


def decode_generated_text(model, tok, prompt: str, max_new_tokens: int) -> Dict[str, object]:
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    generation = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tok.eos_token_id,
    )
    prompt_len = int(input_ids.shape[-1])
    total_len = int(generation.shape[-1])
    generated_len = max(0, total_len - prompt_len)
    full_text = tok.decode(generation[0], skip_special_tokens=True)
    generated_only = tok.decode(generation[0][prompt_len:], skip_special_tokens=True).strip()
    return {
        "full_text": full_text,
        "generated_text": generated_only,
        "generated_token_count": generated_len,
    }


def write_report(path: Path, summary: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Stage56 自然生成窗口探针报告",
        "",
        f"- case_count: {summary.get('case_count', 0)}",
        f"- model_count: {summary.get('model_count', 0)}",
        f"- tail_tokens: {summary.get('tail_tokens', 0)}",
        "",
        "## Per Model",
    ]
    for model_id, block in dict(summary.get("per_model", {})).items():
        lines.append(f"- {model_id}: cases={block.get('case_count', 0)}")
        for axis, axis_block in dict(block.get("per_axis", {})).items():
            lines.append(
                f"  - {axis}: hidden={axis_block.get('dominant_hidden_tail_position', '')}, "
                f"mlp={axis_block.get('dominant_mlp_tail_position', '')}, "
                f"layer={axis_block.get('dominant_hidden_layer', '')}, "
                f"mlp_layer={axis_block.get('dominant_mlp_layer', '')}"
            )
    lines.extend(["", "## Sample Generations"])
    for row in rows[:12]:
        lines.append(
            f"- {row['model_id']} / {row['category']} / {row['axis']} / {row['instance_term']}: "
            f"tokens={row.get('generated_token_count', 0)} / suffix={row.get('generated_text', '')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Probe natural generation trajectories before final closure")
    ap.add_argument(
        "--cases-jsonl",
        default=str(
            ROOT
            / "tests"
            / "codex_temp"
            / "stage56_generation_gate_internal_map_all3_12cat_allpairs_20260319_0118"
            / "cases.jsonl"
        ),
    )
    ap.add_argument("--max-cases-per-model", type=int, default=12)
    ap.add_argument("--tail-tokens", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_natural_generation_window_probe_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    cases = read_jsonl(Path(args.cases_jsonl))
    selected_cases = select_representative_cases(cases, max_cases_per_model=int(args.max_cases_per_model))

    grouped_cases: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for case in selected_cases:
        grouped_cases[str(case["model_id"])].append(case)

    output_rows: List[Dict[str, object]] = []
    for model_id, cases_for_model in sorted(grouped_cases.items()):
        model, tok, model_ref = load_model(
            model_id=model_id,
            dtype_name=args.dtype,
            local_files_only=bool(args.local_files_only),
            device=args.device,
        )
        collector = FullTokenGateCollector(model)
        try:
            for case in cases_for_model:
                term = str(case["instance_term"])
                prompts = natural_prompt_variants(term)
                control_prompt = next(item["prompt"] for item in prompts if str(item["axis"]) == "control")
                control_payload = decode_generated_text(
                    model,
                    tok,
                    control_prompt,
                    max_new_tokens=int(args.max_new_tokens),
                )
                control_text = str(control_payload["full_text"])
                control_trace = run_prompt_trace(model, tok, collector, control_text)
                for axis in AXES:
                    variant_prompt = next(item["prompt"] for item in prompts if str(item["axis"]) == axis)
                    variant_payload = decode_generated_text(
                        model,
                        tok,
                        variant_prompt,
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    variant_text = str(variant_payload["full_text"])
                    variant_trace = run_prompt_trace(model, tok, collector, variant_text)
                    delta = build_trace_delta(control_trace, variant_trace, tail_tokens=int(args.tail_tokens))
                    row = summarize_case_axis(case, axis=axis, axis_rows=[delta], tail_tokens=int(args.tail_tokens))
                    row["model_ref"] = model_ref
                    row["prompt"] = variant_prompt
                    row["generated_full_text"] = variant_text
                    row["generated_text"] = str(variant_payload["generated_text"])
                    row["generated_token_count"] = int(variant_payload["generated_token_count"])
                    row["control_generated_full_text"] = control_text
                    row["control_generated_text"] = str(control_payload["generated_text"])
                    row["control_generated_token_count"] = int(control_payload["generated_token_count"])
                    row["trajectory_source"] = "natural_generation"
                    output_rows.append(row)
        finally:
            collector.close()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = aggregate_case_rows(output_rows, tail_tokens=int(args.tail_tokens))
    summary["runtime_sec"] = float(time.time() - started)
    summary["trajectory_source"] = "natural_generation"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "cases.jsonl", output_rows)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", summary, output_rows)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "case_count": summary.get("case_count", 0),
                "model_count": summary.get("model_count", 0),
                "runtime_sec": summary.get("runtime_sec", 0.0),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
