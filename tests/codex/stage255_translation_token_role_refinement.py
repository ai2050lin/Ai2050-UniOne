#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from translation_parameter_shared import GateCollector, cosine, load_model_and_tokenizer, run_forward


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage255_translation_token_role_refinement_20260324"

BASE_PROMPT = "请把中文翻译为英文，只输出翻译结果：今天天气不错"
VARIANTS = {
    "remove_translate_word": "请把中文处理为英文，只输出结果：今天天气不错",
    "remove_target_word": "请把中文翻译，只输出翻译结果：今天天气不错",
    "remove_source_word": "请翻译为英文，只输出翻译结果：今天天气不错",
    "remove_output_constraint": "请把中文翻译为英文：今天天气不错",
}


def build_summary() -> dict:
    model, tokenizer = load_model_and_tokenizer()
    collector = GateCollector(model)
    try:
        collector.reset()
        _, base_outputs = run_forward(model, tokenizer, BASE_PROMPT)
        base_gates = collector.get()
        base_last = base_outputs.hidden_states[-1][0, -1, :].detach().cpu()
        rows = []
        for name, prompt in VARIANTS.items():
            collector.reset()
            _, outputs = run_forward(model, tokenizer, prompt)
            gates = collector.get()
            last = outputs.hidden_states[-1][0, -1, :].detach().cpu()
            gate_shift = sum((base_gates[i] - gates[i]).norm().item() for i in range(len(base_gates))) / len(base_gates)
            hidden_similarity = cosine(base_last, last)
            rows.append(
                {
                    "variant_name": name,
                    "gate_shift_mean": gate_shift,
                    "hidden_similarity_to_base": hidden_similarity,
                }
            )
    finally:
        collector.close()

    strongest = max(rows, key=lambda row: row["gate_shift_mean"])
    weakest = min(rows, key=lambda row: row["gate_shift_mean"])
    role_score = (
        max(row["gate_shift_mean"] for row in rows) / max(max(row["gate_shift_mean"] for row in rows), 1e-9)
        + sum(1.0 - row["hidden_similarity_to_base"] for row in rows) / len(rows)
    ) / 2.0
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage255_translation_token_role_refinement",
        "title": "翻译指令词角色细化图",
        "status_short": "translation_token_role_refinement_ready",
        "variant_count": len(rows),
        "role_score": role_score,
        "strongest_role_name": strongest["variant_name"],
        "weakest_role_name": weakest["variant_name"],
        "variant_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with (output_dir / "variant_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["variant_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["variant_rows"])
    report = [
        "# Stage255：翻译指令词角色细化图",
        "",
        "## 核心结果",
        f"- 变体数量：{summary['variant_count']}",
        f"- 角色图总分：{summary['role_score']:.4f}",
        f"- 最强词角色：{summary['strongest_role_name']}",
        f"- 最弱词角色：{summary['weakest_role_name']}",
    ]
    (output_dir / "STAGE255_TRANSLATION_TOKEN_ROLE_REFINEMENT_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8-sig"
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="翻译指令词角色细化图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
