#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from translation_parameter_shared import GateCollector, cosine, load_model_and_tokenizer, run_forward


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage252_translation_component_role_map_20260324"

VARIANTS = {
    "full_instruction": "请把中文翻译为英文：今天天气不错",
    "remove_translate": "请把中文处理为英文：今天天气不错",
    "remove_target": "请把中文翻译：今天天气不错",
    "remove_source": "请翻译为英文：今天天气不错",
}


def build_summary() -> dict:
    model, tokenizer = load_model_and_tokenizer()
    collector = GateCollector(model)
    try:
        collector.reset()
        full_inputs, full_outputs = run_forward(model, tokenizer, VARIANTS["full_instruction"])
        full_gates = collector.get()
        full_last = full_outputs.hidden_states[-1][0, -1, :].detach().cpu()
        rows = []
        for name, text in VARIANTS.items():
            if name == "full_instruction":
                continue
            collector.reset()
            _, outputs = run_forward(model, tokenizer, text)
            gates = collector.get()
            last = outputs.hidden_states[-1][0, -1, :].detach().cpu()
            gate_shift = sum((full_gates[i] - gates[i]).norm().item() for i in range(len(full_gates))) / len(full_gates)
            hidden_similarity = cosine(full_last, last)
            rows.append(
                {
                    "variant_name": name,
                    "gate_shift_mean": gate_shift,
                    "hidden_similarity_to_full": hidden_similarity,
                }
            )
        strongest = max(rows, key=lambda row: row["gate_shift_mean"])
        weakest = min(rows, key=lambda row: row["gate_shift_mean"])
        role_score = (
            max(row["gate_shift_mean"] for row in rows) / max(max(row["gate_shift_mean"] for row in rows), 1e-9)
            + min(1.0, 1.0 - weakest["hidden_similarity_to_full"] + 0.5)
        ) / 2.0
        return {
            "schema_version": "agi_research_result.v1",
            "experiment_id": "stage252_translation_component_role_map",
            "title": "翻译指令组成角色图",
            "status_short": "translation_component_role_map_ready",
            "component_count": len(rows),
            "role_score": role_score,
            "strongest_component_name": strongest["variant_name"],
            "weakest_component_name": weakest["variant_name"],
            "component_rows": rows,
        }
    finally:
        collector.close()


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with (output_dir / "component_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["component_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["component_rows"])
    report = [
        "# Stage252：翻译指令组成角色图",
        "",
        "## 核心结果",
        f"- 组成数量：{summary['component_count']}",
        f"- 角色图总分：{summary['role_score']:.4f}",
        f"- 最大作用组件：{summary['strongest_component_name']}",
        f"- 最小作用组件：{summary['weakest_component_name']}",
    ]
    (output_dir / "STAGE252_TRANSLATION_COMPONENT_ROLE_MAP_REPORT.md").write_text("\n".join(report), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="翻译指令组成角色图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
