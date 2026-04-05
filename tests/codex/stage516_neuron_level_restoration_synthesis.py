#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage516_neuron_level_restoration_synthesis_20260404"
)
STAGE514_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage514_multi_family_cross_task_core_protocol_20260404"
    / "summary.json"
)
STAGE515_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage515_cross_task_minimal_causal_circuit_20260404"
    / "summary.json"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    start = time.time()

    stage514 = load_json(STAGE514_PATH)
    stage515 = load_json(STAGE515_PATH)

    family_rows = {}
    for row in stage514["model_rows"]:
        family_rows[row["model_key"]] = row["aggregate"]

    circuit_rows = {}
    for row in stage515["model_rows"]:
        circuit_rows[row["model_key"]] = {
            "final_subset": row["final_subset"],
            "target_drop": row["final_result"]["target_drop"],
            "control_abs_shift": row["final_result"]["control_abs_shift"],
            "utility": row["final_result"]["utility"],
            "baseline_target": row["baseline_target"]["mean_correct_prob"],
            "baseline_control": row["baseline_control"]["mean_correct_prob"],
        }

    model_rows = []
    for model_key, agg in family_rows.items():
        row = {
            "model_key": model_key,
            "family_count": agg["family_count"],
            "mean_core_shared_all": agg["mean_core_shared_all"],
            "mean_core_shared_4plus": agg["mean_core_shared_4plus"],
            "mean_core_shared_ratio": agg["mean_core_shared_ratio"],
            "mean_knowledge_syntax_shared": agg["mean_knowledge_syntax_shared"],
            "mean_knowledge_attribute_shared": agg["mean_knowledge_attribute_shared"],
            "mean_knowledge_association_shared": agg["mean_knowledge_association_shared"],
            "mean_attribute_association_overlap": agg["mean_attribute_association_overlap"],
        }
        if model_key in circuit_rows:
            row["cross_task_minimal_circuit"] = circuit_rows[model_key]
        model_rows.append(row)

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage516_neuron_level_restoration_synthesis",
        "title": "神经元级还原综合摘要",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - start, 3),
        "source_summaries": {
            "stage514": str(STAGE514_PATH),
            "stage515": str(STAGE515_PATH),
        },
        "model_rows": model_rows,
        "working_mechanism_hypothesis": {
            "shared_concept_backbone": "由跨知识、语法、属性任务稳定复用的一批中高频神经元承担，说明概念本体不是每个任务各存一份。",
            "relation_adapters": "由知识-联想共享较低、属性-联想重叠有限的任务专属神经元群承担，说明关系连接更多像局部适配器。",
            "routing_mechanism": "负责把同一概念送入不同任务路径，本轮通过跨任务共享骨干与前序工作中的头主导现象间接支持。",
            "binding_mechanism": "负责把概念骨干、属性通道和关系适配器绑到同一实例，本轮仍以结构支持强于最小因果证据。",
            "low_cost_switch": "通过在共享骨干上叠加小型任务适配器和最小因果子集完成，而不是整套神经元重建。",
        },
        "core_answer": "神经元级还原的关键不是寻找一个概念对应一团固定神经元，而是同时找出共享骨干、任务适配器和最小因果子回路，再看这些部件如何在不同任务中重复组合。",
    }

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "REPORT.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# stage516 神经元级还原综合摘要",
        "",
        "## 核心结论",
        "",
        "共享概念骨干不是苹果个案，而是在水果、动物、工具、组织四类名词家族中都能看到的更一般结构。",
        "跨任务最小因果回路也已经开始显形，说明同一概念跨知识、语法、属性、联想任务的能力，并不只是相关重叠，而是有一小组共享神经元在真正承担因果作用。",
        "",
        "## 家族级复用",
        "",
    ]
    for row in model_rows:
        lines.extend(
            [
                f"### {row['model_key']}",
                "",
                f"- 家族数：{row['family_count']}",
                f"- 平均全任务共享核心：{row['mean_core_shared_all']:.2f}",
                f"- 平均四组以上共享核心：{row['mean_core_shared_4plus']:.2f}",
                f"- 平均共享比例：{row['mean_core_shared_ratio']:.4f}",
                f"- 平均知识-语法共享：{row['mean_knowledge_syntax_shared']:.2f}",
                f"- 平均知识-属性共享：{row['mean_knowledge_attribute_shared']:.2f}",
                f"- 平均知识-联想共享：{row['mean_knowledge_association_shared']:.2f}",
                f"- 平均属性-联想重叠：{row['mean_attribute_association_overlap']:.4f}",
                "",
            ]
        )
        if "cross_task_minimal_circuit" in row:
            circuit = row["cross_task_minimal_circuit"]
            lines.extend(
                [
                    f"- 最小跨任务因果子集：{', '.join(circuit['final_subset'])}",
                    f"- 目标下降：{circuit['target_drop']:.6f}",
                    f"- 控制偏移：{circuit['control_abs_shift']:.6f}",
                    f"- 综合效用：{circuit['utility']:.6f}",
                    "",
                ]
            )

    lines.extend(
        [
            "## 工作模型",
            "",
            "当前更稳的神经元级工作模型是：",
            "",
            "1. 先由共享概念骨干保存概念本体和家族位置。",
            "2. 再由任务适配器把同一概念接到知识、语法、属性、联想等不同任务轨道。",
            "3. 路由机制决定当前任务该调用哪组适配器。",
            "4. 绑定机制把骨干、属性、关系和当前实例拼到一起。",
            "5. 低成本切换依赖共享骨干上的小增量改写，而不是整套神经元重建。",
            "",
            "## 最严格的边界",
            "",
            "1. 当前最小因果回路只在 Qwen3 和 DeepSeek7B 上完成，还没扩到四模型全覆盖。",
            "2. 绑定机制仍然以结构证据为主，强因果闭环还不够硬。",
            "3. 共享数量依然部分依赖 top-k 阈值定义，不能当成自然常数。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
