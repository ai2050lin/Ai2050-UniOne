#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / f"stage489_unified_residual_dynamics_protocol_{time.strftime('%Y%m%d')}"
)

STAGE441_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage441_unified_language_state_equation_20260402" / "summary.json"
STAGE487_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / f"stage487_polysemy_unified_switch_protocol_{time.strftime('%Y%m%d')}" / "summary.json"
STAGE488_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / f"stage488_bridge_minimal_causal_circuit_protocol_{time.strftime('%Y%m%d')}" / "summary.json"
STAGE481_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage481_apple_switch_pair_order_analysis_20260403" / "summary.json"
STAGE482_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage482_apple_switch_direction_tracking_20260403" / "summary.json"
STAGE483_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage483_apple_switch_residual_basis_20260403" / "summary.json"
STAGE484_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage484_apple_switch_signed_residual_basis_20260403" / "summary.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def unit_map(model_row: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {unit["unit_id"]: unit for unit in model_row["units"]}


def build_summary() -> Dict[str, object]:
    stage441 = load_json(STAGE441_SUMMARY_PATH)
    stage487 = load_json(STAGE487_SUMMARY_PATH)
    stage488 = load_json(STAGE488_SUMMARY_PATH)
    stage481 = load_json(STAGE481_SUMMARY_PATH)
    stage482 = load_json(STAGE482_SUMMARY_PATH)
    stage483 = load_json(STAGE483_SUMMARY_PATH)
    stage484 = load_json(STAGE484_SUMMARY_PATH)

    qwen482 = unit_map(stage482["models"]["qwen3"])
    deep482 = unit_map(stage482["models"]["deepseek7b"])
    qwen483 = unit_map(stage483["models"]["qwen3"])
    deep483 = unit_map(stage483["models"]["deepseek7b"])
    qwen484 = unit_map(stage484["models"]["qwen3"])
    deep484 = unit_map(stage484["models"]["deepseek7b"])

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage489_unified_residual_dynamics_protocol",
        "title": "统一残差动力学正式协议",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "equation": {
            "base_symbolic_form": stage441["payload"]["state_equation"]["symbolic_form"],
            "updated_symbolic_form": "h_{t}^{l+1} = h_{t}^{l} + R_l + B_l + S_l + A_l + G_l + O_l, 其中 S_l 可由两类控制杆实现：Qwen3 偏 head skeleton（头骨架），DeepSeek7B 偏 neuron anchor + head boosters（神经元锚点加头增强器）。",
            "term_meaning": stage441["payload"]["state_equation"]["term_meaning"],
        },
        "topology_modes": {
            "qwen3": {
                "mode_name": "head_skeleton_write_then_late_readout",
                "best_order": stage481["models"]["qwen3"]["utility_focus"]["best_order"]["order"],
                "evidence": {
                    "peak_effect_layers": {unit_id: int(row["tracking"]["peak_effect_layer"]) for unit_id, row in qwen482.items()},
                    "peak_alignment_layers": {unit_id: int(row["tracking"]["peak_contrast_alignment_layer"]) for unit_id, row in qwen483.items()},
                    "reverse_peak_layers": {unit_id: int(row["tracking"]["reverse_peak_layer"]) for unit_id, row in qwen484.items()},
                },
                "reading": "Qwen3 更像由敏感层头骨架先写入切换偏置，随后在晚层读出中被放大并收束成清晰语义分叉。",
            },
            "deepseek7b": {
                "mode_name": "anchor_neuron_pin_then_head_boost",
                "best_order": stage481["models"]["deepseek7b"]["utility_focus"]["best_order"]["order"],
                "evidence": {
                    "peak_effect_layers": {unit_id: int(row["tracking"]["peak_effect_layer"]) for unit_id, row in deep482.items()},
                    "peak_alignment_layers": {unit_id: int(row["tracking"]["peak_contrast_alignment_layer"]) for unit_id, row in deep483.items()},
                    "reverse_peak_layers": {unit_id: int(row["tracking"]["reverse_peak_layer"]) for unit_id, row in deep484.items()},
                },
                "reading": "DeepSeek7B 更像由早层神经元锚点先钉住切换主方向，再由同层头群持续增强并把影响传播到中后层。",
            },
        },
        "evidence_bridge": {
            "polysemy_protocol": stage487["cross_model_summary"],
            "bridge_protocol": stage488["aggregate"],
        },
        "mechanism_constraints": [
            "多义词切换不能再被视为普通上下文扰动，统一协议要求它表现出稳定的低重合切换结构。",
            "桥接项 G_l 不能被直接删掉，但也不能先验假定为纯神经元集合；当前更合理的约束是结构规律强、最小因果回路家族差异大。",
            "残差动力学统一时，必须允许不同模型用不同拓扑实现同一抽象状态变量。",
        ],
        "core_answer": "当前最合理的统一残差动力学是：R_l 决定路由，B_l 提供共享底座，S_l 负责多义切换，A_l 写入属性方向，G_l 承担对象-属性绑定，O_l 完成晚层读出。Qwen3 与 DeepSeek7B 共享这套抽象分工，但在 S_l 的实现拓扑上出现分叉：前者偏头骨架写入与晚读出放大，后者偏神经元锚点钉住与头群持续增益。",
        "sources": {
            "stage441": str(STAGE441_SUMMARY_PATH),
            "stage487": str(STAGE487_SUMMARY_PATH),
            "stage488": str(STAGE488_SUMMARY_PATH),
            "stage481": str(STAGE481_SUMMARY_PATH),
            "stage482": str(STAGE482_SUMMARY_PATH),
            "stage483": str(STAGE483_SUMMARY_PATH),
            "stage484": str(STAGE484_SUMMARY_PATH),
        },
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 核心回答",
        summary["core_answer"],
        "",
        "## 方程",
        f"- base = {summary['equation']['base_symbolic_form']}",
        f"- updated = {summary['equation']['updated_symbolic_form']}",
        "",
        "## 两种拓扑模式",
    ]
    for model_key in ["qwen3", "deepseek7b"]:
        row = summary["topology_modes"][model_key]
        lines.append(f"- {model_key}: {row['mode_name']}")
        lines.append(f"  - best_order = {' -> '.join(row['best_order'])}")
        lines.append(f"  - reading = {row['reading']}")
    lines.extend(["", "## 机制约束"])
    for row in summary["mechanism_constraints"]:
        lines.append(f"- {row}")
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一残差动力学正式协议")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary()
    write_outputs(summary, Path(args.output_dir))
    print(json.dumps({"status_short": "stage489_ready", "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
