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
    / "stage441_unified_language_state_equation_20260402"
)

STAGE429_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage429_deepseek7b_pronoun_head_pair_order_validation_20260402"
    / "summary.json"
)
STAGE433_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage433_polysemous_noun_family_generalization_20260402"
    / "summary.json"
)
STAGE434_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage434_apple_polysemy_factorized_switch_20260402"
    / "summary.json"
)
STAGE435_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage435_apple_feature_binding_neuron_channels_20260402"
    / "summary.json"
)
STAGE439_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage439_binding_bridge_causal_ablation_20260402"
    / "summary.json"
)
STAGE440_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage440_attribute_graph_generalization_20260402"
    / "summary.json"
)


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_float(value) -> float:
    return float(value)


def build_equation_payload(
    stage429: Dict[str, object],
    stage433: Dict[str, object],
    stage434: Dict[str, object],
    stage435: Dict[str, object],
    stage439: Dict[str, object],
    stage440: Dict[str, object],
) -> Dict[str, object]:
    route_pair = stage429["route_pair"]
    integrator = stage429["integrator_head"]
    pronoun_margin_full = safe_float(stage429["mechanism_inference"]["route_then_integrate_margin_full"])
    pronoun_margin_heldout = safe_float(stage429["mechanism_inference"]["route_then_integrate_margin_heldout"])

    shared_base_support_rate = safe_float(stage433["cross_model_summary"]["shared_base_support_rate"])
    structured_switch_support_rate = safe_float(stage433["cross_model_summary"]["structured_switch_support_rate"])
    reliable_readout_support_rate = safe_float(stage433["cross_model_summary"]["reliable_readout_support_rate"])

    distinct_switch_axis_rate = safe_float(stage434["cross_model_summary"]["distinct_switch_axis_vote_count"]) / max(
        1.0, float(len(stage434["model_results"]))
    )
    binding_reuse_vote_rate = safe_float(stage435["cross_model_summary"]["binding_reuse_vote_count"]) / max(
        1.0, float(len(stage435["model_results"]))
    )

    binding_causal_support_rate = safe_float(stage439["cross_model_summary"]["binding_causal_support_rate"])
    attribute_graph_support_rate = safe_float(stage440["cross_model_summary"]["attribute_graph_support_rate"])

    evidence_scores = {
        "route_first_support": 0.5 * (pronoun_margin_full + pronoun_margin_heldout),
        "shared_base_support": shared_base_support_rate,
        "sense_switch_support": 0.5 * (structured_switch_support_rate + distinct_switch_axis_rate),
        "binding_reuse_support": 0.5 * (binding_reuse_vote_rate + attribute_graph_support_rate),
        "binding_causal_support": binding_causal_support_rate,
        "readout_support": reliable_readout_support_rate,
    }

    state_equation = {
        "symbolic_form": (
            "h_{t}^{l+1} = h_{t}^{l} + R_l(x_{<=t}) + B_l(lemma_t) + S_l(lemma_t, context_t) "
            "+ A_l(attr_t, context_t) + G_l(B_l, A_l, route_t) + O_l(h_t^l)"
        ),
        "term_meaning": {
            "R_l": "早层 route field（路由场），主要由 pronoun route pair（代词路由对）和 integrator head（整合头）承载，负责先路由后整合。",
            "B_l": "noun backbone（名词骨干），负责公共名词底座与家族复用。",
            "S_l": "sense switch axis（词义切换轴），在共享底座上完成 fruit/brand 等多义切换。",
            "A_l": "attribute modifier channel（属性修饰通道），承载颜色、味道、大小等修饰方向。",
            "G_l": "binding bridge term（绑定桥接项），把名词骨干与属性修饰真正绑定到同一对象上。",
            "O_l": "late readout term（晚层读出项），把前面形成的内部状态压成可输出答案。",
        },
        "mechanism_reading": [
            "先由 R_l 决定谁与谁需要建立关系，尤其是功能词和指代线索如何被提前路由。",
            "再由 B_l 提供共享名词底座，由 S_l 在底座上完成多义切换。",
            "随后 A_l 写入颜色、味道、大小等修饰方向。",
            "最后由 G_l 把名词底座与属性修饰绑定成 apple-red、apple-sweet、apple-fist 这类复合概念，并交给 O_l 读出。",
        ],
    }

    return {
        "evidence_scores": evidence_scores,
        "state_equation": state_equation,
        "mechanism_mapping": {
            "pronoun_route_pair": route_pair,
            "pronoun_integrator_head": integrator,
            "shared_base_support_rate": shared_base_support_rate,
            "structured_switch_support_rate": structured_switch_support_rate,
            "distinct_switch_axis_rate": distinct_switch_axis_rate,
            "binding_reuse_vote_rate": binding_reuse_vote_rate,
            "binding_causal_support_rate": binding_causal_support_rate,
            "attribute_graph_support_rate": attribute_graph_support_rate,
            "reliable_readout_support_rate": reliable_readout_support_rate,
        },
        "core_answer": (
            "当前最合理的统一图景是：语言编码不是一张静态词典，而是 route（路由）- backbone（骨干）- switch（切换）- attribute（属性）- bridge（桥接）- readout（读出）"
            " 六段耦合系统。代词主线告诉我们谁先路由、谁后整合；多义名词主线告诉我们共享底座与词义切换如何避免组合爆炸；属性绑定主线告诉我们概念不是整块重写，而是骨干与修饰在桥接项上完成因果绑定。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    eq = summary["payload"]["state_equation"]
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 核心回答",
        summary["payload"]["core_answer"],
        "",
        "## 统一状态方程",
        f"`{eq['symbolic_form']}`",
        "",
        "## 各项含义",
    ]
    for key, value in eq["term_meaning"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 机制解读"])
    for item in eq["mechanism_reading"]:
        lines.append(f"- {item}")
    lines.extend(["", "## 证据分数"])
    for key, value in summary["payload"]["evidence_scores"].items():
        lines.append(f"- {key}: {value:.4f}")
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一语言状态方程整合脚本")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage429 = load_json(STAGE429_SUMMARY_PATH)
    stage433 = load_json(STAGE433_SUMMARY_PATH)
    stage434 = load_json(STAGE434_SUMMARY_PATH)
    stage435 = load_json(STAGE435_SUMMARY_PATH)
    stage439 = load_json(STAGE439_SUMMARY_PATH)
    stage440 = load_json(STAGE440_SUMMARY_PATH)
    payload = build_equation_payload(stage429, stage433, stage434, stage435, stage439, stage440)
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage441_unified_language_state_equation",
        "title": "统一语言状态方程整合",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "payload": payload,
        "sources": {
            "stage429": str(STAGE429_SUMMARY_PATH),
            "stage433": str(STAGE433_SUMMARY_PATH),
            "stage434": str(STAGE434_SUMMARY_PATH),
            "stage435": str(STAGE435_SUMMARY_PATH),
            "stage439": str(STAGE439_SUMMARY_PATH),
            "stage440": str(STAGE440_SUMMARY_PATH),
        },
    }
    write_outputs(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
