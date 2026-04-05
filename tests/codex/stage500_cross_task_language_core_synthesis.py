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
    / "stage500_cross_task_language_core_synthesis_20260404"
)

PATHS = {
    "stage496_qwen3": PROJECT_ROOT / "tests" / "codex_temp" / "stage496_cross_task_cv_stability_20260404" / "summary_qwen3.json",
    "stage496_deepseek": PROJECT_ROOT / "tests" / "codex_temp" / "stage496_cross_task_cv_stability_20260404" / "summary_deepseek.json",
    "stage497_qwen3": PROJECT_ROOT / "tests" / "codex_temp" / "stage497_causal_chain_20260404" / "summary_qwen3.json",
    "stage497_deepseek": PROJECT_ROOT / "tests" / "codex_temp" / "stage497_causal_chain_20260404" / "summary_deepseek.json",
    "stage498_qwen3": PROJECT_ROOT / "tests" / "codex_temp" / "stage498_hierarchy_20260404" / "summary_qwen3.json",
    "stage498_deepseek": PROJECT_ROOT / "tests" / "codex_temp" / "stage498_hierarchy_20260404" / "summary_deepseek.json",
    "stage499_qwen3": PROJECT_ROOT / "tests" / "codex_temp" / "stage499_cross_token_routing_20260404" / "summary_qwen3.json",
    "stage499_deepseek": PROJECT_ROOT / "tests" / "codex_temp" / "stage499_cross_token_routing_20260404" / "summary_deepseek.json",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model_row(model_key: str, s496: dict, s497: dict, s498: dict, s499: dict) -> dict:
    cv_dist = s496["global_control_variable_distribution"]
    prop = s497["aggregate"]
    hierarchy = s498["aggregate"]
    routing = s499["aggregate"]

    return {
        "control_variable_stability": {
            "four_types_found": s496["four_types_found"],
            "four_types_missing": s496["four_types_missing"],
            "stability_score": s496["stability_score"],
            "distribution": cv_dist,
        },
        "propagation": {
            "dominant_mode": prop["dominant_propagation_mode"],
            "faithful_count": prop["faithful_propagation_count"],
            "rewritten_count": prop["rewritten_propagation_count"],
            "avg_cosine_preservation": prop["avg_cosine_preservation"],
        },
        "hierarchy": {
            "avg_monotonicity_rate": hierarchy["avg_monotonicity_rate"],
            "hierarchies_with_monotonicity": hierarchy["hierarchies_with_monotonicity"],
            "hierarchies_without_monotonicity": hierarchy["hierarchies_without_monotonicity"],
        },
        "cross_token_routing": {
            "route_heads_dominant_count": routing["route_heads_dominant_count"],
            "write_neurons_dominant_count": routing["write_neurons_dominant_count"],
            "mixed_count": routing["mixed_count"],
            "total_tasks": routing["total_tasks"],
            "route_heads_ratio": routing["route_heads_ratio"],
            "avg_attn_to_mlp_ratio": routing["avg_attn_to_mlp_ratio"],
        },
        "core_character": {
            "control_variable_character": (
                "控制变量较稳定但不完备"
                if s496["stability_score"] < 1.0
                else "四类控制变量都稳定出现"
            ),
            "propagation_character": (
                "以忠实传播为主"
                if prop["dominant_propagation_mode"] == "faithful"
                else "以中途重写为主"
            ),
            "hierarchy_character": (
                "概念层次不是简单单调树"
                if hierarchy["avg_monotonicity_rate"] < 0.5
                else "概念层次接近单调树"
            ),
            "cross_token_character": (
                "跨词元任务以混合回路为主"
                if routing["mixed_count"] == routing["total_tasks"]
                else "跨词元任务出现纯路由头主导"
            ),
        },
    }


def build_summary() -> dict:
    s496_q = load_json(PATHS["stage496_qwen3"])
    s496_d = load_json(PATHS["stage496_deepseek"])
    s497_q = load_json(PATHS["stage497_qwen3"])
    s497_d = load_json(PATHS["stage497_deepseek"])
    s498_q = load_json(PATHS["stage498_qwen3"])
    s498_d = load_json(PATHS["stage498_deepseek"])
    s499_q = load_json(PATHS["stage499_qwen3"])
    s499_d = load_json(PATHS["stage499_deepseek"])

    models = {
        "qwen3": model_row("qwen3", s496_q, s497_q, s498_q, s499_q),
        "deepseek7b": model_row("deepseek7b", s496_d, s497_d, s498_d, s499_d),
    }

    shared = {
        "shared_stable_findings": [
            "残差传播在当前跨任务测试中以忠实传播为主，而不是层层完全重写。",
            "跨词元任务没有出现“纯注意力头单独接管”的简单图景，而是头与神经元的混合回路。",
            "概念层次在模型空间里并不呈现简单的单调欧式树结构。",
        ],
        "cross_model_difference": [
            "Qwen3 的跨任务控制变量稳定性评分较低，更集中在写入神经元与晚层读出放大。",
            "DeepSeek7B 更容易露出 mixed_binding_circuits（混合绑定回路），说明其跨任务拓扑更异质。",
        ],
    }

    core_answer = (
        "当前新增证据表明，语言核心编码结构更像“控制变量 + 忠实传播 + 混合回路 + 非简单层次空间”。"
        "也就是说，网络内部既不是纯静态词典，也不是每层都完全重写；更像若干控制变量写入后，经残差流忠实传播，"
        "再由头和神经元的混合回路在不同任务中完成路由、绑定、放大和读出。"
    )

    return {
        "stage": "stage500_cross_task_language_core_synthesis",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
        "shared": shared,
        "core_answer": core_answer,
        "next_focus": [
            "继续寻找真正能让 route_heads 单独抬头的更长距离跨词元任务。",
            "把忠实传播与局部重写分解到更细粒度的残差方向。",
            "把概念层次从距离统计推进到因果路径验证，避免只停在几何现象。",
        ],
    }


def build_report(summary: dict) -> str:
    lines = ["# stage500 跨任务语言核心结构综合摘要", ""]
    lines.append("## 核心答案")
    lines.append("")
    lines.append(f"- {summary['core_answer']}")
    lines.append("")
    lines.append("## 模型摘要")
    lines.append("")
    for model_key, row in summary["models"].items():
        lines.append(f"### {model_key}")
        lines.append("")
        cv = row["control_variable_stability"]
        prop = row["propagation"]
        hierarchy = row["hierarchy"]
        routing = row["cross_token_routing"]
        lines.append(f"- 控制变量稳定性评分: `{cv['stability_score']}`")
        lines.append(f"- 已出现控制变量: `{', '.join(cv['four_types_found'])}`")
        lines.append(f"- 缺失控制变量: `{', '.join(cv['four_types_missing'])}`")
        lines.append(f"- 忠实传播数: `{prop['faithful_count']}`，重写传播数: `{prop['rewritten_count']}`，平均余弦保持: `{prop['avg_cosine_preservation']:.4f}`")
        lines.append(f"- 概念层次平均单调率: `{hierarchy['avg_monotonicity_rate']:.4f}`")
        lines.append(f"- 跨词元任务 mixed（混合）数: `{routing['mixed_count']}/{routing['total_tasks']}`，route_heads 比例: `{routing['route_heads_ratio']:.4f}`")
        lines.append("")
    lines.append("## 跨模型共同结论")
    lines.append("")
    for item in summary["shared"]["shared_stable_findings"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 跨模型差异")
    lines.append("")
    for item in summary["shared"]["cross_model_difference"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 下一阶段重点")
    lines.append("")
    for item in summary["next_focus"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    summary = build_summary()
    report = build_report(summary)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果写入: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
