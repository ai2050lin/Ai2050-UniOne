#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage495_unified_language_control_variable_protocol_20260403"
)

PATHS = {
    "stage426": PROJECT_ROOT / "tests" / "codex_temp" / "stage426_pronoun_minimal_causal_mechanism_20260330" / "summary.json",
    "stage428": PROJECT_ROOT / "tests" / "codex_temp" / "stage428_deepseek7b_pronoun_head_group_stability_20260402" / "summary.json",
    "stage429": PROJECT_ROOT / "tests" / "codex_temp" / "stage429_deepseek7b_pronoun_head_pair_order_validation_20260402" / "summary.json",
    "stage487": PROJECT_ROOT / "tests" / "codex_temp" / "stage487_polysemy_unified_switch_protocol_20260403" / "summary.json",
    "stage488": PROJECT_ROOT / "tests" / "codex_temp" / "stage488_bridge_minimal_causal_circuit_protocol_20260403" / "summary.json",
    "stage489": PROJECT_ROOT / "tests" / "codex_temp" / "stage489_unified_residual_dynamics_protocol_20260403" / "summary.json",
    "stage493": PROJECT_ROOT / "tests" / "codex_temp" / "stage493_chinese_language_master_atlas_20260403" / "summary.json",
    "stage494": PROJECT_ROOT / "tests" / "codex_temp" / "stage494_pattern_specific_control_protocol_20260403" / "summary.json",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_report(summary: dict) -> str:
    lines = ["# stage495 统一语言控制变量协议", ""]
    lines.append("## 总结")
    lines.append("")
    lines.append(f"- 更新后的统一方程：`{summary['equation']['updated_symbolic_form']}`")
    lines.append(
        f"- 重复出现的控制变量：`{', '.join(summary['recurring_control_variables'].keys())}`"
    )
    lines.append("")
    for model_key, model_row in summary["models"].items():
        lines.append(f"### {model_key}")
        lines.append("")
        lines.append(f"- 核心拓扑：`{model_row['topology_mode']}`")
        lines.append(f"- 词元补全图谱：`{json.dumps(model_row['lexical_route_topology_counts'], ensure_ascii=False)}`")
        lines.append(f"- 强控制特异性支持率：`{model_row['lexical_specificity_support_rate']:.4f}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    stage426 = load_json(PATHS["stage426"])
    stage428 = load_json(PATHS["stage428"])
    stage429 = load_json(PATHS["stage429"])
    stage487 = load_json(PATHS["stage487"])
    stage488 = load_json(PATHS["stage488"])
    stage489 = load_json(PATHS["stage489"])
    stage493 = load_json(PATHS["stage493"])
    stage494 = load_json(PATHS["stage494"])

    qwen_lex_counts = stage493["models"]["qwen3"]["model_summary"]["overall_topology_counts"]
    deepseek_lex_counts = stage493["models"]["deepseek7b"]["model_summary"]["overall_topology_counts"]

    summary = {
        "stage": "stage495_unified_language_control_variable_protocol",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "equation": {
            "previous_symbolic_form": stage489["equation"]["updated_symbolic_form"],
            "updated_symbolic_form": (
                "h_{t}^{l+1} = h_{t}^{l} + C_l(form_t, context_t) + R_l(ref_t, context_t) + "
                "B_l(lemma_t) + S_l(lemma_t, context_t) + A_l(attr_t, context_t) + "
                "G_l(B_l, A_l, C_l, R_l) + O_l(h_t^l)"
            ),
            "term_meaning": {
                "C_l": "词元续写控制项，负责把局部字形/词形前缀推进到高概率后续词元。",
                "R_l": "代词和功能词路由项，负责先建立引用/句法路线。",
                "B_l": "名词骨干项，负责共享底座和对象家族复用。",
                "S_l": "词义切换项，负责多义词在共享底座上的低重合跨义切换。",
                "A_l": "属性修饰项，负责颜色、味道、大小等修饰方向。",
                "G_l": "绑定桥接项，负责把对象骨干和属性绑定到同一实例。",
                "O_l": "读出项，负责把前面状态压成最终输出。",
            },
        },
        "recurring_control_variables": {
            "route_heads": {
                "meaning": "负责跨位置搬运、路由和配对的头控制变量。",
                "evidence": {
                    "pronoun_route_pair": stage429["route_pair"],
                    "pronoun_integrator_head": stage429["integrator_head"],
                    "lexical_head_dominant_count_qwen3": int(qwen_lex_counts.get("head_dominant", 0)),
                    "lexical_head_dominant_count_deepseek7b": int(deepseek_lex_counts.get("head_dominant", 0)),
                },
            },
            "write_neurons": {
                "meaning": "负责局部写入、目标方向放大和落点锁定的神经元控制变量。",
                "evidence": {
                    "qwen3_pronoun_top_layers": stage426["models"]["qwen3"]["pronoun_top_layers"],
                    "qwen3_lexical_neuron_dominant_count": int(qwen_lex_counts.get("neuron_dominant", 0)),
                    "deepseek7b_lexical_neuron_dominant_count": int(deepseek_lex_counts.get("neuron_dominant", 0)),
                    "deepseek_switch_anchor_mode": stage489["topology_modes"]["deepseek7b"]["mode_name"],
                },
            },
            "mixed_binding_circuits": {
                "meaning": "负责对象-属性绑定或复杂模式整合的混合回路控制变量。",
                "evidence": {
                    "bridge_family_mixed_support_count": int(stage488["aggregate"]["family_mixed_support_count"]),
                    "bridge_size_final_subset": stage488["bridge_status"]["size"]["family_probe"]["final_subset_ids"],
                    "deepseek_lexical_mixed_count": int(deepseek_lex_counts.get("mixed", 0)),
                },
            },
            "late_readout_amplifiers": {
                "meaning": "负责把早中层写入一路放大到最终答案的晚层读出控制变量。",
                "evidence": {
                    "qwen3_switch_mode": stage489["topology_modes"]["qwen3"]["mode_name"],
                    "qwen3_switch_best_order": stage489["topology_modes"]["qwen3"]["best_order"],
                    "polysemy_gap_qwen3": float(stage487["models"]["qwen3"]["mean_gap"]),
                    "polysemy_gap_deepseek7b": float(stage487["models"]["deepseek7b"]["mean_gap"]),
                },
            },
        },
        "models": {
            "qwen3": {
                "topology_mode": stage489["topology_modes"]["qwen3"]["mode_name"],
                "pronoun_route_character": "晚层 MLP 写入更关键，注意力整层消融不是主因。",
                "polysemy_character": "头骨架先写入，多义切换在晚层读出放大。",
                "lexical_route_topology_counts": qwen_lex_counts,
                "lexical_specificity_support_rate": float(stage494["models"]["qwen3"]["aggregate"]["support_rate"]),
                "core_answer": (
                    "Qwen3 更接近“头负责写入偏置、神经元负责目标字写出、晚层负责放大读出”的风格。"
                ),
            },
            "deepseek7b": {
                "topology_mode": stage489["topology_modes"]["deepseek7b"]["mode_name"],
                "pronoun_route_character": "早层 route pair 加 integrator head 先路由后整合。",
                "polysemy_character": "神经元锚点先钉住切换方向，头群后续增强。",
                "lexical_route_topology_counts": deepseek_lex_counts,
                "lexical_specificity_support_rate": float(stage494["models"]["deepseek7b"]["aggregate"]["support_rate"]),
                "core_answer": (
                    "DeepSeek7B 更接近“头负责开路，锚点/神经元负责落点，混合回路负责整合”的风格。"
                ),
            },
        },
        "cross_task_constraints": [
            "词元补全路线 C_l 不能再被视为噪声，因为高基线模式已经通过强控制特异性验证。",
            "多义切换 S_l 与词元补全 C_l 不是同一件事，但它们都反复显示“头路由 + 神经元写入 + 残差传播”的三段结构。",
            "桥接项 G_l 仍然是最薄弱部分，但当前证据已经支持它通常需要混合回路，而不是单一神经元集合。",
            "统一状态方程必须允许不同模型对同一抽象变量采用不同拓扑实现。"
        ],
        "aggregate": {
            "lexical_specificity_support_rate_qwen3": float(stage494["models"]["qwen3"]["aggregate"]["support_rate"]),
            "lexical_specificity_support_rate_deepseek7b": float(stage494["models"]["deepseek7b"]["aggregate"]["support_rate"]),
            "bridge_structural_law_support_rate": float(stage488["aggregate"]["structural_law_support_rate"]),
            "bridge_pure_bridge_causal_support_rate": float(stage488["aggregate"]["pure_bridge_causal_support_rate"]),
            "polysemy_large_gap_support_rate": float(stage489["evidence_bridge"]["polysemy_protocol"]["shared_large_gap_support_rate"]),
        },
        "core_answer": (
            "当前最稳的统一图景是：语言编码不是静态词典，而是由 route heads、write neurons、mixed binding circuits、late readout amplifiers "
            "这四类反复出现的控制变量共同驱动。词元补全、多义切换、代词路由、属性绑定都在复用这套更高层分工，只是不同模型的实现拓扑不同。"
        ),
        "sources": {key: str(path) for key, path in PATHS.items()},
        "elapsed_seconds": float(time.time() - started),
    }

    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "REPORT.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_report(summary), encoding="utf-8")
    print(f"summary written to {summary_path}")
    print(f"report written to {report_path}")


if __name__ == "__main__":
    main()
