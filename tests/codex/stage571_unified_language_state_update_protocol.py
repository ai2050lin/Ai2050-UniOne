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
    / "stage571_unified_language_state_update_protocol_20260409"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_state_components() -> list[dict]:
    return [
        {"symbol": "O_t", "name_zh": "对象状态", "role": "noun/entity/object content"},
        {"symbol": "A_t", "name_zh": "属性状态", "role": "color/taste/size and other object properties"},
        {"symbol": "R_t", "name_zh": "关系框架", "role": "preposition, role assignment, spatial or event frame"},
        {"symbol": "P_t", "name_zh": "指代状态", "role": "pronoun reference, coreference pointer, discourse target"},
        {"symbol": "M_t", "name_zh": "修饰状态", "role": "adverbial scope, degree, frequency, manner, time"},
        {"symbol": "Q_t", "name_zh": "推理状态", "role": "intermediate conclusions, constraints, candidate worlds"},
        {"symbol": "G_t", "name_zh": "绑定状态", "role": "non-additive glue between objects, relations, and modifiers"},
        {"symbol": "C_t", "name_zh": "上下文条件", "role": "task frame, style, negation, discourse history"},
    ]


def build_experiment_rows() -> list[dict]:
    return [
        {
            "experiment_id": "E1",
            "name_zh": "代词共指实验",
            "focus": "P_t / Q_t",
            "question": "代词是否主要编码为指向前文对象的引用指针，而不是独立名词语义",
            "minimal_pairs": [
                "John thanked Bob because he smiled.",
                "John thanked Bob because he helped.",
                "Alice met Mary after she arrived.",
            ],
            "observables": [
                "candidate antecedent margin",
                "layerwise hidden-state swap effect",
                "coreference classifier shift",
            ],
            "causal_prediction": "ablate reference units and the antecedent choice should destabilize while object words remain readable",
        },
        {
            "experiment_id": "E2",
            "name_zh": "介词关系框架实验",
            "focus": "R_t / G_t",
            "question": "介词是否主要建立对象之间的关系框架，而不是给对象加属性",
            "minimal_pairs": [
                "The apple is on the table.",
                "The apple is under the table.",
                "The apple is near the table.",
            ],
            "observables": [
                "relation-specific logit margins",
                "object-role swap effect",
                "frame reconstruction score",
            ],
            "causal_prediction": "ablate relation-frame units and object identity survives better than the on/under/near distinction",
        },
        {
            "experiment_id": "E3",
            "name_zh": "副词修饰范围实验",
            "focus": "M_t / G_t / Q_t",
            "question": "副词是在修饰动作、整句置信度，还是对象局部属性",
            "minimal_pairs": [
                "The boy quickly opened the door.",
                "The boy probably opened the door.",
                "The very ripe apple fell.",
            ],
            "observables": [
                "predicate readout shift",
                "scope-sensitive token trajectory",
                "confidence or manner probe response",
            ],
            "causal_prediction": "different adverbs should move different targets: manner adverbs mostly change event trajectory, epistemic adverbs mostly change Q_t",
        },
        {
            "experiment_id": "E4",
            "name_zh": "逻辑链路逐层追踪实验",
            "focus": "Q_t / R_t / P_t / C_t",
            "question": "中间结论是否能作为稳定推理状态沿层传播，而不是只在末层突然出现",
            "minimal_pairs": [
                "All fruits are edible. Apple is a fruit. Therefore apple is edible.",
                "If the key is in the box, then the box is on the shelf. The key is in the box. Therefore the key is on the shelf.",
                "No red fruit is green. This apple is red. Therefore this apple is not green.",
            ],
            "observables": [
                "intermediate conclusion probe",
                "constraint satisfaction score",
                "layerwise contradiction resolution",
            ],
            "causal_prediction": "a stable Q_t trajectory should emerge before final token readout and should be perturbable by targeted ablation",
        },
    ]


def build_update_laws() -> list[str]:
    return [
        "S_t,l = (O_t,l, A_t,l, R_t,l, P_t,l, M_t,l, Q_t,l, G_t,l, C_t,l)",
        "S_t,l+1 = F_l(S_t,l)",
        "Q_t,l+1 = F_reason(Q_t,l, O_t,l, R_t,l, P_t,l, M_t,l, C_t,l)",
        "readout_t = R(S_t,L)",
    ]


def build_success_criteria() -> list[str]:
    return [
        "Pronoun disambiguation can be isolated as a reference-state effect rather than a generic noun effect.",
        "Preposition edits change relation frames more than object identity.",
        "Adverb classes separate into event-level and proposition-level modifiers.",
        "Intermediate logical conclusions appear as layerwise states before final-token readout.",
        "The four experiments can be described in the same state-update vocabulary without ad hoc exceptions.",
    ]


def build_failure_modes() -> list[str]:
    return [
        "Pronoun behavior cannot be separated from generic semantic similarity.",
        "Preposition changes are explained entirely by surface token identity with no stable relation frame.",
        "Adverb effects collapse into one undifferentiated context residual.",
        "No stable intermediate reasoning trajectory is detectable before final readout.",
        "The unified state-update law becomes a loose naming scheme rather than a predictive structure.",
    ]


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage571_unified_language_state_update_protocol",
        "title": "统一语言状态更新协议",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "master_state_laws": build_update_laws(),
        "state_components": build_state_components(),
        "experiment_rows": build_experiment_rows(),
        "success_criteria": build_success_criteria(),
        "failure_modes": build_failure_modes(),
        "core_answer": (
            "代词、介词、副词和逻辑推理不应再作为零散补丁加入理论，"
            "而应统一进入对象、属性、关系、指代、修饰、推理状态的同一套状态更新方程；"
            "这样理论才能从概念编码猜想升级成真正的语言计算理论。"
        ),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# stage571 统一语言状态更新协议",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
        "## 主状态方程",
    ]
    for law in summary["master_state_laws"]:
        lines.append(f"- `{law}`")
    lines.append("")
    lines.append("## 状态组件")
    for row in summary["state_components"]:
        lines.append(f"- `{row['symbol']}` / {row['name_zh']}：{row['role']}")
    lines.append("")
    lines.append("## 四类最小实验")
    for row in summary["experiment_rows"]:
        lines.append(f"- {row['experiment_id']} / {row['name_zh']}：{row['question']}")
        lines.append(f"  - 聚焦：`{row['focus']}`")
        for pair in row["minimal_pairs"]:
            lines.append(f"  - 例子：`{pair}`")
        for obs in row["observables"]:
            lines.append(f"  - 观测量：`{obs}`")
        lines.append(f"  - 因果预测：{row['causal_prediction']}")
    lines.append("")
    lines.append("## 成功标准")
    for item in summary["success_criteria"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 失败信号")
    for item in summary["failure_modes"]:
        lines.append(f"- {item}")

    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
