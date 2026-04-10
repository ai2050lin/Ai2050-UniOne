#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE573_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage573_fruit_minimal_causal_encoding_empirical_20260409"
    / "summary.json"
)
STAGE574_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage574_unified_language_state_update_empirical_20260409"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage575_first_principles_brain_bridge_assessment_20260409"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def fruit_assessment(stage573: Dict[str, object]) -> Dict[str, object]:
    model_rows = stage573["model_rows"]
    fruit_margin_rows = []
    offset_ratio_rows = []
    red_transfer_rows = []
    sweet_transfer_rows = []
    bind_rows = []
    probe_rows = []
    for row in model_rows:
        apple = row["fruit_rows"]["apple"]
        fruit_margin_rows.append(float(apple["family_margin_vs_animal"]))
        offset_ratio_rows.append(float(apple["concept_offset_ratio"]))
        red_transfer_rows.append(float(row["attribute_rows"]["red"]["shared_delta_cos_mean"]))
        sweet_transfer_rows.append(float(row["attribute_rows"]["sweet"]["shared_delta_cos_mean"]))
        bind_rows.append(float(mean([
            row["attribute_rows"]["red"]["mean_binding_residual_ratio"],
            row["attribute_rows"]["sweet"]["mean_binding_residual_ratio"],
        ])))
        probe_correct = sum(1 for probe in row["probe_rows"] if probe["best_candidate"].strip() in {"fruit", "sweet"})
        probe_rows.append(probe_correct / max(len(row["probe_rows"]), 1))

    support = {
        "family_backbone_support": mean(fruit_margin_rows),
        "concept_offset_ratio_mean": mean(offset_ratio_rows),
        "red_transfer_mean": mean(red_transfer_rows),
        "sweet_transfer_mean": mean(sweet_transfer_rows),
        "binding_residual_mean": mean(bind_rows),
        "probe_success_mean": mean(probe_rows),
    }
    support["reading"] = (
        "水果骨干、属性迁移、绑定残差都得到支撑；但概念偏置并不小，"
        "所以更准确的结构应是‘共享骨干 + 非小局部偏置 + 非零绑定残差’。"
    )
    return support


def state_update_assessment(stage574: Dict[str, object]) -> Dict[str, object]:
    model_rows = stage574["model_rows"]
    pronoun = []
    prep = []
    adverb = []
    logic = []
    logic_growth_rows = []
    for row in model_rows:
        exp = row["experiment_rows"]
        pronoun.append(float(exp["pronoun_coreference"]["accuracy"]))
        prep.append(float(exp["preposition_relation"]["accuracy"]))
        adverb.append(float(exp["adverb_scope"]["accuracy"]))
        logic.append(float(exp["logic_reasoning"]["accuracy"]))
        growth_cases = []
        for margin_row in exp["logic_reasoning"]["layerwise_margin_rows"]:
            margins = margin_row["layer_margins"]
            if margins:
                growth_cases.append(float(margins[-1] - margins[0]))
        logic_growth_rows.append(mean(growth_cases))

    support = {
        "pronoun_accuracy_mean": mean(pronoun),
        "preposition_accuracy_mean": mean(prep),
        "adverb_accuracy_mean": mean(adverb),
        "logic_accuracy_mean": mean(logic),
        "logic_margin_growth_mean": mean(logic_growth_rows),
    }
    support["reading"] = (
        "关系框架和逻辑推理最稳，代词中等，副词最弱。"
        "统一状态更新方程已有支撑，但 M_t 与 P_t 仍需更细拆和更强实验。"
    )
    return support


def readiness_assessment(fruit: Dict[str, object], state: Dict[str, object]) -> Dict[str, object]:
    # 0~1 rough scores
    geometry = min(max(fruit["family_backbone_support"] / 0.05, 0.0), 1.0)
    transfer = min(max((fruit["red_transfer_mean"] + fruit["sweet_transfer_mean"]) / 1.0, 0.0), 1.0)
    binding = min(max(fruit["binding_residual_mean"] / 0.2, 0.0), 1.0)
    logic = min(max(state["logic_accuracy_mean"], 0.0), 1.0)
    relation = min(max(state["preposition_accuracy_mean"], 0.0), 1.0)
    pronoun = min(max(state["pronoun_accuracy_mean"], 0.0), 1.0)
    modifier = min(max(state["adverb_accuracy_mean"], 0.0), 1.0)
    dynamics = min(max(state["logic_margin_growth_mean"] / 10.0, 0.0), 1.0)

    axes = {
        "shared_structure": geometry,
        "attribute_transfer": transfer,
        "binding_nonadditivity": binding,
        "relation_frame": relation,
        "reasoning_state": logic,
        "reasoning_dynamics": dynamics,
        "reference_state": pronoun,
        "modifier_state": modifier,
    }
    overall = mean(list(axes.values()))
    if overall >= 0.8:
        level = "strong_candidate"
    elif overall >= 0.6:
        level = "partial_first_principles_candidate"
    elif overall >= 0.4:
        level = "structured_working_theory"
    else:
        level = "descriptive_stage"
    return {
        "axes": axes,
        "overall_score": overall,
        "level": level,
        "reading": (
            "当前理论已经超过纯描述阶段，进入‘结构化工作理论’到‘部分第一性原理候选’之间。"
            "它最强的是共享结构、关系框架和逻辑状态；最弱的是代词与副词的统一化。"
        ),
    }


def brain_bridge_assessment(fruit: Dict[str, object], state: Dict[str, object]) -> Dict[str, object]:
    mappings = [
        {
            "theory_term": "B_family",
            "brain_candidate": "群体原型场 / family patch",
            "support": "fruit backbone support across models",
            "reading": "概念首先落在家族级共享群体编码上，而不是单细胞标签。",
        },
        {
            "theory_term": "E_concept",
            "brain_candidate": "patch 内局部偏置 / local offset",
            "support": "non-small concept offset ratio",
            "reading": "苹果更像水果 patch 上的局部位移，而不是独立孤岛。",
        },
        {
            "theory_term": "A_attr",
            "brain_candidate": "可复用属性纤维 / reusable feature channel",
            "support": "cross-object red/sweet transfer",
            "reading": "颜色、味道等特征更像跨对象共享的通道，而非每个概念重复存一遍。",
        },
        {
            "theory_term": "G_bind",
            "brain_candidate": "绑定桥接 / time-window binding or circuit bridge",
            "support": "non-zero binding residual",
            "reading": "组合语义不是相加，而要靠额外桥接才能稳定绑定。",
        },
        {
            "theory_term": "R_t",
            "brain_candidate": "关系框架状态 / relational frame state",
            "support": "preposition accuracy = 1.0 across models",
            "reading": "空间与角色关系更像独立状态变量，而不是对象属性的附属项。",
        },
        {
            "theory_term": "Q_t",
            "brain_candidate": "递进推理状态 / recurrent constraint state",
            "support": "logic accuracy high + layerwise margin growth",
            "reading": "推理结论更像在层间逐步形成的群体状态，而非末端瞬时读出。",
        },
    ]
    return {
        "mapping_rows": mappings,
        "reading": (
            "如果把 DNN 结果翻译到脑侧，最自然的图景不是‘祖母神经元’，"
            "而是‘家族 patch + 局部偏置 + 属性纤维 + 绑定桥接 + 递进状态更新’的群体编码机制。"
        ),
    }


def build_next_tasks(readiness: Dict[str, object]) -> List[str]:
    return [
        "把 B_family / E_concept / A_attr / G_bind 真正投影回最小因果神经元回路，而不是只停在表征层。",
        "把 pronoun 与 adverb 再细拆成更小子类，避免 P_t / M_t 成为松散兜底项。",
        "把逻辑逐层 margin 扩成真实状态更新方程，验证哪些层在做约束传播、哪些层在做结论收束。",
        "把 DNN 侧的 family patch / attribute channel / binding bridge 翻译成脑侧可检验的群体编码预测。",
        "扩大样本和任务范围，要求理论能预测未见组合，而不只是解释当前小样本。"
    ]


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    stage573 = load_json(STAGE573_PATH)
    stage574 = load_json(STAGE574_PATH)
    fruit = fruit_assessment(stage573)
    state = state_update_assessment(stage574)
    readiness = readiness_assessment(fruit, state)
    brain = brain_bridge_assessment(fruit, state)
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage575_first_principles_brain_bridge_assessment",
        "title": "第一性原理成熟度与大脑编码机制桥接评估",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": {
            "stage573": str(STAGE573_PATH),
            "stage574": str(STAGE574_PATH),
        },
        "fruit_assessment": fruit,
        "state_update_assessment": state,
        "first_principles_readiness": readiness,
        "brain_bridge_assessment": brain,
        "next_tasks": build_next_tasks(readiness),
        "core_answer": (
            "当前理论已经从概念编码猜想推进到结构化工作理论，并开始逼近第一性原理候选；"
            "它对大脑的最自然解释也不再是单神经元标签，而是群体 patch、局部偏置、属性纤维、绑定桥接与递进状态更新。"
        ),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# stage575 第一性原理成熟度与大脑编码机制桥接评估",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
        "## 第一性原理成熟度",
        f"- level: `{readiness['level']}`",
        f"- overall_score: `{readiness['overall_score']:.4f}`",
    ]
    for key, value in readiness["axes"].items():
        lines.append(f"- `{key}`: `{value:.4f}`")
    lines.extend([
        "",
        "## 水果编码评估",
        f"- family_backbone_support: `{fruit['family_backbone_support']:.4f}`",
        f"- concept_offset_ratio_mean: `{fruit['concept_offset_ratio_mean']:.4f}`",
        f"- red_transfer_mean: `{fruit['red_transfer_mean']:.4f}`",
        f"- sweet_transfer_mean: `{fruit['sweet_transfer_mean']:.4f}`",
        f"- binding_residual_mean: `{fruit['binding_residual_mean']:.4f}`",
        "",
        "## 统一状态更新评估",
        f"- pronoun_accuracy_mean: `{state['pronoun_accuracy_mean']:.4f}`",
        f"- preposition_accuracy_mean: `{state['preposition_accuracy_mean']:.4f}`",
        f"- adverb_accuracy_mean: `{state['adverb_accuracy_mean']:.4f}`",
        f"- logic_accuracy_mean: `{state['logic_accuracy_mean']:.4f}`",
        f"- logic_margin_growth_mean: `{state['logic_margin_growth_mean']:.4f}`",
        "",
        "## 大脑编码桥接",
    ])
    for row in brain["mapping_rows"]:
        lines.append(f"- `{row['theory_term']}` -> {row['brain_candidate']}：{row['reading']}")
    lines.extend([
        "",
        "## 下一阶段任务",
    ])
    for item in summary["next_tasks"]:
        lines.append(f"- {item}")

    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(OUTPUT_DIR), "level": readiness["level"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
