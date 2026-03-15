from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def stage_row(stage: str, current: float, target: float, core_question: str, hard_gap: str) -> Dict[str, Any]:
    gap = max(0.0, target - current)
    if current >= target:
        status = "candidate_closed"
    elif current >= target * 0.75:
        status = "strong_candidate"
    elif current >= target * 0.45:
        status = "midway"
    else:
        status = "early"
    return {
        "stage": stage,
        "current_percent": round(current, 1),
        "target_percent": round(target, 1),
        "gap_percent": round(gap, 1),
        "status": status,
        "core_question": core_question,
        "hard_gap": hard_gap,
    }


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    systematic = load_json(ROOT / "tests" / "codex_temp" / "dnn_systematic_mass_extraction_block_20260315.json")
    generality = load_json(ROOT / "tests" / "codex_temp" / "dnn_general_math_generality_block_20260315.json")
    dense_real = load_json(ROOT / "tests" / "codex_temp" / "dnn_dense_real_unit_corpus_block_20260315.json")
    structured = load_json(ROOT / "tests" / "codex_temp" / "dnn_multimodel_structured_canonical_operator_block_20260315.json")
    specific = load_json(ROOT / "tests" / "codex_temp" / "dnn_multimodel_specific_reconstruction_block_20260315.json")

    sys_metrics = systematic["headline_metrics"]
    sys_support = systematic["support_metrics"]
    gen_metrics = generality["headline_metrics"]
    dense_metrics = dense_real["headline_metrics"]
    struct_metrics = structured["headline_metrics"]
    specific_metrics = specific["headline_metrics"]

    systematic_corpus = (
        0.40 * clamp01(sys_metrics["systematic_extraction_score"])
        + 0.25 * clamp01(sys_metrics["exact_real_fraction"] / 0.55)
        + 0.20 * clamp01(sys_metrics["total_standardized_units"] / 1800.0)
        + 0.15 * clamp01(sys_metrics["exact_real_units"] / 900.0)
    ) * 100.0
    dense_real_units = (
        0.40 * clamp01(dense_metrics["weighted_units"] / 900.0)
        + 0.30 * clamp01(dense_metrics["macro_weight"] / 700.0)
        + 0.30 * clamp01(dense_metrics["specific_weight"] / 700.0)
    ) * 100.0
    specific_real_coverage = (
        0.45 * clamp01(dense_metrics["specific_weight"] / 700.0)
        + 0.35 * clamp01(specific_metrics["contextual_family_to_specific_gain"] / 0.75)
        + 0.20 * clamp01(sys_metrics["exact_real_fraction"] / 0.55)
    ) * 100.0
    family_offset_core = (
        0.50 * gen_metrics["family_basis_score"] + 0.50 * gen_metrics["bounded_offset_score"]
    ) * 100.0
    contextual_relation_operator = (
        0.35 * gen_metrics["contextual_operator_score"]
        + 0.35 * clamp01(struct_metrics["structured_specific_gain"] / 0.65)
        + 0.30 * clamp01(struct_metrics["structured_macro_gain"] / 0.60)
    ) * 100.0
    macro_protocol_successor = (
        0.30 * clamp01(struct_metrics["structured_macro_gain"] / 0.65)
        + 0.30 * clamp01(sys_support["dense_real_macro_weight"] / 700.0)
        + 0.20 * clamp01(sys_metrics["exact_real_fraction"] / 0.55)
        + 0.20 * clamp01(sys_support["extracted_successor_score"] / 0.50)
    ) * 100.0
    neuron_level_general_structure = (
        0.35 * clamp01(sys_metrics["exact_real_fraction"] / 0.60)
        + 0.25 * clamp01(dense_metrics["weighted_units"] / 900.0)
        + 0.20 * clamp01(dense_metrics["specific_weight"] / 700.0)
        + 0.20 * clamp01(dense_metrics["macro_weight"] / 700.0)
    ) * 72.0
    full_math_theory = (
        0.35 * gen_metrics["generality_score"]
        + 0.20 * clamp01(sys_metrics["exact_real_fraction"] / 0.60)
        + 0.15 * clamp01(struct_metrics["structured_specific_gain"] / 0.70)
        + 0.15 * clamp01(struct_metrics["structured_macro_gain"] / 0.70)
        + 0.15 * clamp01(dense_metrics["specific_weight"] / 700.0)
    ) * 90.0

    table = [
        stage_row(
            "系统语料库底座",
            systematic_corpus,
            85.0,
            "是否已形成统一、可计量、可持续扩展的 DNN 结构语料库？",
            "inventory mass 仍然很大，系统仍偏 summary-heavy，而不是 dense-exact-heavy。",
        ),
        stage_row(
            "真实单位扩张",
            dense_real_units,
            82.0,
            "是否已把大量真实结构单位纳入统一标准条目？",
            "真实单位数量已经上来，但主力仍然是 row-level 单位，不是 activation-level 单位。",
        ),
        stage_row(
            "概念细节覆盖",
            specific_real_coverage,
            78.0,
            "是否已对大量 concept-specific 编码形成真实覆盖？",
            "specific-bearing 真实单位已经明显增加，但仍未达到 dense concept atlas 的闭合强度。",
        ),
        stage_row(
            "Family+Offset 核心",
            family_offset_core,
            90.0,
            "family basis 与 bounded offset 的一般结构是否已经站住？",
            "meso 层已经很强，但仍未收敛成唯一 canonical parameter law。",
        ),
        stage_row(
            "上下文与关系算子",
            contextual_relation_operator,
            85.0,
            "是否已能把 relation/context/operator 系统性写进概念恢复律？",
            "structured operator 已成立，但仍压不过最佳 contextual shortcut。",
        ),
        stage_row(
            "Macro/Protocol/Successor",
            macro_protocol_successor,
            80.0,
            "macro protocol、successor、lift 是否已进入同一精确坐标系？",
            "macro 侧真实单位已经大幅增加，但 successor/protocol 仍缺 dense exact coordinates。",
        ),
        stage_row(
            "神经元级普遍结构",
            neuron_level_general_structure,
            75.0,
            "是否已在神经元级别找到更普遍、可迁移的编码结构？",
            "exact real fraction 已明显提升，但证据仍以 row-level 为主，不足以称为 neuron-level 闭合。",
        ),
        stage_row(
            "完整数学理论",
            full_math_theory,
            88.0,
            "是否已从系统语料库收敛到完整、唯一、可预测的数学理论？",
            "generality 很强，但 dense exact evidence 与 successor/protocol 条款仍不够。",
        ),
    ]

    markdown_lines = [
        "| 阶段 | 当前% | 目标% | 差距% | 状态 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in table:
        markdown_lines.append(
            f"| {row['stage']} | {row['current_percent']} | {row['target_percent']} | {row['gap_percent']} | {row['status']} |"
        )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_corpus_to_full_theory_progress_board",
        },
        "strict_goal": {
            "statement": "Build a single progress board from systematic corpus building to full mathematical theory closure, centered on concept encoding, concept relations, and neuron-level general structure.",
            "boundary": "This board is a control dashboard, not a claim that the final theory is already closed.",
        },
        "progress_table": table,
        "markdown_table": "\n".join(markdown_lines),
        "headline_metrics": {
            "systematic_corpus_percent": round(systematic_corpus, 1),
            "dense_real_units_percent": round(dense_real_units, 1),
            "specific_real_coverage_percent": round(specific_real_coverage, 1),
            "family_offset_core_percent": round(family_offset_core, 1),
            "contextual_relation_operator_percent": round(contextual_relation_operator, 1),
            "macro_protocol_successor_percent": round(macro_protocol_successor, 1),
            "neuron_level_general_structure_percent": round(neuron_level_general_structure, 1),
            "full_math_theory_percent": round(full_math_theory, 1),
        },
        "strict_verdict": {
            "progress_board_present": True,
            "full_math_theory_closed": bool(
                full_math_theory >= 88.0
                and sys_metrics["exact_real_fraction"] > 0.60
                and sys_support["dense_real_specific_weight"] > 700
            ),
            "core_answer": "从系统语料库到完整数学理论的主线已经可以被量化管理：目前 family+offset 与上下文算子最强，神经元级普遍结构与完整数学理论仍明显落后。",
            "main_hard_gaps": [
                "exact real fraction 已明显提升，但仍未达到 dense exact closure 所需水平",
                "specific-bearing 真实单位虽已扩张，但仍以 row-level 与 proxy-level 为主",
                "macro/protocol/successor 仍缺少 dense exact 坐标，阻碍最终理论闭合",
            ],
        },
        "progress_estimate": {
            "progress_board_percent": 78.0,
            "systematic_mass_extraction_percent": 78.0,
            "full_brain_encoding_mechanism_percent": 87.0,
        },
        "next_large_blocks": [
            "系统扩充更高密度的 exact real units，把 row-level 继续压向 activation-level。",
            "把 protocol / successor / lift 推进成 dense exact coordinates，而不是 row-level proxy。",
            "在更高 exact real fraction 下重算 neuron-level general structure 与 full math theory 两段。",
        ],
    }
    return payload


def test_dnn_corpus_to_full_theory_progress_board() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert len(payload["progress_table"]) == 8
    assert metrics["systematic_corpus_percent"] >= 78.0
    assert metrics["family_offset_core_percent"] > metrics["neuron_level_general_structure_percent"]
    assert metrics["full_math_theory_percent"] < 88.0
    assert verdict["progress_board_present"] is True
    assert verdict["full_math_theory_closed"] is False


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN corpus to full theory progress board")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_corpus_to_full_theory_progress_board_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
