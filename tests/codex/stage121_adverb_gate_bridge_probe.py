#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage121: adverb（副词）-gate（门控）桥探针。

目标：
1. 用静态 embedding（嵌入）层近似区分“路由/控制原型”和“内容/属性原型”。
2. 量化 adverb（副词）是否位于两者之间，而不是落回对象层。
3. 为后续 q / b / g（条件门控场 / 上下文偏置 / 门控路由）动态研究提供桥梁候选。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from stage119_gpt2_embedding_full_vocab_scan import OUTPUT_DIR as STAGE119_OUTPUT_DIR
from stage119_gpt2_embedding_full_vocab_scan import load_embedding_weight
from stage119_gpt2_embedding_full_vocab_scan import run_analysis as run_stage119_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage121_adverb_gate_bridge_probe_20260323"
TRACKED_TYPES = ["adverb", "verb", "function", "noun", "adjective"]
CORE_ADVERB_EXCEPTIONS = {
    "also",
    "thus",
    "often",
    "usually",
    "almost",
    "therefore",
    "never",
    "always",
    "rarely",
    "seldom",
    "soon",
    "later",
    "however",
    "perhaps",
    "already",
    "still",
    "mostly",
    "mainly",
    "simply",
    "exactly",
    "nearly",
    "directly",
    "probably",
    "clearly",
    "actually",
    "eventually",
    "finally",
    "generally",
    "basically",
    "definitely",
    "obviously",
    "frequently",
    "typically",
    "virtually",
    "certainly",
}


def ensure_stage119_rows(input_dir: Path) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    summary_path = input_dir / "summary.json"
    rows_path = input_dir / "word_rows.jsonl"
    if summary_path.exists() and rows_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        rows: List[Dict[str, object]] = []
        with rows_path.open("r", encoding="utf-8-sig") as fh:
            for line in fh:
                rows.append(json.loads(line))
        return summary, rows
    return run_stage119_analysis(output_dir=input_dir)


def l2_normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.linalg.norm(vec).clamp_min(1e-8)


def build_prototype(
    rows: Sequence[Dict[str, object]],
    embed_weight: torch.Tensor,
    predicate,
) -> Tuple[torch.Tensor, int]:
    indices = [int(row["token_id"]) for row in rows if predicate(row)]
    if not indices:
        raise RuntimeError("原型集合为空，无法继续构建 adverb-gate bridge probe")
    mat = embed_weight[indices].float()
    centroid = l2_normalize(mat.mean(dim=0))
    return centroid, len(indices)


def compute_row_probe(
    row: Dict[str, object],
    embed_weight: torch.Tensor,
    prototypes: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    vec = l2_normalize(embed_weight[int(row["token_id"])].float())
    verb_sim = float(torch.dot(vec, prototypes["verb"]).item())
    function_sim = float(torch.dot(vec, prototypes["function"]).item())
    noun_sim = float(torch.dot(vec, prototypes["noun"]).item())
    adjective_sim = float(torch.dot(vec, prototypes["adjective"]).item())
    route_score = (verb_sim + function_sim) / 2.0
    content_score = (noun_sim + adjective_sim) / 2.0
    bridge_margin = route_score - content_score
    balance = 1.0 - abs(verb_sim - function_sim) / (abs(verb_sim) + abs(function_sim) + 1e-8)
    gate_bridge_score = 0.65 * bridge_margin + 0.35 * balance
    return {
        "verb_similarity": verb_sim,
        "function_similarity": function_sim,
        "noun_similarity": noun_sim,
        "adjective_similarity": adjective_sim,
        "route_score": route_score,
        "content_score": content_score,
        "bridge_margin": bridge_margin,
        "action_function_balance": balance,
        "gate_bridge_score": gate_bridge_score,
    }


def top_rows(
    rows: Sequence[Dict[str, object]],
    key: str,
    count: int,
    reverse: bool = True,
) -> List[Dict[str, object]]:
    selected = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)[:count]
    return [
        {
            "word": row["word"],
            "band": row["band"],
            "group": row["group"],
            "lexical_type_score": float(row["lexical_type_score"]),
            "bridge_margin": float(row["bridge_margin"]),
            "action_function_balance": float(row["action_function_balance"]),
            "gate_bridge_score": float(row["gate_bridge_score"]),
        }
        for row in selected
    ]


def is_core_adverb(row: Dict[str, object]) -> bool:
    if str(row["lexical_type"]) != "adverb":
        return False
    word = str(row["word"]).lower()
    return word.endswith("ly") or word in CORE_ADVERB_EXCEPTIONS


def build_summary(rows: Sequence[Dict[str, object]], embed_weight: torch.Tensor) -> Dict[str, object]:
    prototypes = {}
    prototype_seed_counts = {}

    prototypes["verb"], prototype_seed_counts["verb"] = build_prototype(
        rows,
        embed_weight,
        lambda row: row["lexical_type"] == "verb"
        and row["band"] == "macro"
        and row["group"] == "macro_action"
        and float(row["lexical_type_score"]) >= 0.55,
    )
    prototypes["function"], prototype_seed_counts["function"] = build_prototype(
        rows,
        embed_weight,
        lambda row: row["lexical_type"] == "function"
        and row["band"] == "macro"
        and float(row["lexical_type_score"]) >= 0.55,
    )
    prototypes["noun"], prototype_seed_counts["noun"] = build_prototype(
        rows,
        embed_weight,
        lambda row: row["lexical_type"] == "noun"
        and row["band"] == "meso"
        and float(row["lexical_type_score"]) >= 0.55,
    )
    prototypes["adjective"], prototype_seed_counts["adjective"] = build_prototype(
        rows,
        embed_weight,
        lambda row: row["lexical_type"] == "adjective"
        and row["band"] == "micro"
        and float(row["lexical_type_score"]) >= 0.55,
    )

    enriched_rows: List[Dict[str, object]] = []
    for row in rows:
        if row["lexical_type"] not in TRACKED_TYPES:
            continue
        probe = compute_row_probe(row, embed_weight, prototypes)
        enriched_rows.append({**row, **probe})

    by_type: Dict[str, List[Dict[str, object]]] = {t: [] for t in TRACKED_TYPES}
    for row in enriched_rows:
        by_type[str(row["lexical_type"])].append(row)

    core_adverb_rows = [row for row in by_type["adverb"] if is_core_adverb(row)]
    by_type["adverb"] = core_adverb_rows

    type_means = {}
    for lexical_type, type_rows in by_type.items():
        n = max(1, len(type_rows))
        type_means[lexical_type] = {
            "count": len(type_rows),
            "mean_gate_bridge_score": sum(float(row["gate_bridge_score"]) for row in type_rows) / n,
            "mean_bridge_margin": sum(float(row["bridge_margin"]) for row in type_rows) / n,
            "mean_action_function_balance": sum(float(row["action_function_balance"]) for row in type_rows) / n,
        }

    control_mean = (
        type_means["verb"]["mean_gate_bridge_score"] + type_means["function"]["mean_gate_bridge_score"]
    ) / 2.0
    content_mean = (
        type_means["noun"]["mean_gate_bridge_score"] + type_means["adjective"]["mean_gate_bridge_score"]
    ) / 2.0
    adverb_mean = type_means["adverb"]["mean_gate_bridge_score"]
    midpoint = (adverb_mean - content_mean) / max(1e-8, control_mean - content_mean)
    adverb_balance = type_means["adverb"]["mean_action_function_balance"]
    bridge_score = 0.60 * max(0.0, min(1.0, midpoint)) + 0.40 * adverb_balance

    adverb_rows = by_type["adverb"]
    action_leaning = sorted(
        adverb_rows,
        key=lambda row: float(row["verb_similarity"] - row["function_similarity"]),
        reverse=True,
    )[:12]
    control_leaning = sorted(
        adverb_rows,
        key=lambda row: float(row["function_similarity"] - row["verb_similarity"]),
        reverse=True,
    )[:12]
    content_heavy = sorted(adverb_rows, key=lambda row: float(row["bridge_margin"]))[:12]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage121_adverb_gate_bridge_probe",
        "title": "Adverb 门控桥探针",
        "status_short": "gpt2_adverb_gate_bridge_probe_ready",
        "source_stage": "stage119_gpt2_embedding_full_vocab_scan",
        "source_output_dir": str(STAGE119_OUTPUT_DIR),
        "prototype_seed_counts": prototype_seed_counts,
        "core_adverb_count": len(adverb_rows),
        "type_means": type_means,
        "control_gate_mean": float(control_mean),
        "content_gate_mean": float(content_mean),
        "adverb_gate_mean": float(adverb_mean),
        "adverb_midpoint_position": float(midpoint),
        "adverb_action_function_balance_mean": float(adverb_balance),
        "adverb_gate_bridge_score": float(bridge_score),
        "top_gate_adverbs": top_rows(adverb_rows, "gate_bridge_score", 20, reverse=True),
        "action_leaning_adverbs": [
            {
                "word": row["word"],
                "verb_minus_function": float(row["verb_similarity"] - row["function_similarity"]),
                "band": row["band"],
                "group": row["group"],
            }
            for row in action_leaning
        ],
        "control_leaning_adverbs": [
            {
                "word": row["word"],
                "function_minus_verb": float(row["function_similarity"] - row["verb_similarity"]),
                "band": row["band"],
                "group": row["group"],
            }
            for row in control_leaning
        ],
        "content_heavy_adverbs": [
            {
                "word": row["word"],
                "bridge_margin": float(row["bridge_margin"]),
                "band": row["band"],
                "group": row["group"],
            }
            for row in content_heavy
        ],
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage121: Adverb 门控桥探针",
        "",
        "## 核心结果",
        f"- 核心副词样本数: {summary['core_adverb_count']}",
        f"- adverb（副词）门控桥分数: {summary['adverb_gate_bridge_score']:.4f}",
        f"- adverb（副词）门控均值: {summary['adverb_gate_mean']:.4f}",
        f"- control（控制原型）均值: {summary['control_gate_mean']:.4f}",
        f"- content（内容原型）均值: {summary['content_gate_mean']:.4f}",
        f"- adverb（副词）中点位置: {summary['adverb_midpoint_position']:.4f}",
        f"- adverb（副词）动作-功能平衡均值: {summary['adverb_action_function_balance_mean']:.4f}",
        "",
        "## 解释",
        "- 如果 adverb（副词）均值高于 noun（名词）/ adjective（形容词），但低于 verb（动词）/ function（功能词），就说明它更像桥而不是核心控制块。",
        "- 如果动作-功能平衡值很高，说明副词不是只贴动作，也不是只贴功能词，而是同时沾到两边。",
        "",
        "## Top Adverbs",
    ]

    for row in summary["top_gate_adverbs"][:12]:
        lines.append(
            "- "
            f"{row['word']}: gate={row['gate_bridge_score']:.4f}, "
            f"margin={row['bridge_margin']:.4f}, balance={row['action_function_balance']:.4f}, "
            f"{row['band']}/{row['group']}"
        )

    lines.extend(
        [
            "",
            "## 理论提示",
            "- 副词若稳定站在 route/control（路由/控制）与 content（内容）之间，就可以成为 q / b / g 链的优先静态入口。",
            "- 这还不是动态证明，但已经把“副词像桥”从印象推进成了可计算指标。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "STAGE121_ADVERB_GATE_BRIDGE_PROBE_REPORT.md"
    top_csv = output_dir / "top_gate_adverbs.csv"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report_path.write_text(build_report(summary), encoding="utf-8-sig")

    with top_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "word",
                "band",
                "group",
                "lexical_type_score",
                "bridge_margin",
                "action_function_balance",
                "gate_bridge_score",
            ],
        )
        writer.writeheader()
        for row in summary["top_gate_adverbs"]:
            writer.writerow(row)

    return {
        "summary": summary_path,
        "report": report_path,
        "top_gate_adverbs_csv": top_csv,
    }


def run_analysis(
    *,
    input_dir: Path = STAGE119_OUTPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, object]:
    _stage119_summary, rows = ensure_stage119_rows(input_dir)
    embed_weight = load_embedding_weight()
    summary = build_summary(rows, embed_weight)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Adverb 门控桥探针")
    parser.add_argument("--input-dir", default=str(STAGE119_OUTPUT_DIR), help="Stage119 输出目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage121 输出目录")
    args = parser.parse_args()

    summary = run_analysis(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "adverb_gate_bridge_score": summary["adverb_gate_bridge_score"],
                "adverb_midpoint_position": summary["adverb_midpoint_position"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
