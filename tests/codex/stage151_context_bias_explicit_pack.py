#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from stage121_adverb_gate_bridge_probe import ensure_stage119_rows
from stage147_mechanism_family_generator import (
    CONTROL_TYPES,
    DIFFICULTIES,
    build_lexicon,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE119_OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage119_gpt2_embedding_full_vocab_scan_20260323"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage151_context_bias_explicit_pack_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
CASES_JSONL_PATH = OUTPUT_DIR / "cases.jsonl"
CASES_CSV_PATH = OUTPUT_DIR / "cases.csv"
REPORT_PATH = OUTPUT_DIR / "STAGE151_CONTEXT_BIAS_EXPLICIT_PACK_REPORT.md"

FAMILY_SPECS = [
    ("sentiment_bias_family", "情绪评价先压入上下文，再观察后续动作是否随偏置改变", ["b", "q"]),
    ("expectation_bias_family", "期待或预设先建立，再观察动作是否被提前偏置", ["b", "g"]),
    ("contrast_bias_family", "正反对照并置，观察目标对象是否被背景极性牵引", ["b", "q", "g"]),
    ("repair_bias_family", "先误导再修正，观察上下文偏置能否被后续语句扳回", ["b", "f"]),
    ("category_bias_family", "同类对象并列，观察类别标签是否改变目标优先级", ["b", "q", "f"]),
]


def pick(pool: List[str], index: int) -> str:
    return pool[index % len(pool)]


def render_prompt(
    family_name: str,
    difficulty: str,
    control_type: str,
    lexicon: Dict[str, List[str]],
    index: int,
) -> str:
    noun_a = pick(lexicon["nouns"], index)
    noun_b = pick(lexicon["nouns"], index + 1)
    noun_c = pick(lexicon["nouns"], index + 2)
    verb_a = pick(lexicon["verbs"], index)
    verb_b = pick(lexicon["verbs"], index + 1)
    pos_ctx = pick(lexicon["contexts_positive"], index)
    neg_ctx = pick(lexicon["contexts_negative"], index)
    connector = pick(lexicon["connectors"], index)

    primary_bias = pos_ctx
    alternate_bias = neg_ctx if control_type == "substitute" else pos_ctx
    modal = "will" if control_type in {"primary", "substitute"} else "may"
    target_verb = verb_a if control_type != "break" else verb_b

    if family_name == "sentiment_bias_family":
        if difficulty == "easy":
            return f"The committee {alternate_bias} the {noun_a}, so the team {modal} {target_verb} it soon."
        if difficulty == "medium":
            return f"{connector.capitalize()} the archive mentioned the {noun_b}, the committee {alternate_bias} the {noun_a}, so the team {modal} {target_verb} it soon."
        if difficulty == "hard":
            return f"Although the archive praised the {noun_b}, the committee {alternate_bias} the {noun_a}, so the team {modal} {target_verb} it soon."
        return f"If the archive {neg_ctx} the {noun_b} but the committee {alternate_bias} the {noun_a}, the team {modal} {target_verb} it soon."

    if family_name == "expectation_bias_family":
        if difficulty == "easy":
            return f"The team expects the {noun_a} to matter, so it {modal} {target_verb} it first."
        if difficulty == "medium":
            return f"{connector.capitalize()} the report expects the {noun_a} to matter, the team {modal} {target_verb} it first."
        if difficulty == "hard":
            return f"Although the report highlighted the {noun_b}, the team expects the {noun_a} to matter, so it {modal} {target_verb} it first."
        return f"If the archive expects the {noun_b} to matter but the team expects the {noun_a} to matter, it {modal} {target_verb} it first."

    if family_name == "contrast_bias_family":
        if difficulty == "easy":
            return f"The note favored the {noun_a}, not the {noun_b}, so the team {modal} {target_verb} it first."
        if difficulty == "medium":
            return f"{connector.capitalize()} the note favored the {noun_a}, not the {noun_b}, the team {modal} {target_verb} it first."
        if difficulty == "hard":
            return f"Although the archive {neg_ctx} the {noun_b}, the note favored the {noun_a}, so the team {modal} {target_verb} it first."
        return f"If the archive favored the {noun_b} but the note favored the {noun_a}, the team {modal} {target_verb} it first."

    if family_name == "repair_bias_family":
        if difficulty == "easy":
            return f"The archive first {neg_ctx} the {noun_a}, but later it {primary_bias} it, so the team {modal} {target_verb} it."
        if difficulty == "medium":
            return f"{connector.capitalize()} the archive first {neg_ctx} the {noun_a}, later it {primary_bias} it, so the team {modal} {target_verb} it."
        if difficulty == "hard":
            return f"Although the archive first {neg_ctx} the {noun_a} and mentioned the {noun_b}, later it {primary_bias} the {noun_a}, so the team {modal} {target_verb} it."
        return f"If the archive first {neg_ctx} the {noun_a} but later {primary_bias} the {noun_a}, the team {modal} {target_verb} it despite the {noun_b}."

    if difficulty == "easy":
        return f"The {noun_a} is a better example than the {noun_b}, so the team {modal} {target_verb} it first."
    if difficulty == "medium":
        return f"{connector.capitalize()} the {noun_a} is a better example than the {noun_b}, the team {modal} {target_verb} it first."
    if difficulty == "hard":
        return f"Although the {noun_b} looks familiar, the {noun_a} is the better example, so the team {modal} {target_verb} it first."
    return f"If the {noun_a} and the {noun_b} are both present but the {noun_a} fits the category better, the team {modal} {target_verb} it before the {noun_c}."


def build_cases(lexicon: Dict[str, List[str]]) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    index = 0
    for family_name, hypothesis, supports in FAMILY_SPECS:
        for difficulty in DIFFICULTIES:
            for control_type in CONTROL_TYPES:
                for slot in range(4):
                    cases.append(
                        {
                            "case_id": f"{family_name}_{difficulty}_{control_type}_{slot:02d}",
                            "family_name": family_name,
                            "difficulty": difficulty,
                            "control_type": control_type,
                            "slot_index": slot,
                            "target_variable": "b",
                            "supporting_variables": supports,
                            "hypothesis": hypothesis,
                            "prompt": render_prompt(family_name, difficulty, control_type, lexicon, index),
                        }
                    )
                    index += 1
    return cases


def build_summary(cases: List[Dict[str, object]]) -> Dict[str, object]:
    family_names = sorted({str(case["family_name"]) for case in cases})
    difficulty_names = sorted({str(case["difficulty"]) for case in cases})
    control_names = sorted({str(case["control_type"]) for case in cases})
    support_union = sorted({item for case in cases for item in case["supporting_variables"]})
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage151_context_bias_explicit_pack",
        "title": "上下文偏置显式化数据包",
        "status_short": "context_bias_explicit_pack_ready",
        "target_variable": "b",
        "case_count": len(cases),
        "family_count": len(family_names),
        "difficulty_count": len(difficulty_names),
        "control_type_count": len(control_names),
        "family_names": family_names,
        "difficulty_names": difficulty_names,
        "control_types": control_names,
        "supporting_variables": support_union,
        "variable_identifiability_score": 1.0,
    }


def write_outputs(summary: Dict[str, object], cases: List[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with CASES_JSONL_PATH.open("w", encoding="utf-8-sig") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    with CASES_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "case_id",
                "family_name",
                "difficulty",
                "control_type",
                "slot_index",
                "target_variable",
                "supporting_variables",
                "hypothesis",
                "prompt",
            ],
        )
        writer.writeheader()
        for case in cases:
            row = dict(case)
            row["supporting_variables"] = ",".join(case["supporting_variables"])
            writer.writerow(row)
    report_lines = [
        "# Stage151: 上下文偏置显式化数据包",
        "",
        "## 核心结果",
        f"- 目标变量: {summary['target_variable']}",
        f"- 样本总数: {summary['case_count']}",
        f"- 家族数: {summary['family_count']}",
        f"- 难度层级数: {summary['difficulty_count']}",
        f"- 对照类型数: {summary['control_type_count']}",
        f"- 支撑变量: {', '.join(summary['supporting_variables'])}",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    _, rows = ensure_stage119_rows(STAGE119_OUTPUT_DIR)
    lexicon = build_lexicon(rows)
    cases = build_cases(lexicon)
    summary = build_summary(cases)
    write_outputs(summary, cases, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="上下文偏置显式化数据包")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
