#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from stage119_gpt2_embedding_full_vocab_scan import OUTPUT_DIR as STAGE119_OUTPUT_DIR
from stage121_adverb_gate_bridge_probe import ensure_stage119_rows
from stage147_mechanism_family_generator import (
    CONTROL_TYPES,
    DIFFICULTIES,
    build_family_catalog,
    build_lexicon,
    render_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage148_variable_identifiability_dataset_pack_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
BUNDLE_PATH = OUTPUT_DIR / "variable_bundle_catalog.json"
PACK_JSONL_PATH = OUTPUT_DIR / "identifiability_cases.jsonl"
PACK_CSV_PATH = OUTPUT_DIR / "identifiability_cases.csv"
REPORT_PATH = OUTPUT_DIR / "STAGE148_VARIABLE_IDENTIFIABILITY_DATASET_PACK_REPORT.md"

VARIABLE_ORDER = ["a", "r", "f", "g", "q", "b"]
VARIABLE_FAMILY_MAP = {
    "a": ["anchor_subject_family"],
    "r": ["pronoun_recovery_family", "ellipsis_recovery_family"],
    "f": ["anchor_subject_family", "pronoun_recovery_family", "ellipsis_recovery_family", "late_repair_family"],
    "g": ["adverb_route_family", "context_bias_family", "late_repair_family"],
    "q": ["adverb_route_family", "context_bias_family"],
    "b": ["context_bias_family", "late_repair_family"],
}
VARIABLE_CONTRAST_MAP = {
    "a": ["primary", "weaken", "break"],
    "r": ["primary", "substitute", "break"],
    "f": ["primary", "weaken", "break"],
    "g": ["primary", "substitute", "break"],
    "q": ["primary", "substitute", "weaken"],
    "b": ["primary", "substitute", "weaken", "break"],
}
VARIABLE_HYPOTHESIS = {
    "a": "在保持语义对象基本不变时，早层定锚应对结构破坏最敏感。",
    "r": "回返一致性应对回指与省略破坏敏感，并在替代指代时出现可区分变化。",
    "f": "后层续接应在弱化样本中保留，但在结构破坏样本中明显下降。",
    "g": "门控路由应对副词与动作选择类对照最敏感，并在破坏样本中断裂。",
    "q": "条件门控场应对上下文条件与弱化条件同时敏感，但不完全等同于路由变量。",
    "b": "上下文偏置应在正负语境替换中发生方向性偏移，并在长链补救中保留影响。",
}


def load_cached_summary(output_dir: Path) -> Dict[str, object] | None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    return None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_variable_bundle_catalog() -> List[Dict[str, object]]:
    family_catalog = {row["family_name"]: row for row in build_family_catalog()}
    bundle_rows: List[Dict[str, object]] = []
    for variable_name in VARIABLE_ORDER:
        families = VARIABLE_FAMILY_MAP[variable_name]
        target_variables = sorted({value for family_name in families for value in family_catalog[family_name]["target_variables"]})
        bundle_rows.append(
            {
                "variable_name": variable_name,
                "family_names": families,
                "contrast_types": VARIABLE_CONTRAST_MAP[variable_name],
                "difficulties": list(DIFFICULTIES),
                "supporting_variables": target_variables,
                "hypothesis": VARIABLE_HYPOTHESIS[variable_name],
            }
        )
    return bundle_rows


def build_pack_rows(bundle_catalog: Sequence[Dict[str, object]], lexicon: Dict[str, List[str]]) -> List[Dict[str, object]]:
    pack_rows: List[Dict[str, object]] = []
    case_index = 0
    for bundle in bundle_catalog:
        variable_name = str(bundle["variable_name"])
        for family_name in bundle["family_names"]:
            for difficulty in bundle["difficulties"]:
                for contrast_type in bundle["contrast_types"]:
                    for lexeme_slot in range(4):
                        prompt = render_prompt(family_name, difficulty, contrast_type, lexicon, lexeme_slot)
                        pack_rows.append(
                            {
                                "case_id": f"stage148_{case_index:05d}",
                                "variable_name": variable_name,
                                "family_name": family_name,
                                "difficulty": difficulty,
                                "contrast_type": contrast_type,
                                "lexeme_slot": lexeme_slot,
                                "supporting_variables": list(bundle["supporting_variables"]),
                                "hypothesis": bundle["hypothesis"],
                                "prompt": prompt,
                            }
                        )
                        case_index += 1
    return pack_rows


def build_summary(bundle_catalog: Sequence[Dict[str, object]], pack_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    family_counter = Counter(str(row["family_name"]) for row in pack_rows)
    difficulty_counter = Counter(str(row["difficulty"]) for row in pack_rows)
    contrast_counter = Counter(str(row["contrast_type"]) for row in pack_rows)
    variable_counter = Counter(str(row["variable_name"]) for row in pack_rows)
    family_span_per_variable = defaultdict(set)
    contrast_span_per_variable = defaultdict(set)
    difficulty_span_per_variable = defaultdict(set)
    support_counter = Counter()

    for row in pack_rows:
        variable_name = str(row["variable_name"])
        family_span_per_variable[variable_name].add(str(row["family_name"]))
        contrast_span_per_variable[variable_name].add(str(row["contrast_type"]))
        difficulty_span_per_variable[variable_name].add(str(row["difficulty"]))
        for value in row["supporting_variables"]:
            support_counter.update([str(value)])

    bundle_rows = []
    for bundle in bundle_catalog:
        variable_name = str(bundle["variable_name"])
        family_coverage = len(family_span_per_variable[variable_name]) / max(1, len(bundle["family_names"]))
        contrast_coverage = len(contrast_span_per_variable[variable_name]) / max(1, len(bundle["contrast_types"]))
        difficulty_coverage = len(difficulty_span_per_variable[variable_name]) / max(1, len(bundle["difficulties"]))
        bundle_score = 0.35 * family_coverage + 0.35 * contrast_coverage + 0.20 * difficulty_coverage + 0.10 * clamp01(variable_counter[variable_name] / 48.0)
        bundle_rows.append(
            {
                **bundle,
                "case_count": int(variable_counter[variable_name]),
                "family_coverage": family_coverage,
                "contrast_coverage": contrast_coverage,
                "difficulty_coverage": difficulty_coverage,
                "variable_identifiability_score": bundle_score,
            }
        )

    overall_score = sum(float(row["variable_identifiability_score"]) for row in bundle_rows) / len(bundle_rows)
    weakest_variable_name = min(bundle_rows, key=lambda row: float(row["variable_identifiability_score"]))["variable_name"]
    strongest_variable_name = max(bundle_rows, key=lambda row: float(row["variable_identifiability_score"]))["variable_name"]
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage148_variable_identifiability_dataset_pack",
        "title": "变量可辨识数据包块",
        "status_short": "variable_identifiability_dataset_ready",
        "variable_count": len(bundle_catalog),
        "case_count": len(pack_rows),
        "family_count": len(family_counter),
        "difficulty_count": len(difficulty_counter),
        "contrast_type_count": len(contrast_counter),
        "overall_identifiability_score": overall_score,
        "weakest_variable_name": weakest_variable_name,
        "strongest_variable_name": strongest_variable_name,
        "family_case_counts": dict(family_counter),
        "difficulty_counts": dict(difficulty_counter),
        "contrast_type_counts": dict(contrast_counter),
        "variable_case_counts": dict(variable_counter),
        "support_variable_counts": dict(support_counter),
        "variable_rows": bundle_rows,
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage148: 变量可辨识数据包块",
        "",
        "## 核心结果",
        f"- 变量数: {summary['variable_count']}",
        f"- 样本总数: {summary['case_count']}",
        f"- 家族数: {summary['family_count']}",
        f"- 难度层数: {summary['difficulty_count']}",
        f"- 对照类型数: {summary['contrast_type_count']}",
        f"- 总体可辨识分数: {summary['overall_identifiability_score']:.4f}",
        f"- 最强变量: {summary['strongest_variable_name']}",
        f"- 最弱变量: {summary['weakest_variable_name']}",
        "",
        "## 变量包目录",
    ]
    for row in summary["variable_rows"]:
        lines.append(
            "- "
            f"{row['variable_name']}: "
            f"families={','.join(row['family_names'])}; "
            f"contrasts={','.join(row['contrast_types'])}; "
            f"score={row['variable_identifiability_score']:.4f}; "
            f"hypothesis={row['hypothesis']}"
        )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], bundle_catalog: Sequence[Dict[str, object]], pack_rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    BUNDLE_PATH.write_text(json.dumps(list(bundle_catalog), ensure_ascii=False, indent=2), encoding="utf-8-sig")
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8-sig")

    with PACK_JSONL_PATH.open("w", encoding="utf-8-sig") as fh:
        for row in pack_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = ["case_id", "variable_name", "family_name", "difficulty", "contrast_type", "lexeme_slot", "supporting_variables", "hypothesis", "prompt"]
    with PACK_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in pack_rows:
            writer.writerow({**row, "supporting_variables": ",".join(row["supporting_variables"])})


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force:
        cached = load_cached_summary(output_dir)
        if cached is not None:
            return cached
    _, rows = ensure_stage119_rows(STAGE119_OUTPUT_DIR)
    lexicon = build_lexicon(rows)
    bundle_catalog = build_variable_bundle_catalog()
    pack_rows = build_pack_rows(bundle_catalog, lexicon)
    summary = build_summary(bundle_catalog, pack_rows)
    write_outputs(summary, bundle_catalog, pack_rows, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="变量可辨识数据包块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重算")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
