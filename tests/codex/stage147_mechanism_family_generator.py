#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from stage119_gpt2_embedding_full_vocab_scan import OUTPUT_DIR as STAGE119_OUTPUT_DIR
from stage121_adverb_gate_bridge_probe import ensure_stage119_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage147_mechanism_family_generator_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
CATALOG_PATH = OUTPUT_DIR / "family_catalog.json"
CASES_JSONL_PATH = OUTPUT_DIR / "cases.jsonl"
CASES_CSV_PATH = OUTPUT_DIR / "cases.csv"
REPORT_PATH = OUTPUT_DIR / "STAGE147_MECHANISM_FAMILY_GENERATOR_REPORT.md"

DIFFICULTIES = ["easy", "medium", "hard", "adversarial"]
CONTROL_TYPES = ["primary", "substitute", "weaken", "break"]


def load_cached_summary(output_dir: Path) -> Dict[str, object] | None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    return None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def select_preferred(rows: Sequence[Dict[str, object]], words: Sequence[str], lexical_type: str, limit: int) -> List[str]:
    available = {str(row["word"]).lower(): row for row in rows if str(row["lexical_type"]) == lexical_type}
    picked: List[str] = []
    for word in words:
        if word in available and word not in picked:
            picked.append(word)
        if len(picked) >= limit:
            return picked

    fallback = [
        row for row in rows
        if str(row["lexical_type"]) == lexical_type
        and str(row["word"]).isalpha()
        and len(str(row["word"])) >= 3
    ]
    fallback.sort(
        key=lambda row: (
            float(row.get("lexical_type_score", 0.0)),
            float(row.get("effective_encoding_score", 0.0)),
        ),
        reverse=True,
    )
    for row in fallback:
        word = str(row["word"]).lower()
        if word not in picked:
            picked.append(word)
        if len(picked) >= limit:
            break
    return picked


def build_lexicon(rows: Sequence[Dict[str, object]]) -> Dict[str, List[str]]:
    return {
        "nouns": select_preferred(
            rows,
            ["apple", "engine", "teacher", "planet", "language", "justice", "river", "robot", "garden", "memory"],
            "noun",
            8,
        ),
        "verbs": select_preferred(
            rows,
            ["run", "build", "change", "create", "keep", "move", "give", "make"],
            "verb",
            6,
        ),
        "adverbs": select_preferred(
            rows,
            ["quickly", "slowly", "clearly", "suddenly", "actually", "eventually"],
            "adverb",
            6,
        ),
        "adjectives": select_preferred(
            rows,
            ["red", "large", "cold", "smooth", "bright", "simple"],
            "adjective",
            6,
        ),
        "contexts_positive": ["praised", "trusted", "favored", "recommended"],
        "contexts_negative": ["doubted", "ignored", "delayed", "blocked"],
        "connectors": ["if", "unless", "although", "because", "while", "but"],
    }


def pick_word(pool: Sequence[str], index: int) -> str:
    return pool[index % len(pool)]


def render_prompt(family_name: str, difficulty: str, control_type: str, lexicon: Dict[str, List[str]], index: int) -> str:
    noun_a = pick_word(lexicon["nouns"], index)
    noun_b = pick_word(lexicon["nouns"], index + 1)
    noun_c = pick_word(lexicon["nouns"], index + 2)
    verb_a = pick_word(lexicon["verbs"], index)
    verb_b = pick_word(lexicon["verbs"], index + 1)
    adverb_a = pick_word(lexicon["adverbs"], index)
    adjective_a = pick_word(lexicon["adjectives"], index)
    pos_ctx = pick_word(lexicon["contexts_positive"], index)
    neg_ctx = pick_word(lexicon["contexts_negative"], index)
    connector = pick_word(lexicon["connectors"], index)

    if family_name == "anchor_subject_family":
        prompt_map = {
            ("easy", "primary"): f"The {noun_a} stayed nearby while the team described the {noun_b}.",
            ("easy", "substitute"): f"The {noun_a} stayed nearby while the team described the old {noun_b}.",
            ("easy", "weaken"): f"The {noun_a} stayed nearby while the team briefly mentioned the {noun_b}.",
            ("easy", "break"): f"Nearby stayed the {noun_a} while the {noun_b} described the team.",
            ("medium", "primary"): f"The {noun_a} that the archive stored stayed nearby while the team described the {noun_b}.",
            ("medium", "substitute"): f"The {noun_a} that the archive stored stayed nearby while the team described the {adjective_a} {noun_b}.",
            ("medium", "weaken"): f"The {noun_a} that the archive stored was noted while the team described the {noun_b}.",
            ("medium", "break"): f"The archive stored the {noun_b} while nearby stayed the {noun_a} described by the team.",
            ("hard", "primary"): f"Although the {noun_b} looked ordinary, the {noun_a} stayed nearby while the team described the report.",
            ("hard", "substitute"): f"Although the {noun_b} looked ordinary, the {noun_a} stayed nearby while the team described the {adjective_a} report.",
            ("hard", "weaken"): f"Although the {noun_b} looked ordinary, the {noun_a} was later noted while the team described the report.",
            ("hard", "break"): f"Although ordinary looked the {noun_b}, nearby the {noun_a} stayed while the report described the team.",
            ("adversarial", "primary"): f"The {noun_a}, not the {noun_b}, stayed nearby while the team described the {noun_c}.",
            ("adversarial", "substitute"): f"The {noun_a}, not the {noun_b}, stayed nearby while the team described the {adjective_a} {noun_c}.",
            ("adversarial", "weaken"): f"The {noun_a}, not the {noun_b}, was only briefly noted while the team described the {noun_c}.",
            ("adversarial", "break"): f"Not the {noun_b} but nearby stayed the {noun_a} while the {noun_c} described the team.",
        }
        return prompt_map[(difficulty, control_type)]

    if family_name == "adverb_route_family":
        prompt_map = {
            ("easy", "primary"): f"The team will {adverb_a} {verb_a} the {noun_a} today.",
            ("easy", "substitute"): f"The team will {adjective_a} {verb_a} the {noun_a} today.",
            ("easy", "weaken"): f"The team may {adverb_a} {verb_a} the {noun_a} today.",
            ("easy", "break"): f"The team will {verb_a} {adverb_a} the {noun_a} today.",
            ("medium", "primary"): f"Because the report changed, the team will {adverb_a} {verb_a} the {noun_a} today.",
            ("medium", "substitute"): f"Because the report changed, the team will {adjective_a} {verb_a} the {noun_a} today.",
            ("medium", "weaken"): f"Because the report changed, the team may {adverb_a} {verb_a} the {noun_a} today.",
            ("medium", "break"): f"Because the report changed, the team will {verb_a} the {noun_a} {adverb_a} today.",
            ("hard", "primary"): f"Although the archive mentioned the {noun_b}, the team will {adverb_a} {verb_a} the {noun_a} today.",
            ("hard", "substitute"): f"Although the archive mentioned the {noun_b}, the team will {adjective_a} {verb_a} the {noun_a} today.",
            ("hard", "weaken"): f"Although the archive mentioned the {noun_b}, the team may {adverb_a} {verb_a} the {noun_a} today.",
            ("hard", "break"): f"Although the archive mentioned the {noun_b}, the team will {verb_a} the {noun_a} {adverb_a} today.",
            ("adversarial", "primary"): f"If the team {verb_a} the {noun_a}, it will {adverb_a} {verb_b} the {noun_b} tomorrow.",
            ("adversarial", "substitute"): f"If the team {verb_a} the {noun_a}, it will {adjective_a} {verb_b} the {noun_b} tomorrow.",
            ("adversarial", "weaken"): f"If the team {verb_a} the {noun_a}, it may {adverb_a} {verb_b} the {noun_b} tomorrow.",
            ("adversarial", "break"): f"If the team {verb_a} the {noun_a}, it will {verb_b} the {noun_b} {adverb_a} tomorrow.",
        }
        return prompt_map[(difficulty, control_type)]

    if family_name == "pronoun_recovery_family":
        prompt_map = {
            ("easy", "primary"): f"The {noun_a} appeared in the archive. Later the team rechecked it before sunset.",
            ("easy", "substitute"): f"The {noun_a} appeared in the archive. Later the team rechecked that object before sunset.",
            ("easy", "weaken"): f"The {noun_a} appeared in the archive. Much later the team rechecked it before sunset.",
            ("easy", "break"): f"The {noun_a} appeared in the archive. Later the team rechecked them before sunset.",
            ("medium", "primary"): f"The {noun_a} appeared beside the {noun_b}. Later the team rechecked it before sunset.",
            ("medium", "substitute"): f"The {noun_a} appeared beside the {noun_b}. Later the team rechecked that object before sunset.",
            ("medium", "weaken"): f"The {noun_a} appeared beside the {noun_b}. Much later the team rechecked it before sunset.",
            ("medium", "break"): f"The {noun_a} appeared beside the {noun_b}. Later the team rechecked them before sunset.",
            ("hard", "primary"): f"When the analyst mentioned the {noun_b}, the {noun_a} still appeared in the archive. Later the team rechecked it.",
            ("hard", "substitute"): f"When the analyst mentioned the {noun_b}, the {noun_a} still appeared in the archive. Later the team rechecked that object.",
            ("hard", "weaken"): f"When the analyst mentioned the {noun_b}, the {noun_a} still appeared in the archive. Much later the team rechecked it.",
            ("hard", "break"): f"When the analyst mentioned the {noun_b}, the {noun_a} still appeared in the archive. Later the team rechecked them.",
            ("adversarial", "primary"): f"The {noun_a} moved past the {noun_b}, and the {noun_c} remained hidden. Later the team rechecked it.",
            ("adversarial", "substitute"): f"The {noun_a} moved past the {noun_b}, and the {noun_c} remained hidden. Later the team rechecked that object.",
            ("adversarial", "weaken"): f"The {noun_a} moved past the {noun_b}, and the {noun_c} remained hidden. Much later the team rechecked it.",
            ("adversarial", "break"): f"The {noun_a} moved past the {noun_b}, and the {noun_c} remained hidden. Later the team rechecked them.",
        }
        return prompt_map[(difficulty, control_type)]

    if family_name == "ellipsis_recovery_family":
        prompt_map = {
            ("easy", "primary"): f"We studied the {noun_a}, and the planners did too.",
            ("easy", "substitute"): f"We studied the {noun_a}, and the planners studied that object too.",
            ("easy", "weaken"): f"We studied the {noun_a}, and much later the planners did too.",
            ("easy", "break"): f"We studied the {noun_a}, and the planners did nothing too.",
            ("medium", "primary"): f"We studied the {noun_a} after the {noun_b}, and the planners did too.",
            ("medium", "substitute"): f"We studied the {noun_a} after the {noun_b}, and the planners studied that object too.",
            ("medium", "weaken"): f"We studied the {noun_a} after the {noun_b}, and much later the planners did too.",
            ("medium", "break"): f"We studied the {noun_a} after the {noun_b}, and the planners did nothing too.",
            ("hard", "primary"): f"Although the report highlighted the {noun_b}, we studied the {noun_a}, and the planners did too.",
            ("hard", "substitute"): f"Although the report highlighted the {noun_b}, we studied the {noun_a}, and the planners studied that object too.",
            ("hard", "weaken"): f"Although the report highlighted the {noun_b}, we studied the {noun_a}, and much later the planners did too.",
            ("hard", "break"): f"Although the report highlighted the {noun_b}, we studied the {noun_a}, and the planners did nothing too.",
            ("adversarial", "primary"): f"We studied the {noun_a}, not the {noun_b}, and the planners did too.",
            ("adversarial", "substitute"): f"We studied the {noun_a}, not the {noun_b}, and the planners studied that object too.",
            ("adversarial", "weaken"): f"We studied the {noun_a}, not the {noun_b}, and much later the planners did too.",
            ("adversarial", "break"): f"We studied the {noun_a}, not the {noun_b}, and the planners did nothing too.",
        }
        return prompt_map[(difficulty, control_type)]

    if family_name == "context_bias_family":
        prompt_map = {
            ("easy", "primary"): f"The committee {pos_ctx} the {noun_a}, so the team will {verb_a} it soon.",
            ("easy", "substitute"): f"The committee {neg_ctx} the {noun_a}, so the team will {verb_a} it soon.",
            ("easy", "weaken"): f"The committee slightly {pos_ctx} the {noun_a}, so the team may {verb_a} it soon.",
            ("easy", "break"): f"The committee {pos_ctx} the {noun_a}, so soon it will the team {verb_a}.",
            ("medium", "primary"): f"{connector.capitalize()} the committee {pos_ctx} the {noun_a}, the team will {verb_a} it soon.",
            ("medium", "substitute"): f"{connector.capitalize()} the committee {neg_ctx} the {noun_a}, the team will {verb_a} it soon.",
            ("medium", "weaken"): f"{connector.capitalize()} the committee slightly {pos_ctx} the {noun_a}, the team may {verb_a} it soon.",
            ("medium", "break"): f"{connector.capitalize()} the committee {pos_ctx} the {noun_a}, soon it will the team {verb_a}.",
            ("hard", "primary"): f"Although the archive doubted the {noun_b}, the committee {pos_ctx} the {noun_a}, so the team will {verb_a} it soon.",
            ("hard", "substitute"): f"Although the archive praised the {noun_b}, the committee {neg_ctx} the {noun_a}, so the team will {verb_a} it soon.",
            ("hard", "weaken"): f"Although the archive doubted the {noun_b}, the committee slightly {pos_ctx} the {noun_a}, so the team may {verb_a} it soon.",
            ("hard", "break"): f"Although the archive doubted the {noun_b}, the committee {pos_ctx} the {noun_a}, so soon it will the team {verb_a}.",
            ("adversarial", "primary"): f"If the committee {pos_ctx} the {noun_a} but the archive {neg_ctx} the {noun_b}, the team will {verb_a} it soon.",
            ("adversarial", "substitute"): f"If the committee {neg_ctx} the {noun_a} but the archive {pos_ctx} the {noun_b}, the team will {verb_a} it soon.",
            ("adversarial", "weaken"): f"If the committee slightly {pos_ctx} the {noun_a} but the archive {neg_ctx} the {noun_b}, the team may {verb_a} it soon.",
            ("adversarial", "break"): f"If the committee {pos_ctx} the {noun_a} but the archive {neg_ctx} the {noun_b}, soon it will the team {verb_a}.",
        }
        return prompt_map[(difficulty, control_type)]

    if family_name == "late_repair_family":
        prompt_map = {
            ("easy", "primary"): f"At first the team missed the {noun_a}, but later it clearly {verb_a} the record.",
            ("easy", "substitute"): f"At first the team missed the {noun_a}, but later it vaguely {verb_a} the record.",
            ("easy", "weaken"): f"At first the team almost missed the {noun_a}, but later it clearly {verb_a} the record.",
            ("easy", "break"): f"At first the team missed the {noun_a}, but later the record clearly {verb_a} it.",
            ("medium", "primary"): f"At first the team missed the {noun_a} near the {noun_b}, but later it clearly {verb_a} the record.",
            ("medium", "substitute"): f"At first the team missed the {noun_a} near the {noun_b}, but later it vaguely {verb_a} the record.",
            ("medium", "weaken"): f"At first the team almost missed the {noun_a} near the {noun_b}, but later it clearly {verb_a} the record.",
            ("medium", "break"): f"At first the team missed the {noun_a} near the {noun_b}, but later the record clearly {verb_a} it.",
            ("hard", "primary"): f"Although the archive first highlighted the {noun_b}, the team missed the {noun_a}, but later it clearly {verb_a} the record.",
            ("hard", "substitute"): f"Although the archive first highlighted the {noun_b}, the team missed the {noun_a}, but later it vaguely {verb_a} the record.",
            ("hard", "weaken"): f"Although the archive first highlighted the {noun_b}, the team almost missed the {noun_a}, but later it clearly {verb_a} the record.",
            ("hard", "break"): f"Although the archive first highlighted the {noun_b}, the team missed the {noun_a}, but later the record clearly {verb_a} it.",
            ("adversarial", "primary"): f"At first the team missed the {noun_a} and the {noun_b}, but later it clearly {verb_a} only the record for the {noun_a}.",
            ("adversarial", "substitute"): f"At first the team missed the {noun_a} and the {noun_b}, but later it vaguely {verb_a} only the record for the {noun_a}.",
            ("adversarial", "weaken"): f"At first the team almost missed the {noun_a} and the {noun_b}, but later it clearly {verb_a} only the record for the {noun_a}.",
            ("adversarial", "break"): f"At first the team missed the {noun_a} and the {noun_b}, but later only the record clearly {verb_a} it.",
        }
        return prompt_map[(difficulty, control_type)]

    raise KeyError(f"未知机制家族: {family_name}")


def build_family_catalog() -> List[Dict[str, object]]:
    return [
        {
            "family_name": "anchor_subject_family",
            "target_variables": ["a", "f"],
            "hypothesis": "句中目标名词的早层定锚应在多种干扰下保持可观测。",
        },
        {
            "family_name": "adverb_route_family",
            "target_variables": ["g", "q"],
            "hypothesis": "副词应优先改变动词位选路，而不是仅仅提供静态属性。",
        },
        {
            "family_name": "pronoun_recovery_family",
            "target_variables": ["r", "f"],
            "hypothesis": "回指更可能在后层被补救，而不是在早层闭环完成。",
        },
        {
            "family_name": "ellipsis_recovery_family",
            "target_variables": ["r", "f"],
            "hypothesis": "省略恢复应依赖回返一致性与后层续接，而不是表面模板匹配。",
        },
        {
            "family_name": "context_bias_family",
            "target_variables": ["b", "q", "g"],
            "hypothesis": "上下文偏置应改变后续动作选择，并和门控场共同作用。",
        },
        {
            "family_name": "late_repair_family",
            "target_variables": ["f", "b", "g"],
            "hypothesis": "系统应能在前段错误后，通过后层补救与偏置修正恢复正确链路。",
        },
    ]


def build_case_rows(catalog: Sequence[Dict[str, object]], lexicon: Dict[str, List[str]]) -> List[Dict[str, object]]:
    case_rows: List[Dict[str, object]] = []
    case_index = 0
    for family in catalog:
        family_name = str(family["family_name"])
        for difficulty in DIFFICULTIES:
            for control_type in CONTROL_TYPES:
                for lexeme_index in range(4):
                    prompt = render_prompt(family_name, difficulty, control_type, lexicon, lexeme_index)
                    case_rows.append(
                        {
                            "case_id": f"stage147_{case_index:05d}",
                            "family_name": family_name,
                            "difficulty": difficulty,
                            "control_type": control_type,
                            "target_variables": list(family["target_variables"]),
                            "hypothesis": str(family["hypothesis"]),
                            "prompt": prompt,
                            "model_scope": "shared_gpt2_qwen3_deepseek",
                        }
                    )
                    case_index += 1
    return case_rows


def build_summary(catalog: Sequence[Dict[str, object]], case_rows: Sequence[Dict[str, object]], lexicon: Dict[str, List[str]]) -> Dict[str, object]:
    variable_counter = Counter()
    for family in catalog:
        variable_counter.update(str(value) for value in family["target_variables"])
    difficulty_counter = Counter(str(row["difficulty"]) for row in case_rows)
    control_counter = Counter(str(row["control_type"]) for row in case_rows)
    family_counter = Counter(str(row["family_name"]) for row in case_rows)

    variable_coverage = len(variable_counter) / 6.0
    difficulty_coverage = len(difficulty_counter) / len(DIFFICULTIES)
    control_coverage = len(control_counter) / len(CONTROL_TYPES)
    family_balance = min(family_counter.values()) / max(family_counter.values())
    generator_score = (
        0.30 * clamp01(variable_coverage)
        + 0.25 * clamp01(difficulty_coverage)
        + 0.20 * clamp01(control_coverage)
        + 0.15 * clamp01(family_balance)
        + 0.10 * clamp01(len(case_rows) / 384.0)
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage147_mechanism_family_generator",
        "title": "机制家族生成器块",
        "status_short": "mechanism_family_generator_ready",
        "family_count": len(catalog),
        "case_count": len(case_rows),
        "difficulty_count": len(difficulty_counter),
        "control_type_count": len(control_counter),
        "target_variable_count": len(variable_counter),
        "mechanism_family_generator_score": generator_score,
        "difficulty_counts": dict(difficulty_counter),
        "control_type_counts": dict(control_counter),
        "family_case_counts": dict(family_counter),
        "target_variable_counts": dict(variable_counter),
        "lexicon": lexicon,
        "family_rows": list(catalog),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage147: 机制家族生成器块",
        "",
        "## 核心结果",
        f"- 机制家族数: {summary['family_count']}",
        f"- 样本总数: {summary['case_count']}",
        f"- 难度层数: {summary['difficulty_count']}",
        f"- 对照类型数: {summary['control_type_count']}",
        f"- 目标变量数: {summary['target_variable_count']}",
        f"- 生成器分数: {summary['mechanism_family_generator_score']:.4f}",
        "",
        "## 家族目录",
    ]
    for row in summary["family_rows"]:
        lines.append(
            "- "
            f"{row['family_name']}: "
            f"variables={','.join(row['target_variables'])}, "
            f"hypothesis={row['hypothesis']}"
        )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], catalog: Sequence[Dict[str, object]], case_rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    CATALOG_PATH.write_text(json.dumps(list(catalog), ensure_ascii=False, indent=2), encoding="utf-8-sig")
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8-sig")

    with CASES_JSONL_PATH.open("w", encoding="utf-8-sig") as fh:
        for row in case_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = ["case_id", "family_name", "difficulty", "control_type", "target_variables", "hypothesis", "prompt", "model_scope"]
    with CASES_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in case_rows:
            writer.writerow({**row, "target_variables": ",".join(row["target_variables"])})


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force:
        cached = load_cached_summary(output_dir)
        if cached is not None:
            return cached
    _, rows = ensure_stage119_rows(STAGE119_OUTPUT_DIR)
    lexicon = build_lexicon(rows)
    catalog = build_family_catalog()
    case_rows = build_case_rows(catalog, lexicon)
    summary = build_summary(catalog, case_rows, lexicon)
    write_outputs(summary, catalog, case_rows, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="机制家族生成器块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重算")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
