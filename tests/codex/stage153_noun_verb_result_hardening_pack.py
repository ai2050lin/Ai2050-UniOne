#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from stage121_adverb_gate_bridge_probe import ensure_stage119_rows
from stage147_mechanism_family_generator import CONTROL_TYPES, DIFFICULTIES, build_lexicon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE119_OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage119_gpt2_embedding_full_vocab_scan_20260323"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage153_noun_verb_result_hardening_pack_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
CASES_JSONL_PATH = OUTPUT_DIR / "cases.jsonl"
CASES_CSV_PATH = OUTPUT_DIR / "cases.csv"
REPORT_PATH = OUTPUT_DIR / "STAGE153_NOUN_VERB_RESULT_HARDENING_PACK_REPORT.md"

FAMILY_SPECS = [
    ("fruit_action_result_family", "水果类名词进入动作与结果链，观察苹果与其他水果的分化", ["a", "g", "f"]),
    ("category_swap_result_family", "同动作下交换类别，观察结果位是否随名词类别改变", ["a", "g"]),
    ("tool_target_result_family", "工具和目标并置，观察结果位是否优先回收真正目标名词", ["a", "g", "f"]),
    ("double_candidate_result_family", "双候选名词竞争，观察动作后结果到底落回哪个名词", ["a", "g", "q"]),
    ("repair_to_result_family", "先误导再修正，观察结果位是否完成正确回收", ["a", "f", "q"]),
    ("adversarial_result_family", "对抗顺序与干扰项压入结果链，观察链条是否断裂", ["a", "g", "f", "q"]),
]

POSITIVE_RESULTS = ["fresh", "ready", "saved", "shared", "served", "chosen"]
NEGATIVE_RESULTS = ["lost", "broken", "delayed", "ignored", "wasted", "dropped"]


def pick(pool: List[str], index: int) -> str:
    return pool[index % len(pool)]


def fruit_pool(rows: List[Dict[str, object]]) -> List[str]:
    fruits = [
        str(row["word"]).lower()
        for row in rows
        if row["lexical_type"] == "noun" and row["group"] == "meso_fruit" and str(row["word"]).isalpha()
    ]
    out: List[str] = []
    seen = set()
    for word in ["apple", "orange", "banana", "pear", "peach", "grape", "melon", "berry"] + fruits:
        if word not in seen:
            seen.add(word)
            out.append(word)
        if len(out) >= 8:
            break
    return out


def render_prompt(
    family_name: str,
    difficulty: str,
    control_type: str,
    fruits: List[str],
    lexicon: Dict[str, List[str]],
    index: int,
) -> str:
    fruit_a = pick(fruits, index)
    fruit_b = pick(fruits, index + 1)
    fruit_c = pick(fruits, index + 2)
    noun_a = pick(lexicon["nouns"], index)
    verb_a = pick(lexicon["verbs"], index)
    verb_b = pick(lexicon["verbs"], index + 1)
    good = pick(POSITIVE_RESULTS, index)
    bad = pick(NEGATIVE_RESULTS, index)
    result_word = good if control_type in {"primary", "weaken"} else bad
    modal = "will" if control_type in {"primary", "substitute"} else "may"
    action = verb_a if control_type != "break" else verb_b

    if family_name == "fruit_action_result_family":
        if difficulty == "easy":
            return f"The team will {action} the {fruit_a}, so it becomes {result_word}."
        if difficulty == "medium":
            return f"After the team checks the {fruit_b}, it will {action} the {fruit_a}, so it becomes {result_word}."
        if difficulty == "hard":
            return f"Although the {fruit_b} looked useful, the team will {action} the {fruit_a}, so it becomes {result_word}."
        return f"If the team sees both the {fruit_a} and the {fruit_b}, it will {action} the {fruit_a}, so it becomes {result_word} before the {fruit_c}."

    if family_name == "category_swap_result_family":
        if difficulty == "easy":
            return f"The team will {action} the {fruit_a}, not the {noun_a}, so it becomes {result_word}."
        if difficulty == "medium":
            return f"After comparing the {fruit_a} with the {noun_a}, the team will {action} the {fruit_a}, so it becomes {result_word}."
        if difficulty == "hard":
            return f"Although the {noun_a} seems important, the team will {action} the {fruit_a}, so it becomes {result_word}."
        return f"If the {fruit_a} and the {noun_a} both remain, the team will {action} the {fruit_a}, so it becomes {result_word} first."

    if family_name == "tool_target_result_family":
        if difficulty == "easy":
            return f"The team used the {noun_a} to {action} the {fruit_a}, so it becomes {result_word}."
        if difficulty == "medium":
            return f"After moving the {noun_a}, the team used it to {action} the {fruit_a}, so it becomes {result_word}."
        if difficulty == "hard":
            return f"Although the {noun_a} was loud, the team used it to {action} the {fruit_a}, so it becomes {result_word}."
        return f"If the {noun_a} is beside the {fruit_a}, the team uses the {noun_a} to {action} the {fruit_a}, so it becomes {result_word}."

    if family_name == "double_candidate_result_family":
        if difficulty == "easy":
            return f"The team will {action} the {fruit_a} before the {fruit_b}, so it becomes {result_word}."
        if difficulty == "medium":
            return f"After comparing the {fruit_a} and the {fruit_b}, the team will {action} the {fruit_a}, so it becomes {result_word}."
        if difficulty == "hard":
            return f"Although the {fruit_b} looked brighter, the team will {action} the {fruit_a}, so it becomes {result_word}."
        return f"If both the {fruit_a} and the {fruit_b} are present, the team will {action} the {fruit_a}, so it becomes {result_word} before the {fruit_c}."

    if family_name == "repair_to_result_family":
        if difficulty == "easy":
            return f"The team first chose the {fruit_b}, but then it chose the {fruit_a} and will {action} it, so it becomes {result_word}."
        if difficulty == "medium":
            return f"After first choosing the {fruit_b}, the team corrected itself, chose the {fruit_a}, and will {action} it, so it becomes {result_word}."
        if difficulty == "hard":
            return f"Although the report first pushed the {fruit_b}, the team corrected itself, chose the {fruit_a}, and will {action} it, so it becomes {result_word}."
        return f"If the archive first pushed the {fruit_b} but the team later chose the {fruit_a}, it will {action} it, so it becomes {result_word}."

    if difficulty == "easy":
        return f"The team may {action} the {fruit_a}, but the {fruit_b} is a distraction, so it becomes {result_word}."
    if difficulty == "medium":
        return f"After the archive mentions the {fruit_b}, the team may {action} the {fruit_a}, so it becomes {result_word}."
    if difficulty == "hard":
        return f"Although the archive highlights the {fruit_b} and the {noun_a}, the team may {action} the {fruit_a}, so it becomes {result_word}."
    return f"If the archive highlights the {fruit_b}, the {noun_a}, and the {fruit_c}, the team may {action} the {fruit_a}, so it becomes {result_word}."


def build_cases(fruits: List[str], lexicon: Dict[str, List[str]]) -> List[Dict[str, object]]:
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
                            "target_variables": supports,
                            "hypothesis": hypothesis,
                            "prompt": render_prompt(family_name, difficulty, control_type, fruits, lexicon, index),
                        }
                    )
                    index += 1
    return cases


def build_summary(cases: List[Dict[str, object]], fruits: List[str]) -> Dict[str, object]:
    family_names = sorted({str(case["family_name"]) for case in cases})
    support_union = sorted({item for case in cases for item in case["target_variables"]})
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage153_noun_verb_result_hardening_pack",
        "title": "名词到动词到结果强化数据包",
        "status_short": "noun_verb_result_hardening_pack_ready",
        "case_count": len(cases),
        "family_count": len(family_names),
        "difficulty_count": len(DIFFICULTIES),
        "control_type_count": len(CONTROL_TYPES),
        "fruit_seed_count": len(fruits),
        "fruit_seeds": fruits,
        "family_names": family_names,
        "target_variables": support_union,
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
                "target_variables",
                "hypothesis",
                "prompt",
            ],
        )
        writer.writeheader()
        for case in cases:
            row = dict(case)
            row["target_variables"] = ",".join(case["target_variables"])
            writer.writerow(row)
    lines = [
        "# Stage153: 名词到动词到结果强化数据包",
        "",
        "## 核心结果",
        f"- 样本总数: {summary['case_count']}",
        f"- 家族数: {summary['family_count']}",
        f"- 水果种子数: {summary['fruit_seed_count']}",
        f"- 目标变量: {', '.join(summary['target_variables'])}",
        f"- 水果种子: {', '.join(summary['fruit_seeds'])}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    _, rows = ensure_stage119_rows(STAGE119_OUTPUT_DIR)
    lexicon = build_lexicon(rows)
    fruits = fruit_pool(rows)
    cases = build_cases(fruits, lexicon)
    summary = build_summary(cases, fruits)
    write_outputs(summary, cases, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="名词到动词到结果强化数据包")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
