from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


def read_source_items(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            term = row[0].strip()
            if not term or term.startswith("#") or len(row) < 2:
                continue
            category = row[1].strip()
            if not category:
                continue
            out[category].append(term)
    return dict(out)


def parse_categories(raw: str, available: Sequence[str]) -> List[str]:
    if not raw.strip():
        return sorted(available)
    chosen = [x.strip() for x in raw.split(",") if x.strip()]
    missing = [x for x in chosen if x not in available]
    if missing:
        raise ValueError(f"unknown categories: {missing}")
    return chosen


def category_word_for_terms(terms: Sequence[str], category: str) -> str | None:
    for term in terms:
        if term.strip().lower() == category.strip().lower():
            return term
    return None


def select_terms_for_category(
    terms: Sequence[str],
    category: str,
    terms_per_category: int,
    seed: int,
    require_category_word: bool,
) -> List[str]:
    if terms_per_category <= 0:
        raise ValueError("terms_per_category must be positive")
    category_word = category_word_for_terms(terms, category)
    if require_category_word and category_word is None:
        raise ValueError(f"category word not found for category={category}")
    others = [term for term in terms if term != category_word]
    if terms_per_category == 1 and category_word is not None:
        return [category_word]
    needed_other_count = terms_per_category - 1 if category_word is not None else terms_per_category
    if len(others) < needed_other_count:
        raise ValueError(
            f"category={category} has only {len(others)} instance terms, cannot satisfy terms_per_category={terms_per_category}"
        )
    rng = random.Random(seed)
    shuffled = list(others)
    rng.shuffle(shuffled)
    chosen = ([category_word] if category_word is not None else []) + shuffled[:needed_other_count]
    return chosen


def build_inventory(
    source_by_category: Dict[str, List[str]],
    categories: Sequence[str],
    terms_per_category: int,
    seed: int,
    require_category_word: bool = False,
) -> Dict[str, List[str]]:
    plan: Dict[str, List[str]] = {}
    for offset, category in enumerate(categories):
        if category not in source_by_category:
            raise ValueError(f"missing category={category}")
        plan[category] = select_terms_for_category(
            source_by_category[category],
            category=category,
            terms_per_category=terms_per_category,
            seed=seed + offset * 101,
            require_category_word=require_category_word,
        )
    return plan


def write_inventory_csv(path: Path, plan: Dict[str, List[str]]) -> None:
    lines = ["# noun,category"]
    for category in sorted(plan):
        for term in plan[category]:
            lines.append(f"{term},{category}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, plan: Dict[str, List[str]], source_file: str, seed: int) -> None:
    payload = {
        "record_type": "stage56_large_scale_discovery_inventory",
        "source_file": source_file,
        "seed": seed,
        "category_count": len(plan),
        "term_count": sum(len(terms) for terms in plan.values()),
        "categories": [
            {
                "category": category,
                "count": len(terms),
                "has_category_word": category_word_for_terms(terms, category) is not None,
                "category_word": category_word_for_terms(terms, category),
                "instance_terms": [term for term in terms if term != category],
                "terms": terms,
            }
            for category, terms in sorted(plan.items())
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(path: Path, plan: Dict[str, List[str]]) -> None:
    lines = [
        "# Stage56 Large Scale Discovery Inventory",
        "",
        "## Category Plan",
    ]
    for category, terms in sorted(plan.items()):
        lines.append(f"- {category}: {len(terms)} terms")
        category_word = category_word_for_terms(terms, category)
        lines.append(f"  - category word: {category_word if category_word is not None else 'MISSING'}")
        lines.append(f"  - instances: {', '.join(term for term in terms if term != category)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a balanced large-scale discovery inventory for stage56 analysis")
    ap.add_argument("--source-file", default="tests/codex/deepseek7b_nouns_english_520_clean.csv")
    ap.add_argument("--categories", default="")
    ap.add_argument("--terms-per-category", type=int, default=9)
    ap.add_argument("--require-category-word", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-file", default="tests/codex_temp/stage56_large_scale_discovery_items.csv")
    ap.add_argument("--manifest-file", default="tests/codex_temp/stage56_large_scale_discovery_manifest.json")
    ap.add_argument("--report-file", default="tests/codex_temp/stage56_large_scale_discovery_report.md")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    source_by_category = read_source_items(args.source_file)
    categories = parse_categories(args.categories, source_by_category.keys())
    plan = build_inventory(
        source_by_category=source_by_category,
        categories=categories,
        terms_per_category=args.terms_per_category,
        seed=args.seed,
        require_category_word=bool(args.require_category_word),
    )
    write_inventory_csv(Path(args.output_file), plan)
    write_manifest(Path(args.manifest_file), plan, source_file=args.source_file, seed=args.seed)
    write_report(Path(args.report_file), plan)
    print(
        json.dumps(
            {
                "output_file": args.output_file,
                "manifest_file": args.manifest_file,
                "report_file": args.report_file,
                "category_count": len(plan),
                "term_count": sum(len(terms) for terms in plan.values()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
