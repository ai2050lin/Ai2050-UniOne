from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from stage56_mass_scan_io import row_term, scan_term_rows


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MASS_JSONS = [
    "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed101/mass_noun_encoding_scan.json",
    "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed202/mass_noun_encoding_scan.json",
    "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed303/mass_noun_encoding_scan.json",
    "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed404/mass_noun_encoding_scan.json",
    "tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed505/mass_noun_encoding_scan.json",
]

DEFAULT_ACTION_TERMS = [
    "run", "jump", "walk", "move", "think", "speak", "write", "read", "build", "grow",
    "eat", "drink", "sleep", "learn", "teach", "drive", "fly", "swim", "climb", "push",
    "pull", "throw", "catch", "open", "close", "lift", "drop", "carry", "watch", "listen",
    "remember", "forget", "plan", "decide", "search", "find", "create", "destroy", "help", "lead",
    "follow", "change", "start", "stop", "return", "travel", "explore", "measure", "compare", "solve",
    "reason", "imagine",
]

DEFAULT_WEATHER_TERMS = [
    "rain", "snow", "wind", "storm", "cloud", "fog", "mist", "drizzle", "hail", "sleet",
    "thunder", "lightning", "humidity", "temperature", "breeze", "cyclone", "tornado", "monsoon", "frost", "dew",
    "sunshine", "downpour", "blizzard", "heatwave", "coldfront", "warmfront", "drought", "rainbow", "gale", "typhoon",
]

EXTENSION_TERMS: Dict[str, List[str]] = {
    "action": DEFAULT_ACTION_TERMS,
    "weather": DEFAULT_WEATHER_TERMS,
}


def read_source_items(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#") or len(row) < 2:
                continue
            term = row[0].strip().lower()
            category = row[1].strip().lower()
            if term and category:
                out[category].append(term)
    return dict(out)


def read_mass_categories(paths: Sequence[Path]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for path in paths:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        for row in scan_term_rows(obj):
            category = str(row.get("category", "")).strip().lower()
            if category:
                counter[category] += 1
    return dict(counter)


def read_mass_terms(paths: Sequence[Path]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for path in paths:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        for row in scan_term_rows(obj):
            term = row_term(row).strip().lower()
            category = str(row.get("category", "")).strip().lower()
            if term and category:
                out[category].append(term)
    return {category: sorted(dict.fromkeys(terms)) for category, terms in out.items()}


def sample_terms(terms: Sequence[str], count: int, seed: int) -> List[str]:
    unique_terms = []
    seen = set()
    for term in terms:
        if term not in seen:
            unique_terms.append(term)
            seen.add(term)
    if count > len(unique_terms):
        raise ValueError(f"requested {count} terms but only {len(unique_terms)} available")
    rng = random.Random(seed)
    pool = list(unique_terms)
    rng.shuffle(pool)
    return sorted(pool[:count])


def sample_terms_capped(terms: Sequence[str], count: int, seed: int) -> List[str]:
    unique_count = len(dict.fromkeys(terms))
    target = min(int(count), unique_count)
    return sample_terms(terms, target, seed)


def category_pool(
    category: str,
    source_by_category: Dict[str, List[str]],
    current_mass_terms: Dict[str, List[str]],
) -> List[str]:
    return (
        list(source_by_category.get(category, []))
        + list(current_mass_terms.get(category, []))
        + list(EXTENSION_TERMS.get(category, []))
    )


def build_inventory(
    source_by_category: Dict[str, List[str]],
    current_mass_categories: Dict[str, int],
    current_mass_terms: Dict[str, List[str]],
    terms_per_category: int,
    seed: int,
) -> Dict[str, List[str]]:
    inventory: Dict[str, List[str]] = {}
    categories = sorted(set(source_by_category) | set(current_mass_categories) | set(EXTENSION_TERMS))
    for idx, category in enumerate(categories):
        pool = category_pool(category, source_by_category, current_mass_terms)
        inventory[category] = sample_terms_capped(pool, terms_per_category, seed + idx * 17)
    return inventory


def build_summary(
    source_by_category: Dict[str, List[str]],
    current_mass_categories: Dict[str, int],
    inventory: Dict[str, List[str]],
    terms_per_category: int,
) -> Dict[str, object]:
    source_categories = sorted(source_by_category)
    current_categories = sorted(current_mass_categories)
    inventory_categories = sorted(inventory)
    missing_in_mass = [category for category in source_categories if category not in current_mass_categories]
    added_by_inventory = [category for category in inventory_categories if category not in current_mass_categories]
    available_term_counts = {category: len(terms) for category, terms in sorted(inventory.items())}
    capped_categories = [
        category for category, count in available_term_counts.items() if count < int(terms_per_category)
    ]
    return {
        "source_category_count": len(source_categories),
        "current_mass_category_count": len(current_categories),
        "inventory_category_count": len(inventory_categories),
        "terms_per_category": terms_per_category,
        "inventory_term_count": sum(len(terms) for terms in inventory.values()),
        "source_categories": source_categories,
        "current_mass_categories": current_categories,
        "missing_in_current_mass": missing_in_mass,
        "added_by_inventory": added_by_inventory,
        "extension_categories": sorted(EXTENSION_TERMS),
        "available_term_counts": available_term_counts,
        "capped_categories": capped_categories,
        "coverage_gap_statement": (
            "当前真实 mass scan 缺少 abstract，action 与 weather 仍主要依赖扩展词表补齐。"
            if ("abstract" in missing_in_mass or "action" in added_by_inventory or "weather" in added_by_inventory)
            else "当前真实 mass scan 已覆盖 source 词表中的主体类别，扩展词表主要用于放大容量。"
        ),
    }


def write_inventory_csv(path: Path, inventory: Dict[str, List[str]]) -> None:
    lines = ["# term,category"]
    for category in sorted(inventory):
        for term in inventory[category]:
            lines.append(f"{term},{category}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, summary: Dict[str, object], inventory: Dict[str, List[str]]) -> None:
    lines = [
        "# ICSPB 扩容清单报告",
        "",
        "## 覆盖状态",
        f"- source_category_count: {summary['source_category_count']}",
        f"- current_mass_category_count: {summary['current_mass_category_count']}",
        f"- inventory_category_count: {summary['inventory_category_count']}",
        f"- terms_per_category: {summary['terms_per_category']}",
        f"- inventory_term_count: {summary['inventory_term_count']}",
        f"- missing_in_current_mass: {', '.join(summary['missing_in_current_mass'])}",
        f"- added_by_inventory: {', '.join(summary['added_by_inventory'])}",
        f"- capped_categories: {', '.join(summary['capped_categories'])}",
        "",
        "## 推荐实验清单",
    ]
    for category in sorted(inventory):
        preview = ", ".join(inventory[category][:8])
        lines.append(f"- {category}: {preview}")
    lines.extend(
        [
            "",
            "## 判断",
            f"- {summary['coverage_gap_statement']}",
            "- 这份清单适合直接进入下一轮大规模扫描，用于扩大 abstract、action、weather 的真实覆盖。",
            "- 当前输出已经统一为 term 口径，但下游仍有部分脚本保留 noun 命名兼容层，后续需要继续去名词偏置。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build an expanded ICSPB inventory with abstract, action, and weather coverage")
    ap.add_argument("--source-file", default="tests/codex/deepseek7b_nouns_english_520_clean.csv")
    ap.add_argument("--mass-json", action="append", default=[])
    ap.add_argument("--terms-per-category", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_icspb_expanded_inventory_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    mass_jsons = [Path(item) for item in (args.mass_json or DEFAULT_MASS_JSONS)]
    source_by_category = read_source_items(Path(args.source_file))
    current_mass_categories = read_mass_categories(mass_jsons)
    current_mass_terms = read_mass_terms(mass_jsons)
    inventory = build_inventory(
        source_by_category=source_by_category,
        current_mass_categories=current_mass_categories,
        current_mass_terms=current_mass_terms,
        terms_per_category=int(args.terms_per_category),
        seed=int(args.seed),
    )
    summary = build_summary(
        source_by_category=source_by_category,
        current_mass_categories=current_mass_categories,
        inventory=inventory,
        terms_per_category=int(args.terms_per_category),
    )

    payload = {
        "record_type": "stage56_icspb_expanded_inventory",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "config": {
            "source_file": args.source_file,
            "mass_jsons": [str(path) for path in mass_jsons],
            "terms_per_category": int(args.terms_per_category),
            "seed": int(args.seed),
        },
        "summary": summary,
        "inventory": inventory,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_inventory_csv(out_dir / "items.csv", inventory)
    (out_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir / "REPORT.md", summary, inventory)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "items_csv": str(out_dir / "items.csv"),
                "manifest_json": str(out_dir / "manifest.json"),
                "report_md": str(out_dir / "REPORT.md"),
                "inventory_category_count": summary["inventory_category_count"],
                "inventory_term_count": summary["inventory_term_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
