from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def read_items(path: Path, max_terms: int = 0) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
    lines: List[str] = []
    for idx, line in enumerate(raw_lines):
        text = line.strip()
        if not text:
            continue
        if idx == 0 and text.startswith("#"):
            text = text.lstrip("#").strip()
        lines.append(text)
    reader = csv.DictReader(lines)
    if not reader.fieldnames:
        return rows
    reader.fieldnames = [str(name).strip().lstrip("#").strip() for name in reader.fieldnames]
    for row in reader:
        if not isinstance(row, dict):
            continue
        term = str(row.get("term", "")).strip()
        category = str(row.get("category", "")).strip()
        if not term or not category:
            continue
        rows.append({"term": term, "category": category})
        if max_terms > 0 and len(rows) >= max_terms:
            break
    return rows


def category_phrase(category: str) -> str:
    mapping = {
        "abstract": "an abstract concept",
        "action": "an action",
        "animal": "an animal",
        "celestial": "a celestial object",
        "food": "a food",
        "fruit": "a fruit",
        "human": "a human role",
        "nature": "a natural thing",
        "object": "an object",
        "tech": "a technical concept",
        "vehicle": "a vehicle",
        "weather": "a weather phenomenon",
    }
    return mapping.get(str(category).strip().lower(), f"a {category}")


def term_prompts(term: str, category: str) -> Dict[str, Dict[str, str]]:
    phrase = category_phrase(category)
    category_text = str(category).strip().lower()
    if category_text == "action":
        return {
            "style": {
                "a": f"In casual conversation, people would say that to {term} is a kind of action because",
                "b": f"In a formal linguistic description, to {term} denotes an action that",
            },
            "logic": {
                "a": f"Because to {term} is one example within the broader class of actions, we can conclude that to {term} is",
                "b": f"Even though to {term} is one example within the broader class of actions, we should deny that to {term} is",
            },
            "syntax": {
                "a": f"To {term} is a kind of action that",
                "b": f"A kind of action is to {term}, which",
            },
        }

    return {
        "style": {
            "a": f"In casual conversation, people would say that {term} is {phrase} because",
            "b": f"In a formal reference entry, {term} is classified as {phrase} because",
        },
        "logic": {
            "a": f"Because {term} is one member of the broader class of {category}, we can conclude that {term} is",
            "b": f"Even though {term} is one member of the broader class of {category}, we should deny that {term} is",
        },
        "syntax": {
            "a": f"{term} belongs to the class of {category} because",
            "b": f"The class to which {term} belongs is {category} because",
        },
    }


def build_pairs(items: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {"style": [], "logic": [], "syntax": []}
    for idx, item in enumerate(items):
        term = item["term"]
        category = item["category"]
        prompts = term_prompts(term, category)
        for axis in ("style", "logic", "syntax"):
            out[axis].append(
                {
                    "id": f"{axis}_{category}_{idx:04d}_{term}",
                    "term": term,
                    "category": category,
                    "a": prompts[axis]["a"],
                    "b": prompts[axis]["b"],
                }
            )
    return out


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(path: Path, payload: Dict[str, object]) -> None:
    lines = [
        "# Stage56 自然语料三轴对照构建",
        "",
        f"- term_count: {payload['term_count']}",
        f"- categories: {', '.join(payload['categories'])}",
        "",
        "## Axis Counts",
    ]
    for axis, rows in dict(payload["pairs"]).items():
        lines.append(f"- {axis}: {len(rows)}")
    lines.extend(["", "## Sample Pairs"])
    for axis in ("style", "logic", "syntax"):
        sample = list(payload["pairs"][axis])[:3]
        for row in sample:
            lines.append(f"- {axis} / {row['term']} / {row['category']}: A={row['a']} | B={row['b']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build large natural-corpus contrast pairs for style/logic/syntax probes")
    ap.add_argument(
        "--items-csv",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_icspb_expanded_inventory_20260318_1525" / "items.csv"),
    )
    ap.add_argument("--max-terms", type=int, default=0)
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_natural_corpus_contrast_builder_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    items = read_items(Path(args.items_csv), max_terms=int(args.max_terms))
    pairs = build_pairs(items)
    payload = {
        "record_type": "stage56_natural_corpus_contrast_pairs",
        "term_count": len(items),
        "categories": sorted({row["category"] for row in items}),
        "pairs": pairs,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "pairs.json", payload)
    write_report(out_dir / "REPORT.md", payload)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "term_count": payload["term_count"],
                "style_pairs": len(pairs["style"]),
                "logic_pairs": len(pairs["logic"]),
                "syntax_pairs": len(pairs["syntax"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
