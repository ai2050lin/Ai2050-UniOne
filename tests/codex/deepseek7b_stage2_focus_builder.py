from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def read_source_items(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            head = row[0].strip()
            if not head or head.startswith("#") or len(row) < 2:
                continue
            term = head
            category = row[1].strip()
            if not category:
                continue
            out[category].append(term)
    return dict(out)


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def dedupe_keep_order(rows: Iterable[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    seen = set()
    for term, category, role in rows:
        key = (term, category)
        if key in seen:
            continue
        seen.add(key)
        out.append((term, category, role))
    return out


def build_focus_plan(
    source_by_category: Dict[str, List[str]],
    records: Sequence[Dict[str, object]],
    closure_candidates: Sequence[Dict[str, object]],
    anchors_per_category: int,
    challengers_per_category: int,
    supports_per_category: int,
) -> Dict[str, List[Dict[str, object]]]:
    deep_by_category: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in records:
        if str(row.get("pool")) == "deep":
            deep_by_category[str(row["item"]["category"])].append(row)

    closure_by_category: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in closure_candidates:
        if str(row.get("pool")) == "closure":
            closure_by_category[str(row["item"]["category"])].append(row)

    plan: Dict[str, List[Dict[str, object]]] = {}
    all_categories = sorted(source_by_category)
    for category in all_categories:
        chosen: List[Tuple[str, str, str]] = []
        chosen_terms = set()

        closure_rows = sorted(
            closure_by_category.get(category, []),
            key=lambda x: (float(x["exact_closure_proxy"]), float(x["wrong_family_margin"])),
            reverse=True,
        )
        for row in closure_rows[: max(0, anchors_per_category)]:
            term = str(row["item"]["term"])
            chosen.append((term, category, "anchor"))
            chosen_terms.add(term)

        challenge_rows = sorted(
            closure_by_category.get(category, []),
            key=lambda x: (float(x["wrong_family_margin"]), float(x["exact_closure_proxy"])),
        )
        for row in challenge_rows:
            if sum(1 for _, _, role in chosen if role == "challenger") >= max(0, challengers_per_category):
                break
            term = str(row["item"]["term"])
            if term in chosen_terms:
                continue
            chosen.append((term, category, "challenger"))
            chosen_terms.add(term)

        deep_rows = sorted(
            deep_by_category.get(category, []),
            key=lambda x: (float(x["aggregate"]["prompt_stability_jaccard_mean"]), float(x["aggregate"]["top3_layer_ratio"])),
            reverse=True,
        )
        weak_deep_rows = sorted(
            deep_by_category.get(category, []),
            key=lambda x: (float(x["aggregate"]["prompt_stability_jaccard_mean"]), float(x["aggregate"]["top3_layer_ratio"])),
        )
        while sum(1 for _, _, role in chosen if role == "challenger") < max(0, challengers_per_category):
            appended = False
            for row in weak_deep_rows:
                term = str(row["item"]["term"])
                if term in chosen_terms:
                    continue
                chosen.append((term, category, "challenger"))
                chosen_terms.add(term)
                appended = True
                break
            if not appended:
                break

        for row in deep_rows[: max(0, supports_per_category)]:
            term = str(row["item"]["term"])
            if term in chosen_terms:
                continue
            chosen.append((term, category, "support"))
            chosen_terms.add(term)

        ordered = dedupe_keep_order(chosen)
        chosen_terms = {term for term, _, _ in ordered}
        for term in source_by_category.get(category, []):
            if len(ordered) >= anchors_per_category + challengers_per_category + supports_per_category:
                break
            if term in chosen_terms:
                continue
            ordered.append((term, category, "fill"))
            chosen_terms.add(term)

        plan[category] = [
            {"term": term, "category": category, "role": role}
            for term, category, role in ordered
        ]
    return plan


def write_focus_items_csv(path: Path, plan: Dict[str, List[Dict[str, object]]]) -> None:
    lines = ["# noun,category"]
    for category in sorted(plan):
        for row in plan[category]:
            lines.append(f"{row['term']},{row['category']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_focus_manifest_json(path: Path, plan: Dict[str, List[Dict[str, object]]], source_info: Dict[str, object]) -> None:
    payload = {
        "record_type": "stage2_focus_manifest",
        "source": source_info,
        "categories": [
            {
                "category": category,
                "count": len(rows),
                "roles": rows,
            }
            for category, rows in sorted(plan.items())
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_focus_report_md(path: Path, plan: Dict[str, List[Dict[str, object]]]) -> None:
    lines = [
        "# DeepSeek Stage2 Focus Builder Report",
        "",
        "## Category Focus",
    ]
    for category, rows in sorted(plan.items()):
        lines.append(f"- {category}: {len(rows)} items")
        for row in rows:
            lines.append(f"  - {row['term']} [{row['role']}]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build stage2 focus inventory from stage1 DeepSeek scan outputs")
    ap.add_argument("--source-items-file", default="tests/codex/deepseek7b_nouns_english_500plus.csv")
    ap.add_argument("--records-file", default="tempdata/deepseek7b_three_pool_stage1_500plus_bf16_20260316/records.jsonl")
    ap.add_argument(
        "--closure-candidates-file",
        default="tempdata/deepseek7b_three_pool_stage1_500plus_bf16_20260316/closure_candidates.jsonl",
    )
    ap.add_argument("--anchors-per-category", type=int, default=2)
    ap.add_argument("--challengers-per-category", type=int, default=2)
    ap.add_argument("--supports-per-category", type=int, default=2)
    ap.add_argument("--output-dir", default="tempdata/deepseek7b_stage2_focus_20260316")
    args = ap.parse_args()

    source_by_category = read_source_items(args.source_items_file)
    records = read_jsonl(args.records_file)
    closure_candidates = read_jsonl(args.closure_candidates_file)
    plan = build_focus_plan(
        source_by_category=source_by_category,
        records=records,
        closure_candidates=closure_candidates,
        anchors_per_category=args.anchors_per_category,
        challengers_per_category=args.challengers_per_category,
        supports_per_category=args.supports_per_category,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_focus_items_csv(out_dir / "focus_items.csv", plan)
    write_focus_manifest_json(
        out_dir / "focus_manifest.json",
        plan,
        source_info={
            "source_items_file": args.source_items_file,
            "records_file": args.records_file,
            "closure_candidates_file": args.closure_candidates_file,
            "anchors_per_category": args.anchors_per_category,
            "challengers_per_category": args.challengers_per_category,
            "supports_per_category": args.supports_per_category,
        },
    )
    write_focus_report_md(out_dir / "focus_report.md", plan)

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "focus_items_file": str(out_dir / "focus_items.csv"),
                "focus_manifest_file": str(out_dir / "focus_manifest.json"),
                "category_count": len(plan),
                "item_count": sum(len(rows) for rows in plan.values()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
