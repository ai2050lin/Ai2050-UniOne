from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Sequence, Tuple


def read_rows(path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            term = row[0].strip()
            category = row[1].strip()
            if not term or not category or term.startswith("#"):
                continue
            rows.append((term, category))
    return rows


def is_ascii_term(term: str) -> bool:
    return bool(term) and all(ord(ch) < 128 for ch in term)


def dedupe_keep_order(rows: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen = set()
    for term, category in rows:
        key = (term, category)
        if key in seen:
            continue
        seen.add(key)
        out.append((term, category))
    return out


def balanced_round_robin(rows: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    by_category: Dict[str, Deque[str]] = defaultdict(deque)
    for term, category in rows:
        by_category[category].append(term)

    ordered: List[Tuple[str, str]] = []
    categories = sorted(by_category)
    while True:
        progressed = False
        for category in categories:
            if by_category[category]:
                ordered.append((by_category[category].popleft(), category))
                progressed = True
        if not progressed:
            break
    return ordered


def build_clean_inventory(source_rows: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    ascii_rows = [(term, category) for term, category in source_rows if is_ascii_term(term)]
    unique_rows = dedupe_keep_order(ascii_rows)
    return balanced_round_robin(unique_rows)


def write_csv(path: Path, rows: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# noun", "category"])
        for term, category in rows:
            writer.writerow([term, category])


def per_category_counts(rows: Sequence[Tuple[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for _, category in rows:
        counts[category] += 1
    return dict(sorted(counts.items()))


def diff_rows(
    rows: Sequence[Tuple[str, str]],
    compare_rows: Sequence[Tuple[str, str]],
) -> Dict[str, List[Dict[str, str]]]:
    row_set = set(rows)
    compare_set = set(compare_rows)
    return {
        "missing_in_compare": [
            {"term": term, "category": category}
            for term, category in sorted(row_set - compare_set)
        ],
        "extra_in_compare": [
            {"term": term, "category": category}
            for term, category in sorted(compare_set - row_set)
        ],
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# DeepSeek Clean Vocab Builder Report",
        "",
        f"- Source rows: {summary['source_row_count']}",
        f"- Clean rows: {summary['clean_row_count']}",
        f"- Category count: {summary['category_count']}",
        "",
        "## Category Coverage",
    ]
    for category, count in summary["category_counts"].items():
        lines.append(f"- {category}: {count}")
    compare = summary.get("compare_diff", {})
    missing = compare.get("missing_in_compare", [])
    extra = compare.get("extra_in_compare", [])
    if missing or extra:
        lines.extend(
            [
                "",
                "## Compare Diff",
                f"- Missing in compare: {len(missing)}",
                f"- Extra in compare: {len(extra)}",
            ]
        )
        for row in missing[:20]:
            lines.append(f"- added: {row['term']} [{row['category']}]")
        for row in extra[:20]:
            lines.append(f"- only_in_compare: {row['term']} [{row['category']}]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a clean ASCII DeepSeek noun inventory from the 1000+ source")
    ap.add_argument("--source-file", default="tests/codex/deepseek7b_bilingual_nouns_1000plus.csv")
    ap.add_argument("--target-file", default="tests/codex/deepseek7b_nouns_english_520_clean.csv")
    ap.add_argument("--compare-file", default="tests/codex/deepseek7b_nouns_english_500plus.csv")
    ap.add_argument("--output-dir", default="tempdata/deepseek7b_clean_vocab_520_20260316")
    args = ap.parse_args()

    t0 = time.time()
    source_rows = read_rows(args.source_file)
    clean_rows = build_clean_inventory(source_rows)

    target_path = Path(args.target_file)
    write_csv(target_path, clean_rows)

    compare_diff: Dict[str, object] = {}
    if args.compare_file and Path(args.compare_file).exists():
        compare_rows = dedupe_keep_order(read_rows(args.compare_file))
        compare_diff = diff_rows(clean_rows, compare_rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "record_type": "clean_vocab_builder_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "source_file": args.source_file,
        "target_file": args.target_file,
        "compare_file": args.compare_file,
        "source_row_count": len(source_rows),
        "clean_row_count": len(clean_rows),
        "category_count": len({category for _, category in clean_rows}),
        "category_counts": per_category_counts(clean_rows),
        "compare_diff": compare_diff,
    }
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", summary)

    print(
        json.dumps(
            {
                "target_file": str(target_path),
                "summary": str(out_dir / "summary.json"),
                "clean_row_count": len(clean_rows),
                "category_count": summary["category_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
