from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Set


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def collect_terms(joined_rows: List[Dict[str, object]]) -> Set[str]:
    terms: Set[str] = set()
    for row in joined_rows:
        terms.add(str(row.get("prototype_term", "")))
        terms.add(str(row.get("instance_term", "")))
    return {term for term in terms if term}


def subset_pairs(source_pairs: Dict[str, List[Dict[str, object]]], keep_terms: Set[str]) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    for axis, rows in source_pairs.items():
        out[str(axis)] = [row for row in rows if str(row.get("term", "")) in keep_terms]
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build natural corpus pair subset for exact stage6 pair terms")
    ap.add_argument("--pairs-json", required=True)
    ap.add_argument("--joined-rows", required=True)
    ap.add_argument("--output-dir", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"tests/codex_temp/stage56_natural_pair_subset_builder_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.pairs_json).read_text(encoding="utf-8"))
    joined_rows = read_jsonl(Path(args.joined_rows))
    keep_terms = collect_terms(joined_rows)
    subset = subset_pairs(dict(payload.get("pairs") or {}), keep_terms)
    categories = sorted(
        {
            str(row.get("category", ""))
            for rows in subset.values()
            for row in rows
            if str(row.get("category", ""))
        }
    )
    output = {
        "record_type": "stage56_natural_pair_subset",
        "source_pairs_json": str(args.pairs_json),
        "source_joined_rows": str(args.joined_rows),
        "term_count": len(keep_terms),
        "terms": sorted(keep_terms),
        "categories": categories,
        "pairs": subset,
    }
    (out_dir / "pairs.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "term_count": len(keep_terms),
                "categories": categories,
                "pair_count_per_axis": {axis: len(rows) for axis, rows in subset.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": out_dir.as_posix(), "term_count": len(keep_terms)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
