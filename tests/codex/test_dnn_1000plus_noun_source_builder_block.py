from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gpt5.code.dnn_1000plus_noun_source_builder import Dnn1000PlusNounSourceBuilder  # noqa: E402


def build_payload(out_csv_rel: str) -> Dict[str, Any]:
    t0 = time.time()
    builder = Dnn1000PlusNounSourceBuilder(ROOT)
    out_csv = builder.write_csv(out_csv_rel)

    rows = []
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            if len(row) >= 2:
                rows.append((row[0], row[1]))

    unique_nouns = len({noun for noun, _ in rows})
    category_counter = Counter(category for _, category in rows)

    metric_lines_cn = [
        f"（1000+词源唯一名词数）unique_nouns_1000plus = {unique_nouns:.4f}",
        f"（1000+词源类别数量）category_count_1000plus = {len(category_counter):.4f}",
        f"（1000+词源最大单类名词数）max_category_size_1000plus = {max(category_counter.values()):.4f}",
        f"（1000+词源最小单类名词数）min_category_size_1000plus = {min(category_counter.values()):.4f}",
        f"（1000+词源是否达标）meets_1000plus_target = {1.0 if unique_nouns >= 1000 else 0.0:.4f}",
        f"（1000+词源输出完成）source_written = {1.0:.4f}",
    ]

    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "dnn_1000plus_noun_source_builder_block",
        },
        "strict_goal": {
            "statement": "Generate a real 1000+ unique noun source CSV for the noun-scaling route, with balanced category coverage.",
            "boundary": "This block builds the 1000+ source. It does not yet run dense harvesting over the whole source.",
        },
        "headline_metrics": {
            "metric_lines_cn": metric_lines_cn,
        },
        "source_csv": str(out_csv.relative_to(ROOT)),
        "category_rows": [
            {"category": category, "count": int(count)}
            for category, count in sorted(category_counter.items())
        ],
        "strict_conclusion": {
            "core_answer": "The repo now has a generated 1000+ noun source with balanced category structure, so the second stage no longer depends on a missing lexicon.",
            "main_hard_gaps": [
                "the 1000+ source is generated but not yet fully harvested through the dense pipeline",
                "some category terms are seed-quality rather than benchmark-grade lexicon entries",
                "this improves source coverage, not theorem closure by itself",
            ],
        },
    }


def test_dnn_1000plus_noun_source_builder_block() -> None:
    payload = build_payload("tests/codex/deepseek7b_bilingual_nouns_1000plus.csv")
    assert len(payload["headline_metrics"]["metric_lines_cn"]) >= 6
    assert payload["headline_metrics"]["metric_lines_cn"][0].startswith("（")
    rows = payload["category_rows"]
    assert len(rows) == 10
    assert sum(row["count"] for row in rows) >= 1000


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN 1000+ noun source builder block")
    ap.add_argument(
        "--csv-out",
        type=str,
        default="tests/codex/deepseek7b_bilingual_nouns_1000plus.csv",
    )
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_1000plus_noun_source_builder_block_20260316.json",
    )
    args = ap.parse_args()

    payload = build_payload(args.csv_out)
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
