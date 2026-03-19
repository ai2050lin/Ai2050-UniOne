from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]

ROLE_PRIORITY = {
    "weak_bridge_positive": 4,
    "weak_partial_bridge": 3,
    "weak_dominant_positive": 2,
    "weak_drag_or_conflict": 1,
}


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_score(row: Dict[str, object]) -> Tuple[float, float, float, float]:
    best_mixed = float(row["best_mixed"]["metrics"]["joint_adv_mean"])
    stage6_union = float(row["stage6_reference"]["union_joint_adv"])
    stage6_synergy = float(row["stage6_reference"]["union_synergy_joint"])
    role_priority = float(ROLE_PRIORITY.get(str(row["case_role"]), 0))
    return (role_priority, best_mixed, stage6_union, stage6_synergy)


def select_representative_cases(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_id"]), str(row["category"]))].append(row)

    selected: List[Dict[str, object]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        chosen = sorted(bucket, key=row_score, reverse=True)[0]
        selected.append(chosen)
    return selected


def build_summary(source_rows: Sequence[Dict[str, object]], selected_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "record_type": "stage56_taxonomy_case_selector_summary",
        "source_case_count": len(source_rows),
        "selected_case_count": len(selected_rows),
        "model_count": len({str(row["model_id"]) for row in selected_rows}),
        "category_count": len({str(row["category"]) for row in selected_rows}),
        "per_model": {
            model_id: {
                "case_count": len([row for row in selected_rows if str(row["model_id"]) == model_id]),
                "categories": sorted({str(row["category"]) for row in selected_rows if str(row["model_id"]) == model_id}),
            }
            for model_id in sorted({str(row["model_id"]) for row in selected_rows})
        },
    }


def write_report(path: Path, summary: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Stage56 Taxonomy Case Selector Report",
        "",
        f"- source_case_count: {summary['source_case_count']}",
        f"- selected_case_count: {summary['selected_case_count']}",
        f"- model_count: {summary['model_count']}",
        f"- category_count: {summary['category_count']}",
        "",
        "## Selected Cases",
    ]
    for row in rows:
        lines.append(
            f"- {row['group_label']} / {row['category']} / proto={row['prototype_term']} / inst={row['instance_term']} "
            f"/ role={row['case_role']} / best_mixed={row['best_mixed']['metrics']['joint_adv_mean']:.6f} "
            f"/ stage6_union={row['stage6_reference']['union_joint_adv']:.6f} "
            f"/ stage6_synergy={row['stage6_reference']['union_synergy_joint']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Select one representative taxonomy case per model/category")
    ap.add_argument("--taxonomy-cases", required=True)
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_taxonomy_case_selector_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(Path(args.taxonomy_cases))
    selected_rows = select_representative_cases(source_rows)
    summary = build_summary(source_rows, selected_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "selected_cases.jsonl", selected_rows)
    write_report(out_dir / "REPORT.md", summary, selected_rows)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "selected_case_count": len(selected_rows),
                "model_count": summary["model_count"],
                "category_count": summary["category_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
