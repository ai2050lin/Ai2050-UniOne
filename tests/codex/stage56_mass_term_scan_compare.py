from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def model_tag_for_scan(scan_dir: Path) -> str:
    return scan_dir.name


def load_closure_rows(scan_dir: Path) -> List[Dict[str, object]]:
    rows = read_jsonl(scan_dir / "closure_candidates.jsonl")
    return [row for row in rows if str(row.get("pool", "")) == "closure"]


def top_rows_by_category(scan_dir: Path) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in load_closure_rows(scan_dir):
        category = str(dict(row.get("item", {})).get("category", "")).strip().lower()
        if category:
            grouped[category].append(row)
    out: Dict[str, Dict[str, object]] = {}
    for category, rows in grouped.items():
        out[category] = max(rows, key=lambda row: float(row.get("exact_closure_proxy", 0.0)))
    return out


def build_model_row(scan_dir: Path) -> Dict[str, object]:
    manifest = read_json(scan_dir / "manifest.json")
    summary = read_json(scan_dir / "summary.json")
    top_rows = top_rows_by_category(scan_dir)
    recurring_terms = [
        {
            "category": category,
            "term": dict(row.get("item", {})).get("term", ""),
            "exact_closure_proxy": float(row.get("exact_closure_proxy", 0.0)),
            "wrong_family_margin": float(row.get("wrong_family_margin", 0.0)),
        }
        for category, row in sorted(top_rows.items())
    ]
    return {
        "model_tag": model_tag_for_scan(scan_dir),
        "model_id": manifest.get("model_id", ""),
        "input_items": int(dict(manifest.get("counts", {})).get("input_items", 0)),
        "record_count": int(dict(summary.get("headline_metrics", {})).get("record_count", 0)),
        "family_count": int(dict(summary.get("headline_metrics", {})).get("family_count", 0)),
        "closure_candidate_count": int(dict(summary.get("headline_metrics", {})).get("closure_candidate_count", 0)),
        "mean_prompt_stability_survey": float(dict(summary.get("headline_metrics", {})).get("mean_prompt_stability_survey", 0.0)),
        "mean_prompt_stability_deep": float(dict(summary.get("headline_metrics", {})).get("mean_prompt_stability_deep", 0.0)),
        "mean_prompt_stability_closure": float(dict(summary.get("headline_metrics", {})).get("mean_prompt_stability_closure", 0.0)),
        "category_coverage_survey": dict(summary.get("category_coverage_survey", {})),
        "top_terms_by_category": recurring_terms,
    }


def build_consensus_rows(scan_dirs: Sequence[Path]) -> List[Dict[str, object]]:
    per_scan_top = {model_tag_for_scan(scan_dir): top_rows_by_category(scan_dir) for scan_dir in scan_dirs}
    all_categories = sorted({category for tops in per_scan_top.values() for category in tops})
    consensus_rows: List[Dict[str, object]] = []
    for category in all_categories:
        term_counter: Counter[str] = Counter()
        term_scores: Dict[str, List[float]] = defaultdict(list)
        per_model_rows = []
        for model_tag, tops in sorted(per_scan_top.items()):
            row = tops.get(category)
            if not row:
                continue
            item = dict(row.get("item", {}))
            term = str(item.get("term", ""))
            score = float(row.get("exact_closure_proxy", 0.0))
            margin = float(row.get("wrong_family_margin", 0.0))
            term_counter[term] += 1
            term_scores[term].append(score)
            per_model_rows.append(
                {
                    "model_tag": model_tag,
                    "term": term,
                    "exact_closure_proxy": score,
                    "wrong_family_margin": margin,
                }
            )
        ranked_terms = sorted(
            (
                {
                    "term": term,
                    "support_count": count,
                    "mean_exact_closure_proxy": average(term_scores[term]),
                }
                for term, count in term_counter.items()
            ),
            key=lambda row: (row["support_count"], row["mean_exact_closure_proxy"], row["term"]),
            reverse=True,
        )
        leader = ranked_terms[0] if ranked_terms else {"term": "", "support_count": 0, "mean_exact_closure_proxy": 0.0}
        consensus_rows.append(
            {
                "category": category,
                "model_count": len(per_model_rows),
                "consensus_term": leader["term"],
                "consensus_support_count": leader["support_count"],
                "consensus_support_ratio": float(leader["support_count"] / len(per_model_rows)) if per_model_rows else 0.0,
                "consensus_mean_exact_closure_proxy": leader["mean_exact_closure_proxy"],
                "mean_category_exact_closure_proxy": average([row["exact_closure_proxy"] for row in per_model_rows]),
                "mean_category_wrong_family_margin": average([row["wrong_family_margin"] for row in per_model_rows]),
                "term_candidates": ranked_terms,
                "per_model": per_model_rows,
            }
        )
    consensus_rows.sort(
        key=lambda row: (
            row["consensus_support_count"],
            row["consensus_mean_exact_closure_proxy"],
            row["mean_category_wrong_family_margin"],
        ),
        reverse=True,
    )
    return consensus_rows


def build_summary(scan_dirs: Sequence[Path], per_model_rows: Sequence[Dict[str, object]], consensus_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    model_tags = [row["model_tag"] for row in per_model_rows]
    strong_consensus = [row for row in consensus_rows if int(row["consensus_support_count"]) >= max(2, len(scan_dirs))]
    medium_consensus = [row for row in consensus_rows if int(row["consensus_support_count"]) >= 2]
    return {
        "record_type": "stage56_mass_term_scan_compare_summary",
        "model_count": len(scan_dirs),
        "models": model_tags,
        "total_input_items": sum(int(row["input_items"]) for row in per_model_rows),
        "total_record_count": sum(int(row["record_count"]) for row in per_model_rows),
        "mean_prompt_stability_survey": average([float(row["mean_prompt_stability_survey"]) for row in per_model_rows]),
        "mean_prompt_stability_deep": average([float(row["mean_prompt_stability_deep"]) for row in per_model_rows]),
        "mean_prompt_stability_closure": average([float(row["mean_prompt_stability_closure"]) for row in per_model_rows]),
        "strong_consensus_categories": [row["category"] for row in strong_consensus],
        "medium_consensus_categories": [row["category"] for row in medium_consensus],
        "top_consensus_rows": consensus_rows[:12],
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, summary: Dict[str, object], per_model_rows: Sequence[Dict[str, object]], consensus_rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Stage56 Mass Term Scan Compare Report",
        "",
        f"- model_count: {summary['model_count']}",
        f"- total_input_items: {summary['total_input_items']}",
        f"- total_record_count: {summary['total_record_count']}",
        f"- mean_prompt_stability_survey: {summary['mean_prompt_stability_survey']:.6f}",
        f"- mean_prompt_stability_deep: {summary['mean_prompt_stability_deep']:.6f}",
        f"- mean_prompt_stability_closure: {summary['mean_prompt_stability_closure']:.6f}",
        "",
        "## Per Model",
    ]
    for row in per_model_rows:
        lines.append(
            "- "
            f"{row['model_tag']} / items={row['input_items']}"
            f" / records={row['record_count']}"
            f" / survey={row['mean_prompt_stability_survey']:.6f}"
            f" / deep={row['mean_prompt_stability_deep']:.6f}"
            f" / closure={row['mean_prompt_stability_closure']:.6f}"
        )
    lines.extend(["", "## Category Consensus"])
    for row in consensus_rows[:12]:
        lines.append(
            "- "
            f"{row['category']} / consensus_term={row['consensus_term']}"
            f" / support={row['consensus_support_count']}/{row['model_count']}"
            f" / mean_exact_closure={row['consensus_mean_exact_closure_proxy']:.6f}"
            f" / mean_margin={row['mean_category_wrong_family_margin']:.6f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare multiple stage56 mass-term structure scans")
    ap.add_argument("--scan-dir", action="append", required=True)
    ap.add_argument("--summary-file", required=True)
    ap.add_argument("--per-model-file", required=True)
    ap.add_argument("--per-category-file", required=True)
    ap.add_argument("--report-file", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    scan_dirs = [Path(item) for item in args.scan_dir]
    per_model_rows = [build_model_row(scan_dir) for scan_dir in scan_dirs]
    consensus_rows = build_consensus_rows(scan_dirs)
    summary = build_summary(scan_dirs, per_model_rows, consensus_rows)
    write_json(Path(args.summary_file), summary)
    write_jsonl(Path(args.per_model_file), per_model_rows)
    write_jsonl(Path(args.per_category_file), consensus_rows)
    write_report(Path(args.report_file), summary, per_model_rows, consensus_rows)
    print(
        json.dumps(
            {
                "summary_file": args.summary_file,
                "per_model_file": args.per_model_file,
                "per_category_file": args.per_category_file,
                "report_file": args.report_file,
                "model_count": summary["model_count"],
                "total_record_count": summary["total_record_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
