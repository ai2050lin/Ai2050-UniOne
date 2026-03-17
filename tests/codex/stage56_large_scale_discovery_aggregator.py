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


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_model_roots(output_root: Path) -> List[Path]:
    roots: List[Path] = []
    if not output_root.exists():
        return roots
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        marker = child / "stage6_prototype_instance_decomposition" / "summary.json"
        if marker.exists():
            roots.append(child)
    return roots


def sum_layer_counts(rows: Iterable[Dict[str, object]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for layer, value in dict(row.get("candidate_layer_distribution", {})).items():
            counts[str(layer)] += int(value)
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], int(kv[0]))))


def top_layer_rows(layer_counts: Dict[str, int], limit: int = 8) -> List[Dict[str, object]]:
    rows = [{"layer": int(layer), "count": int(count)} for layer, count in layer_counts.items()]
    rows.sort(key=lambda row: (-row["count"], row["layer"]))
    return rows[:limit]


def average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def aggregate_output_root(output_root: Path) -> Dict[str, object]:
    model_roots = find_model_roots(output_root)
    per_model_rows: List[Dict[str, object]] = []
    per_category_rows: List[Dict[str, object]] = []
    category_totals: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "category": "",
            "model_count": 0,
            "pair_count": 0,
            "strict_positive_pair_count": 0,
            "mean_union_joint_adv_values": [],
            "mean_union_synergy_joint_values": [],
            "models": [],
        }
    )

    total_pairs = 0
    total_strict_positive_pairs = 0
    total_margin_zero_pairs = 0

    for model_root in model_roots:
        stage3_summary = read_json(model_root / "stage3_causal_closure" / "summary.json")
        stage5_prototype_summary = read_json(model_root / "stage5_prototype" / "summary.json")
        stage5_instance_summary = read_json(model_root / "stage5_instance" / "summary.json")
        stage6_summary = read_json(model_root / "stage6_prototype_instance_decomposition" / "summary.json")
        prototype_candidates = read_jsonl(model_root / "stage5_prototype" / "candidates.jsonl")
        instance_candidates = read_jsonl(model_root / "stage5_instance" / "candidates.jsonl")
        stage6_rows = read_jsonl(model_root / "stage6_prototype_instance_decomposition" / "results.jsonl")

        prototype_layer_counts = sum_layer_counts(prototype_candidates)
        instance_layer_counts = sum_layer_counts(instance_candidates)
        strict_positive_rows = [row for row in stage6_rows if bool(row.get("strict_positive_synergy"))]
        margin_zero_rows = [row for row in stage6_rows if float(row.get("union_margin_adv", 0.0)) == 0.0]

        per_model_rows.append(
            {
                "record_type": "stage56_large_scale_discovery_model_summary",
                "model_tag": model_root.name,
                "model_id": stage6_summary["model_id"],
                "selected_category_count": len(stage3_summary.get("selected_categories", [])),
                "selected_categories": stage3_summary.get("selected_categories", []),
                "prototype_candidate_count": stage5_prototype_summary["candidate_count"],
                "instance_candidate_count": stage5_instance_summary["candidate_count"],
                "prototype_mean_strict_joint_adv": stage5_prototype_summary["mean_candidate_full_strict_joint_adv"],
                "instance_mean_strict_joint_adv": stage5_instance_summary["mean_candidate_full_strict_joint_adv"],
                "prototype_strict_positive_micro_circuit_count": stage5_prototype_summary["strict_positive_micro_circuit_count"],
                "instance_strict_positive_micro_circuit_count": stage5_instance_summary["strict_positive_micro_circuit_count"],
                "pair_count": stage6_summary["pair_count"],
                "strict_positive_pair_count": stage6_summary["strict_positive_synergy_pair_count"],
                "strict_positive_pair_ratio": safe_ratio(
                    stage6_summary["strict_positive_synergy_pair_count"],
                    stage6_summary["pair_count"],
                ),
                "mean_union_joint_adv": stage6_summary["mean_union_joint_adv"],
                "mean_union_synergy_joint": stage6_summary["mean_union_synergy_joint"],
                "margin_zero_pair_count": len(margin_zero_rows),
                "margin_zero_pair_ratio": safe_ratio(len(margin_zero_rows), len(stage6_rows)),
                "top_prototype_layers": top_layer_rows(prototype_layer_counts),
                "top_instance_layers": top_layer_rows(instance_layer_counts),
                "strict_positive_categories": stage6_summary["strict_positive_synergy_categories"],
            }
        )

        total_pairs += len(stage6_rows)
        total_strict_positive_pairs += len(strict_positive_rows)
        total_margin_zero_pairs += len(margin_zero_rows)

        category_stats: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in stage6_rows:
            category_stats[str(row["category"])].append(row)
        for category, rows in sorted(category_stats.items()):
            strict_count = int(sum(1 for row in rows if bool(row.get("strict_positive_synergy"))))
            union_joint_values = [float(row["union_joint_adv"]) for row in rows]
            synergy_values = [float(row["union_synergy_joint"]) for row in rows]
            overlap_ratios = [
                safe_ratio(float(row["overlap_neuron_count"]), float(row["union_neuron_count"]))
                for row in rows
            ]
            top_row = sorted(
                rows,
                key=lambda row: (
                    1 if bool(row.get("strict_positive_synergy")) else 0,
                    float(row["union_joint_adv"]),
                ),
                reverse=True,
            )[0]
            per_category_rows.append(
                {
                    "record_type": "stage56_large_scale_discovery_category_summary",
                    "model_tag": model_root.name,
                    "model_id": stage6_summary["model_id"],
                    "category": category,
                    "pair_count": len(rows),
                    "strict_positive_pair_count": strict_count,
                    "strict_positive_pair_ratio": safe_ratio(strict_count, len(rows)),
                    "mean_union_joint_adv": average(union_joint_values),
                    "mean_union_synergy_joint": average(synergy_values),
                    "mean_overlap_ratio": average(overlap_ratios),
                    "mean_union_neuron_count": average([float(row["union_neuron_count"]) for row in rows]),
                    "mean_prototype_neuron_count": average([float(row["prototype_neuron_count"]) for row in rows]),
                    "mean_instance_neuron_count": average([float(row["instance_neuron_count"]) for row in rows]),
                    "top_instance_term": top_row["instance_term"],
                    "top_union_joint_adv": top_row["union_joint_adv"],
                    "top_union_synergy_joint": top_row["union_synergy_joint"],
                    "top_row_is_strict_positive": bool(top_row.get("strict_positive_synergy")),
                }
            )

            totals = category_totals[category]
            totals["category"] = category
            totals["model_count"] = int(totals["model_count"]) + 1
            totals["pair_count"] = int(totals["pair_count"]) + len(rows)
            totals["strict_positive_pair_count"] = int(totals["strict_positive_pair_count"]) + strict_count
            totals["mean_union_joint_adv_values"].append(average(union_joint_values))
            totals["mean_union_synergy_joint_values"].append(average(synergy_values))
            totals["models"].append(model_root.name)

    consensus_rows = []
    for category, totals in sorted(category_totals.items()):
        consensus_rows.append(
            {
                "category": category,
                "model_count": totals["model_count"],
                "models": sorted(totals["models"]),
                "pair_count": totals["pair_count"],
                "strict_positive_pair_count": totals["strict_positive_pair_count"],
                "strict_positive_pair_ratio": safe_ratio(
                    float(totals["strict_positive_pair_count"]),
                    float(totals["pair_count"]),
                ),
                "mean_union_joint_adv": average(totals["mean_union_joint_adv_values"]),
                "mean_union_synergy_joint": average(totals["mean_union_synergy_joint_values"]),
            }
        )
    consensus_rows.sort(
        key=lambda row: (
            row["strict_positive_pair_count"],
            row["mean_union_synergy_joint"],
            row["mean_union_joint_adv"],
        ),
        reverse=True,
    )

    summary = {
        "record_type": "stage56_large_scale_discovery_summary",
        "output_root": str(output_root),
        "model_count": len(per_model_rows),
        "models": [row["model_tag"] for row in per_model_rows],
        "total_pair_count": total_pairs,
        "total_strict_positive_pair_count": total_strict_positive_pairs,
        "strict_positive_pair_ratio": safe_ratio(total_strict_positive_pairs, total_pairs),
        "total_margin_zero_pair_count": total_margin_zero_pairs,
        "margin_zero_pair_ratio": safe_ratio(total_margin_zero_pairs, total_pairs),
        "categories_with_any_strict_positive": [
            row["category"] for row in consensus_rows if int(row["strict_positive_pair_count"]) > 0
        ],
        "categories_with_cross_model_strict_positive": [
            row["category"]
            for row in consensus_rows
            if int(row["strict_positive_pair_count"]) > 0 and int(row["model_count"]) >= 2
        ],
        "top_consensus_categories": consensus_rows[:10],
    }
    return {
        "summary": summary,
        "per_model_rows": per_model_rows,
        "per_category_rows": per_category_rows,
        "consensus_rows": consensus_rows,
    }


def write_report(
    path: Path,
    summary: Dict[str, object],
    per_model_rows: Sequence[Dict[str, object]],
    consensus_rows: Sequence[Dict[str, object]],
) -> None:
    lines = [
        "# Stage56 Large Scale Discovery Report",
        "",
        f"- model_count: {summary['model_count']}",
        f"- total_pair_count: {summary['total_pair_count']}",
        f"- total_strict_positive_pair_count: {summary['total_strict_positive_pair_count']}",
        f"- strict_positive_pair_ratio: {summary['strict_positive_pair_ratio']:.6f}",
        f"- total_margin_zero_pair_count: {summary['total_margin_zero_pair_count']}",
        f"- margin_zero_pair_ratio: {summary['margin_zero_pair_ratio']:.6f}",
        "",
        "## Per Model",
    ]
    for row in per_model_rows:
        lines.append(
            "- "
            f"{row['model_tag']} / pairs={row['pair_count']}"
            f" / strict_positive={row['strict_positive_pair_count']}"
            f" / mean_union_joint={row['mean_union_joint_adv']:.6f}"
            f" / mean_union_synergy={row['mean_union_synergy_joint']:.6f}"
            f" / margin_zero_ratio={row['margin_zero_pair_ratio']:.6f}"
        )
    lines.extend(["", "## Consensus Categories"])
    for row in consensus_rows[:10]:
        lines.append(
            "- "
            f"{row['category']} / models={row['model_count']}"
            f" / pairs={row['pair_count']}"
            f" / strict_positive={row['strict_positive_pair_count']}"
            f" / mean_union_joint={row['mean_union_joint_adv']:.6f}"
            f" / mean_union_synergy={row['mean_union_synergy_joint']:.6f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate large-scale discovery outputs across models and categories")
    ap.add_argument("--output-root", default="tempdata/stage56_large_scale_discovery")
    ap.add_argument("--summary-file", default="tempdata/stage56_large_scale_discovery/discovery_summary.json")
    ap.add_argument("--report-file", default="tempdata/stage56_large_scale_discovery/DISCOVERY_REPORT.md")
    ap.add_argument("--per-model-file", default="tempdata/stage56_large_scale_discovery/discovery_per_model.jsonl")
    ap.add_argument("--per-category-file", default="tempdata/stage56_large_scale_discovery/discovery_per_category.jsonl")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    payload = aggregate_output_root(Path(args.output_root))
    write_json(Path(args.summary_file), payload["summary"])
    write_jsonl(Path(args.per_model_file), payload["per_model_rows"])
    write_jsonl(Path(args.per_category_file), payload["per_category_rows"])
    write_report(Path(args.report_file), payload["summary"], payload["per_model_rows"], payload["consensus_rows"])
    print(
        json.dumps(
            {
                "summary_file": args.summary_file,
                "report_file": args.report_file,
                "per_model_file": args.per_model_file,
                "per_category_file": args.per_category_file,
                "model_count": payload["summary"]["model_count"],
                "total_pair_count": payload["summary"]["total_pair_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
