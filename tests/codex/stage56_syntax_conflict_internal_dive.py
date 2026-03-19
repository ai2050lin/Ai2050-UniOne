from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def average(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def positive_part(value: float) -> float:
    return value if value > 0.0 else 0.0


def pair_key(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    return (
        str(row["model_id"]),
        str(row["category"]),
        str(row["prototype_term"]),
        str(row["instance_term"]),
    )


def weighted_mode(labels: Iterable[str], weights: Iterable[float], default: str) -> str:
    totals: Dict[str, float] = {}
    for label, weight in zip(labels, weights):
        if label:
            totals[label] = totals.get(label, 0.0) + weight
    if not totals:
        return default
    return max(sorted(totals), key=lambda label: totals[label])


def weighted_profile(profiles: Sequence[Sequence[float]], weights: Sequence[float]) -> List[float]:
    if not profiles or not weights:
        return []
    width = max(len(profile) for profile in profiles)
    totals = [0.0] * width
    total_weight = 0.0
    for profile, weight in zip(profiles, weights):
        if weight <= 0.0:
            continue
        total_weight += weight
        for idx, value in enumerate(profile):
            totals[idx] += safe_float(value) * weight
    if total_weight == 0.0:
        return [0.0] * width
    return [value / total_weight for value in totals]


def weighted_peak_label(profile: Sequence[float], prefix: str) -> str:
    if not profile:
        return f"{prefix}_0"
    best_idx = max(range(len(profile)), key=lambda idx: profile[idx])
    return f"{prefix}_{best_idx}"


def join_rows(
    pair_rows: Sequence[Dict[str, object]],
    internal_cases: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    case_map = {pair_key(row): row for row in internal_cases}
    joined: List[Dict[str, object]] = []
    for pair_row in pair_rows:
        key = pair_key(pair_row)
        if key not in case_map:
            continue
        syntax_x = positive_part(
            safe_float(pair_row["axes"]["syntax"]["conflict_field_proxy"])
        )
        if syntax_x <= 0.0:
            continue
        synergy = safe_float(pair_row["union_synergy_joint"])
        strict_positive = bool(pair_row["strict_positive_synergy"])
        syntax_internal = dict(case_map[key]["axis_internal_summary"]["syntax"])
        joined.append(
            {
                "model_id": key[0],
                "category": key[1],
                "prototype_term": key[2],
                "instance_term": key[3],
                "syntax_conflict_proxy": syntax_x,
                "union_joint_adv": safe_float(pair_row["union_joint_adv"]),
                "union_synergy_joint": synergy,
                "strict_positive_synergy": strict_positive,
                "constraint_weight": syntax_x if (synergy > 0.0 or strict_positive) else 0.0,
                "destructive_weight": syntax_x if (synergy <= 0.0 and not strict_positive) else 0.0,
                "syntax_internal_summary": syntax_internal,
            }
        )
    return joined


def aggregate_subset(rows: Sequence[Dict[str, object]], weight_key: str) -> Dict[str, object]:
    selected = [row for row in rows if safe_float(row[weight_key]) > 0.0]
    weights = [safe_float(row[weight_key]) for row in selected]
    hidden_profiles = [
        list(row["syntax_internal_summary"]["mean_hidden_shift_profile"]) for row in selected
    ]
    mlp_profiles = [
        list(row["syntax_internal_summary"]["mean_mlp_layer_delta_profile"]) for row in selected
    ]
    dominant_hidden_labels = [
        str(row["syntax_internal_summary"]["dominant_hidden_layer"]) for row in selected
    ]
    dominant_mlp_labels = [
        str(row["syntax_internal_summary"]["dominant_mlp_layer"]) for row in selected
    ]
    dominant_head_labels = [
        str(row["syntax_internal_summary"]["dominant_attention_head"]) for row in selected
    ]
    hidden_profile = weighted_profile(hidden_profiles, weights)
    mlp_profile = weighted_profile(mlp_profiles, weights)
    ordered_cases = sorted(
        selected,
        key=lambda row: (
            safe_float(row[weight_key]),
            safe_float(row["union_synergy_joint"]),
            safe_float(row["union_joint_adv"]),
        ),
        reverse=True,
    )
    return {
        "case_count": len(selected),
        "weight_sum": sum(weights),
        "mean_syntax_conflict_proxy": average([safe_float(row["syntax_conflict_proxy"]) for row in selected]),
        "mean_union_synergy_joint": average([safe_float(row["union_synergy_joint"]) for row in selected]),
        "mean_union_joint_adv": average([safe_float(row["union_joint_adv"]) for row in selected]),
        "dominant_hidden_layer_mode": weighted_mode(dominant_hidden_labels, weights, "layer_0"),
        "dominant_mlp_layer_mode": weighted_mode(dominant_mlp_labels, weights, "layer_0"),
        "dominant_attention_head_mode": weighted_mode(dominant_head_labels, weights, "layer_0_head_0"),
        "peak_hidden_layer_from_profile": weighted_peak_label(hidden_profile, "layer"),
        "peak_mlp_layer_from_profile": weighted_peak_label(mlp_profile, "layer"),
        "weighted_hidden_shift_profile": hidden_profile,
        "weighted_mlp_layer_delta_profile": mlp_profile,
        "top_cases": [
            {
                "model_id": row["model_id"],
                "category": row["category"],
                "prototype_term": row["prototype_term"],
                "instance_term": row["instance_term"],
                "syntax_conflict_proxy": safe_float(row["syntax_conflict_proxy"]),
                "weight": safe_float(row[weight_key]),
                "union_synergy_joint": safe_float(row["union_synergy_joint"]),
                "dominant_hidden_layer": row["syntax_internal_summary"]["dominant_hidden_layer"],
                "dominant_mlp_layer": row["syntax_internal_summary"]["dominant_mlp_layer"],
                "dominant_attention_head": row["syntax_internal_summary"]["dominant_attention_head"],
            }
            for row in ordered_cases[:8]
        ],
    }


def aggregate_per_model(rows: Sequence[Dict[str, object]], weight_key: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for model_id in sorted({str(row["model_id"]) for row in rows}):
        out[model_id] = aggregate_subset(
            [row for row in rows if str(row["model_id"]) == model_id],
            weight_key=weight_key,
        )
    return out


def build_summary(joined_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "record_type": "stage56_syntax_conflict_internal_dive_summary",
        "joined_row_count": len(joined_rows),
        "constraint_conflict": {
            "overall": aggregate_subset(joined_rows, "constraint_weight"),
            "per_model": aggregate_per_model(joined_rows, "constraint_weight"),
        },
        "destructive_conflict": {
            "overall": aggregate_subset(joined_rows, "destructive_weight"),
            "per_model": aggregate_per_model(joined_rows, "destructive_weight"),
        },
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    support = dict(summary["constraint_conflict"]["overall"])
    damage = dict(summary["destructive_conflict"]["overall"])
    lines = [
        "# Stage56 Syntax Conflict Internal Dive",
        "",
        f"- joined_row_count: {summary['joined_row_count']}",
        "",
        "## Constraint Conflict",
        f"- case_count: {support['case_count']}",
        f"- dominant_hidden_layer_mode: {support['dominant_hidden_layer_mode']}",
        f"- dominant_mlp_layer_mode: {support['dominant_mlp_layer_mode']}",
        f"- dominant_attention_head_mode: {support['dominant_attention_head_mode']}",
        f"- peak_hidden_layer_from_profile: {support['peak_hidden_layer_from_profile']}",
        f"- peak_mlp_layer_from_profile: {support['peak_mlp_layer_from_profile']}",
        "",
        "## Destructive Conflict",
        f"- case_count: {damage['case_count']}",
        f"- dominant_hidden_layer_mode: {damage['dominant_hidden_layer_mode']}",
        f"- dominant_mlp_layer_mode: {damage['dominant_mlp_layer_mode']}",
        f"- dominant_attention_head_mode: {damage['dominant_attention_head_mode']}",
        f"- peak_hidden_layer_from_profile: {damage['peak_hidden_layer_from_profile']}",
        f"- peak_mlp_layer_from_profile: {damage['peak_mlp_layer_from_profile']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dive syntax->X closure signal into hidden, MLP, and attention coordinates")
    ap.add_argument(
        "--pair-joined-rows",
        default=str(
            ROOT
            / "tests"
            / "codex_temp"
            / "stage56_generation_gate_stage6_pair_link_all3_12cat_pairs_20260318_2120"
            / "joined_rows.jsonl"
        ),
    )
    ap.add_argument(
        "--internal-cases",
        default=str(
            ROOT
            / "tests"
            / "codex_temp"
            / "stage56_generation_gate_internal_map_20260318_1338"
            / "cases.jsonl"
        ),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_syntax_conflict_internal_dive_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pair_rows = read_jsonl(Path(args.pair_joined_rows))
    internal_cases = read_jsonl(Path(args.internal_cases))
    joined_rows = join_rows(pair_rows, internal_cases)
    summary = build_summary(joined_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "joined_rows.json", {"rows": joined_rows})
    write_report(out_dir / "REPORT.md", summary)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "joined_row_count": len(joined_rows),
                "constraint_cases": summary["constraint_conflict"]["overall"]["case_count"],
                "destructive_cases": summary["destructive_conflict"]["overall"]["case_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
