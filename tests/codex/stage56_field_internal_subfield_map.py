from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]

COMPONENT_SPECS = (
    ("logic_prototype", "logic", "prototype_positive"),
    ("syntax_constraint_conflict", "syntax", "constraint_conflict"),
    ("logic_fragile_bridge", "logic", "fragile_bridge"),
)


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
    best_idx = max(range(len(profile)), key=lambda idx: abs(safe_float(profile[idx])))
    return f"{prefix}_{best_idx}"


def component_weight(row: Dict[str, object], axis: str, component_name: str) -> float:
    if component_name == "prototype_positive":
        return positive_part(safe_float(dict(row.get("axes", {})).get(axis, {}).get("prototype_field_proxy")))
    return positive_part(safe_float(dict(row.get("rewritten_axes", {})).get(axis, {}).get(component_name)))


def join_rows(
    rewritten_rows: Sequence[Dict[str, object]],
    internal_cases: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    case_map = {pair_key(row): row for row in internal_cases}
    joined: List[Dict[str, object]] = []
    for rewritten in rewritten_rows:
        key = pair_key(rewritten)
        case = case_map.get(key)
        if case is None:
            continue
        axis_map = dict(case.get("axis_internal_summary", {}))
        for component_label, axis, component_name in COMPONENT_SPECS:
            axis_summary = dict(axis_map.get(axis, {}))
            joined.append(
                {
                    "component_label": component_label,
                    "axis": axis,
                    "model_id": key[0],
                    "category": key[1],
                    "prototype_term": key[2],
                    "instance_term": key[3],
                    "weight": component_weight(rewritten, axis=axis, component_name=component_name),
                    "union_joint_adv": safe_float(rewritten.get("union_joint_adv")),
                    "union_synergy_joint": safe_float(rewritten.get("union_synergy_joint")),
                    "strict_positive_synergy": bool(rewritten.get("strict_positive_synergy")),
                    "dominant_hidden_layer": str(axis_summary.get("dominant_hidden_layer", "layer_0")),
                    "dominant_mlp_layer": str(axis_summary.get("dominant_mlp_layer", "layer_0")),
                    "dominant_attention_head": str(axis_summary.get("dominant_attention_head", "layer_0_head_0")),
                    "hidden_profile": list(axis_summary.get("mean_hidden_shift_profile", [])),
                    "mlp_profile": list(axis_summary.get("mean_mlp_layer_delta_profile", [])),
                }
            )
    return joined


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    selected = [row for row in rows if safe_float(row.get("weight")) > 0.0]
    weights = [safe_float(row["weight"]) for row in selected]
    hidden_profiles = [list(row.get("hidden_profile", [])) for row in selected]
    mlp_profiles = [list(row.get("mlp_profile", [])) for row in selected]
    hidden_profile = weighted_profile(hidden_profiles, weights)
    mlp_profile = weighted_profile(mlp_profiles, weights)
    ordered = sorted(
        selected,
        key=lambda row: (
            safe_float(row["weight"]),
            safe_float(row["union_synergy_joint"]),
            safe_float(row["union_joint_adv"]),
        ),
        reverse=True,
    )
    return {
        "case_count": len(selected),
        "weight_sum": sum(weights),
        "mean_weight": average(weights),
        "mean_union_synergy_joint": average([safe_float(row["union_synergy_joint"]) for row in selected]),
        "mean_union_joint_adv": average([safe_float(row["union_joint_adv"]) for row in selected]),
        "dominant_hidden_layer_mode": weighted_mode(
            [str(row["dominant_hidden_layer"]) for row in selected],
            weights,
            "layer_0",
        ),
        "dominant_mlp_layer_mode": weighted_mode(
            [str(row["dominant_mlp_layer"]) for row in selected],
            weights,
            "layer_0",
        ),
        "dominant_attention_head_mode": weighted_mode(
            [str(row["dominant_attention_head"]) for row in selected],
            weights,
            "layer_0_head_0",
        ),
        "peak_hidden_layer_from_profile": weighted_peak_label(hidden_profile, "layer"),
        "peak_mlp_layer_from_profile": weighted_peak_label(mlp_profile, "layer"),
        "weighted_hidden_profile": hidden_profile,
        "weighted_mlp_profile": mlp_profile,
        "top_cases": [
            {
                "model_id": row["model_id"],
                "category": row["category"],
                "prototype_term": row["prototype_term"],
                "instance_term": row["instance_term"],
                "weight": safe_float(row["weight"]),
                "union_synergy_joint": safe_float(row["union_synergy_joint"]),
                "union_joint_adv": safe_float(row["union_joint_adv"]),
                "dominant_hidden_layer": row["dominant_hidden_layer"],
                "dominant_mlp_layer": row["dominant_mlp_layer"],
                "dominant_attention_head": row["dominant_attention_head"],
            }
            for row in ordered[:8]
        ],
    }


def aggregate_per_model(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for model_id in sorted({str(row["model_id"]) for row in rows}):
        out[model_id] = aggregate_rows([row for row in rows if str(row["model_id"]) == model_id])
    return out


def build_summary(joined_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    per_component: Dict[str, object] = {}
    for component_label, _axis, _component_name in COMPONENT_SPECS:
        component_rows = [row for row in joined_rows if str(row["component_label"]) == component_label]
        per_component[component_label] = {
            "overall": aggregate_rows(component_rows),
            "per_model": aggregate_per_model(component_rows),
        }
    return {
        "record_type": "stage56_field_internal_subfield_map_summary",
        "joined_row_count": len(joined_rows),
        "component_labels": [label for label, _axis, _name in COMPONENT_SPECS],
        "per_component": per_component,
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 定向分量到内部子场联立",
        "",
        f"- joined_row_count: {summary['joined_row_count']}",
        "",
        "## Overall",
    ]
    for component_label in summary["component_labels"]:
        overall = dict(summary["per_component"][component_label]["overall"])
        lines.append(
            f"- {component_label}: "
            f"case_count={overall['case_count']}, "
            f"hidden_mode={overall['dominant_hidden_layer_mode']}, "
            f"mlp_mode={overall['dominant_mlp_layer_mode']}, "
            f"head_mode={overall['dominant_attention_head_mode']}, "
            f"peak_hidden={overall['peak_hidden_layer_from_profile']}, "
            f"peak_mlp={overall['peak_mlp_layer_from_profile']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Map logic_P, syntax_CX, and logic_fragile_bridge to internal subfields")
    ap.add_argument(
        "--rewritten-rows-jsonl",
        default=str(
            ROOT / "tests" / "codex_temp" / "stage56_bxm_rewrite_20260318_2222" / "rewritten_rows.jsonl"
        ),
    )
    ap.add_argument(
        "--internal-cases-jsonl",
        default=str(
            ROOT / "tests" / "codex_temp" / "stage56_generation_gate_internal_map_20260318_1338" / "cases.jsonl"
        ),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_field_internal_subfield_map_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rewritten_rows = read_jsonl(Path(args.rewritten_rows_jsonl))
    internal_cases = read_jsonl(Path(args.internal_cases_jsonl))
    joined_rows = join_rows(rewritten_rows, internal_cases)
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
                "components": {
                    label: summary["per_component"][label]["overall"]["case_count"]
                    for label in summary["component_labels"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
