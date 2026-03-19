from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def weighted_peak_label(profile: Sequence[float], labels: Sequence[str], default: str) -> str:
    if not profile or not labels:
        return default
    best_idx = max(range(min(len(profile), len(labels))), key=lambda idx: abs(safe_float(profile[idx])))
    return str(labels[best_idx])


def trajectory_key(row: Dict[str, object]) -> Tuple[str, str, str, str, str]:
    return (
        str(row["model_id"]),
        str(row["category"]),
        str(row["prototype_term"]),
        str(row["instance_term"]),
        str(row["axis"]),
    )


def component_axis(component_label: str) -> str:
    if component_label.startswith("logic_"):
        return "logic"
    if component_label.startswith("syntax_"):
        return "syntax"
    raise ValueError(f"Unsupported component label: {component_label}")


def join_rows(
    component_rows: Sequence[Dict[str, object]],
    trajectory_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    trajectory_map = {trajectory_key(row): row for row in trajectory_rows}
    joined: List[Dict[str, object]] = []
    for row in component_rows:
        axis = component_axis(str(row["component_label"]))
        key = (
            str(row["model_id"]),
            str(row["category"]),
            str(row["prototype_term"]),
            str(row["instance_term"]),
            axis,
        )
        traj = trajectory_map.get(key)
        if traj is None:
            continue
        joined.append(
            {
                "component_label": str(row["component_label"]),
                "axis": axis,
                "model_id": str(row["model_id"]),
                "category": str(row["category"]),
                "prototype_term": str(row["prototype_term"]),
                "instance_term": str(row["instance_term"]),
                "weight": safe_float(row.get("weight")),
                "union_joint_adv": safe_float(row.get("union_joint_adv")),
                "union_synergy_joint": safe_float(row.get("union_synergy_joint")),
                "dominant_hidden_tail_position": str(traj.get("dominant_hidden_tail_position", "tail_pos_-1")),
                "dominant_mlp_tail_position": str(traj.get("dominant_mlp_tail_position", "tail_pos_-1")),
                "hidden_late_focus": safe_float(traj.get("hidden_late_focus")),
                "mlp_late_focus": safe_float(traj.get("mlp_late_focus")),
                "hidden_total": safe_float(traj.get("hidden_total")),
                "mlp_total": safe_float(traj.get("mlp_total")),
                "tail_position_labels": list(traj.get("tail_position_labels", [])),
                "hidden_token_profile": list(traj.get("mean_hidden_token_profile", [])),
                "mlp_token_profile": list(traj.get("mean_mlp_token_profile", [])),
            }
        )
    return joined


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    selected = [row for row in rows if safe_float(row.get("weight")) > 0.0]
    weights = [safe_float(row["weight"]) for row in selected]
    hidden_profiles = [list(row.get("hidden_token_profile", [])) for row in selected]
    mlp_profiles = [list(row.get("mlp_token_profile", [])) for row in selected]
    labels = list(selected[0].get("tail_position_labels", [])) if selected else []
    weighted_hidden = weighted_profile(hidden_profiles, weights)
    weighted_mlp = weighted_profile(mlp_profiles, weights)
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
        "mean_hidden_late_focus": average([safe_float(row["hidden_late_focus"]) for row in selected]),
        "mean_mlp_late_focus": average([safe_float(row["mlp_late_focus"]) for row in selected]),
        "dominant_hidden_tail_position_mode": weighted_mode(
            [str(row["dominant_hidden_tail_position"]) for row in selected],
            weights,
            "tail_pos_-1",
        ),
        "dominant_mlp_tail_position_mode": weighted_mode(
            [str(row["dominant_mlp_tail_position"]) for row in selected],
            weights,
            "tail_pos_-1",
        ),
        "peak_hidden_tail_position_from_profile": weighted_peak_label(
            weighted_hidden,
            labels,
            "tail_pos_-1",
        ),
        "peak_mlp_tail_position_from_profile": weighted_peak_label(
            weighted_mlp,
            labels,
            "tail_pos_-1",
        ),
        "weighted_hidden_token_profile": weighted_hidden,
        "weighted_mlp_token_profile": weighted_mlp,
        "top_cases": [
            {
                "model_id": row["model_id"],
                "category": row["category"],
                "prototype_term": row["prototype_term"],
                "instance_term": row["instance_term"],
                "weight": safe_float(row["weight"]),
                "union_synergy_joint": safe_float(row["union_synergy_joint"]),
                "dominant_hidden_tail_position": row["dominant_hidden_tail_position"],
                "dominant_mlp_tail_position": row["dominant_mlp_tail_position"],
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
    labels = sorted({str(row["component_label"]) for row in joined_rows})
    return {
        "record_type": "stage56_component_trajectory_window_map_summary",
        "joined_row_count": len(joined_rows),
        "component_labels": labels,
        "per_component": {
            label: {
                "overall": aggregate_rows([row for row in joined_rows if str(row["component_label"]) == label]),
                "per_model": aggregate_per_model([row for row in joined_rows if str(row["component_label"]) == label]),
            }
            for label in labels
        },
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 定向分量到词元轨迹窗口联立",
        "",
        f"- joined_row_count: {summary['joined_row_count']}",
        "",
        "## Overall",
    ]
    for label in summary["component_labels"]:
        overall = dict(summary["per_component"][label]["overall"])
        lines.append(
            f"- {label}: "
            f"case_count={overall['case_count']}, "
            f"hidden_mode={overall['dominant_hidden_tail_position_mode']}, "
            f"mlp_mode={overall['dominant_mlp_tail_position_mode']}, "
            f"hidden_peak={overall['peak_hidden_tail_position_from_profile']}, "
            f"mlp_peak={overall['peak_mlp_tail_position_from_profile']}, "
            f"mean_synergy={overall['mean_union_synergy_joint']:.6f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Map component-level subfields to token trajectory windows")
    ap.add_argument(
        "--component-joined-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_field_internal_subfield_map_20260319_0047" / "joined_rows.json"),
    )
    ap.add_argument(
        "--trajectory-cases-jsonl",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_token_trajectory_equation_all12_20260319_0020" / "cases.jsonl"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_component_trajectory_window_map_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    component_rows = list(read_json(Path(args.component_joined_json)).get("rows", []))
    trajectory_rows = read_jsonl(Path(args.trajectory_cases_jsonl))
    joined_rows = join_rows(component_rows, trajectory_rows)
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
