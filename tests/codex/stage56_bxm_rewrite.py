from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
AXES = ("style", "logic", "syntax")


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


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def average(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def positive_part(value: float) -> float:
    return value if value > 0.0 else 0.0


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = average(xs)
    mean_y = average(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    cov = sum(a * b for a, b in zip(dx, dy))
    std_x = sum(a * a for a in dx) ** 0.5
    std_y = sum(b * b for b in dy) ** 0.5
    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    return float(cov / (std_x * std_y))


def decompose_axis_fields(row: Dict[str, object], axis: str) -> Dict[str, float]:
    axis_values = dict(row["axes"][axis])
    bridge = positive_part(safe_float(axis_values.get("bridge_field_proxy")))
    conflict = positive_part(safe_float(axis_values.get("conflict_field_proxy")))
    mismatch = positive_part(safe_float(axis_values.get("mismatch_field_proxy")))
    joint_adv = safe_float(row.get("union_joint_adv"))
    synergy = safe_float(row.get("union_synergy_joint"))
    strict_positive = bool(row.get("strict_positive_synergy"))

    stable_bridge = bridge if joint_adv > 0.0 and synergy > 0.0 else 0.0
    fragile_bridge = bridge if bridge > 0.0 and stable_bridge == 0.0 else 0.0
    constraint_conflict = conflict if (synergy > 0.0 or strict_positive) else 0.0
    destructive_conflict = conflict if conflict > 0.0 and constraint_conflict == 0.0 else 0.0
    mismatch_exposure = mismatch if joint_adv > 0.0 and synergy >= 0.0 else 0.0
    mismatch_damage = mismatch if mismatch > 0.0 and mismatch_exposure == 0.0 else 0.0

    return {
        "bridge_raw": bridge,
        "stable_bridge": stable_bridge,
        "fragile_bridge": fragile_bridge,
        "conflict_raw": conflict,
        "constraint_conflict": constraint_conflict,
        "destructive_conflict": destructive_conflict,
        "mismatch_raw": mismatch,
        "mismatch_exposure": mismatch_exposure,
        "mismatch_damage": mismatch_damage,
    }


def rewrite_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        rewritten = dict(row)
        rewritten_axes = {}
        for axis in AXES:
            rewritten_axes[axis] = decompose_axis_fields(row, axis)
        rewritten["rewritten_axes"] = rewritten_axes
        out.append(rewritten)
    return out


def summarize_axis(rows: Sequence[Dict[str, object]], axis: str) -> Dict[str, object]:
    components = (
        "stable_bridge",
        "fragile_bridge",
        "constraint_conflict",
        "destructive_conflict",
        "mismatch_exposure",
        "mismatch_damage",
    )
    synergy_targets = [safe_float(row.get("union_synergy_joint")) for row in rows]
    joint_adv_targets = [safe_float(row.get("union_joint_adv")) for row in rows]
    axis_rows = [dict(row["rewritten_axes"][axis]) for row in rows]

    summary: Dict[str, object] = {
        "row_count": len(rows),
        "mean_bridge_raw": average([safe_float(row["bridge_raw"]) for row in axis_rows]),
        "mean_conflict_raw": average([safe_float(row["conflict_raw"]) for row in axis_rows]),
        "mean_mismatch_raw": average([safe_float(row["mismatch_raw"]) for row in axis_rows]),
    }
    for component in components:
        values = [safe_float(row[component]) for row in axis_rows]
        raw_total = sum(values)
        if component.endswith("bridge"):
            denom = sum(safe_float(row["bridge_raw"]) for row in axis_rows)
        elif component.endswith("conflict"):
            denom = sum(safe_float(row["conflict_raw"]) for row in axis_rows)
        else:
            denom = sum(safe_float(row["mismatch_raw"]) for row in axis_rows)
        summary[component] = {
            "mean_value": average(values),
            "nonzero_count": sum(1 for value in values if value > 0.0),
            "share_within_parent": (raw_total / denom) if denom > 0.0 else 0.0,
            "corr_to_union_synergy_joint": pearson_corr(values, synergy_targets),
            "corr_to_union_joint_adv": pearson_corr(values, joint_adv_targets),
        }
    return summary


def build_top_findings(axis_summary: Dict[str, object]) -> Dict[str, object]:
    findings: List[Dict[str, object]] = []
    for axis in AXES:
        block = dict(axis_summary[axis])
        for component in (
            "stable_bridge",
            "fragile_bridge",
            "constraint_conflict",
            "destructive_conflict",
            "mismatch_exposure",
            "mismatch_damage",
        ):
            item = dict(block[component])
            findings.append(
                {
                    "axis": axis,
                    "component": component,
                    "mean_value": safe_float(item["mean_value"]),
                    "share_within_parent": safe_float(item["share_within_parent"]),
                    "corr_to_union_synergy_joint": safe_float(item["corr_to_union_synergy_joint"]),
                    "corr_to_union_joint_adv": safe_float(item["corr_to_union_joint_adv"]),
                }
            )
    return {
        "top_mean_components": sorted(findings, key=lambda row: row["mean_value"], reverse=True)[:8],
        "top_positive_closure_components": sorted(
            findings,
            key=lambda row: (
                row["corr_to_union_synergy_joint"],
                row["corr_to_union_joint_adv"],
                row["mean_value"],
            ),
            reverse=True,
        )[:8],
    }


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    axis_summary = {axis: summarize_axis(rows, axis) for axis in AXES}
    return {
        "record_type": "stage56_bxm_rewrite_summary",
        "row_count": len(rows),
        "definitions": {
            "stable_bridge": "B>0，且 union_joint_adv>0，且 union_synergy_joint>0",
            "fragile_bridge": "B>0，但未满足稳定桥接闭包条件",
            "constraint_conflict": "X>0，且 union_synergy_joint>0，或 strict_positive_synergy 为真",
            "destructive_conflict": "X>0，且不属于约束型冲突",
            "mismatch_exposure": "M>0，且 union_joint_adv>0，且 union_synergy_joint>=0",
            "mismatch_damage": "M>0，且不属于失配暴露",
        },
        "per_axis": axis_summary,
        "top_findings": build_top_findings(axis_summary),
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 B/X/M Rewrite Report",
        "",
        f"- row_count: {summary['row_count']}",
        "",
        "## Definitions",
    ]
    for key, value in dict(summary["definitions"]).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Per Axis"])
    for axis in AXES:
        block = dict(summary["per_axis"][axis])
        lines.append(
            f"- {axis}: "
            f"stable_bridge={safe_float(block['stable_bridge']['mean_value']):.6f}, "
            f"fragile_bridge={safe_float(block['fragile_bridge']['mean_value']):.6f}, "
            f"constraint_conflict={safe_float(block['constraint_conflict']['mean_value']):.6f}, "
            f"destructive_conflict={safe_float(block['destructive_conflict']['mean_value']):.6f}, "
            f"mismatch_exposure={safe_float(block['mismatch_exposure']['mean_value']):.6f}, "
            f"mismatch_damage={safe_float(block['mismatch_damage']['mean_value']):.6f}"
        )
    lines.extend(["", "## Top Positive Closure Components"])
    for row in summary["top_findings"]["top_positive_closure_components"]:
        lines.append(
            f"- {row['axis']} / {row['component']}: "
            f"mean={row['mean_value']:.6f}, "
            f"share={row['share_within_parent']:.4f}, "
            f"corr_synergy={row['corr_to_union_synergy_joint']:.4f}, "
            f"corr_joint_adv={row['corr_to_union_joint_adv']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rewrite B/X/M into stable-fragile and supportive-destructive subfields")
    ap.add_argument(
        "--joined-rows",
        default=str(
            ROOT
            / "tests"
            / "codex_temp"
            / "stage56_generation_gate_stage6_pair_link_all3_12cat_pairs_20260318_2120"
            / "joined_rows.jsonl"
        ),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_bxm_rewrite_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    joined_rows = read_jsonl(Path(args.joined_rows))
    rewritten_rows = rewrite_rows(joined_rows)
    summary = build_summary(rewritten_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "rewritten_rows.jsonl", rewritten_rows)
    write_report(out_dir / "REPORT.md", summary)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "row_count": len(rewritten_rows),
                "top_positive_closure_components": summary["top_findings"]["top_positive_closure_components"][:3],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
