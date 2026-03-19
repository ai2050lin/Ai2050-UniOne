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
    if component_label.startswith("style_"):
        return "style"
    raise ValueError(f"Unsupported component label: {component_label}")


def zone_split(length: int, generated_token_count: int) -> Tuple[int, int]:
    generated = max(0, min(int(generated_token_count), int(length)))
    prompt_end = max(0, int(length) - generated)
    return prompt_end, generated


def abs_sum(values: Sequence[float]) -> float:
    return float(sum(abs(safe_float(value)) for value in values))


def slice_zone(values: Sequence[float], start: int, end: int) -> float:
    return abs_sum(list(values)[start:end])


def dominant_zone(values: Sequence[float], prompt_end: int) -> str:
    items = list(values)
    if not items:
        return "none"
    peak_idx = max(range(len(items)), key=lambda idx: abs(safe_float(items[idx])))
    return "prompt" if peak_idx < prompt_end else "generated"


def attach_zone_metrics(row: Dict[str, object]) -> Dict[str, object]:
    labels = list(row.get("tail_position_labels", []))
    hidden_profile = list(row.get("mean_hidden_token_profile", row.get("hidden_token_profile", [])))
    mlp_profile = list(row.get("mean_mlp_token_profile", row.get("mlp_token_profile", [])))
    length = len(labels) if labels else max(len(hidden_profile), len(mlp_profile))
    prompt_end, generated_count = zone_split(length=length, generated_token_count=int(row.get("generated_token_count", 0)))
    hidden_prompt = slice_zone(hidden_profile, 0, prompt_end)
    hidden_generated = slice_zone(hidden_profile, prompt_end, length)
    mlp_prompt = slice_zone(mlp_profile, 0, prompt_end)
    mlp_generated = slice_zone(mlp_profile, prompt_end, length)
    hidden_total = hidden_prompt + hidden_generated
    mlp_total = mlp_prompt + mlp_generated
    out = dict(row)
    out.update(
        {
            "tail_token_count": length,
            "prompt_window_token_count": prompt_end,
            "generated_window_token_count": generated_count,
            "hidden_prompt_sum": hidden_prompt,
            "hidden_generated_sum": hidden_generated,
            "mlp_prompt_sum": mlp_prompt,
            "mlp_generated_sum": mlp_generated,
            "hidden_prompt_share": hidden_prompt / hidden_total if hidden_total else 0.0,
            "hidden_generated_share": hidden_generated / hidden_total if hidden_total else 0.0,
            "mlp_prompt_share": mlp_prompt / mlp_total if mlp_total else 0.0,
            "mlp_generated_share": mlp_generated / mlp_total if mlp_total else 0.0,
            "dominant_hidden_zone": dominant_zone(hidden_profile, prompt_end),
            "dominant_mlp_zone": dominant_zone(mlp_profile, prompt_end),
        }
    )
    return out


def count_values(labels: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    return counts


def summarize_zone_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "case_count": 0,
            "mean_hidden_prompt_share": 0.0,
            "mean_hidden_generated_share": 0.0,
            "mean_mlp_prompt_share": 0.0,
            "mean_mlp_generated_share": 0.0,
            "dominant_hidden_zone_counts": {},
            "dominant_mlp_zone_counts": {},
            "generated_token_count_mean": 0.0,
        }
    return {
        "case_count": len(rows),
        "mean_hidden_prompt_share": average([safe_float(row.get("hidden_prompt_share")) for row in rows]),
        "mean_hidden_generated_share": average([safe_float(row.get("hidden_generated_share")) for row in rows]),
        "mean_mlp_prompt_share": average([safe_float(row.get("mlp_prompt_share")) for row in rows]),
        "mean_mlp_generated_share": average([safe_float(row.get("mlp_generated_share")) for row in rows]),
        "dominant_hidden_zone_counts": count_values(str(row.get("dominant_hidden_zone", "none")) for row in rows),
        "dominant_mlp_zone_counts": count_values(str(row.get("dominant_mlp_zone", "none")) for row in rows),
        "generated_token_count_mean": average([safe_float(row.get("generated_token_count")) for row in rows]),
    }


def join_component_rows(
    component_rows: Sequence[Dict[str, object]],
    natural_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    natural_map = {trajectory_key(row): row for row in natural_rows}
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
        natural = natural_map.get(key)
        if natural is None:
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
                "union_synergy_joint": safe_float(row.get("union_synergy_joint")),
                "union_joint_adv": safe_float(row.get("union_joint_adv")),
                "hidden_prompt_sum": safe_float(natural.get("hidden_prompt_sum")),
                "hidden_generated_sum": safe_float(natural.get("hidden_generated_sum")),
                "mlp_prompt_sum": safe_float(natural.get("mlp_prompt_sum")),
                "mlp_generated_sum": safe_float(natural.get("mlp_generated_sum")),
                "hidden_prompt_share": safe_float(natural.get("hidden_prompt_share")),
                "hidden_generated_share": safe_float(natural.get("hidden_generated_share")),
                "mlp_prompt_share": safe_float(natural.get("mlp_prompt_share")),
                "mlp_generated_share": safe_float(natural.get("mlp_generated_share")),
                "dominant_hidden_zone": str(natural.get("dominant_hidden_zone", "none")),
                "dominant_mlp_zone": str(natural.get("dominant_mlp_zone", "none")),
                "generated_token_count": int(natural.get("generated_token_count", 0)),
            }
        )
    return joined


def summarize_component_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    positive = [row for row in rows if safe_float(row.get("weight")) > 0.0]
    weights = [safe_float(row.get("weight")) for row in positive]
    weight_sum = sum(weights)

    def weighted_average(key: str) -> float:
        if not positive or weight_sum == 0.0:
            return 0.0
        return float(sum(safe_float(row.get(key)) * safe_float(row.get("weight")) for row in positive) / weight_sum)

    return {
        "case_count": len(positive),
        "weight_sum": weight_sum,
        "mean_union_synergy_joint": average([safe_float(row.get("union_synergy_joint")) for row in positive]),
        "mean_union_joint_adv": average([safe_float(row.get("union_joint_adv")) for row in positive]),
        "weighted_hidden_prompt_share": weighted_average("hidden_prompt_share"),
        "weighted_hidden_generated_share": weighted_average("hidden_generated_share"),
        "weighted_mlp_prompt_share": weighted_average("mlp_prompt_share"),
        "weighted_mlp_generated_share": weighted_average("mlp_generated_share"),
        "dominant_hidden_zone_counts": count_values(str(row.get("dominant_hidden_zone", "none")) for row in positive),
        "dominant_mlp_zone_counts": count_values(str(row.get("dominant_mlp_zone", "none")) for row in positive),
        "corr_hidden_prompt_to_synergy": pearson_corr(
            [safe_float(row.get("hidden_prompt_sum")) for row in positive],
            [safe_float(row.get("union_synergy_joint")) for row in positive],
        ),
        "corr_hidden_generated_to_synergy": pearson_corr(
            [safe_float(row.get("hidden_generated_sum")) for row in positive],
            [safe_float(row.get("union_synergy_joint")) for row in positive],
        ),
        "corr_mlp_prompt_to_synergy": pearson_corr(
            [safe_float(row.get("mlp_prompt_sum")) for row in positive],
            [safe_float(row.get("union_synergy_joint")) for row in positive],
        ),
        "corr_mlp_generated_to_synergy": pearson_corr(
            [safe_float(row.get("mlp_generated_sum")) for row in positive],
            [safe_float(row.get("union_synergy_joint")) for row in positive],
        ),
    }


def build_summary(
    natural_rows: Sequence[Dict[str, object]],
    component_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    enriched_rows = [attach_zone_metrics(row) for row in natural_rows]
    joined_rows = join_component_rows(component_rows, enriched_rows)
    axes = sorted({str(row["axis"]) for row in enriched_rows})
    labels = sorted({str(row["component_label"]) for row in joined_rows})
    return {
        "record_type": "stage56_natural_generation_decoupling_summary",
        "case_count": len(enriched_rows),
        "component_joined_row_count": len(joined_rows),
        "per_axis": {
            axis: summarize_zone_rows([row for row in enriched_rows if str(row["axis"]) == axis])
            for axis in axes
        },
        "per_component": {
            label: summarize_component_rows([row for row in joined_rows if str(row["component_label"]) == label])
            for label in labels
        },
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 自然生成解耦报告",
        "",
        f"- case_count: {int(summary.get('case_count', 0))}",
        f"- component_joined_row_count: {int(summary.get('component_joined_row_count', 0))}",
        "",
        "## 按轴汇总",
    ]
    for axis, row in dict(summary.get("per_axis", {})).items():
        lines.append(
            f"- {axis}: hidden_prompt_share={safe_float(row.get('mean_hidden_prompt_share')):.4f}, "
            f"hidden_generated_share={safe_float(row.get('mean_hidden_generated_share')):.4f}, "
            f"mlp_prompt_share={safe_float(row.get('mean_mlp_prompt_share')):.4f}, "
            f"mlp_generated_share={safe_float(row.get('mean_mlp_generated_share')):.4f}, "
            f"hidden_zone_counts={row.get('dominant_hidden_zone_counts', {})}"
        )
    lines.extend(["", "## 按组件汇总"])
    for label, row in dict(summary.get("per_component", {})).items():
        lines.append(
            f"- {label}: hidden_prompt_share={safe_float(row.get('weighted_hidden_prompt_share')):.4f}, "
            f"hidden_generated_share={safe_float(row.get('weighted_hidden_generated_share')):.4f}, "
            f"corr_hidden_prompt_to_synergy={safe_float(row.get('corr_hidden_prompt_to_synergy')):.4f}, "
            f"corr_hidden_generated_to_synergy={safe_float(row.get('corr_hidden_generated_to_synergy')):.4f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Decouple prompt skeleton windows from generated windows in natural generation traces")
    ap.add_argument(
        "--natural-cases-jsonl",
        default=str(
            ROOT
            / "tests"
            / "codex_temp"
            / "stage56_natural_generation_window_probe_all3_12cat_allpairs_20260319_0648"
            / "cases.jsonl"
        ),
    )
    ap.add_argument(
        "--component-joined-json",
        default=str(
            ROOT
            / "tests"
            / "codex_temp"
            / "stage56_field_internal_subfield_map_all3_12cat_allpairs_20260319_0122"
            / "joined_rows.json"
        ),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_natural_generation_decoupling_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    natural_rows = read_jsonl(Path(args.natural_cases_jsonl))
    component_rows = list(read_json(Path(args.component_joined_json)).get("rows", []))
    summary = build_summary(natural_rows, component_rows)
    enriched_rows = [attach_zone_metrics(row) for row in natural_rows]
    joined_rows = join_component_rows(component_rows, enriched_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "enriched_rows.json", {"rows": enriched_rows})
    write_json(out_dir / "joined_rows.json", {"rows": joined_rows})
    write_report(out_dir / "REPORT.md", summary)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "case_count": summary["case_count"],
                "component_joined_row_count": summary["component_joined_row_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
