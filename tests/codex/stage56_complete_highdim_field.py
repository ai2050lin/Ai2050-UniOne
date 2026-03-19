from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from stage56_natural_generation_decoupling import attach_zone_metrics, read_jsonl as read_natural_jsonl

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs or not ys or len(xs) != len(ys):
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = safe_float(x) - mx
        dy = safe_float(y) - my
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    if den_x <= 0.0 or den_y <= 0.0:
        return 0.0
    return num / ((den_x ** 0.5) * (den_y ** 0.5))


def natural_key(row: Dict[str, object]) -> Tuple[str, str, str, str, str]:
    return (
        str(row["model_id"]),
        str(row["category"]),
        str(row["prototype_term"]),
        str(row["instance_term"]),
        str(row["axis"]),
    )


def join_rows(
    component_rows: Sequence[Dict[str, object]],
    natural_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    natural_map = {natural_key(row): row for row in natural_rows}
    out: List[Dict[str, object]] = []
    for row in component_rows:
        key = natural_key(row)
        natural = natural_map.get(key)
        if natural is None:
            continue
        hidden_generated_share = safe_float(natural.get("hidden_generated_share"))
        mlp_generated_share = safe_float(natural.get("mlp_generated_share"))
        hidden_prompt_share = safe_float(natural.get("hidden_prompt_share"))
        mlp_prompt_share = safe_float(natural.get("mlp_prompt_share"))
        layer_window_hidden_energy = safe_float(row.get("layer_window_hidden_energy"))
        layer_window_mlp_energy = safe_float(row.get("layer_window_mlp_energy"))
        layer_window_cross_energy = safe_float(row.get("layer_window_cross_energy"))

        generated_energy = (
            layer_window_hidden_energy * hidden_generated_share
            + layer_window_mlp_energy * mlp_generated_share
            + layer_window_cross_energy * mean([hidden_generated_share, mlp_generated_share])
        )
        prompt_energy = (
            layer_window_hidden_energy * hidden_prompt_share
            + layer_window_mlp_energy * mlp_prompt_share
            + layer_window_cross_energy * mean([hidden_prompt_share, mlp_prompt_share])
        )
        out.append(
            {
                **row,
                "hidden_generated_share": hidden_generated_share,
                "mlp_generated_share": mlp_generated_share,
                "hidden_prompt_share": hidden_prompt_share,
                "mlp_prompt_share": mlp_prompt_share,
                "generated_prompt_gap_hidden": safe_float(natural.get("hidden_generated_sum"))
                - safe_float(natural.get("hidden_prompt_sum")),
                "generated_prompt_gap_mlp": safe_float(natural.get("mlp_generated_sum"))
                - safe_float(natural.get("mlp_prompt_sum")),
                "complete_generated_energy": generated_energy,
                "complete_prompt_energy": prompt_energy,
                "complete_energy_gap": generated_energy - prompt_energy,
                "generated_dominance_score": mean([hidden_generated_share, mlp_generated_share]),
            }
        )
    return out


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    feature_names = [
        "weight",
        "preferred_density",
        "hidden_layer_center",
        "mlp_layer_center",
        "hidden_window_center",
        "mlp_window_center",
        "layer_window_hidden_energy",
        "layer_window_mlp_energy",
        "layer_window_cross_energy",
        "hidden_generated_share",
        "mlp_generated_share",
        "complete_generated_energy",
        "complete_prompt_energy",
        "complete_energy_gap",
        "generated_dominance_score",
    ]
    targets = ("union_joint_adv", "union_synergy_joint", "strict_positive_synergy")
    findings: List[Dict[str, object]] = []
    per_component: Dict[str, object] = {}
    for component_label in sorted({str(row["component_label"]) for row in rows}):
        subset = [row for row in rows if str(row["component_label"]) == component_label]
        positives = [row for row in subset if bool(row["strict_positive_synergy"])]
        negatives = [row for row in subset if not bool(row["strict_positive_synergy"])]
        block: Dict[str, object] = {"case_count": len(subset), "feature_stats": {}}
        for feature_name in feature_names:
            xs = [safe_float(row.get(feature_name)) for row in subset]
            pos_xs = [safe_float(row.get(feature_name)) for row in positives]
            neg_xs = [safe_float(row.get(feature_name)) for row in negatives]
            feature_stat = {
                "mean_value": mean(xs),
                "positive_pair_mean": mean(pos_xs),
                "non_positive_pair_mean": mean(neg_xs),
                "positive_pair_gap": mean(pos_xs) - mean(neg_xs),
                "targets": {},
            }
            for target_name in targets:
                ys = (
                    [1.0 if bool(row["strict_positive_synergy"]) else 0.0 for row in subset]
                    if target_name == "strict_positive_synergy"
                    else [safe_float(row.get(target_name)) for row in subset]
                )
                corr = pearson(xs, ys)
                feature_stat["targets"][target_name] = {"pearson_corr": corr}
                findings.append(
                    {
                        "component_label": component_label,
                        "feature_name": feature_name,
                        "target_name": target_name,
                        "corr": corr,
                        "positive_pair_gap": feature_stat["positive_pair_gap"],
                    }
                )
            block["feature_stats"][feature_name] = feature_stat
        per_component[component_label] = block
    findings.sort(key=lambda row: abs(safe_float(row["corr"])), reverse=True)
    return {
        "record_type": "stage56_complete_highdim_field_summary",
        "joined_row_count": len(rows),
        "component_labels": sorted({str(row["component_label"]) for row in rows}),
        "per_component": per_component,
        "top_abs_correlations": findings[:30],
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 \u5b8c\u6574\u9ad8\u7ef4\u573a\u6458\u8981",
        "",
        f"- joined_row_count: {summary['joined_row_count']}",
        f"- component_labels: {', '.join(summary['component_labels'])}",
        "",
        "## Top Correlations",
    ]
    for row in summary["top_abs_correlations"]:
        lines.append(
            f"- {row['component_label']} / {row['feature_name']} -> {row['target_name']}: "
            f"corr={safe_float(row['corr']):+.4f}, positive_gap={safe_float(row['positive_pair_gap']):+.4f}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Join component-specific highdim field with natural generation source splits")
    ap.add_argument(
        "--component-joined-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_component_specific_highdim_field_20260319_1628" / "joined_rows.json"),
    )
    ap.add_argument(
        "--natural-cases-jsonl",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_natural_generation_window_probe_all3_12cat_allpairs_20260319_0648" / "cases.jsonl"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_complete_highdim_field_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    component_rows = list(read_json(Path(args.component_joined_json)).get("rows", []))
    natural_rows = [attach_zone_metrics(row) for row in read_natural_jsonl(Path(args.natural_cases_jsonl))]
    joined_rows = join_rows(component_rows, natural_rows)
    summary = build_summary(joined_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "joined_rows.json").write_text(
        json.dumps({"rows": joined_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "joined_row_count": len(joined_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
