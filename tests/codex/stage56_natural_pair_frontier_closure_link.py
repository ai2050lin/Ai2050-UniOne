from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from stage56_density_frontier_closure_link import (
    AXES,
    mean,
    normalize_model_label,
    pearson,
    read_jsonl,
    safe_float,
)
from stage56_density_frontier_curve import infer_model_label, load_json


def load_pairs_manifest(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for axis, rows in (payload.get("pairs") or {}).items():
        for row in rows or []:
            pair_id = str(row.get("id", ""))
            out[(str(axis), pair_id)] = {
                "term": str(row.get("term", "")),
                "category": str(row.get("category", "")),
            }
    return out


def rank_desc(values: Sequence[float], current: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    idx = ordered.index(current)
    if len(ordered) == 1:
        return 1.0
    return float(1.0 - (idx / (len(ordered) - 1)))


def zscore(values: Sequence[float], current: float) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    var = mean([(x - mu) ** 2 for x in values])
    std = var ** 0.5
    if std <= 1e-12:
        return 0.0
    return float((current - mu) / std)


def build_term_metrics(
    probe_json_paths: Sequence[Path],
    manifest: Dict[Tuple[str, str], Dict[str, str]],
) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
    raw_rows: List[Dict[str, object]] = []
    for probe_path in probe_json_paths:
        probe = load_json(probe_path)
        model_label = infer_model_label(probe, probe_path)
        for axis, dim in (probe.get("dimensions") or {}).items():
            for pair_row in dim.get("pairs") or []:
                pair_id = str(pair_row.get("id", ""))
                info = manifest.get((str(axis), pair_id))
                if not info:
                    continue
                raw_rows.append(
                    {
                        "model_label": model_label,
                        "axis": str(axis),
                        "category": str(info["category"]),
                        "term": str(info["term"]),
                        "pair_id": pair_id,
                        "delta_l2": safe_float(pair_row.get("delta_l2")),
                        "delta_mean_abs": safe_float(pair_row.get("delta_mean_abs")),
                    }
                )

    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["model_label"]), str(row["axis"]), str(row["category"]))].append(row)

    out: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    for group_key, rows in grouped.items():
        l2_values = [safe_float(row["delta_l2"]) for row in rows]
        abs_values = [safe_float(row["delta_mean_abs"]) for row in rows]
        for row in rows:
            key = (group_key[0], group_key[1], group_key[2], str(row["term"]))
            l2_value = safe_float(row["delta_l2"])
            abs_value = safe_float(row["delta_mean_abs"])
            out[key] = {
                "delta_l2": l2_value,
                "delta_mean_abs": abs_value,
                "delta_l2_topness": rank_desc(l2_values, l2_value),
                "delta_mean_abs_topness": rank_desc(abs_values, abs_value),
                "delta_l2_zscore": zscore(l2_values, l2_value),
                "delta_mean_abs_zscore": zscore(abs_values, abs_value),
                "category_term_count": float(len(rows)),
            }
    return out


def join_pair_rows(
    joined_rows_path: Path,
    term_metrics: Dict[Tuple[str, str, str, str], Dict[str, float]],
) -> List[Dict[str, object]]:
    rows = read_jsonl(joined_rows_path)
    joined: List[Dict[str, object]] = []
    for row in rows:
        model_label = normalize_model_label(str(row.get("model_id", "")))
        category = str(row.get("category", ""))
        prototype_term = str(row.get("prototype_term", ""))
        instance_term = str(row.get("instance_term", ""))
        axis_block: Dict[str, object] = {}
        ok = True
        for axis in AXES:
            proto_key = (model_label, axis, category, prototype_term)
            inst_key = (model_label, axis, category, instance_term)
            if proto_key not in term_metrics or inst_key not in term_metrics:
                ok = False
                break
            proto = term_metrics[proto_key]
            inst = term_metrics[inst_key]
            axis_block[axis] = {
                "prototype_delta_l2": proto["delta_l2"],
                "instance_delta_l2": inst["delta_l2"],
                "pair_mean_delta_l2": mean([proto["delta_l2"], inst["delta_l2"]]),
                "pair_gap_delta_l2": abs(proto["delta_l2"] - inst["delta_l2"]),
                "prototype_delta_mean_abs": proto["delta_mean_abs"],
                "instance_delta_mean_abs": inst["delta_mean_abs"],
                "pair_mean_delta_mean_abs": mean([proto["delta_mean_abs"], inst["delta_mean_abs"]]),
                "pair_gap_delta_mean_abs": abs(proto["delta_mean_abs"] - inst["delta_mean_abs"]),
                "prototype_delta_l2_topness": proto["delta_l2_topness"],
                "instance_delta_l2_topness": inst["delta_l2_topness"],
                "pair_mean_delta_l2_topness": mean([proto["delta_l2_topness"], inst["delta_l2_topness"]]),
                "prototype_delta_mean_abs_topness": proto["delta_mean_abs_topness"],
                "instance_delta_mean_abs_topness": inst["delta_mean_abs_topness"],
                "pair_mean_delta_mean_abs_topness": mean([proto["delta_mean_abs_topness"], inst["delta_mean_abs_topness"]]),
                "prototype_delta_l2_zscore": proto["delta_l2_zscore"],
                "instance_delta_l2_zscore": inst["delta_l2_zscore"],
                "pair_mean_delta_l2_zscore": mean([proto["delta_l2_zscore"], inst["delta_l2_zscore"]]),
                "prototype_delta_mean_abs_zscore": proto["delta_mean_abs_zscore"],
                "instance_delta_mean_abs_zscore": inst["delta_mean_abs_zscore"],
                "pair_mean_delta_mean_abs_zscore": mean([proto["delta_mean_abs_zscore"], inst["delta_mean_abs_zscore"]]),
            }
        if not ok:
            continue
        joined.append(
            {
                "model_id": str(row.get("model_id", "")),
                "model_label": model_label,
                "category": category,
                "prototype_term": prototype_term,
                "instance_term": instance_term,
                "strict_positive_synergy": bool(row.get("strict_positive_synergy")),
                "union_joint_adv": safe_float(row.get("union_joint_adv")),
                "union_synergy_joint": safe_float(row.get("union_synergy_joint")),
                "axes": axis_block,
            }
        )
    return joined


def build_axis_feature_rows(joined_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in joined_rows:
        for axis in AXES:
            axis_values = dict(row["axes"][axis])
            out.append(
                {
                    "model_label": row["model_label"],
                    "category": row["category"],
                    "prototype_term": row["prototype_term"],
                    "instance_term": row["instance_term"],
                    "axis": axis,
                    "strict_positive_synergy": row["strict_positive_synergy"],
                    "union_joint_adv": row["union_joint_adv"],
                    "union_synergy_joint": row["union_synergy_joint"],
                    **axis_values,
                }
            )
    return out


def build_summary(joined_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    axis_rows = build_axis_feature_rows(joined_rows)
    feature_names = [
        "prototype_delta_l2",
        "instance_delta_l2",
        "pair_mean_delta_l2",
        "pair_gap_delta_l2",
        "prototype_delta_mean_abs",
        "instance_delta_mean_abs",
        "pair_mean_delta_mean_abs",
        "pair_gap_delta_mean_abs",
        "prototype_delta_l2_topness",
        "instance_delta_l2_topness",
        "pair_mean_delta_l2_topness",
        "prototype_delta_mean_abs_topness",
        "instance_delta_mean_abs_topness",
        "pair_mean_delta_mean_abs_topness",
        "prototype_delta_l2_zscore",
        "instance_delta_l2_zscore",
        "pair_mean_delta_l2_zscore",
        "prototype_delta_mean_abs_zscore",
        "instance_delta_mean_abs_zscore",
        "pair_mean_delta_mean_abs_zscore",
    ]
    target_names = [
        "union_joint_adv",
        "union_synergy_joint",
        "strict_positive_synergy",
    ]
    findings: List[Dict[str, object]] = []
    axis_feature_stats: Dict[str, Dict[str, object]] = {}
    for axis in AXES:
        rows = [row for row in axis_rows if str(row["axis"]) == axis]
        axis_feature_stats[axis] = {}
        positives = [row for row in rows if bool(row["strict_positive_synergy"])]
        negatives = [row for row in rows if not bool(row["strict_positive_synergy"])]
        for feature_name in feature_names:
            values = [safe_float(row.get(feature_name)) for row in rows]
            pos_values = [safe_float(row.get(feature_name)) for row in positives]
            neg_values = [safe_float(row.get(feature_name)) for row in negatives]
            feature_block = {
                "mean_value": mean(values),
                "positive_pair_mean": mean(pos_values),
                "non_positive_pair_mean": mean(neg_values),
                "positive_pair_gap": mean(pos_values) - mean(neg_values),
                "targets": {},
            }
            for target_name in target_names:
                if target_name == "strict_positive_synergy":
                    target_values = [1.0 if bool(row["strict_positive_synergy"]) else 0.0 for row in rows]
                else:
                    target_values = [safe_float(row.get(target_name)) for row in rows]
                corr = pearson(values, target_values)
                feature_block["targets"][target_name] = {
                    "pearson_corr": corr,
                }
                findings.append(
                    {
                        "axis": axis,
                        "feature_name": feature_name,
                        "target_name": target_name,
                        "corr": corr,
                        "positive_pair_gap": feature_block["positive_pair_gap"],
                    }
                )
            axis_feature_stats[axis][feature_name] = feature_block
    findings.sort(key=lambda row: abs(safe_float(row["corr"])), reverse=True)
    return {
        "record_type": "stage56_natural_pair_frontier_closure_link_summary",
        "joined_pair_count": len(joined_rows),
        "axis_row_count": len(axis_rows),
        "model_count": len({str(row["model_label"]) for row in axis_rows}),
        "category_count": len({str(row["category"]) for row in axis_rows}),
        "strict_positive_pair_ratio": mean([1.0 if bool(row["strict_positive_synergy"]) else 0.0 for row in joined_rows]),
        "axis_feature_stats": axis_feature_stats,
        "top_abs_correlations": findings[:18],
    }


def build_markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# 自然语料 Pair 级前沿到闭包联立",
        "",
        f"- joined_pair_count: {summary['joined_pair_count']}",
        f"- axis_row_count: {summary['axis_row_count']}",
        f"- model_count: {summary['model_count']}",
        f"- category_count: {summary['category_count']}",
        f"- strict_positive_pair_ratio: {summary['strict_positive_pair_ratio']:.6f}",
        "",
        "## Top Correlations",
    ]
    for row in summary["top_abs_correlations"]:
        lines.append(
            f"- {row['axis']} / {row['feature_name']} -> {row['target_name']}: "
            f"corr={row['corr']:+.4f}, positive_gap={row['positive_pair_gap']:+.4f}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Link natural corpus pair-level contrast metrics with exact stage6 pair closure")
    ap.add_argument("--pairs-json", required=True)
    ap.add_argument("--probe-json", action="append", required=True)
    ap.add_argument("--joined-rows", required=True)
    ap.add_argument("--output-dir", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"tests/codex_temp/stage56_natural_pair_frontier_closure_link_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_pairs_manifest(Path(args.pairs_json))
    term_metrics = build_term_metrics([Path(x) for x in args.probe_json], manifest)
    joined_rows = join_pair_rows(Path(args.joined_rows), term_metrics)
    summary = build_summary(joined_rows)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "joined_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in joined_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "SUMMARY.md").write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_dir": out_dir.as_posix(), "joined_pair_count": len(joined_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
