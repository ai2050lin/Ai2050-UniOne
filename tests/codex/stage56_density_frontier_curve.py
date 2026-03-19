from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_model_label(probe: Dict[str, object], path: Path) -> str:
    model_id = str(probe.get("model_id", "")).lower()
    source = f"{model_id} {path.as_posix().lower()}"
    runtime = probe.get("runtime_config") or {}
    total_neurons = int(runtime.get("total_neurons", 0))
    if "deepseek" in source or total_neurons == 530432:
        return "DeepSeek-7B"
    if "qwen" in source or total_neurons == 350208:
        return "Qwen3-4B"
    if "glm" in source or total_neurons == 547840:
        return "GLM-4-9B"
    if model_id:
        return Path(str(probe.get("model_id"))).name
    return path.parent.name


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def auc_gap(curve: Sequence[Dict[str, object]]) -> float:
    if len(curve) < 2:
        return 0.0
    xs = [safe_float(row["mass_ratio"]) for row in curve]
    ys = [safe_float(row["mass_ratio"]) - safe_float(row["compaction_ratio"]) for row in curve]
    area = 0.0
    for i in range(1, len(xs)):
        width = xs[i] - xs[i - 1]
        area += width * (ys[i] + ys[i - 1]) * 0.5
    return float(area)


def knee_mass_ratio(curve: Sequence[Dict[str, object]], max_mass_ratio: float = 0.50) -> float:
    filtered = [row for row in curve if safe_float(row["mass_ratio"]) <= max_mass_ratio]
    if len(filtered) < 3:
        return 0.0
    x0 = safe_float(filtered[0]["mass_ratio"])
    y0 = safe_float(filtered[0]["compaction_ratio"])
    x1 = safe_float(filtered[-1]["mass_ratio"])
    y1 = safe_float(filtered[-1]["compaction_ratio"])
    dx = x1 - x0
    dy = y1 - y0
    norm = (dx * dx + dy * dy) ** 0.5
    if norm <= 1e-12:
        return x0
    best_ratio = x0
    best_dist = -1.0
    for row in filtered[1:-1]:
        x = safe_float(row["mass_ratio"])
        y = safe_float(row["compaction_ratio"])
        dist = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / norm
        if dist > best_dist:
            best_dist = dist
            best_ratio = x
    return float(best_ratio)


def first_mass_ratio_for_coverage(curve: Sequence[Dict[str, object]], threshold: float) -> float:
    for row in curve:
        if safe_float(row.get("layer_coverage_ratio")) >= threshold:
            return safe_float(row["mass_ratio"])
    return 1.0


def first_mass_ratio_for_jaccard(curve: Sequence[Dict[str, object]], threshold: float) -> float:
    for row in curve:
        if safe_float(row.get("jaccard")) >= threshold:
            return safe_float(row["mass_ratio"])
    return 1.0


def min_jaccard_point(curve: Sequence[Dict[str, object]]) -> Dict[str, float]:
    if not curve:
        return {"mass_ratio": 0.0, "jaccard": 0.0}
    best = min(curve, key=lambda row: safe_float(row.get("jaccard")))
    return {"mass_ratio": safe_float(best["mass_ratio"]), "jaccard": safe_float(best["jaccard"])}


def average_curve(curves: Sequence[Sequence[Dict[str, object]]], value_key: str) -> List[Dict[str, float]]:
    if not curves:
        return []
    ratios = [safe_float(row["mass_ratio"]) for row in curves[0]]
    out: List[Dict[str, float]] = []
    for idx, ratio in enumerate(ratios):
        values = [safe_float(curve[idx][value_key]) for curve in curves if idx < len(curve)]
        out.append({"mass_ratio": ratio, value_key: mean(values)})
    return out


def summarize_dimension(dim_obj: Dict[str, object]) -> Dict[str, object]:
    curve = list(dim_obj.get("frontier_curve") or [])
    return {
        "selected_neuron_count": int(dim_obj.get("selected_neuron_count", 0)),
        "mass10_compaction_ratio": safe_float(next((row["compaction_ratio"] for row in curve if safe_float(row["mass_ratio"]) == 0.10), 0.0)),
        "mass25_compaction_ratio": safe_float(next((row["compaction_ratio"] for row in curve if safe_float(row["mass_ratio"]) == 0.25), 0.0)),
        "mass50_compaction_ratio": safe_float(next((row["compaction_ratio"] for row in curve if safe_float(row["mass_ratio"]) == 0.50), 0.0)),
        "frontier_sharpness_auc": auc_gap(curve),
        "knee_mass_ratio": knee_mass_ratio(curve),
        "full_layer_coverage_mass_ratio": first_mass_ratio_for_coverage(curve, 1.0),
        "pair_delta_cosine_mean": safe_float(dim_obj.get("pair_delta_cosine_mean", 0.0)),
        "specific_selected_ratio": safe_float(dim_obj.get("specific_selected_ratio", 0.0)),
        "frontier_curve": curve,
    }


def summarize_model(probe: Dict[str, object], path: Path) -> Dict[str, object]:
    dims = {name: summarize_dimension(obj) for name, obj in (probe.get("dimensions") or {}).items()}
    cross = probe.get("cross_dimension") or {}
    cross_curves = [list(obj.get("frontier_curve_jaccard") or []) for obj in cross.values()]
    avg_cross_curve = average_curve(cross_curves, "jaccard")
    return {
        "label": infer_model_label(probe, path),
        "probe_json": path.as_posix(),
        "total_neurons": int((probe.get("runtime_config") or {}).get("total_neurons", 0)),
        "per_dimension": dims,
        "average_cross_frontier_jaccard_curve": avg_cross_curve,
        "cross_merge_mass_ratio_50": first_mass_ratio_for_jaccard(avg_cross_curve, 0.50),
        "cross_merge_mass_ratio_25": first_mass_ratio_for_jaccard(avg_cross_curve, 0.25),
        "cross_min_jaccard_point": min_jaccard_point(avg_cross_curve),
    }


def derive_shared_claims(model_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    avg_cross_curves = [row["average_cross_frontier_jaccard_curve"] for row in model_rows]
    shared_curve = average_curve(avg_cross_curves, "jaccard")
    stable_axis = []
    broad_axis = []
    knee_values = []
    for row in model_rows:
        dims = row["per_dimension"]
        stable_axis.append(max(dims.items(), key=lambda item: item[1]["pair_delta_cosine_mean"])[0])
        broad_axis.append(max(dims.items(), key=lambda item: item[1]["specific_selected_ratio"])[0])
        knee_values.append(mean([safe_float(dim["knee_mass_ratio"]) for dim in dims.values()]))
    return {
        "shared_stable_axis_dimension": stable_axis[0] if stable_axis and all(x == stable_axis[0] for x in stable_axis) else "mixed",
        "shared_broad_reconfiguration_dimension": broad_axis[0] if broad_axis and all(x == broad_axis[0] for x in broad_axis) else "mixed",
        "mean_cross_merge_mass_ratio_25": mean([safe_float(row["cross_merge_mass_ratio_25"]) for row in model_rows]),
        "mean_cross_merge_mass_ratio_50": mean([safe_float(row["cross_merge_mass_ratio_50"]) for row in model_rows]),
        "mean_knee_mass_ratio": mean(knee_values),
        "shared_average_cross_frontier_jaccard_curve": shared_curve,
        "shared_min_jaccard_point": min_jaccard_point(shared_curve),
    }


def build_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# 三模型连续密度前沿分析",
        "",
        "## 共享结果",
        f"- 共享最稳定轴: {report['shared_claims']['shared_stable_axis_dimension']}",
        f"- 共享最广重排维度: {report['shared_claims']['shared_broad_reconfiguration_dimension']}",
        f"- 平均 25% 汇合点: {report['shared_claims']['mean_cross_merge_mass_ratio_25']:.4f}",
        f"- 平均 50% 汇合点: {report['shared_claims']['mean_cross_merge_mass_ratio_50']:.4f}",
        f"- 平均前沿拐点: {report['shared_claims']['mean_knee_mass_ratio']:.4f}",
        "",
        "## 模型摘要",
    ]
    for row in report["models"]:
        lines.append(
            f"- {row['label']}: total={row['total_neurons']}, "
            f"cross_merge25={row['cross_merge_mass_ratio_25']:.4f}, "
            f"cross_merge50={row['cross_merge_mass_ratio_50']:.4f}, "
            f"cross_min_jaccard={row['cross_min_jaccard_point']['jaccard']:.4f}@{row['cross_min_jaccard_point']['mass_ratio']:.2f}"
        )
        for dim_name, dim in row["per_dimension"].items():
            lines.append(
                f"  {dim_name}: auc={dim['frontier_sharpness_auc']:.4f}, "
                f"knee={dim['knee_mass_ratio']:.4f}, "
                f"full_coverage={dim['full_layer_coverage_mass_ratio']:.4f}, "
                f"pair_cos={dim['pair_delta_cosine_mean']:.4f}, "
                f"specific_ratio={dim['specific_selected_ratio']:.4f}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-json", action="append", required=True)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"tests/codex_temp/stage56_density_frontier_curve_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model_rows = []
    for raw in args.probe_json:
        path = Path(raw)
        probe = load_json(path)
        model_rows.append(summarize_model(probe, path))
    model_rows.sort(key=lambda row: str(row["label"]))

    report = {
        "models": model_rows,
        "shared_claims": derive_shared_claims(model_rows),
    }

    json_path = out_dir / "summary.json"
    md_path = out_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"summary_json": json_path.as_posix(), "summary_md": md_path.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
