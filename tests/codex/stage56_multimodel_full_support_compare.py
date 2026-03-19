from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_model_label(summary: Dict[str, object], path: Path) -> str:
    runtime = summary.get("runtime_config") or {}
    model_id = str(runtime.get("model_id", "")).lower()
    source = f"{model_id} {path.as_posix().lower()}"
    if "deepseek" in source:
        return "DeepSeek-7B"
    if "qwen" in source:
        return "Qwen3-4B"
    if "glm" in source:
        return "GLM-4-9B"
    total_neurons = int(runtime.get("total_neurons", 0))
    if total_neurons == 530432:
        return "DeepSeek-7B"
    if total_neurons == 350208:
        return "Qwen3-4B"
    if total_neurons == 547840:
        return "GLM-4-9B"
    if model_id:
        return Path(str(runtime.get("model_id"))).name
    return path.parent.name


def pick_dimension_extreme(
    per_dimension: Dict[str, Dict[str, object]],
    metric: str,
    reverse: bool,
) -> Tuple[str, float]:
    items = [(dim, float(obj.get(metric, 0.0))) for dim, obj in per_dimension.items()]
    if not items:
        return "unknown", 0.0
    items.sort(key=lambda item: item[1], reverse=reverse)
    return items[0]


def summarize_model(summary: Dict[str, object], path: Path) -> Dict[str, object]:
    per_dimension = summary.get("per_dimension") or {}
    system_claims = summary.get("system_claims") or {}

    narrow_core_dim, narrow_core_value = pick_dimension_extreme(
        per_dimension, "mass10_compaction_ratio", reverse=False
    )
    stable_axis_dim, stable_axis_value = pick_dimension_extreme(
        per_dimension, "pair_delta_cosine_mean", reverse=True
    )
    broad_reconfig_dim, broad_reconfig_value = pick_dimension_extreme(
        per_dimension, "specific_selected_ratio", reverse=True
    )

    return {
        "label": infer_model_label(summary, path),
        "summary_path": path.as_posix(),
        "total_neurons": int((summary.get("runtime_config") or {}).get("total_neurons", 0)),
        "support_shape": str(system_claims.get("support_shape", "unknown")),
        "system_law": str(system_claims.get("system_law", "")),
        "mean_effective_support_jaccard": float(system_claims.get("mean_effective_support_jaccard", 0.0)),
        "mean_mass10_jaccard": float(system_claims.get("mean_mass10_jaccard", 0.0)),
        "mean_mass25_jaccard": float(system_claims.get("mean_mass25_jaccard", 0.0)),
        "mean_mass50_jaccard": float(system_claims.get("mean_mass50_jaccard", 0.0)),
        "mean_mass80_jaccard": float(system_claims.get("mean_mass80_jaccard", 0.0)),
        "mean_mass10_compaction": float(system_claims.get("mean_mass10_compaction", 0.0)),
        "mean_mass25_compaction": float(system_claims.get("mean_mass25_compaction", 0.0)),
        "mean_mass50_compaction": float(system_claims.get("mean_mass50_compaction", 0.0)),
        "mean_layer_profile_corr": float(system_claims.get("mean_layer_profile_corr", 0.0)),
        "narrow_core_dimension": narrow_core_dim,
        "narrow_core_value": narrow_core_value,
        "stable_axis_dimension": stable_axis_dim,
        "stable_axis_value": stable_axis_value,
        "broad_reconfiguration_dimension": broad_reconfig_dim,
        "broad_reconfiguration_value": broad_reconfig_value,
        "per_dimension": per_dimension,
    }


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def unanimous_dimension(rows: List[Dict[str, object]], key: str) -> str:
    values = [str(row.get(key, "")) for row in rows if row.get(key)]
    if not values:
        return "none"
    first = values[0]
    return first if all(v == first for v in values) else "mixed"


def derive_shared_claims(model_rows: List[Dict[str, object]]) -> Dict[str, object]:
    support_shapes = [str(row.get("support_shape", "")) for row in model_rows]
    if support_shapes and all(shape.startswith("广支撑") for shape in support_shapes):
        shared_support_shape = "广支撑家族"
    else:
        shared_support_shape = unanimous_dimension(model_rows, "support_shape")
    shared_narrow_core_dim = unanimous_dimension(model_rows, "narrow_core_dimension")
    shared_stable_axis_dim = unanimous_dimension(model_rows, "stable_axis_dimension")
    shared_broad_reconfig_dim = unanimous_dimension(model_rows, "broad_reconfiguration_dimension")

    mean_effective = mean([float(row["mean_effective_support_jaccard"]) for row in model_rows])
    mean_mass10 = mean([float(row["mean_mass10_jaccard"]) for row in model_rows])
    mean_mass25 = mean([float(row["mean_mass25_jaccard"]) for row in model_rows])
    mean_mass50 = mean([float(row["mean_mass50_jaccard"]) for row in model_rows])
    mean_mass10_compaction = mean([float(row["mean_mass10_compaction"]) for row in model_rows])
    mean_mass25_compaction = mean([float(row["mean_mass25_compaction"]) for row in model_rows])
    mean_layer_corr = mean([float(row["mean_layer_profile_corr"]) for row in model_rows])

    if mean_effective > 0.95 and mean_mass25_compaction < 0.16 and mean_mass10 < 0.20:
        shared_support_law = "三模型共享广支撑底座，但高质量核心明显分化。"
    elif mean_effective > 0.80 and mean_mass25_compaction < 0.20:
        shared_support_law = "三模型大体共享广支撑结构，但核心分化仍较明显。"
    else:
        shared_support_law = "三模型之间尚未形成稳定的广支撑-窄核心共识。"

    return {
        "model_count": len(model_rows),
        "shared_support_shape": shared_support_shape,
        "shared_narrow_core_dimension": shared_narrow_core_dim,
        "shared_stable_axis_dimension": shared_stable_axis_dim,
        "shared_broad_reconfiguration_dimension": shared_broad_reconfig_dim,
        "mean_effective_support_jaccard": mean_effective,
        "mean_mass10_jaccard": mean_mass10,
        "mean_mass25_jaccard": mean_mass25,
        "mean_mass50_jaccard": mean_mass50,
        "mean_mass10_compaction": mean_mass10_compaction,
        "mean_mass25_compaction": mean_mass25_compaction,
        "mean_layer_profile_corr": mean_layer_corr,
        "shared_support_law": shared_support_law,
    }


def derive_model_private_signals(model_rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    avg_mass10 = mean([float(row["mean_mass10_compaction"]) for row in model_rows])
    avg_layer_corr = mean([float(row["mean_layer_profile_corr"]) for row in model_rows])
    avg_stable_axis = mean([float(row["stable_axis_value"]) for row in model_rows])
    avg_reconfig = mean([float(row["broad_reconfiguration_value"]) for row in model_rows])
    for row in model_rows:
        label = str(row["label"])
        out[label] = {
            "mass10_compaction_delta_vs_mean": float(row["mean_mass10_compaction"]) - avg_mass10,
            "layer_profile_corr_delta_vs_mean": float(row["mean_layer_profile_corr"]) - avg_layer_corr,
            "stable_axis_strength_delta_vs_mean": float(row["stable_axis_value"]) - avg_stable_axis,
            "broad_reconfiguration_delta_vs_mean": float(row["broad_reconfiguration_value"]) - avg_reconfig,
        }
    return out


def build_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# 三模型全支撑比较",
        "",
        "## 共享规律",
        f"- 支撑主型: {report['shared_claims']['shared_support_shape']}",
        f"- 系统结论: {report['shared_claims']['shared_support_law']}",
        f"- 共同最窄核心维度: {report['shared_claims']['shared_narrow_core_dimension']}",
        f"- 共同最稳定轴: {report['shared_claims']['shared_stable_axis_dimension']}",
        f"- 共同最广重排维度: {report['shared_claims']['shared_broad_reconfiguration_dimension']}",
        "",
        "## 模型摘要",
    ]
    for row in report["models"]:
        lines.append(
            f"- {row['label']}: total={row['total_neurons']}, "
            f"mass10压缩={row['mean_mass10_compaction']:.4f}, "
            f"mass25压缩={row['mean_mass25_compaction']:.4f}, "
            f"层谱相关={row['mean_layer_profile_corr']:.4f}, "
            f"最窄核心={row['narrow_core_dimension']}, "
            f"最稳定轴={row['stable_axis_dimension']}, "
            f"最广重排={row['broad_reconfiguration_dimension']}"
        )
    lines.extend(["", "## 模型私有偏移"])
    for label, row in report["model_private_signals"].items():
        lines.append(
            f"- {label}: mass10偏移={row['mass10_compaction_delta_vs_mean']:+.4f}, "
            f"层谱偏移={row['layer_profile_corr_delta_vs_mean']:+.4f}, "
            f"稳定轴强度偏移={row['stable_axis_strength_delta_vs_mean']:+.4f}, "
            f"广重排偏移={row['broad_reconfiguration_delta_vs_mean']:+.4f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", action="append", required=True)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"tests/codex_temp/stage56_multimodel_full_support_compare_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model_rows = []
    for raw_path in args.summary_json:
        path = Path(raw_path)
        summary = load_json(path)
        model_rows.append(summarize_model(summary, path))
    model_rows.sort(key=lambda row: str(row["label"]))

    report = {
        "models": model_rows,
        "shared_claims": derive_shared_claims(model_rows),
        "model_private_signals": derive_model_private_signals(model_rows),
    }

    json_path = out_dir / "summary.json"
    md_path = out_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"summary_json": json_path.as_posix(), "summary_md": md_path.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
