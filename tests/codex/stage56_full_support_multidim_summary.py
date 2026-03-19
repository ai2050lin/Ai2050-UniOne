from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def summarize_dimension(dim_obj: Dict[str, object], total_neurons: int) -> Dict[str, object]:
    selected = int(dim_obj.get("selected_neuron_count", 0))
    mass10 = int(dim_obj.get("mass10_neuron_count", 0))
    mass25 = int(dim_obj.get("mass25_neuron_count", 0))
    mass50 = int(dim_obj.get("mass50_neuron_count", 0))
    mass80 = int(dim_obj.get("mass80_neuron_count", 0))
    mass95 = int(dim_obj.get("mass95_neuron_count", 0))
    return {
        "selection_mode": dim_obj.get("selection_mode", "unknown"),
        "selected_neuron_count": selected,
        "selected_neuron_ratio": ratio(selected, total_neurons),
        "mass10_neuron_count": mass10,
        "mass25_neuron_count": mass25,
        "mass50_neuron_count": mass50,
        "mass80_neuron_count": mass80,
        "mass95_neuron_count": mass95,
        "mass10_compaction_ratio": ratio(mass10, selected),
        "mass25_compaction_ratio": ratio(mass25, selected),
        "mass50_compaction_ratio": ratio(mass50, selected),
        "mass80_compaction_ratio": ratio(mass80, selected),
        "mass95_compaction_ratio": ratio(mass95, selected),
        "mass10_layer_coverage_ratio": float(
            (dim_obj.get("mass10_layer_coverage") or {}).get("covered_layer_ratio", 0.0)
        ),
        "mass25_layer_coverage_ratio": float(
            (dim_obj.get("mass25_layer_coverage") or {}).get("covered_layer_ratio", 0.0)
        ),
        "mass50_layer_coverage_ratio": float(
            (dim_obj.get("mass50_layer_coverage") or {}).get("covered_layer_ratio", 0.0)
        ),
        "mass80_layer_coverage_ratio": float(
            (dim_obj.get("mass80_layer_coverage") or {}).get("covered_layer_ratio", 0.0)
        ),
        "mass95_layer_coverage_ratio": float(
            (dim_obj.get("mass95_layer_coverage") or {}).get("covered_layer_ratio", 0.0)
        ),
        "pair_delta_cosine_mean": float(dim_obj.get("pair_delta_cosine_mean", 0.0)),
        "specific_selected_count": int(dim_obj.get("specific_selected_count", 0)),
        "specific_selected_ratio": float(dim_obj.get("specific_selected_ratio", 0.0)),
    }


def derive_system_claims(per_dim: Dict[str, Dict[str, object]], cross: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    mean_effective = sum(float(v["selected_neuron_ratio"]) for v in per_dim.values()) / max(1, len(per_dim))
    mean_mass10_compaction = sum(float(v["mass10_compaction_ratio"]) for v in per_dim.values()) / max(1, len(per_dim))
    mean_mass25_compaction = sum(float(v["mass25_compaction_ratio"]) for v in per_dim.values()) / max(1, len(per_dim))
    mean_mass50_compaction = sum(float(v["mass50_compaction_ratio"]) for v in per_dim.values()) / max(1, len(per_dim))
    mean_layer_corr = sum(float(v.get("layer_profile_corr", 0.0)) for v in cross.values()) / max(1, len(cross))
    mean_effective_jaccard = sum(float(v.get("effective_support_jaccard", 0.0)) for v in cross.values()) / max(1, len(cross))
    mean_mass10_jaccard = sum(float(v.get("mass10_jaccard", 0.0)) for v in cross.values()) / max(1, len(cross))
    mean_mass25_jaccard = sum(float(v.get("mass25_jaccard", 0.0)) for v in cross.values()) / max(1, len(cross))
    mean_mass50_jaccard = sum(float(v.get("mass50_jaccard", 0.0)) for v in cross.values()) / max(1, len(cross))
    mean_mass80_jaccard = sum(float(v.get("mass80_jaccard", 0.0)) for v in cross.values()) / max(1, len(cross))

    if mean_effective > 0.85 and mean_mass25_compaction < 0.12:
        support_shape = "广支撑-窄核心"
    elif mean_effective > 0.85 and mean_mass10_compaction < 0.08:
        support_shape = "广支撑-尖峰核心"
    elif mean_effective > 0.50:
        support_shape = "中广支撑-中等核心"
    else:
        support_shape = "窄支撑-窄核心"

    if mean_layer_corr > 0.5 and mean_mass25_jaccard < mean_effective_jaccard:
        law = "维度共享广泛层级底座，但高质量核心仍分化。"
    else:
        law = "维度之间的共享底座与核心分化尚不充分。"

    return {
        "support_shape": support_shape,
        "mean_effective_support_jaccard": mean_effective_jaccard,
        "mean_mass10_jaccard": mean_mass10_jaccard,
        "mean_mass25_jaccard": mean_mass25_jaccard,
        "mean_mass50_jaccard": mean_mass50_jaccard,
        "mean_mass80_jaccard": mean_mass80_jaccard,
        "mean_mass10_compaction": mean_mass10_compaction,
        "mean_mass25_compaction": mean_mass25_compaction,
        "mean_mass50_compaction": mean_mass50_compaction,
        "mean_layer_profile_corr": mean_layer_corr,
        "system_law": law,
    }


def build_markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# 全支撑多维编码系统摘要",
        "",
        "## 总体判断",
        f"- 主结构: {summary['system_claims']['support_shape']}",
        f"- 系统结论: {summary['system_claims']['system_law']}",
        "",
        "## 各维度",
    ]
    for dim, obj in summary["per_dimension"].items():
        lines.append(
            f"- {dim}: 有效支撑占比={obj['selected_neuron_ratio']:.4f}, "
            f"10%质量压缩比={obj['mass10_compaction_ratio']:.4f}, "
            f"25%质量压缩比={obj['mass25_compaction_ratio']:.4f}, "
            f"50%质量压缩比={obj['mass50_compaction_ratio']:.4f}, "
            f"50%质量层覆盖={obj['mass50_layer_coverage_ratio']:.4f}, "
            f"特异支撑占比={obj['specific_selected_ratio']:.4f}"
        )
    lines.extend(["", "## 跨维关系"])
    for pair, obj in summary["cross_dimension"].items():
        lines.append(
            f"- {pair}: broad_jaccard={obj['effective_support_jaccard']:.4f}, "
            f"mass10_jaccard={obj['mass10_jaccard']:.4f}, "
            f"mass25_jaccard={obj['mass25_jaccard']:.4f}, "
            f"mass50_jaccard={obj['mass50_jaccard']:.4f}, "
            f"mass80_jaccard={obj['mass80_jaccard']:.4f}, "
            f"layer_corr={obj['layer_profile_corr']:.4f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-json", required=True)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    probe = load_json(Path(args.probe_json))
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tests/codex_temp/stage56_full_support_multidim_summary_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    total_neurons = int((probe.get("runtime_config") or {}).get("total_neurons", 0))
    per_dim = {
        dim: summarize_dimension(obj, total_neurons)
        for dim, obj in (probe.get("dimensions") or {}).items()
    }
    cross = {
        key: {
            "top_neuron_jaccard": float(val.get("top_neuron_jaccard", 0.0)),
            "effective_support_jaccard": float(val.get("effective_support_jaccard", 0.0)),
            "mass10_jaccard": float(val.get("mass10_jaccard", 0.0)),
            "mass25_jaccard": float(val.get("mass25_jaccard", 0.0)),
            "mass50_jaccard": float(val.get("mass50_jaccard", 0.0)),
            "mass80_jaccard": float(val.get("mass80_jaccard", 0.0)),
            "layer_profile_corr": float(val.get("layer_profile_corr", 0.0)),
        }
        for key, val in (probe.get("cross_dimension") or {}).items()
    }
    system_claims = derive_system_claims(per_dim, cross)

    summary = {
        "probe_json": str(Path(args.probe_json).as_posix()),
        "runtime_config": probe.get("runtime_config", {}),
        "per_dimension": per_dim,
        "cross_dimension": cross,
        "system_claims": system_claims,
    }

    json_path = out_dir / "summary.json"
    md_path = out_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"summary_json": json_path.as_posix(), "summary_md": md_path.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
