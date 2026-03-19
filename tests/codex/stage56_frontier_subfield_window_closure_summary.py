from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def target_corr(feature_stat: Dict[str, object], target_name: str) -> float:
    return safe_float(dict(feature_stat.get("targets", {})).get(target_name, {}).get("pearson_corr"))


def best_axis_feature(
    per_axis: Dict[str, object],
    target_name: str,
    maximize: bool,
) -> Dict[str, object]:
    best = {
        "axis": "",
        "feature_name": "",
        "corr": float("-inf") if maximize else float("inf"),
        "positive_pair_gap": 0.0,
        "mean_value": 0.0,
    }
    for axis_name, axis_block_obj in per_axis.items():
        axis_block = dict(axis_block_obj)
        feature_stats = dict(axis_block.get("feature_stats", {}))
        for feature_name, feature_stat_obj in feature_stats.items():
            feature_stat = dict(feature_stat_obj)
            corr = target_corr(feature_stat, target_name)
            better = corr > safe_float(best["corr"]) if maximize else corr < safe_float(best["corr"])
            if better:
                best = {
                    "axis": axis_name,
                    "feature_name": feature_name,
                    "corr": corr,
                    "positive_pair_gap": safe_float(feature_stat.get("positive_pair_gap")),
                    "mean_value": safe_float(feature_stat.get("mean_value")),
                }
    return best


def build_density_frontier_section(
    pair_density_summary: Dict[str, object],
    law_summary: Dict[str, object],
) -> Dict[str, object]:
    per_axis = dict(pair_density_summary.get("per_axis", {}))
    laws = dict(law_summary.get("laws", {}))
    return {
        "broad_support_base": safe_float(laws.get("broad_support_base")),
        "long_separation_frontier": safe_float(laws.get("long_separation_frontier")),
        "strongest_positive_frontier": best_axis_feature(per_axis, "strict_positive_synergy", maximize=True),
        "strongest_negative_frontier": best_axis_feature(per_axis, "strict_positive_synergy", maximize=False),
        "principle": (
            "\u5bc6\u5ea6\u524d\u6cbf\u63cf\u8ff0\u7684\u662f\u9ad8\u8d28\u91cf\u795e\u7ecf\u5143\u652f\u6491\u6cbf\u8d28\u91cf\u6bd4\u4f8b\u5982\u4f55\u5c55\u5f00\u3002"
            "\u5b83\u4e0d\u518d\u95ee\u54ea\u4e9b\u795e\u7ecf\u5143\u975e\u96f6\uff0c\u800c\u662f\u95ee\u771f\u6b63\u6709\u5224\u522b\u529b\u7684\u6838\u5fc3\u5728\u54ea\u91cc\u5f00\u59cb\u538b\u7f29\u3001\u5206\u79bb\u6216\u7834\u574f\u95ed\u5305\u3002"
        ),
    }


def build_internal_subfield_section(complete_summary: Dict[str, object]) -> Dict[str, object]:
    per_component = dict(complete_summary.get("per_component", {}))
    components = []
    for component_label in sorted(per_component):
        component_block = dict(per_component[component_label])
        feature_stats = dict(component_block.get("feature_stats", {}))
        best_positive_feature = ""
        best_positive_corr = float("-inf")
        best_negative_feature = ""
        best_negative_corr = float("inf")
        for feature_name, feature_stat_obj in feature_stats.items():
            feature_stat = dict(feature_stat_obj)
            corr = target_corr(feature_stat, "union_synergy_joint")
            if corr > best_positive_corr:
                best_positive_corr = corr
                best_positive_feature = feature_name
            if corr < best_negative_corr:
                best_negative_corr = corr
                best_negative_feature = feature_name
        components.append(
            {
                "component_label": component_label,
                "case_count": safe_int(component_block.get("case_count")),
                "best_positive_feature": best_positive_feature,
                "best_positive_corr_to_synergy": 0.0 if best_positive_corr == float("-inf") else best_positive_corr,
                "best_negative_feature": best_negative_feature,
                "best_negative_corr_to_synergy": 0.0 if best_negative_corr == float("inf") else best_negative_corr,
            }
        )
    return {
        "component_count": len(components),
        "components": components,
        "principle": (
            "\u5185\u90e8\u5b50\u573a\u4e0d\u662f style\uff08\u98ce\u683c\uff09\u3001logic\uff08\u903b\u8f91\uff09\u3001syntax\uff08\u53e5\u6cd5\uff09\u4e09\u8f74\u672c\u8eab\uff0c"
            "\u800c\u662f\u8fd9\u4e9b\u8f74\u5185\u90e8\u771f\u6b63\u6267\u884c\u529f\u80fd\u7684\u7ec6\u5206\u673a\u5236\uff0c\u6bd4\u5982\u7acb\u9aa8\u67b6\u3001\u8106\u5f31\u6865\u63a5\u3001\u7ea6\u675f\u578b\u51b2\u7a81\u3002"
        ),
    }


def build_token_window_section(window_summary: Dict[str, object]) -> Dict[str, object]:
    per_component = dict(window_summary.get("per_component", {}))
    components = []
    for component_label in sorted(per_component):
        overall = dict(dict(per_component[component_label]).get("overall", {}))
        components.append(
            {
                "component_label": component_label,
                "dominant_hidden_tail_position": str(overall.get("dominant_hidden_tail_position_mode", "")),
                "dominant_mlp_tail_position": str(overall.get("dominant_mlp_tail_position_mode", "")),
                "peak_hidden_tail_position": str(overall.get("peak_hidden_tail_position_from_profile", "")),
                "peak_mlp_tail_position": str(overall.get("peak_mlp_tail_position_from_profile", "")),
                "mean_union_synergy_joint": safe_float(overall.get("mean_union_synergy_joint")),
            }
        )
    return {
        "component_count": len(components),
        "components": components,
        "principle": (
            "\u8bcd\u5143\u7a97\u53e3\u56de\u7b54\u7684\u662f\u673a\u5236\u5728\u4ec0\u4e48\u65f6\u5019\u8d77\u4f5c\u7528\u3002"
            "\u771f\u6b63\u7684\u6536\u675f\u901a\u5e38\u53d1\u751f\u5728\u53e5\u5c3e\u524d\u82e5\u5e72\u8bcd\u5143\u7a97\u53e3\uff0c\u800c\u4e0d\u662f\u6700\u540e\u4e00\u4e2a\u8bcd\u5143\u672c\u8eab\u3002"
        ),
    }


def build_closure_section(pair_link_summary: Dict[str, object]) -> Dict[str, object]:
    axis_target_stats = dict(pair_link_summary.get("axis_target_stats", {}))
    best_positive = {"axis": "", "field_name": "", "corr": float("-inf")}
    best_negative = {"axis": "", "field_name": "", "corr": float("inf")}
    for axis_name, axis_block_obj in axis_target_stats.items():
        axis_block = dict(axis_block_obj)
        for field_name, field_stat_obj in axis_block.items():
            field_stat = dict(field_stat_obj)
            corr = safe_float(dict(field_stat.get("targets", {})).get("union_synergy_joint", {}).get("pearson_corr"))
            if corr > safe_float(best_positive["corr"]):
                best_positive = {"axis": axis_name, "field_name": field_name, "corr": corr}
            if corr < safe_float(best_negative["corr"]):
                best_negative = {"axis": axis_name, "field_name": field_name, "corr": corr}
    return {
        "pair_positive_ratio": safe_float(pair_link_summary.get("pair_positive_ratio")),
        "mean_union_joint_adv": safe_float(pair_link_summary.get("mean_union_joint_adv")),
        "mean_union_synergy_joint": safe_float(pair_link_summary.get("mean_union_synergy_joint")),
        "strongest_positive_field_to_synergy": best_positive,
        "strongest_negative_field_to_synergy": best_negative,
        "principle": (
            "\u95ed\u5305\u91cf\u4e0d\u662f\u770b\u5c40\u90e8\u6fc0\u6d3b\u662f\u5426\u51fa\u73b0\uff0c\u800c\u662f\u770b\u539f\u578b\u3001\u5b9e\u4f8b\u3001\u5173\u7cfb\u548c\u63a7\u5236\u9879\u6700\u540e\u662f\u5426\u771f\u7684\u6536\u675f\u6210\u7a33\u5b9a\u8054\u5408\u8868\u793a\u3002"
        ),
    }


def build_summary(
    pair_density_summary: Dict[str, object],
    complete_summary: Dict[str, object],
    window_summary: Dict[str, object],
    pair_link_summary: Dict[str, object],
    law_summary: Dict[str, object],
) -> Dict[str, object]:
    density_frontier = build_density_frontier_section(pair_density_summary, law_summary)
    internal_subfield = build_internal_subfield_section(complete_summary)
    token_window = build_token_window_section(window_summary)
    closure = build_closure_section(pair_link_summary)
    return {
        "record_type": "stage56_frontier_subfield_window_closure_summary",
        "density_frontier": density_frontier,
        "internal_subfield": internal_subfield,
        "token_window": token_window,
        "closure": closure,
        "unified_principle": (
            "\u5bc6\u5ea6\u524d\u6cbf\u56de\u7b54\u9ad8\u8d28\u91cf\u652f\u6491\u5728\u54ea\u91cc\uff0c\u5185\u90e8\u5b50\u573a\u56de\u7b54\u662f\u8c01\u5728\u8d77\u4f5c\u7528\uff0c"
            "\u8bcd\u5143\u7a97\u53e3\u56de\u7b54\u5b83\u5728\u4ec0\u4e48\u65f6\u5019\u8d77\u4f5c\u7528\uff0c\u95ed\u5305\u91cf\u56de\u7b54\u8fd9\u4e9b\u4f5c\u7528\u6700\u540e\u6709\u6ca1\u6709\u5f62\u6210\u7a33\u5b9a\u8f93\u51fa\u3002"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    density_frontier = dict(summary.get("density_frontier", {}))
    internal_subfield = dict(summary.get("internal_subfield", {}))
    token_window = dict(summary.get("token_window", {}))
    closure = dict(summary.get("closure", {}))
    strongest_positive = dict(density_frontier.get("strongest_positive_frontier", {}))
    strongest_negative = dict(density_frontier.get("strongest_negative_frontier", {}))
    positive_field = dict(closure.get("strongest_positive_field_to_synergy", {}))
    negative_field = dict(closure.get("strongest_negative_field_to_synergy", {}))

    lines = [
        "# Stage56 \u56db\u6982\u5ff5\u8054\u7acb\u6458\u8981",
        "",
        "## \u7edf\u4e00\u539f\u5219",
        f"- {summary.get('unified_principle', '')}",
        "",
        "## \u5bc6\u5ea6\u524d\u6cbf",
        f"- broad_support_base: {safe_float(density_frontier.get('broad_support_base')):+.4f}",
        f"- long_separation_frontier: {safe_float(density_frontier.get('long_separation_frontier')):+.4f}",
        f"- strongest_positive_frontier: {strongest_positive.get('axis', '')} / {strongest_positive.get('feature_name', '')} / corr={safe_float(strongest_positive.get('corr')):+.4f}",
        f"- strongest_negative_frontier: {strongest_negative.get('axis', '')} / {strongest_negative.get('feature_name', '')} / corr={safe_float(strongest_negative.get('corr')):+.4f}",
        "",
        "## \u5185\u90e8\u5b50\u573a",
    ]
    for row in list(internal_subfield.get("components", [])):
        row = dict(row)
        lines.append(
            f"- {row.get('component_label', '')}: "
            f"best_positive={row.get('best_positive_feature', '')}({safe_float(row.get('best_positive_corr_to_synergy')):+.4f}), "
            f"best_negative={row.get('best_negative_feature', '')}({safe_float(row.get('best_negative_corr_to_synergy')):+.4f})"
        )
    lines.extend(["", "## \u8bcd\u5143\u7a97\u53e3"])
    for row in list(token_window.get("components", [])):
        row = dict(row)
        lines.append(
            f"- {row.get('component_label', '')}: "
            f"hidden={row.get('dominant_hidden_tail_position', '')}, "
            f"mlp={row.get('dominant_mlp_tail_position', '')}, "
            f"mean_synergy={safe_float(row.get('mean_union_synergy_joint')):+.4f}"
        )
    lines.extend(
        [
            "",
            "## \u95ed\u5305\u91cf",
            f"- pair_positive_ratio: {safe_float(closure.get('pair_positive_ratio')):+.4f}",
            f"- mean_union_joint_adv: {safe_float(closure.get('mean_union_joint_adv')):+.4f}",
            f"- mean_union_synergy_joint: {safe_float(closure.get('mean_union_synergy_joint')):+.4f}",
            f"- strongest_positive_field: {positive_field.get('axis', '')} / {positive_field.get('field_name', '')} / corr={safe_float(positive_field.get('corr')):+.4f}",
            f"- strongest_negative_field: {negative_field.get('axis', '')} / {negative_field.get('field_name', '')} / corr={safe_float(negative_field.get('corr')):+.4f}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize density frontier, internal subfield, token window, and closure")
    ap.add_argument(
        "--pair-density-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_pair_density_tensor_field_20260319_1512" / "summary.json"),
    )
    ap.add_argument(
        "--complete-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_complete_highdim_field_20260319_1640" / "summary.json"),
    )
    ap.add_argument(
        "--window-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_component_trajectory_window_map_all3_12cat_allpairs_20260319_0137" / "summary.json"),
    )
    ap.add_argument(
        "--pair-link-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_generation_gate_stage6_pair_link_all3_12cat_allpairs_20260319_0121" / "summary.json"),
    )
    ap.add_argument(
        "--law-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_simple_generator_laws_20260319_1702" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_frontier_subfield_window_closure_summary_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        pair_density_summary=read_json(Path(args.pair_density_summary_json)),
        complete_summary=read_json(Path(args.complete_summary_json)),
        window_summary=read_json(Path(args.window_summary_json)),
        pair_link_summary=read_json(Path(args.pair_link_summary_json)),
        law_summary=read_json(Path(args.law_summary_json)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "record_type": summary["record_type"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
