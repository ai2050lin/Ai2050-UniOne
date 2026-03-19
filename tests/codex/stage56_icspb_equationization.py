from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def argmax_label(mapping: Dict[str, float], default: str) -> str:
    if not mapping:
        return default
    return max(mapping.items(), key=lambda item: item[1])[0]


def build_equations(
    relation_summary: Dict[str, object],
    unified_atlas: Dict[str, object],
    bxm_summary: Dict[str, object],
    pair_link_summary: Dict[str, object],
    internal_map: Dict[str, object],
    triplet_json: Dict[str, object],
    apple_dossier: Dict[str, object],
) -> Dict[str, object]:
    pair_axes = dict(pair_link_summary.get("axis_target_stats", {}))
    bxm_axes = dict(bxm_summary.get("per_axis", {}))
    triplet_metrics = dict(triplet_json.get("metrics", {}))
    apple_metrics = dict(apple_dossier.get("metrics", {}))
    model_private = dict(unified_atlas.get("model_private_implementations", {}))
    per_model_internal = dict(internal_map.get("per_model", {}))

    closure_coeffs = {
        "logic_P": safe_float(
            pair_axes.get("logic", {})
            .get("prototype_field_proxy", {})
            .get("targets", {})
            .get("union_synergy_joint", {})
            .get("pearson_corr")
        ),
        "style_I": safe_float(
            pair_axes.get("style", {})
            .get("instance_field_proxy", {})
            .get("targets", {})
            .get("union_synergy_joint", {})
            .get("pearson_corr")
        ),
        "style_SB": safe_float(bxm_axes.get("style", {}).get("stable_bridge", {}).get("corr_to_union_synergy_joint")),
        "logic_FB": abs(
            safe_float(bxm_axes.get("logic", {}).get("fragile_bridge", {}).get("corr_to_union_synergy_joint"))
        ),
        "syntax_CX": safe_float(
            bxm_axes.get("syntax", {}).get("constraint_conflict", {}).get("corr_to_union_synergy_joint")
        ),
        "syntax_MD": abs(
            safe_float(bxm_axes.get("syntax", {}).get("mismatch_damage", {}).get("corr_to_union_synergy_joint"))
        ),
    }

    relation_coeffs = {
        "axis_specificity": safe_float(triplet_metrics.get("axis_specificity_index")),
        "triplet_separability": safe_float(triplet_metrics.get("triplet_separability_index")),
        "hierarchy_gain": safe_float(apple_metrics.get("apple_meso_to_macro_jaccard_mean"))
        - safe_float(apple_metrics.get("apple_micro_to_meso_jaccard_mean")),
        "control_decoupling": safe_float(apple_metrics.get("cross_dim_decoupling_index")),
        "local_linear_ratio": safe_float(dict(relation_summary.get("counts_by_interpretation", {})).get("local_linear", 0))
        / max(1.0, safe_float(relation_summary.get("group_count", 0))),
    }

    layer_predictions: Dict[str, Dict[str, str]] = {}
    for model_id, block in per_model_internal.items():
        per_axis = dict(block.get("per_axis", {}))
        hidden_scores = {
            str(axis): float(str(dict(axis_block).get("dominant_hidden_layer", "layer_0")).split("_")[-1])
            for axis, axis_block in per_axis.items()
        }
        mlp_scores = {
            str(axis): float(str(dict(axis_block).get("dominant_mlp_layer", "layer_0")).split("_")[-1])
            for axis, axis_block in per_axis.items()
        }
        hidden_axis = argmax_label(hidden_scores, "style")
        mlp_axis = argmax_label(mlp_scores, "style")
        hidden_block = dict(per_axis.get(hidden_axis, {}))
        mlp_block = dict(per_axis.get(mlp_axis, {}))
        layer_predictions[model_id] = {
            "predicted_hidden_spine": hidden_axis,
            "predicted_hidden_layer": str(hidden_block.get("dominant_hidden_layer", "layer_0")),
            "predicted_mlp_spine": mlp_axis,
            "predicted_mlp_layer": str(mlp_block.get("dominant_mlp_layer", "layer_0")),
            "predicted_attention_head": str(hidden_block.get("dominant_attention_head", "layer_0_head_0")),
            "summary_reading": str(dict(model_private.get(model_id, {})).get("reading", "")),
        }

    equations = {
        "anchor_fiber_relation_equation": {
            "symbolic_form": "H(term,ctx)=A_anchor(term)+F_modifier(term)+R_relation(term,ctx)+G_control(ctx)",
            "coefficients": relation_coeffs,
        },
        "closure_equation": {
            "symbolic_form": "C = +logic_P + style_I + style_SB - logic_FB + syntax_CX - syntax_MD",
            "coefficients": closure_coeffs,
        },
        "relation_projection_equation": {
            "symbolic_form": "Pi_relation = axis_specificity * local_linearity - hierarchy_gain * bundle_load",
            "coefficients": relation_coeffs,
        },
        "layer_head_mlp_equation": {
            "symbolic_form": "L* = argmax_l [Spine_model(l) + syntax_CX*X_l + logic_P*P_l - logic_FB*FB_l]",
            "per_model_predictions": layer_predictions,
        },
    }
    checks = {
        "closure_positive_terms_present": bool(
            closure_coeffs["logic_P"] > 0.0 and closure_coeffs["syntax_CX"] > 0.0
        ),
        "closure_negative_terms_present": bool(
            closure_coeffs["logic_FB"] > 0.0 and closure_coeffs["syntax_MD"] > 0.0
        ),
        "relation_axis_is_local_not_global": bool(
            relation_coeffs["axis_specificity"] > 0.0 and relation_coeffs["local_linear_ratio"] < 0.5
        ),
    }
    return {"record_type": "stage56_icspb_equationization_summary", "equations": equations, "checks": checks}


def write_report(path: Path, summary: Dict[str, object]) -> None:
    equations = dict(summary.get("equations", {}))
    lines = ["# Stage56 ICSPB 方程化块", "", "## 方程"]
    for key, row in equations.items():
        lines.append(f"- {key}: {row['symbolic_form']}")
    lines.extend(["", "## 检验"])
    for key, value in dict(summary.get("checks", {})).items():
        lines.append(f"- {key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Turn ICSPB language structure findings into testable equations")
    ap.add_argument(
        "--relation-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_large_relation_atlas_20260318" / "summary.json"),
    )
    ap.add_argument(
        "--unified-atlas-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_multimodel_language_unified_atlas_20260318" / "summary.json"),
    )
    ap.add_argument(
        "--bxm-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_bxm_rewrite_20260318_2222" / "summary.json"),
    )
    ap.add_argument(
        "--pair-link-summary-json",
        default=str(
            ROOT
            / "tests"
            / "codex_temp"
            / "stage56_generation_gate_stage6_pair_link_all3_12cat_pairs_20260318_2120"
            / "summary.json"
        ),
    )
    ap.add_argument(
        "--internal-map-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_generation_gate_internal_map_20260318_1338" / "summary.json"),
    )
    ap.add_argument(
        "--triplet-json",
        default=str(ROOT / "tempdata" / "deepseek7b_triplet_probe_20260306_150637" / "apple_king_queen_triplet_probe.json"),
    )
    ap.add_argument(
        "--apple-dossier-json",
        default=str(
            ROOT
            / "tempdata"
            / "deepseek7b_apple_encoding_law_dossier_20260306_223055"
            / "apple_multiaxis_encoding_law_dossier.json"
        ),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_icspb_equationization_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_equations(
        relation_summary=read_json(Path(args.relation_summary_json)),
        unified_atlas=read_json(Path(args.unified_atlas_summary_json)),
        bxm_summary=read_json(Path(args.bxm_summary_json)),
        pair_link_summary=read_json(Path(args.pair_link_summary_json)),
        internal_map=read_json(Path(args.internal_map_summary_json)),
        triplet_json=read_json(Path(args.triplet_json)),
        apple_dossier=read_json(Path(args.apple_dossier_json)),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", summary)
    print(json.dumps({"output_dir": str(out_dir), "checks": summary["checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
