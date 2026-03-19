from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

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


def build_summary(
    structure_atlas: Dict[str, object],
    relation_summary: Dict[str, object],
    wordclass_summary: Dict[str, object],
    unified_atlas: Dict[str, object],
    window_summary: Dict[str, object],
    natural_decoupling: Dict[str, object],
) -> Dict[str, object]:
    relation_group_count = max(1.0, safe_float(relation_summary.get("group_count", 0)))
    local_linear_count = safe_float(dict(relation_summary.get("counts_by_interpretation", {})).get("local_linear", 0))
    path_bundle_count = safe_float(dict(relation_summary.get("counts_by_interpretation", {})).get("path_bundle", 0))
    hybrid_count = safe_float(dict(relation_summary.get("counts_by_interpretation", {})).get("hybrid", 0))

    parts_of_speech = dict(structure_atlas.get("parts_of_speech", {}))
    common_laws = dict(unified_atlas.get("common_laws", {}))
    control_laws = dict(common_laws.get("control_laws", {}))
    per_component = dict(window_summary.get("per_component", {}))
    natural_axes = dict(natural_decoupling.get("per_axis", {}))
    natural_components = dict(natural_decoupling.get("per_component", {}))

    logic_window = dict(per_component.get("logic_prototype", {}))
    fragile_window = dict(per_component.get("logic_fragile_bridge", {}))
    syntax_window = dict(per_component.get("syntax_constraint_conflict", {}))
    syntax_natural = dict(natural_axes.get("syntax", {}))
    syntax_natural_component = dict(natural_components.get("syntax_constraint_conflict", {}))

    generalized_equations = {
        "semantic_state": "S(term,ctx)=A_anchor+F_modifier+R_bundle+G_control",
        "closure_window": (
            "C(term,ctx)≈+P_logic[-9..-8]-FB_logic[-2..-1]"
            "+CX_syntax_hidden[-8..-5]+CX_syntax_mlp[-6..-3]+..."
        ),
        "natural_generation_split": "T_total = T_prompt_skeleton ⊕ T_generated_suffix",
    }

    return {
        "record_type": "stage56_general_encoding_mechanism_summary",
        "system_encoding_law": {
            "dominant_form": "anchor_fiber_path_bundle_with_windowed_closure",
            "structure_scores": {
                "local_linear_ratio": local_linear_count / relation_group_count,
                "path_bundle_ratio": path_bundle_count / relation_group_count,
                "control_mixed_ratio": hybrid_count / relation_group_count,
            },
            "structure_claims": {
                "noun_concept": str(dict(parts_of_speech.get("noun", {})).get("claim", "")),
                "adjective": str(dict(parts_of_speech.get("adjective", {})).get("claim", "")),
                "verb": str(dict(parts_of_speech.get("verb", {})).get("claim", "")),
                "abstract_noun": str(dict(parts_of_speech.get("abstract_noun", {})).get("claim", "")),
                "adverb": str(dict(parts_of_speech.get("adverb", {})).get("claim", "")),
            },
            "global_system_claim": str(structure_atlas.get("system_claim", "")),
            "shared_closure_categories": list(common_laws.get("shared_closure_categories", [])),
        },
        "closure_dynamics": {
            "control_laws": control_laws,
            "logic_prototype_window": {
                "joint_adv_hidden_window": str(dict(logic_window.get("hidden_to_joint_adv", {})).get("best_window", "")),
                "joint_adv_hidden_corr": safe_float(dict(logic_window.get("hidden_to_joint_adv", {})).get("best_corr")),
                "joint_adv_mlp_window": str(dict(logic_window.get("mlp_to_joint_adv", {})).get("best_window", "")),
                "joint_adv_mlp_corr": safe_float(dict(logic_window.get("mlp_to_joint_adv", {})).get("best_corr")),
            },
            "logic_fragile_bridge_window": {
                "synergy_hidden_window": str(dict(fragile_window.get("hidden_to_synergy", {})).get("best_window", "")),
                "synergy_hidden_corr": safe_float(dict(fragile_window.get("hidden_to_synergy", {})).get("best_corr")),
                "joint_adv_hidden_window": str(dict(fragile_window.get("hidden_to_joint_adv", {})).get("best_window", "")),
                "joint_adv_hidden_corr": safe_float(dict(fragile_window.get("hidden_to_joint_adv", {})).get("best_corr")),
            },
            "syntax_constraint_conflict_window": {
                "synergy_hidden_window": str(dict(syntax_window.get("hidden_to_synergy", {})).get("best_window", "")),
                "synergy_hidden_corr": safe_float(dict(syntax_window.get("hidden_to_synergy", {})).get("best_corr")),
                "synergy_mlp_window": str(dict(syntax_window.get("mlp_to_synergy", {})).get("best_window", "")),
                "synergy_mlp_corr": safe_float(dict(syntax_window.get("mlp_to_synergy", {})).get("best_corr")),
            },
        },
        "natural_generation_decoupling": {
            "style_hidden_generated_share": safe_float(dict(natural_axes.get("style", {})).get("mean_hidden_generated_share")),
            "logic_hidden_generated_share": safe_float(dict(natural_axes.get("logic", {})).get("mean_hidden_generated_share")),
            "syntax_hidden_generated_share": safe_float(dict(syntax_natural).get("mean_hidden_generated_share")),
            "syntax_hidden_prompt_share": safe_float(dict(syntax_natural).get("mean_hidden_prompt_share")),
            "syntax_component_hidden_prompt_share": safe_float(
                dict(syntax_natural_component).get("weighted_hidden_prompt_share")
            ),
            "syntax_component_hidden_generated_share": safe_float(
                dict(syntax_natural_component).get("weighted_hidden_generated_share")
            ),
            "judgement": (
                "syntax_prompt_contaminated"
                if safe_float(dict(syntax_natural).get("mean_hidden_prompt_share"))
                > safe_float(dict(syntax_natural).get("mean_hidden_generated_share"))
                else "syntax_generation_dominant"
            ),
        },
        "wordclass_mechanisms": dict(wordclass_summary.get("probes", {})),
        "generalized_equations": generalized_equations,
        "general_principles": [
            "语言系统主结构不是统一线性空间，而是锚点、纤维、关系束与控制轴的复合体。",
            "局部线性只在少数关系家族成立，系统主结构由路径束主导。",
            "闭包动力学必须在连续窗口上建模，粗总量变量已经失效。",
            "自然生成分析必须先把提示骨架区和新增生成区解耦，否则会混入伪早窗信号。",
        ],
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    law = dict(summary.get("system_encoding_law", {}))
    closure = dict(summary.get("closure_dynamics", {}))
    decoupling = dict(summary.get("natural_generation_decoupling", {}))
    lines = [
        "# Stage56 一般编码机制摘要",
        "",
        f"- dominant_form: {law.get('dominant_form', '')}",
        f"- structure_scores: {law.get('structure_scores', {})}",
        f"- shared_closure_categories: {law.get('shared_closure_categories', [])}",
        "",
        "## 闭包动力学",
        f"- logic_prototype_window: {closure.get('logic_prototype_window', {})}",
        f"- logic_fragile_bridge_window: {closure.get('logic_fragile_bridge_window', {})}",
        f"- syntax_constraint_conflict_window: {closure.get('syntax_constraint_conflict_window', {})}",
        "",
        "## 自然生成解耦",
        f"- natural_generation_decoupling: {decoupling}",
        "",
        "## 一般原则",
    ]
    for line in summary.get("general_principles", []):
        lines.append(f"- {line}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a more general mechanism-level summary of language encoding structure")
    ap.add_argument(
        "--structure-atlas-json",
        default=str(
            ROOT
            / "tests"
            / "codex_temp"
            / "stage56_language_system_structure_atlas_20260318_2146"
            / "language_system_structure_atlas.json"
        ),
    )
    ap.add_argument(
        "--relation-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_large_relation_atlas_20260318_2251" / "summary.json"),
    )
    ap.add_argument(
        "--wordclass-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_wordclass_causal_probe_20260318_2237" / "summary.json"),
    )
    ap.add_argument(
        "--unified-atlas-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_multimodel_language_unified_atlas_20260318_2252" / "summary.json"),
    )
    ap.add_argument(
        "--window-summary-json",
        default=str(
            ROOT
            / "tests"
            / "codex_temp"
            / "stage56_window_variable_rewrite_all3_12cat_allpairs_20260319_0610"
            / "summary.json"
        ),
    )
    ap.add_argument(
        "--natural-decoupling-summary-json",
        default=str(
            ROOT
            / "tests"
            / "codex_temp"
            / "stage56_natural_generation_decoupling_20260319"
            / "summary.json"
        ),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_general_encoding_mechanism_summary_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        structure_atlas=read_json(Path(args.structure_atlas_json)),
        relation_summary=read_json(Path(args.relation_summary_json)),
        wordclass_summary=read_json(Path(args.wordclass_summary_json)),
        unified_atlas=read_json(Path(args.unified_atlas_summary_json)),
        window_summary=read_json(Path(args.window_summary_json)),
        natural_decoupling=read_json(Path(args.natural_decoupling_summary_json)),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", summary)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "dominant_form": dict(summary["system_encoding_law"]).get("dominant_form", ""),
                "general_principle_count": len(summary.get("general_principles", [])),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
