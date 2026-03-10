#!/usr/bin/env python
"""
Cross-check whether apple-concept mechanism decomposition is consistent between:
1) Qwen3-4B fresh/local result artifacts
2) DeepSeek-7B historical/local result artifacts

Target decomposition:
- shared basis
- individual offset
- gating term G
- relation term R
- representation space H
- topology/routing space T

Important boundary:
- Qwen3-4B side has direct H/T/G/R result files from local experiments.
- DeepSeek-7B side now has direct local attention-topology measurement on this
  machine, so T no longer needs to rely only on proxy routing evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def status_from_bool(ok: bool) -> str:
    return "consistent" if ok else "not_consistent"


def load_qwen_metrics(paths: Dict[str, Path]) -> Dict[str, Any]:
    basis = read_json(paths["qwen_basis"])["models"]["qwen3_4b"]
    topo = read_json(paths["qwen_topology"])["models"]["qwen3_4b"]
    align = read_json(paths["qwen_repr_topology"])["models"]["qwen3_4b"]
    rg = read_json(paths["qwen_relation_gating"])["models"]["qwen3_4b"]
    nat = read_json(paths["qwen_natural_offset"])["models"]["qwen3_4b"]

    apple_fit = basis["apple_affine_fit"]
    qwen = {
        "shared_basis": {
            "fruit_compactness": float(basis["family_compactness"]["fruit"]["mean_residual_ratio"]),
            "apple_to_fruit_residual": float(apple_fit["fruit"]["residual_ratio"]),
            "apple_to_animal_residual": float(apple_fit["animal"]["residual_ratio"]),
            "apple_to_abstract_residual": float(apple_fit["abstract"]["residual_ratio"]),
            "apple_gap_vs_animal": float(basis["nested_metrics"]["apple_vs_fruit_minus_animal_residual_gap"]),
            "family_nested_world": bool(basis["hypotheses"]["H4_family_nested_in_world_basis"]),
        },
        "offset": {
            "delta_top64_energy_ratio": float(apple_fit["fruit"]["delta_top64_energy_ratio"]),
            "delta_top256_energy_ratio": float(apple_fit["fruit"]["delta_top256_energy_ratio"]),
            "natural_offset_supported": bool(
                next(x for x in nat["targets"] if x["word"] == "apple")["supports_natural_dict_sparse_offset"]
            ),
            "natural_offset_gap_top4": float(
                next(x for x in nat["targets"] if x["word"] == "apple")["matched_vs_wrong_top4_gap"]
            ),
        },
        "H": {
            "apple_closer_to_fruit_than_animal": float(apple_fit["fruit"]["residual_ratio"])
            < float(apple_fit["animal"]["residual_ratio"]),
            "apple_closer_to_fruit_than_abstract": float(apple_fit["fruit"]["residual_ratio"])
            < float(apple_fit["abstract"]["residual_ratio"]),
        },
        "T": {
            "apple_supports_family_topology_basis": bool(topo["probe_fits"]["apple"]["supports_family_topology_basis"]),
            "fruit_topology_residual": float(topo["probe_fits"]["apple"]["fit"]["fruit"]["residual_ratio"]),
            "animal_topology_residual": float(topo["probe_fits"]["apple"]["fit"]["animal"]["residual_ratio"]),
            "abstract_topology_residual": float(topo["probe_fits"]["apple"]["fit"]["abstract"]["residual_ratio"]),
            "topology_layers": [int(x) for x in align["layer_role_summary"]["best_topology_layers"][:5]],
            "repr_layers": [int(x) for x in align["layer_role_summary"]["best_repr_layers"][:5]],
        },
        "G": {
            "repr_gating_layers": [int(x) for x in rg["layer_role_summary"]["repr_gating_layers"][:5]],
            "topo_gating_layers": [int(x) for x in rg["layer_role_summary"]["topo_gating_layers"][:5]],
            "early_topo_gating_strength": mean(
                [float(x) for x in rg["gating_topo_avg_by_layer"][:8]]
            ),
            "early_topo_relation_strength": mean(
                [float(x) for x in rg["relation_topo_avg_by_layer"][:8]]
            ),
        },
        "R": {
            "repr_relation_layers": [int(x) for x in rg["layer_role_summary"]["repr_relation_layers"][:5]],
            "topo_relation_layers": [int(x) for x in rg["layer_role_summary"]["topo_relation_layers"][:5]],
            "deep_repr_relation_strength": mean(
                [float(x) for x in rg["relation_repr_avg_by_layer"][20:32]]
            ),
            "deep_repr_gating_strength": mean(
                [float(x) for x in rg["gating_repr_avg_by_layer"][20:32]]
            ),
        },
    }
    return qwen


def load_deepseek_metrics(paths: Dict[str, Path]) -> Dict[str, Any]:
    family = read_json(paths["deepseek_family"])
    dossier = read_json(paths["deepseek_dossier"])
    ablation = read_json(paths["deepseek_gating"])
    route = read_json(paths["deepseek_route"])
    invariant = read_json(paths["deepseek_invariant"])
    triplet = read_json(paths["deepseek_triplet"])
    topo = read_json(paths["deepseek_topology"])["models"]["deepseek_7b"]

    deepseek = {
        "shared_basis": {
            "apple_shared_base_ratio_mean": float(family["metrics"]["apple_chain_summary"]["shared_base_ratio_vs_micro_union"]["mean"]),
            "apple_micro_to_meso_jaccard_mean": float(family["metrics"]["apple_chain_summary"]["micro_to_meso_jaccard"]["mean"]),
            "apple_meso_to_macro_jaccard_mean": float(family["metrics"]["apple_chain_summary"]["meso_to_macro_jaccard"]["mean"]),
        },
        "offset": {
            "axis_specificity_index": float(dossier["metrics"]["axis_specificity_index"]),
            "cross_dim_decoupling_index": float(dossier["metrics"]["cross_dim_decoupling_index"]),
            "apple_shared_base_ratio_mean": float(dossier["metrics"]["apple_shared_base_ratio_mean"]),
        },
        "H": {
            "hierarchy_closure_pass": bool(
                next(h for h in dossier["hypotheses"] if h["id"] == "H3_apple_hierarchy_closure")["pass"]
            ),
            "layer_anchor_pass": bool(
                next(h for h in dossier["hypotheses"] if h["id"] == "H5_apple_has_layer_anchor")["pass"]
            ),
        },
        "G": {
            "diagonal_advantage": {k: float(v) for k, v in ablation["diagonal_advantage"].items()},
            "style_self": float(ablation["suppression_matrix_mean"]["style"]["style"]),
            "style_logic": float(ablation["suppression_matrix_mean"]["style"]["logic"]),
            "logic_self": float(ablation["suppression_matrix_mean"]["logic"]["logic"]),
            "logic_style": float(ablation["suppression_matrix_mean"]["logic"]["style"]),
            "syntax_self": float(ablation["suppression_matrix_mean"]["syntax"]["syntax"]),
            "syntax_logic": float(ablation["suppression_matrix_mean"]["syntax"]["logic"]),
        },
        "R": {
            "route_index": float(route["base_metrics"]["route_index"]),
            "minimal_subset_size": int(route["minimal_subset"]["size"]),
            "hop3_selectivity": float(route["base_metrics"]["hop3_selectivity"]),
            "triplet_positive_ratio": float(triplet["metrics"]["global_positive_causal_margin_ratio"]),
            "triplet_seq_logprob_margin": float(triplet["metrics"]["global_mean_causal_margin_seq_logprob"]),
        },
        "T": {
            "proxy_only": False,
            "apple_supports_family_topology_basis": bool(topo["probe_fits"]["apple"]["supports_family_topology_basis"]),
            "fruit_topology_residual": float(topo["probe_fits"]["apple"]["fit"]["fruit"]["residual_ratio"]),
            "animal_topology_residual": float(topo["probe_fits"]["apple"]["fit"]["animal"]["residual_ratio"]),
            "abstract_topology_residual": float(topo["probe_fits"]["apple"]["fit"]["abstract"]["residual_ratio"]),
            "graph_geometry_alignment": float(invariant["summary"]["readiness_score"]),
            "same_type_gate_overlap": float(invariant["axis_isolation"]["micro_attr__same_type"]["gate_dim_jaccard_mean"]),
            "super_shared_dims_fruit": int(invariant["group_shared_dims"]["fruit"]["super_type"]["count"]),
            "super_shared_dims_animal": int(invariant["group_shared_dims"]["animal"]["super_type"]["count"]),
            "route_index": float(route["base_metrics"]["route_index"]),
        },
    }
    return deepseek


def build_verdict(qwen: Dict[str, Any], deepseek: Dict[str, Any]) -> Dict[str, Any]:
    shared_basis_ok = (
        qwen["shared_basis"]["apple_to_fruit_residual"] < qwen["shared_basis"]["apple_to_animal_residual"]
        and qwen["shared_basis"]["apple_to_fruit_residual"] < qwen["shared_basis"]["apple_to_abstract_residual"]
        and deepseek["shared_basis"]["apple_shared_base_ratio_mean"] > 0.02
        and deepseek["shared_basis"]["apple_meso_to_macro_jaccard_mean"] > 0.2
    )

    offset_ok = (
        qwen["offset"]["natural_offset_supported"]
        and qwen["offset"]["natural_offset_gap_top4"] > 0.01
        and deepseek["offset"]["axis_specificity_index"] > 0.5
        and deepseek["offset"]["cross_dim_decoupling_index"] > 0.6
    )

    h_ok = (
        qwen["H"]["apple_closer_to_fruit_than_animal"]
        and qwen["H"]["apple_closer_to_fruit_than_abstract"]
        and deepseek["H"]["hierarchy_closure_pass"]
        and deepseek["H"]["layer_anchor_pass"]
    )

    g_ok = (
        qwen["G"]["early_topo_gating_strength"] > qwen["G"]["early_topo_relation_strength"]
        and all(v > 0.0 for v in deepseek["G"]["diagonal_advantage"].values())
    )

    r_ok = (
        qwen["R"]["deep_repr_relation_strength"] > qwen["R"]["deep_repr_gating_strength"]
        and deepseek["R"]["route_index"] > 0.005
        and deepseek["R"]["triplet_seq_logprob_margin"] > 0.0
    )

    t_qwen_ok = (
        qwen["T"]["apple_supports_family_topology_basis"]
        and qwen["T"]["fruit_topology_residual"] < qwen["T"]["animal_topology_residual"]
        and qwen["T"]["fruit_topology_residual"] < qwen["T"]["abstract_topology_residual"]
    )
    t_deepseek_ok = (
        deepseek["T"]["apple_supports_family_topology_basis"]
        and deepseek["T"]["fruit_topology_residual"] < deepseek["T"]["animal_topology_residual"]
        and deepseek["T"]["fruit_topology_residual"] < deepseek["T"]["abstract_topology_residual"]
    )

    component_status = {
        "shared_basis": status_from_bool(shared_basis_ok),
        "individual_offset": status_from_bool(offset_ok),
        "H_representation": status_from_bool(h_ok),
        "G_gating": status_from_bool(g_ok),
        "R_relation": status_from_bool(r_ok),
        "T_topology": status_from_bool(t_qwen_ok and t_deepseek_ok),
    }

    n_consistent = sum(1 for v in component_status.values() if v == "consistent")
    n_partial = sum(1 for v in component_status.values() if v == "partially_consistent")
    overall = "mostly_consistent" if n_consistent >= 5 else "partially_consistent"
    if n_consistent < 4:
        overall = "mixed"

    return {
        "component_status": component_status,
        "overall_verdict": overall,
        "summary": {
            "n_consistent": n_consistent,
            "n_partial": n_partial,
            "topology_boundary": "DeepSeek-7B now has direct same-protocol attention-topology measurement on this machine.",
        },
    }


def default_paths() -> Dict[str, Path]:
    return {
        "qwen_basis": Path("tests/codex_temp/gpt2_qwen3_basis_hierarchy_compare_20260308.json"),
        "qwen_topology": Path("tests/codex_temp/gpt2_qwen3_attention_topology_basis_20260308.json"),
        "qwen_repr_topology": Path("tests/codex_temp/gpt2_qwen3_repr_topology_layer_alignment_20260308.json"),
        "qwen_relation_gating": Path("tests/codex_temp/gpt2_qwen3_relation_gating_layer_separation_20260308.json"),
        "qwen_natural_offset": Path("tests/codex_temp/gpt2_qwen3_natural_offset_dictionary_20260308.json"),
        "deepseek_family": Path("tempdata/deepseek7b_concept_family_parallel_latest/concept_family_parallel_scale.json"),
        "deepseek_dossier": Path("tempdata/deepseek7b_apple_encoding_law_dossier_20260306_223055/apple_multiaxis_encoding_law_dossier.json"),
        "deepseek_gating": Path("tempdata/deepseek7b_multidim_causal_ablation_v2_allpos/multidim_causal_ablation.json"),
        "deepseek_route": Path("tempdata/deepseek7b_multihop_route_20260302_140900/multihop_route_results.json"),
        "deepseek_invariant": Path("tempdata/deepseek7b_encoding_invariant_probe_v1/encoding_invariant_probe.json"),
        "deepseek_triplet": Path("tempdata/deepseek7b_triplet_causal_targeted_20260306_153738/triplet_targeted_causal_scan.json"),
        "deepseek_topology": Path("tests/codex_temp/qwen3_deepseek7b_attention_topology_basis_20260309.json"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Qwen3-4B and DeepSeek-7B apple mechanism consistency")
    ap.add_argument(
        "--json-out",
        default="tests/codex_temp/qwen3_deepseek7b_apple_mechanism_consistency_20260309.json",
    )
    args = ap.parse_args()

    paths = default_paths()
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    qwen = load_qwen_metrics(paths)
    deepseek = load_deepseek_metrics(paths)
    verdict = build_verdict(qwen, deepseek)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scope": "Qwen3-4B direct local artifacts vs DeepSeek-7B historical local artifacts",
        },
        "paths": {k: str(v).replace("\\", "/") for k, v in paths.items()},
        "qwen3_4b": qwen,
        "deepseek_7b": deepseek,
        "verdict": verdict,
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(out_path), "overall_verdict": verdict["overall_verdict"], "component_status": verdict["component_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
