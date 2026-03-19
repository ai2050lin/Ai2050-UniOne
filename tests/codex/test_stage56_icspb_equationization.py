from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_icspb_equationization import build_equations  # noqa: E402


def test_build_equations_keeps_positive_and_negative_terms() -> None:
    summary = build_equations(
        relation_summary={"group_count": 100, "counts_by_interpretation": {"local_linear": 40}},
        unified_atlas={"model_private_implementations": {}},
        bxm_summary={"per_axis": {"style": {"stable_bridge": {"corr_to_union_synergy_joint": 0.15}}, "logic": {"fragile_bridge": {"corr_to_union_synergy_joint": -0.40}}, "syntax": {"constraint_conflict": {"corr_to_union_synergy_joint": 0.27}, "mismatch_damage": {"corr_to_union_synergy_joint": -0.50}}}},
        pair_link_summary={"axis_target_stats": {"logic": {"prototype_field_proxy": {"targets": {"union_synergy_joint": {"pearson_corr": 0.44}}}}, "style": {"instance_field_proxy": {"targets": {"union_synergy_joint": {"pearson_corr": 0.29}}}}}},
        internal_map={"per_model": {}},
        triplet_json={"metrics": {"axis_specificity_index": 0.63, "triplet_separability_index": 0.09}},
        apple_dossier={"metrics": {"apple_meso_to_macro_jaccard_mean": 0.37, "apple_micro_to_meso_jaccard_mean": 0.02, "cross_dim_decoupling_index": 0.68}},
    )
    coeffs = summary["equations"]["closure_equation"]["coefficients"]
    assert coeffs["logic_P"] > 0.0
    assert coeffs["logic_FB"] > 0.0
    assert coeffs["syntax_CX"] > 0.0
    assert coeffs["syntax_MD"] > 0.0
    assert summary["checks"]["closure_positive_terms_present"] is True
