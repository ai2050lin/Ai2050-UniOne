from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track inventory to A coupling")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_to_A_coupling_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inv_math = load("theory_track_inventory_math_structure_formalization_20260312.json")
    operators = load("theory_track_family_conditioned_projection_operators_20260312.json")
    stress = load("theory_track_inventory_stress_profiling_20260312.json")
    explicit = load("theory_track_explicit_A_Mfeas_formalization_20260312.json")

    mean_novel = float(stress["headline_metrics"]["mean_novelty_pressure"])
    mean_ret_risk = float(stress["headline_metrics"]["mean_retention_risk"])
    mean_rel_cap = float(stress["headline_metrics"]["mean_relation_lift_capacity"])

    family_coupling = {}
    for family, operator_row in operators["core_operators"].items():
        family_coupling[family] = {
            "K_ret_family": {
                "projector": "P_mem_family",
                "support_dims": operator_row["P_mem_family"]["support_dims"],
                "stress_gate": f"sigma_ret(c) <= {mean_ret_risk + 0.02:.4f}",
            },
            "K_id_family": {
                "projector": "P_id_family",
                "support_dims": operator_row["P_id_family"]["support_dims"],
                "stress_gate": f"sigma_novel(c) <= {mean_novel + 0.02:.4f}",
            },
            "K_read_family": {
                "projector": "P_disc_family",
                "support_dims": operator_row["P_disc_family"]["support_dims"],
                "stress_gate": "readout update only if concept remains inside restricted object-disc overlap",
            },
            "K_bridge_family": {
                "projector": "P_obj_family",
                "support_dims": operator_row["P_obj_family"]["support_dims"],
                "stress_gate": f"sigma_rel(c) >= {max(1.0, mean_rel_cap * 0.5):.4f}",
            },
        }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_to_A_coupling",
        },
        "starting_form": explicit["explicit_A"]["high_level_form"],
        "inventory_conditioned_form": (
            "A(I) = INTERSECT_f [K_ret^(f)(I) INTERSECT K_id^(f)(I) "
            "INTERSECT K_read^(f)(I) INTERSECT K_bridge^(f)(I)] INTERSECT K_phase"
        ),
        "coupling_rule": {
            "core_statement": "Inventory entries do not just live inside A; they parameterize the family-conditioned admissible cones themselves.",
            "entry_level_statement": "Each concept stress field S_c gates which family-conditioned directions remain admissible.",
        },
        "family_coupling": family_coupling,
        "mathematical_meaning": {
            "A_from_inventory": [
                "family patch picks the local operator family",
                "concept offset determines local admissible directions",
                "attribute axes refine local directional decomposition",
                "stress field determines whether those directions are open or closed",
            ],
            "why_important": "This upgrades inventory from descriptive atlas to admissibility-generating structure.",
        },
        "verdict": {
            "core_answer": "The inventory can now be coupled directly to A: admissible cones should be treated as family-conditioned and stress-gated by inventory entries.",
            "next_theory_target": "perform the same upgrade for M_feas so inventory also parameterizes viability charts and overlap widths",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
