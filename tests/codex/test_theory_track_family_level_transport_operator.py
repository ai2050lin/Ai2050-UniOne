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
    ap = argparse.ArgumentParser(description="Theory-track family-level transport operator")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_family_level_transport_operator_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    transport = load("theory_track_inventory_stress_to_readout_transport_coupling_20260312.json")
    overlap = load("theory_track_restricted_overlap_maps_20260312.json")
    operators = load("theory_track_inventory_operator_family_closure_20260312.json")

    family_rows: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[dict[str, object]]] = {}
    for concept, row in transport["concept_transport_profiles"].items():
        grouped.setdefault(str(row["family"]), []).append(row)

    for family, rows in grouped.items():
        mean_budget = float(sum(float(row["transport_budget"]) for row in rows) / len(rows))
        mean_novelty = float(sum(float(row["novelty_pressure"]) for row in rows) / len(rows))
        mean_ret = float(sum(float(row["retention_risk"]) for row in rows) / len(rows))
        overlap_width = float(overlap["restricted_overlap_maps"][family]["object_disc_overlap"])
        disc_dims = operators["family_operator_blocks"][family]["readout_block"]["disc_operator"]
        family_rows[family] = {
            "transport_operator": f"Tau_read^({family})",
            "disc_support_dims": disc_dims,
            "object_disc_overlap": overlap_width,
            "mean_transport_budget": mean_budget,
            "mean_novelty_pressure": mean_novelty,
            "mean_retention_risk": mean_ret,
            "formal_form": f"Tau_read^({family}) = {overlap_width:.4f} - 0.5*sigma_novel^({family}) - 1.0*sigma_ret^({family})",
            "status": "open" if mean_budget > 0.10 else "narrow" if mean_budget > 0.05 else "fragile",
        }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_family_level_transport_operator",
        },
        "family_transport_operator": {
            "core_statement": "Concept-local readout budgets can now be promoted into family-level transport operators.",
            "formal_family": family_rows,
            "high_level_form": "Tau_read_family(I) = overlap_family - lambda_n * sigma_novel_family - lambda_r * sigma_ret_family",
        },
        "mathematical_meaning": {
            "core_statement": "Readout transport is not only concept-local; it can be summarized as a family-patch operator with family-specific overlap and stress load.",
            "why_useful": "This compresses many concept-local transport budgets into a smaller set of reusable family transport laws.",
        },
        "verdict": {
            "core_answer": "Family-level transport operators can now be defined on top of concept-local transport budgets.",
            "next_theory_target": "compose family transport operators with phase operators to model switching-sensitive transport",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
