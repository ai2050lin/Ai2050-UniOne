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
    ap = argparse.ArgumentParser(description="Theory-track modality-unified reasoning law")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_modality_unified_reasoning_law_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    clue = load("theory_track_conscious_modality_unification_clue_20260312.json")
    icspb = load("theory_track_icspb_axiom_layer_20260312.json")
    path_law = load("theory_track_path_conditioned_encoding_law_20260312.json")
    inventory = load("theory_track_inventory_unified_system_formalization_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_modality_unified_reasoning_law",
        },
        "law_name": "modality-conditioned family-unified reasoning law",
        "high_level_form": (
            "Reason(c, m_in -> m_out) = "
            "(Lift_mod^(f_c)(x_m), Section_c^(f_c), W_reason^(f_c), Tau_reason(c, m_in -> m_out), chi_A, chi_M)"
        ),
        "components": {
            "Lift_mod^(f_c)(x_m)": "modality-conditioned entry operator from raw modality stream into a family patch",
            "Section_c^(f_c)": "concept section carried on the family patch",
            "W_reason^(f_c)": "family-conditioned shared reasoning slice",
            "Tau_reason(c, m_in -> m_out)": "path-conditioned transport budget for reasoning and cross-modal readout",
            "chi_A": "admissibility gate inherited from A(I)",
            "chi_M": "viability gate inherited from M_feas(I)",
        },
        "interpretation": {
            "not_one_global_loop": "A fully shared global central loop is not supported by the current evidence.",
            "preferred_mechanism": clue["core_inference"]["more_plausible_form"],
            "icspb_link": icspb["compressed_view"]["core_claim"],
            "system_link": inventory["unified_object"]["high_level_form"],
            "path_link": path_law["path_conditioned_encoding_law"]["high_level_form"],
        },
        "verdict": {
            "core_answer": "A plausible consciousness-compatible law is now available: modality-conditioned entry plus family-unified reasoning slices inside ICSPB.",
            "next_theory_target": "derive falsifiable predictions for modality substitution, reasoning preservation, and cross-modal transfer.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
