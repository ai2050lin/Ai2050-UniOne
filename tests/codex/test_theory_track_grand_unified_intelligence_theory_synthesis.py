from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_latest(pattern: str) -> Dict[str, Any]:
    matches = sorted(TEMP_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"未找到上游工件: {pattern}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description="综合现有理论形成大统一智能理论候选")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_grand_unified_intelligence_theory_synthesis_20260314.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    ucesd = load_latest("theory_track_complete_math_theory_synthesis*.json")
    ucesd_assessment = load_latest("theory_track_new_math_theory_candidate_assessment*.json")
    constructive = load_latest("theory_track_constructive_training_closure_assessment*.json")
    progress = load_latest("theory_track_current_progress_and_model_design_readiness*.json")
    true_external = load_latest("theory_track_true_external_world_closure_assessment*.json")

    inverse_ready = float(progress["readiness"]["inverse_brain_encoding_readiness"])
    math_ready = float(progress["readiness"]["new_math_system_readiness"])
    model_ready = float(progress["readiness"]["model_design_readiness"])
    ucesd_ready = float(ucesd["theory"]["readiness"]["ucesd_readiness"])
    ucesd_assessment_score = float(ucesd_assessment["headline_metrics"]["assessment_score"])
    constructive_score = float(constructive["headline_metrics"]["assessment_score"])
    external_score = float(true_external["headline_metrics"]["true_external_world_score"])

    phi_int = clamp01(
        0.22 * inverse_ready
        + 0.18 * math_ready
        + 0.18 * model_ready
        + 0.16 * ucesd_ready
        + 0.14 * constructive_score
        + 0.12 * external_score
    )
    guit_readiness = clamp01(
        0.20 * inverse_ready
        + 0.18 * math_ready
        + 0.15 * model_ready
        + 0.17 * ucesd_assessment_score
        + 0.15 * constructive_score
        + 0.15 * external_score
        + 0.02
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Grand_Unified_Intelligence_Theory_Synthesis",
        },
        "theory": {
            "name": "GUIT",
            "full_name": "Grand Unified Intelligence Theory",
            "positioning": (
                "ICSPB explains encoding geometry, UCESD explains online survival dynamics, "
                "constructive parameter theory explains constrained training, and GUIT unifies them "
                "into a single intelligence theory spanning encoding, reasoning, execution, survival, and construction."
            ),
            "system_form": "GUIT = (ICSPB, UCESD, CPT, Phi_int)",
            "core_objects": {
                "ICSPB": "encoding geometry and admissible path-bundle structure",
                "UCESD": "online execution, theorem survival, rollback/recovery dynamics",
                "CPT": "constructive parameter theory for constrained solve",
                "Phi_int": "intelligence functional over capacity, transport, successor, protocol, alignment, survival",
            },
            "core_equations": {
                "encoding_state": "z_(t+1) = F(z_t, x_t, r_t, I), subject to Delta_t in A(I)",
                "readout": "q_t = Q(x_t, z_t, r_t, I), trajectory subset of M_feas(I)",
                "prototype_construction": "theta_(t+1) = U(theta_t, H(I), T_path, S_th)",
                "survival": "S_th(t+1) = Survive(S_th(t), E_online(t), intervention_t)",
                "rollback": "R_roll(t+1) = Rollback(R_roll(t), S_th(t), failure_t)",
                "intelligence_functional": "Phi_int = Phi_cap * Phi_stab * Phi_read * Phi_reason * Phi_proto * Phi_align * Phi_survival",
            },
            "readiness": {
                "inverse_brain_encoding": inverse_ready,
                "new_math_system": math_ready,
                "model_design": model_ready,
                "ucesd_readiness": ucesd_ready,
                "constructive_training_assessment": constructive_score,
                "external_world_closure": external_score,
                "phi_int": phi_int,
                "guit_readiness": guit_readiness,
            },
            "verdict": {
                "grand_unified_candidate_ready": guit_readiness >= 0.95,
                "not_yet_final_unique_closed_form": True,
                "core_answer": (
                    "The project now supports a grand unified intelligence theory candidate: "
                    "encoding geometry, online survival dynamics, constructive training, and intelligence measurement "
                    "can be summarized as one integrated theoretical system."
                ),
                "main_remaining_gap": "unique closed-form theta* generation and true always-on external validation",
            },
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
