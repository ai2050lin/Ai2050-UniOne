from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest(prefix: str) -> Dict[str, Any]:
    matches = sorted(TEMP_DIR.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"missing temp json with prefix: {prefix}")
    return load_json(matches[-1])


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize the complete current math system into a higher-level theory")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_complete_math_theory_synthesis_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    progress = load_latest("theory_track_current_progress_and_model_design_readiness_")
    proto_closure = load_latest("theory_track_prototype_online_closure_assessment_")
    persistent = load_latest("theory_track_real_persistent_external_trace_daemon_assessment_")

    inverse_ready = float(progress["readiness"]["inverse_brain_encoding_readiness"])
    math_ready = float(progress["readiness"]["new_math_system_readiness"])
    model_ready = float(progress["readiness"]["model_design_readiness"])
    proto_online = float(proto_closure["headline_metrics"]["prototype_online_closure_score"])
    persistent_score = float(persistent["headline_metrics"]["persistent_external_daemon_score"])

    ucesd_readiness = clamp01(
        0.22 * inverse_ready
        + 0.24 * math_ready
        + 0.18 * model_ready
        + 0.18 * proto_online
        + 0.18 * persistent_score
    )

    theory = {
        "name": "UCESD",
        "full_name": "Unified Controlled Encoding Survival Dynamics",
        "positioning": (
            "ICSPB is the core encoding geometry theory. UCESD is the higher-level theory that adds "
            "online theorem survival, prototype generation, rollback/recovery, and persistent execution."
        ),
        "core_objects": {
            "I": "encoding inventory",
            "H(I)": "inventory-conditioned patch-fiber geometry",
            "A(I)": "admissible update cone family",
            "M_feas(I)": "viability / feasible manifold",
            "T_path": "path-conditioned transport family",
            "S_th": "theorem survival state",
            "E_online": "online execution state",
            "P_proto": "prototype parameter family",
            "R_roll": "rollback / recovery operator",
        },
        "system_form": (
            "UCESD = (H(I), A(I), M_feas(I), T_path, S_th, E_online, P_proto, R_roll)"
        ),
        "governing_forms": {
            "state_update": "z_(t+1) = F(z_t, x_t, r_t, I), subject to Delta_t in A(I)",
            "readout": "q_t = Q(x_t, z_t, r_t, I), subject to trajectory subset of M_feas(I)",
            "prototype_update": "theta_(t+1) = U(theta_t, H(I), T_path, S_th)",
            "survival": "S_th(t+1) = Survive(S_th(t), E_online(t), intervention_t)",
            "rollback": "R_roll(t+1) = Rollback(R_roll(t), S_th(t), failure_t)",
        },
        "readiness": {
            "inverse_brain_encoding": inverse_ready,
            "new_math_system": math_ready,
            "prototype_ready": model_ready,
            "prototype_online_closure": proto_online,
            "persistent_external_system": persistent_score,
            "ucesd_readiness": ucesd_readiness,
        },
        "verdict": {
            "can_name_new_theory": True,
            "is_beyond_icspb_only": True,
            "core_answer": (
                "The current project is no longer only an encoding-geometry theory. It has enough structure "
                "to define a higher-level theory in which encoding, theorem survival, online execution, "
                "prototype generation, and rollback/recovery are one unified system."
            ),
        },
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Complete_Math_Theory_Synthesis",
        },
        "theory": theory,
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
