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
    ap = argparse.ArgumentParser(description="Assess current progress and model-design readiness")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_current_progress_and_model_design_readiness_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    protocol = load_latest("stage_protocol_bridge_transport_online_execution_")
    p4 = load_latest("stage_p4_online_brain_causal_execution_")
    route = load_latest("theory_track_inverse_brain_math_route_")
    cross_model = load_latest("stage_cross_model_real_long_chain_trace_capture_")

    p_final = protocol["final_projection"]
    b_final = p4["final_projection"]
    c_final = cross_model["final_projection"]
    route_prog = route["route_progress"]

    protocol_score = float(p_final["protocol"])
    successor_score = float(b_final["successor"])
    brain_score = float(b_final["brain"])
    online_trace = float(b_final["online_trace_validation"])
    theorem_recovery = float(b_final["theorem_survival_recovery"])

    inverse_reconstruction = clamp01(
        0.20 * float(route_prog["dnn_extraction_to_inverse_brain_encoding"])
        + 0.20 * float(c_final["successor"])
        + 0.20 * protocol_score
        + 0.20 * online_trace
        + 0.20 * theorem_recovery
    )
    new_math_readiness = clamp01(
        0.25 * float(route_prog["dnn_extraction_to_new_math_closure"])
        + 0.20 * theorem_recovery
        + 0.20 * online_trace
        + 0.20 * brain_score
        + 0.15 * protocol_score
    )
    model_design_readiness = clamp01(
        0.22 * protocol_score
        + 0.22 * successor_score
        + 0.18 * online_trace
        + 0.18 * theorem_recovery
        + 0.20 * inverse_reconstruction
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_current_progress_and_model_design_readiness",
        },
        "current_axes": {
            "protocol": protocol_score,
            "successor": successor_score,
            "brain": brain_score,
            "online_trace_validation": online_trace,
            "theorem_survival_recovery": theorem_recovery,
        },
        "readiness": {
            "inverse_brain_encoding_readiness": inverse_reconstruction,
            "new_math_system_readiness": new_math_readiness,
            "model_design_readiness": model_design_readiness,
        },
        "verdict": {
            "can_design_new_model_family": model_design_readiness >= 0.85,
            "core_answer": (
                "Current progress is strong enough to propose a new model family driven by the extracted coding principles, but not yet strong enough to claim the theory is fully closed."
            ),
            "main_remaining_gap": "global theorem survival rollback and real rolling online execution",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
