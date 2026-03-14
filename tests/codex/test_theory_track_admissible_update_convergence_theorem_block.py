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


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Formalize admissible update convergence theorem block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_admissible_update_convergence_theorem_block_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    persistent_train = load_json(TEMP_DIR / "icspb_v2_openwebtext_persistent_continual_daemon_assessment.json")
    real_curve = load_json(TEMP_DIR / "icspb_v2_openwebtext_real_training_curve_assessment.json")
    proto_online = load_json(TEMP_DIR / "theory_track_prototype_online_closure_assessment_20260313.json")

    daemon_stability = float(persistent_train["daemon_stability"])
    online_delta = float(persistent_train["online_delta_total"])
    rollback_error = float(persistent_train["rollback_error"])
    train_score = float(real_curve["total_score"])
    online_engine = float(proto_online["headline_metrics"]["online_engine_score"])

    theorem_score = clamp01(
        0.30 * daemon_stability
        + 0.25 * (1.0 - min(1.0, online_delta))
        + 0.15 * (1.0 - min(1.0, rollback_error))
        + 0.15 * train_score
        + 0.15 * online_engine
        - 0.20
    )

    theorem = {
        "name": "admissible_update_convergence_theorem",
        "statement": (
            "Under guarded-write, stable-read, rollback, and theorem survival constraints, admissible online updates appear to "
            "converge inside a recoverable manifold, but current evidence still falls short of a strong global convergence proof."
        ),
        "score": theorem_score,
        "strict_pass": theorem_score >= 0.90,
        "status": "partial_dynamic_support" if theorem_score >= 0.65 else "weak_support",
        "main_risk": "current evidence shows recoverable stability, not yet full global convergence under unrestricted real data flow",
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Admissible_Update_Convergence_Theorem_Block",
        },
        "theorem": theorem,
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
