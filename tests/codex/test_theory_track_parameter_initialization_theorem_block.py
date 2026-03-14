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
    ap = argparse.ArgumentParser(description="Formalize parameter initialization theorem block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_parameter_initialization_theorem_block_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    proto_online = load_latest("theory_track_prototype_online_closure_assessment_")
    daemon = load_latest("theory_track_real_persistent_external_trace_daemon_assessment_")
    train_assess = load_json(TEMP_DIR / "icspb_v2_openwebtext_real_training_curve_assessment.json")

    online_engine = float(proto_online["headline_metrics"]["online_engine_score"])
    theorem_recovery = float(proto_online["headline_metrics"]["rolling_survival_score"])
    persistent_daemon = float(daemon["headline_metrics"]["persistent_external_daemon_score"])
    training_score = float(train_assess["total_score"])

    theorem_score = clamp01(
        0.30 * online_engine
        + 0.25 * theorem_recovery
        + 0.20 * persistent_daemon
        + 0.25 * training_score
        - 0.18
    )

    theorem = {
        "name": "parameter_initialization_theorem",
        "statement": (
            "There exists an initialization family aligned with family patches, transport scaffold, and theorem monitor priors "
            "that places training inside a recoverable basin, but current evidence is not yet strong enough to claim uniqueness "
            "or closed-form theta* generation."
        ),
        "score": theorem_score,
        "strict_pass": theorem_score >= 0.90,
        "status": "partial_constructive_support" if theorem_score >= 0.65 else "weak_support",
        "main_risk": "initial basin sufficiency is plausible, but uniqueness and closed-form parameter generation remain open",
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Parameter_Initialization_Theorem_Block",
        },
        "theorem": theorem,
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
