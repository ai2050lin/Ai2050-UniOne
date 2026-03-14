from __future__ import annotations

import argparse
import json
import math
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


def geo_mean(values: list[float]) -> float:
    vals = [max(1e-6, min(1.0, float(v))) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description="GUIT 下智能一般定义泛函块")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_guit_general_intelligence_functional_block_20260314.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    progress = load_latest("theory_track_current_progress_and_model_design_readiness*.json")
    guit = load_latest("theory_track_grand_unified_intelligence_theory_synthesis*.json")
    closure = load_latest("theory_track_grand_unified_intelligence_closure_update*.json")
    constructive = load_latest("theory_track_constructive_training_closure_assessment*.json")
    gauge = load_latest("theory_track_gauge_freedom_removal_theorem_block*.json")
    external = load_latest("theory_track_true_external_world_closure_assessment*.json")

    inverse_ready = float(progress["readiness"]["inverse_brain_encoding_readiness"])
    model_ready = float(progress["readiness"]["model_design_readiness"])
    phi_int = float(guit["theory"]["readiness"]["phi_int"])
    guit_readiness = float(guit["theory"]["readiness"]["guit_readiness"])
    constructive_score = float(constructive["headline_metrics"]["assessment_score"])
    closure_score = float(closure["headline_metrics"]["grand_unified_closure_score"])
    gauge_score = float(gauge["headline_metrics"]["gauge_freedom_removal_score"])
    external_score = float(external["headline_metrics"]["true_external_world_score"])

    phi_cap = inverse_ready
    phi_stab = constructive_score
    phi_read = clamp01(0.70 * phi_int + 0.30 * closure_score)
    phi_reason = guit_readiness
    phi_proto = model_ready
    phi_align = external_score
    phi_survival = closure_score
    phi_general = clamp01(0.55 * guit_readiness + 0.25 * phi_int + 0.20 * gauge_score)

    intelligence_general_score = geo_mean(
        [
            phi_cap,
            phi_stab,
            phi_read,
            phi_reason,
            phi_proto,
            phi_align,
            phi_survival,
            phi_general,
        ]
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_GUIT_General_Intelligence_Functional_Block",
        },
        "functional": {
            "name": "Phi_int_general",
            "definition": "general intelligence as constrained path-system viability and extensibility",
            "formal": "Phi_int_general = GM(Phi_cap, Phi_stab, Phi_read, Phi_reason, Phi_proto, Phi_align, Phi_survival, Phi_general)",
            "axes": {
                "Phi_cap": phi_cap,
                "Phi_stab": phi_stab,
                "Phi_read": phi_read,
                "Phi_reason": phi_reason,
                "Phi_proto": phi_proto,
                "Phi_align": phi_align,
                "Phi_survival": phi_survival,
                "Phi_general": phi_general,
            },
            "score": intelligence_general_score,
        },
        "verdict": {
            "general_intelligence_definition_ready": intelligence_general_score >= 0.93,
            "core_answer": (
                "Intelligence should now be defined more generally as the system's ability to form, preserve, "
                "transport, extend, align, and recover viable reasoning paths under admissible constraints."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
