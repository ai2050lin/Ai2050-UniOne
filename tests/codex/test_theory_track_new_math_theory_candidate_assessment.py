from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest(prefix: str) -> Dict[str, Any]:
    matches = sorted(TEMP_DIR.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"missing temp json with prefix: {prefix}")
    return load_json(matches[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Assess the new higher-level math theory candidate")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_new_math_theory_candidate_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    synthesis = load_latest("theory_track_complete_math_theory_synthesis_")

    readiness = synthesis["theory"]["readiness"]
    inverse_ready = float(readiness["inverse_brain_encoding"])
    math_ready = float(readiness["new_math_system"])
    proto_ready = float(readiness["prototype_ready"])
    proto_online = float(readiness["prototype_online_closure"])
    persistent = float(readiness["persistent_external_system"])
    ucesd_readiness = float(readiness["ucesd_readiness"])

    strict_math_pass = ucesd_readiness >= 0.94
    generative_pass = proto_ready >= 0.96 and proto_online >= 0.95
    online_pass = persistent >= 0.99

    assessment_score = clamp01(
        0.18 * inverse_ready
        + 0.22 * math_ready
        + 0.20 * proto_ready
        + 0.20 * proto_online
        + 0.20 * persistent
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_New_Math_Theory_Candidate_Assessment",
        },
        "headline_metrics": {
            "ucesd_readiness": ucesd_readiness,
            "assessment_score": assessment_score,
            "inverse_brain_encoding_readiness": inverse_ready,
            "new_math_system_readiness": math_ready,
            "prototype_ready": proto_ready,
            "prototype_online_closure": proto_online,
            "persistent_external_system": persistent,
        },
        "verdict": {
            "strict_math_pass": strict_math_pass,
            "generative_pass": generative_pass,
            "online_pass": online_pass,
            "can_summarize_entire_system_as_new_theory": strict_math_pass and generative_pass and online_pass,
            "core_answer": (
                "UCESD can now be treated as the higher-level mathematical theory above ICSPB: "
                "ICSPB explains encoding geometry, while UCESD explains how encoding geometry, theorem survival, "
                "prototype generation, and online execution form one unified research system."
            ),
            "main_remaining_gap": "real long-term always-on execution rather than artifact-fed persistent daemon operation",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
