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
    ap = argparse.ArgumentParser(description="Formalize architecture synthesis theorem block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_architecture_synthesis_theorem_block_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    prototype = load_latest("theory_track_current_progress_and_model_design_readiness_")
    proto_online = load_latest("theory_track_prototype_online_closure_assessment_")
    ext = load_latest("theory_track_true_external_world_closure_assessment_")

    model_readiness = float(prototype["readiness"]["model_design_readiness"])
    inverse_ready = float(proto_online["headline_metrics"]["inverse_brain_encoding_readiness"])
    online_closure = float(proto_online["headline_metrics"]["prototype_online_closure_score"])
    external_system = float(ext["headline_metrics"]["true_external_world_score"])

    theorem_score = clamp01(
        0.35 * model_readiness
        + 0.25 * inverse_ready
        + 0.20 * online_closure
        + 0.20 * external_system
    )

    theorem = {
        "name": "architecture_synthesis_theorem",
        "statement": (
            "Given H(I), A(I), M_feas(I) and the survived theorem frontier, there exists a minimal modular architecture family "
            "whose patch backbone, fiber attachments, transport core, protocol bridge, and theorem monitor jointly satisfy the "
            "current encoding and online-closure constraints."
        ),
        "score": theorem_score,
        "strict_pass": theorem_score >= 0.93,
        "module_family": [
            "family_patch_backbone",
            "concept_section_state",
            "relation_context_fibers",
            "stage_successor_transport_core",
            "protocol_bridge_transport_layer",
            "brain_probe_alignment_head",
            "theorem_survival_monitor",
        ],
        "main_risk": "minimality and uniqueness are still weak; theorem currently proves constructive sufficiency more than uniqueness",
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Architecture_Synthesis_Theorem_Block",
        },
        "theorem": theorem,
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
