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
        fallback = TEMP_DIR / "theory_track_current_progress_and_model_design_readiness_20260313.json"
        if prefix == "theory_track_current_progress_and_model_design_readiness_" and fallback.exists():
            return load_json(fallback)
        raise FileNotFoundError(f"missing temp json with prefix: {prefix}")
    return load_json(matches[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate ICSPB-inspired architecture proposal")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_icspb_model_architecture_proposal_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    readiness = load_latest("theory_track_current_progress_and_model_design_readiness_")
    axes = readiness["current_axes"]

    proposal = {
        "model_family_name": "ICSPB-Backbone-v1",
        "core_modules": [
            "family_patch_backbone",
            "concept_section_state",
            "relation_context_fibers",
            "stage_successor_transport_core",
            "protocol_bridge_transport_layer",
            "brain_probe_alignment_head",
            "theorem_survival_monitor",
        ],
        "update_modes": {
            "read_mode": "path-conditioned transport/access over restricted overlaps",
            "write_mode": "stress-gated admissible plastic update over the same path substrate",
        },
        "key_laws": {
            "readout_law": "restricted overlap + recurrent scaffold + protocol bridge",
            "reasoning_law": "modality-conditioned entry + family-conditioned shared reasoning slice + successor alignment",
            "update_law": "family-conditioned + stress-gated + relation-sensitive + stage-aware admissibility",
        },
        "expected_strengths": {
            "protocol": axes["protocol"],
            "successor": axes["successor"],
            "brain_alignment": axes["brain"],
        },
        "build_readiness": readiness["readiness"]["model_design_readiness"],
        "verdict": {
            "proposal_is_ready_for_prototyping": readiness["verdict"]["can_design_new_model_family"],
            "core_answer": (
                "The extracted principles are now strong enough to define a new neural architecture family whose inductive biases follow ICSPB rather than a plain uniform transformer-style geometry."
            ),
            "main_risk": "theorem survival rollback and real rolling online execution are not yet part of the model itself",
        },
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_ICSPB_model_architecture_proposal",
        },
        "proposal": proposal,
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
