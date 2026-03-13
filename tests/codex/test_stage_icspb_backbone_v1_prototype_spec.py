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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ICSPB-Backbone-v1 prototype spec")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_icspb_backbone_v1_prototype_spec_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    readiness = load_latest("theory_track_current_progress_and_model_design_readiness_")
    proposal = load_latest("theory_track_icspb_model_architecture_proposal_")
    p4 = load_latest("stage_p4_online_brain_causal_execution_")

    axes = readiness["current_axes"]
    build_ready = float(readiness["readiness"]["model_design_readiness"])
    successor = float(axes["successor"])
    protocol = float(axes["protocol"])
    online_trace = float(axes["online_trace_validation"])

    spec = {
        "model_family_name": proposal["proposal"]["model_family_name"],
        "prototype_name": "ICSPB-Backbone-v1-Proto",
        "core_modules": proposal["proposal"]["core_modules"],
        "training_stages": [
            "stage_1_patch_pretrain",
            "stage_2_relation_context_fiber_binding",
            "stage_3_stage_successor_transport_alignment",
            "stage_4_protocol_bridge_online_execution_tuning",
            "stage_5_theorem_survival_monitoring",
        ],
        "loss_stack": [
            "family_patch_compactness_loss",
            "cross_family_margin_loss",
            "restricted_readout_transport_loss",
            "stage_successor_alignment_loss",
            "protocol_bridge_consistency_loss",
            "brain_probe_alignment_loss",
            "theorem_survival_regularizer",
        ],
        "monitoring_axes": {
            "protocol": protocol,
            "successor": successor,
            "online_trace_validation": online_trace,
            "brain_online_closure": float(p4["final_projection"]["brain_online_closure_score"]),
        },
        "prototype_build_readiness": build_ready,
        "verdict": {
            "prototype_spec_ready": build_ready >= 0.95,
            "core_answer": (
                "The theory is now strong enough to define a staged prototype spec whose modules, losses, and monitoring axes are all derived from ICSPB rather than from a plain dense uniform architecture."
            ),
            "main_remaining_risk": "global theorem survival rollback and real rolling online execution are not yet embedded in training itself",
        },
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "Stage_ICSPB_Backbone_v1_Prototype_Spec",
        },
        "prototype_spec": spec,
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
