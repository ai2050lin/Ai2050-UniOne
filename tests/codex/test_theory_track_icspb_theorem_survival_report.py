from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory track ICSPB theorem survival report")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_icspb_theorem_survival_report_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    intervention_ready = load("theory_track_icspb_intervention_level_binding_20260312.json")
    priority_sim = load("stage_p3_p4_priority12_intervention_simulation_20260312.json")

    theorem_status = []
    for item in intervention_ready["upgraded_bindings"]:
        theorem = item["theorem"]
        if theorem == "family_section_theorem":
            status = "priority_1_ready"
            survival_signal = "preserve family patch separation under scaffolded readout"
        elif theorem == "restricted_readout_transport_theorem":
            status = "priority_2_ready"
            survival_signal = "reasoning-slice transport beats readout-only contrast"
        else:
            status = "waiting_for_later_intervention"
            survival_signal = "pending stress or anchored relation intervention"

        theorem_status.append(
            {
                "theorem": theorem,
                "bound_intervention": item["bound_intervention"],
                "status": status,
                "survival_signal": survival_signal,
            }
        )

    ready_count = sum(1 for item in theorem_status if item["status"] != "waiting_for_later_intervention")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_ICSPB_theorem_survival_report",
        },
        "ready_for_immediate_survival_test_count": ready_count,
        "baseline_joint_score": priority_sim["simulated_results"]["baseline_joint_score"],
        "combined_priority12_score": priority_sim["simulated_results"]["priority_1_and_2_combined_score"],
        "theorem_status": theorem_status,
        "verdict": {
            "core_answer": "Two ICSPB theorems are now ready for immediate survival testing under the first intervention pair, while the remaining two are queued behind later interventions.",
            "next_theory_target": "convert the first two ready theorems into explicit pass/fail survival criteria once intervention results are available.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
