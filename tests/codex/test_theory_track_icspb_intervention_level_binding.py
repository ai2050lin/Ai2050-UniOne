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
    ap = argparse.ArgumentParser(description="Theory track ICSPB intervention-level binding")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_icspb_intervention_level_binding_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    theorem_binding = load("theory_track_icspb_theorem_to_p4_binding_20260312.json")
    intervention_design = load("stage_p3_p4_joint_intervention_design_20260312.json")
    stronger_closure = load("theory_track_icspb_stronger_closure_20260312.json")

    intervention_lookup = {item["brain_side_block"]: item["name"] for item in intervention_design["interventions"]}

    upgraded_bindings = []
    for item in theorem_binding["theorem_bindings"]:
        upgraded_bindings.append(
            {
                "theorem": item["theorem"],
                "bound_falsification_block": item["bound_falsification_block"],
                "bound_intervention": intervention_lookup[item["bound_falsification_block"]],
                "binding_level": "intervention_ready",
                "survival_test": f"{item['theorem']} must remain predictive after {intervention_lookup[item['bound_falsification_block']]} is executed",
            }
        )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_ICSPB_intervention_level_binding",
        },
        "previous_theorem_closure_readiness": stronger_closure["theorem_closure_readiness"],
        "upgraded_binding_count": len(upgraded_bindings),
        "upgraded_bindings": upgraded_bindings,
        "next_closure_step": "survival_under_falsification",
        "verdict": {
            "core_answer": "ICSPB theorem candidates are now upgraded from falsification-block binding to intervention-ready binding.",
            "next_theory_target": "measure which theorems survive once the matching interventions are actually executed.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
