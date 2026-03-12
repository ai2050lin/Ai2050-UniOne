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
    ap = argparse.ArgumentParser(description="Theory-track stress-coupled write/read law")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_stress_coupled_write_read_law_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stress = load("theory_track_inventory_stress_profiling_20260312.json")
    path_law = load("theory_track_path_conditioned_encoding_law_20260312.json")
    inv_A = load("theory_track_inventory_to_A_coupling_20260312.json")

    stress_rows = stress["stress_rows"]
    concept_entries = path_law["concept_path_entries"]

    open_write = 0
    guarded_write = 0
    stable_read = 0
    write_read_profiles: dict[str, dict] = {}

    for concept, row in stress_rows.items():
        novelty = float(row["novelty_pressure"])
        retention = float(row["retention_risk"])
        centered_norm = float(row["centered_norm"])
        family = str(row["family"])
        path_sig = concept_entries[concept]["path_signature"]

        write_gain = max(0.0, 1.0 - novelty * 8.0)
        read_gain = max(0.0, 1.0 - (retention * 10.0 + novelty * 2.0))
        write_mode = "open" if novelty < 0.01 else "guarded"
        read_mode = "stable" if retention <= 0.005 else "fragile"

        if write_mode == "open":
            open_write += 1
        else:
            guarded_write += 1
        if read_mode == "stable":
            stable_read += 1

        write_read_profiles[concept] = {
            "family": family,
            "novelty_pressure": novelty,
            "retention_risk": retention,
            "write_gain": float(write_gain),
            "read_gain": float(read_gain),
            "write_mode": write_mode,
            "read_mode": read_mode,
            "memory_operator": path_sig["memory_operator"],
            "identity_operator": path_sig["identity_operator"],
            "local_state_radius": centered_norm,
        }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_stress_coupled_write_read_law",
        },
        "stress_coupled_write_read_law": {
            "core_statement": "Write and read must be jointly gated by concept-local novelty pressure and retention risk.",
            "formal_form": "WR(c) = (g_write(c), g_read(c)) where g_write = psi_w(sigma_novel, delta_c), g_read = psi_r(sigma_ret, sigma_novel, delta_c)",
            "expanded_form": {
                "write_gate": "g_write(c) = max(0, 1 - alpha_n * sigma_novel(c))",
                "read_gate": "g_read(c) = max(0, 1 - alpha_r * sigma_ret(c) - beta_n * sigma_novel(c))",
                "update_constraint": "Delta_write(c) in A(I) only if g_write(c) > tau_w ; readout(c) is stable only if g_read(c) > tau_r",
            },
            "A_link": inv_A["inventory_conditioned_form"],
        },
        "concept_profiles": write_read_profiles,
        "headline_metrics": {
            "concept_count": int(len(write_read_profiles)),
            "open_write_count": int(open_write),
            "guarded_write_count": int(guarded_write),
            "stable_read_count": int(stable_read),
            "mean_novelty_pressure": stress["headline_metrics"]["mean_novelty_pressure"],
            "mean_retention_risk": stress["headline_metrics"]["mean_retention_risk"],
        },
        "mathematical_meaning": {
            "core_answer": "Q2 and Q3 should now be treated as stress-coupled gates over the local path, not as separate generic mechanisms.",
            "why_it_helps": [
                "binds write/read directly to inventory stress instead of treating them as free parameters",
                "explains why novelty phases should narrow write and why read stability depends on retention budget",
                "gives engineering a direct write/read filtering rule for future update-law searches",
            ],
        },
        "verdict": {
            "core_answer": "The write/read side of the theory track can now be rewritten as a stress-coupled local path law.",
            "next_theory_target": "push this law into update-law engineering filters and novelty/retention stress experiments",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
