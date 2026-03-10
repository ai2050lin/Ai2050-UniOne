#!/usr/bin/env python
"""
Task block A:
long-horizon validation for the learnable state machine.

This extends the current best A-system to longer horizons and higher-seed
validation, asking whether its advantage survives beyond the first win at L=32.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np

import test_real_multistep_memory_learnable_state_machine as lsm


def configs() -> Dict[str, Dict[str, object]]:
    base = lsm.system_configs()
    return {
        "single_anchor_beta_086": base["single_anchor_beta_086"],
        "learnable_state_machine_h12": base["learnable_state_machine_h12"],
    }


def summarize(rows):
    return {
        "real_closure_score": float(np.mean([row["overall_episode_success"] for row in rows])),
        "retention_after_phase2": float(np.mean([row["retention_after_phase2"] for row in rows])),
        "gate_entropy": float(np.mean([row["gate_entropy"] for row in rows])),
        "gate_peak": float(np.mean([row["gate_peak"] for row in rows])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Long-horizon validation for learnable state machine")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.36)
    ap.add_argument("--dropout-p", type=float, default=0.15)
    ap.add_argument("--lengths", type=int, nargs="+", default=[32, 40, 48])
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_memory_learnable_state_machine_long_validation_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    lengths = [int(v) for v in args.lengths]
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "lengths": lengths,
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "runtime_sec": 0.0,
        },
        "systems": {},
    }

    for name, cfg in configs().items():
        per_length = {}
        for length in lengths:
            rows = []
            for offset in range(int(args.num_seeds)):
                rows.append(lsm.run_for_length(name, length, int(args.seed) + offset, float(args.noise), float(args.dropout_p)))
            per_length[str(length)] = summarize(rows)
        results["systems"][name] = {"config": cfg, "per_length": per_length}

    deltas = {}
    for length in lengths:
        base_score = results["systems"]["single_anchor_beta_086"]["per_length"][str(length)]["real_closure_score"]
        learned_score = results["systems"]["learnable_state_machine_h12"]["per_length"][str(length)]["real_closure_score"]
        deltas[str(length)] = float(learned_score - base_score)

    results["gains"] = {
        "per_length_vs_single_anchor": deltas,
        "mean_gain_vs_single_anchor": float(np.mean(list(deltas.values()))),
        "min_gain_vs_single_anchor": float(min(deltas.values())),
    }
    results["hypotheses"] = {
        "H1_learned_beats_single_anchor_on_average": bool(results["gains"]["mean_gain_vs_single_anchor"] > 0.0),
        "H2_learned_beats_single_anchor_at_all_lengths": bool(results["gains"]["min_gain_vs_single_anchor"] > 0.0),
    }
    results["meta"]["runtime_sec"] = float(time.time() - t0)

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["gains"], ensure_ascii=False, indent=2))
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
