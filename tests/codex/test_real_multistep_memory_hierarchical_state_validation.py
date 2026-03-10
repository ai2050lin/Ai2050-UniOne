#!/usr/bin/env python
"""
Validate the best hierarchical-state candidate found in the L=32 sweep.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np

import test_real_multistep_memory_hierarchical_state_scan as hierarchical_scan


def configs() -> Dict[str, Dict[str, object]]:
    base = hierarchical_scan.system_configs()
    return {
        "single_anchor_beta_086": base["single_anchor_beta_086"],
        "segment_summary_ref": base["gated_triple_tau_joint_ultra_oracle_segment_summary"],
        "best_hierarchical_candidate": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "joint_softroute_hardtail",
            "state_mode": "hierarchical",
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 1.0,
            "head_lr_scale": 1.0,
            "gate_lr": 0.09,
        },
    }


def run_candidate(name: str, cfg: Dict[str, object], length: int, seed: int, noise: float, dropout_p: float) -> Dict[str, float]:
    original = hierarchical_scan.system_configs
    try:
        hierarchical_scan.system_configs = lambda: {name: cfg}  # type: ignore[assignment]
        return hierarchical_scan.run_for_length(name, length, seed, noise, dropout_p)
    finally:
        hierarchical_scan.system_configs = original  # type: ignore[assignment]


def summarize(rows):
    return {
        "real_closure_score": float(np.mean([row["overall_episode_success"] for row in rows])),
        "retention_after_phase2": float(np.mean([row["retention_after_phase2"] for row in rows])),
        "gate_entropy": float(np.mean([row["gate_entropy"] for row in rows])),
        "gate_peak": float(np.mean([row["gate_peak"] for row in rows])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate best hierarchical state candidate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[24, 28, 32])
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_memory_hierarchical_state_validation_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "runtime_sec": 0.0}, "systems": {}}
    for name, cfg in configs().items():
        per_length = {}
        for length in [int(v) for v in args.lengths]:
            rows = []
            for offset in range(int(args.num_seeds)):
                rows.append(run_candidate(name, cfg, length, int(args.seed) + offset, float(args.noise), float(args.dropout_p)))
            per_length[str(length)] = summarize(rows)
        results["systems"][name] = {"config": cfg, "per_length": per_length}

    max_length = str(max(int(v) for v in args.lengths))
    base = results["systems"]["single_anchor_beta_086"]["per_length"][max_length]["real_closure_score"]
    segment = results["systems"]["segment_summary_ref"]["per_length"][max_length]["real_closure_score"]
    best = results["systems"]["best_hierarchical_candidate"]["per_length"][max_length]["real_closure_score"]
    results["gains"] = {
        "best_hierarchical_vs_single_anchor_at_max_length": float(best - base),
        "best_hierarchical_vs_segment_at_max_length": float(best - segment),
    }
    results["hypotheses"] = {
        "H1_best_hierarchical_beats_single_anchor_at_max_length": bool(best > base),
        "H2_best_hierarchical_beats_segment_at_max_length": bool(best > segment),
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
