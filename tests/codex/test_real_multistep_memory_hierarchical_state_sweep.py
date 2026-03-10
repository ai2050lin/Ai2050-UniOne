#!/usr/bin/env python
"""
Focused sweep for task block A.

We only optimize for the hardest point: L=32.
Goal:
- find whether some hierarchical-state configuration can beat the current
  single-anchor baseline at max length.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import test_real_multistep_memory_hierarchical_state_scan as hierarchical_scan


def candidate_configs() -> List[Dict[str, object]]:
    manual = [
        ("joint_ultra_oracle", 0.16, 0.96, 0.072, 1.0),
        ("joint_ultra_oracle", 0.16, 1.00, 0.080, 1.0),
        ("joint_ultra_oracle", 0.14, 1.05, 0.085, 1.0),
        ("joint_ultra_oracle", 0.17, 1.00, 0.090, 1.0),
        ("joint_softroute_hardtail", 0.16, 0.96, 0.072, 1.0),
        ("joint_softroute_hardtail", 0.16, 1.00, 0.080, 1.0),
        ("joint_softroute_hardtail", 0.14, 1.05, 0.085, 1.0),
        ("joint_softroute_hardtail", 0.17, 1.00, 0.090, 1.0),
        ("joint_ultra_oracle", 0.18, 0.96, 0.072, 2.0),
        ("joint_softroute_hardtail", 0.18, 0.96, 0.072, 2.0),
    ]
    rows = []
    for policy, stability, head_lr_scale, gate_lr, phase2_replay_stride in manual:
        rows.append(
            {
                "betas": [0.50, 0.80, 0.92],
                "use_gate": True,
                "policy": policy,
                "state_mode": "hierarchical",
                "stability": stability,
                "phase1_replay_stride": 1.0,
                "phase2_replay_stride": phase2_replay_stride,
                "head_lr_scale": head_lr_scale,
                "gate_lr": gate_lr,
            }
        )
    return rows


def run_candidate(cfg: Dict[str, object], seed: int, noise: float, dropout_p: float, length: int) -> Dict[str, float]:
    original = hierarchical_scan.system_configs
    try:
        hierarchical_scan.system_configs = lambda: {"candidate": cfg}  # type: ignore[assignment]
        return hierarchical_scan.run_for_length("candidate", length, seed, noise, dropout_p)
    finally:
        hierarchical_scan.system_configs = original  # type: ignore[assignment]


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep hierarchical-state configs at L=32")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_memory_hierarchical_state_sweep_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    baseline_cfg = hierarchical_scan.system_configs()["single_anchor_beta_086"]
    baseline_rows = [run_candidate(baseline_cfg, int(args.seed) + offset, float(args.noise), float(args.dropout_p), int(args.length)) for offset in range(int(args.num_seeds))]
    baseline_mean = float(np.mean([row["overall_episode_success"] for row in baseline_rows]))

    ranking = []
    for idx, cfg in enumerate(candidate_configs()):
        rows = [run_candidate(cfg, int(args.seed) + offset, float(args.noise), float(args.dropout_p), int(args.length)) for offset in range(int(args.num_seeds))]
        mean_closure = float(np.mean([row["overall_episode_success"] for row in rows]))
        mean_retention = float(np.mean([row["retention_after_phase2"] for row in rows]))
        ranking.append(
            {
                "candidate_id": idx,
                "config": cfg,
                "mean_closure_score": mean_closure,
                "mean_retention_score": mean_retention,
                "gain_vs_single_anchor": float(mean_closure - baseline_mean),
            }
        )
        print(f"[candidate {idx}] closure={mean_closure:.4f} gain={mean_closure - baseline_mean:+.4f}")

    ranking.sort(key=lambda row: row["mean_closure_score"], reverse=True)
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "length": int(args.length),
            "runtime_sec": float(time.time() - t0),
        },
        "baseline": {
            "config": baseline_cfg,
            "mean_closure_score": baseline_mean,
        },
        "ranking": ranking[:12],
        "best": ranking[0],
        "hypotheses": {
            "H1_some_hierarchical_candidate_beats_single_anchor_at_L32": bool(ranking[0]["mean_closure_score"] > baseline_mean),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["best"], ensure_ascii=False, indent=2))
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
