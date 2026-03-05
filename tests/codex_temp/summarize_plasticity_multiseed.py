#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stats(xs: List[float], z: float = 1.96) -> Dict[str, float]:
    if not xs:
        return {"n": 0, "mean": 0.0, "std": 0.0, "sem": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    arr = np.asarray(xs, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sem = float(std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "n": int(len(arr)),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": float(mean - z * sem),
        "ci_high": float(mean + z * sem),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize plasticity benchmark over multiseed runs")
    ap.add_argument(
        "--pattern",
        default="tempdata/deepseek7b_plasticity_efficiency_benchmark_v2_seed*/plasticity_efficiency_benchmark.json",
    )
    ap.add_argument("--output-json", default="tempdata/deepseek7b_plasticity_efficiency_benchmark_v2_multiseed_summary.json")
    ap.add_argument("--output-md", default="tempdata/deepseek7b_plasticity_efficiency_benchmark_v2_multiseed_summary.md")
    args = ap.parse_args()

    files = sorted(Path(".").glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.pattern}")

    runs = []
    all_steps = set()
    for fp in files:
        d = read_json(fp)
        step_stats = d.get("sgd_step_stats", {})
        all_steps.update(int(k) for k in step_stats.keys())
        runs.append(
            {
                "path": fp.as_posix(),
                "seed": int(fp.parent.name.split("seed")[-1]) if "seed" in fp.parent.name else None,
                "hebbian_one_shot_acc": float(d.get("hebbian_one_shot_acc", 0.0)),
                "steps_to_match_hebbian": d.get("steps_to_match_hebbian", None),
                "sgd_step_mean_acc": {int(k): float(v.get("mean_acc", 0.0)) for k, v in step_stats.items()},
            }
        )

    step_list = sorted(all_steps)
    step_curve_stats = {}
    for s in step_list:
        vals = [float(r["sgd_step_mean_acc"].get(s, 0.0)) for r in runs]
        step_curve_stats[s] = stats(vals)

    hebbian_stats = stats([float(r["hebbian_one_shot_acc"]) for r in runs])
    # not reached is encoded as null; treat as +inf in report
    steps_to_match_vals = [float(r["steps_to_match_hebbian"]) for r in runs if r["steps_to_match_hebbian"] is not None]
    steps_to_match_stats = stats(steps_to_match_vals) if steps_to_match_vals else None
    not_reached_ratio = float(
        np.mean([1.0 if r["steps_to_match_hebbian"] is None else 0.0 for r in runs])
    )

    summary = {
        "pattern": args.pattern,
        "n_runs": len(runs),
        "hebbian_one_shot_acc_stats": hebbian_stats,
        "sgd_step_curve_stats": {int(k): v for k, v in step_curve_stats.items()},
        "steps_to_match_stats": steps_to_match_stats,
        "not_reached_ratio": not_reached_ratio,
        "runs": runs,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Plasticity Efficiency Multiseed Summary",
        "",
        f"- Runs: {len(runs)}",
        f"- Pattern: `{args.pattern}`",
        f"- Hebbian one-shot acc (mean): {hebbian_stats['mean']:.6f}",
        f"- Not reached ratio (SGD fails to match Hebbian): {not_reached_ratio:.4f}",
        "",
        "## SGD Curve Stats",
    ]
    for s in step_list:
        st = step_curve_stats[s]
        lines.append(f"- step={s}: mean={st['mean']:.6f}, std={st['std']:.6f}, 95%CI=[{st['ci_low']:.6f}, {st['ci_high']:.6f}]")

    lines.extend(["", "## Per Run"])
    for r in runs:
        step_repr = ", ".join([f"{k}:{r['sgd_step_mean_acc'].get(k,0):.4f}" for k in step_list])
        lines.append(
            f"- seed={r['seed']}, hebbian={r['hebbian_one_shot_acc']:.6f}, "
            f"steps_to_match={r['steps_to_match_hebbian']}, sgd=[{step_repr}]"
        )

    out_md = Path(args.output_md)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] JSON: {out_json}")
    print(f"[OK] MD: {out_md}")


if __name__ == "__main__":
    main()

