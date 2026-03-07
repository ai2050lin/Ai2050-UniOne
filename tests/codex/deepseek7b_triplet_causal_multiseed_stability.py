#!/usr/bin/env python
"""
Triplet targeted causal multi-seed stability runner.

Run `deepseek7b_triplet_causal_targeted_scan.py` for multiple seeds and
aggregate stability statistics into a single schema-v1 payload.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from agi_research_result_schema import build_result_payload, write_result_bundle


KEY_METRICS = [
    "triplet_minimal_records",
    "triplet_counterfactual_records",
    "apple_recovery_ratio_mean",
    "king_recovery_ratio_mean",
    "queen_recovery_ratio_mean",
    "apple_specificity_margin_mean",
    "king_specificity_margin_mean",
    "queen_specificity_margin_mean",
    "global_mean_causal_margin_prob",
    "global_mean_causal_margin_seq_logprob",
    "global_positive_causal_margin_ratio",
]


def safe_mean(xs: List[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def safe_std(xs: List[float]) -> float:
    return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def run_seed(seed: int, root: Path, base_out: Path) -> Dict:
    seed_dir = base_out / f"seed_{seed}"
    cmd = [
        sys.executable,
        "tests/codex/deepseek7b_triplet_causal_targeted_scan.py",
        "--seed",
        str(seed),
        "--output-dir",
        str(seed_dir),
    ]
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(root), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"seed {seed} failed, exit code={proc.returncode}")
    out_json = seed_dir / "triplet_targeted_causal_scan.json"
    if not out_json.exists():
        raise FileNotFoundError(f"missing output for seed {seed}: {out_json}")
    payload = load_json(out_json)
    return {
        "seed": seed,
        "json": out_json,
        "payload": payload,
        "metrics": payload.get("metrics") or {},
    }


def aggregate(rows: List[Dict]) -> Dict:
    out = {
        "n_runs": len(rows),
        "seeds": [int(r["seed"]) for r in rows],
    }
    for key in KEY_METRICS:
        vals = []
        for r in rows:
            v = r["metrics"].get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        out[key] = {
            "mean": safe_mean(vals),
            "std": safe_std(vals),
            "min": float(min(vals)) if vals else 0.0,
            "max": float(max(vals)) if vals else 0.0,
        }
    return out


def render_md(payload: Dict) -> str:
    m = payload["metrics"]
    lines = [
        "# 三元组定向因果多 seed 稳定性报告",
        "",
        f"- 运行数: {m['n_runs']}",
        f"- Seeds: {', '.join(str(x) for x in m['seeds'])}",
        "",
        "## 核心稳定性",
        f"- triplet_minimal_records: mean={m['triplet_minimal_records']['mean']:.4f} std={m['triplet_minimal_records']['std']:.4f}",
        f"- triplet_counterfactual_records: mean={m['triplet_counterfactual_records']['mean']:.4f} std={m['triplet_counterfactual_records']['std']:.4f}",
        f"- global_mean_causal_margin_seq_logprob: mean={m['global_mean_causal_margin_seq_logprob']['mean']:.6f} std={m['global_mean_causal_margin_seq_logprob']['std']:.6f}",
        f"- global_positive_causal_margin_ratio: mean={m['global_positive_causal_margin_ratio']['mean']:.6f} std={m['global_positive_causal_margin_ratio']['std']:.6f}",
        "",
        "## 解释",
        "- 若 seq_logprob 边际均值 > 0 且方差较小，说明三元组因果信号具跨 seed 稳定性。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Triplet targeted causal multi-seed stability")
    ap.add_argument("--seeds", default="101,202,303,404,505")
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if not seeds:
        raise ValueError("No valid seeds provided.")

    root = Path(__file__).resolve().parents[2]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else root / f"tempdata/deepseek7b_triplet_multiseed_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for sd in seeds:
        per_seed.append(run_seed(sd, root=root, base_out=out_dir))

    agg = aggregate(per_seed)
    seed_artifacts = []
    for r in per_seed:
        seed_artifacts.append(
            {
                "seed": int(r["seed"]),
                "result_json": r["json"].resolve().relative_to(root).as_posix(),
            }
        )

    payload = build_result_payload(
        experiment_id="triplet_targeted_multiseed_stability_v1",
        title="三元组定向因果多 seed 稳定性",
        config={"seeds": seeds},
        metrics=agg,
        hypotheses=[
            {
                "id": "H_triplet_coverage_stable",
                "rule": "triplet_counterfactual_records.mean >= 3",
                "pass": bool(agg["triplet_counterfactual_records"]["mean"] >= 3),
            },
            {
                "id": "H_seq_margin_positive",
                "rule": "global_mean_causal_margin_seq_logprob.mean > 0",
                "pass": bool(agg["global_mean_causal_margin_seq_logprob"]["mean"] > 0),
            },
            {
                "id": "H_margin_ratio_nontrivial",
                "rule": "global_positive_causal_margin_ratio.mean >= 0.33",
                "pass": bool(agg["global_positive_causal_margin_ratio"]["mean"] >= 0.33),
            },
        ],
        artifacts={"per_seed": seed_artifacts},
        notes=["多 seed 稳定性用于判断三元组因果证据是否可复现。"],
    )

    paths = write_result_bundle(
        out_dir=out_dir,
        base_name="triplet_targeted_multiseed_stability",
        payload=payload,
        report_md=render_md(payload),
    )
    print(f"[OK] {paths['json']}")
    print(f"[OK] {paths['md']}")


if __name__ == "__main__":
    main()

