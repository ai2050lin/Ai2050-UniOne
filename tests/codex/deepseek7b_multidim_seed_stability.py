#!/usr/bin/env python
"""
Run multi-seed stability experiments for style/logic/syntax multidim probe + causal ablation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def stats(values: List[float], z: float = 1.96) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "sem": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    arr = np.asarray(values, dtype=np.float64)
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


def run_cmd(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-pairs-per-dim", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--top-n-ablate", type=int, default=128)
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--output-dir", default="tempdata/deepseek7b_multidim_multiseed_v1")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    runs = []
    for seed in seeds:
        probe_dir = out_dir / f"seed{seed}" / "probe"
        ab_dir = out_dir / f"seed{seed}" / "ablation"
        probe_dir.mkdir(parents=True, exist_ok=True)
        ab_dir.mkdir(parents=True, exist_ok=True)

        run_cmd(
            [
                sys.executable,
                "tests/codex/deepseek7b_multidim_encoding_probe.py",
                "--model-id", args.model_id,
                "--dtype", args.dtype,
                "--max-pairs-per-dim", str(args.max_pairs_per_dim),
                "--top-k", str(args.top_k),
                "--seed", str(seed),
                "--output-dir", str(probe_dir),
            ]
        )
        run_cmd(
            [
                sys.executable,
                "tests/codex/deepseek7b_multidim_causal_ablation.py",
                "--model-id", args.model_id,
                "--dtype", args.dtype,
                "--probe-json", str(probe_dir / "multidim_encoding_probe.json"),
                "--top-n", str(args.top_n_ablate),
                "--ablate-all-positions",
                "--output-dir", str(ab_dir),
            ]
        )

        probe = load_json(probe_dir / "multidim_encoding_probe.json")
        ab = load_json(ab_dir / "multidim_causal_ablation.json")
        run = {
            "seed": seed,
            "probe_json": (probe_dir / "multidim_encoding_probe.json").as_posix(),
            "ablation_json": (ab_dir / "multidim_causal_ablation.json").as_posix(),
            "specificity_margin_style": float(probe["specificity"]["style"]["specificity_margin"]),
            "specificity_margin_logic": float(probe["specificity"]["logic"]["specificity_margin"]),
            "specificity_margin_syntax": float(probe["specificity"]["syntax"]["specificity_margin"]),
            "cross_jaccard_style_logic": float(probe["cross_dimension"]["style__logic"]["top_neuron_jaccard"]),
            "cross_jaccard_style_syntax": float(probe["cross_dimension"]["style__syntax"]["top_neuron_jaccard"]),
            "cross_jaccard_logic_syntax": float(probe["cross_dimension"]["logic__syntax"]["top_neuron_jaccard"]),
            "diag_adv_style": float(ab["diagonal_advantage"]["style"]),
            "diag_adv_logic": float(ab["diagonal_advantage"]["logic"]),
            "diag_adv_syntax": float(ab["diagonal_advantage"]["syntax"]),
        }
        runs.append(run)

    agg = {
        "specificity_margin_style": stats([r["specificity_margin_style"] for r in runs]),
        "specificity_margin_logic": stats([r["specificity_margin_logic"] for r in runs]),
        "specificity_margin_syntax": stats([r["specificity_margin_syntax"] for r in runs]),
        "diag_adv_style": stats([r["diag_adv_style"] for r in runs]),
        "diag_adv_logic": stats([r["diag_adv_logic"] for r in runs]),
        "diag_adv_syntax": stats([r["diag_adv_syntax"] for r in runs]),
        "cross_jaccard_style_logic": stats([r["cross_jaccard_style_logic"] for r in runs]),
        "cross_jaccard_style_syntax": stats([r["cross_jaccard_style_syntax"] for r in runs]),
        "cross_jaccard_logic_syntax": stats([r["cross_jaccard_logic_syntax"] for r in runs]),
    }

    hypotheses = {
        "H1_style_specificity_positive": {
            "metric": "specificity_margin_style",
            "decision": "pass" if float(agg["specificity_margin_style"]["ci_low"]) > 0 else "fail",
            "rule": "95% CI lower bound > 0",
        },
        "H2_logic_specificity_positive": {
            "metric": "specificity_margin_logic",
            "decision": "pass" if float(agg["specificity_margin_logic"]["ci_low"]) > 0 else "fail",
            "rule": "95% CI lower bound > 0",
        },
        "H3_syntax_specificity_positive": {
            "metric": "specificity_margin_syntax",
            "decision": "pass" if float(agg["specificity_margin_syntax"]["ci_low"]) > 0 else "fail",
            "rule": "95% CI lower bound > 0",
        },
        "H4_style_diag_adv_positive": {
            "metric": "diag_adv_style",
            "decision": "pass" if float(agg["diag_adv_style"]["ci_low"]) > 0 else "fail",
            "rule": "95% CI lower bound > 0",
        },
        "H5_logic_diag_adv_positive": {
            "metric": "diag_adv_logic",
            "decision": "pass" if float(agg["diag_adv_logic"]["ci_low"]) > 0 else "fail",
            "rule": "95% CI lower bound > 0",
        },
        "H6_syntax_diag_adv_positive": {
            "metric": "diag_adv_syntax",
            "decision": "pass" if float(agg["diag_adv_syntax"]["ci_low"]) > 0 else "fail",
            "rule": "95% CI lower bound > 0",
        },
    }

    result = {
        "config": {
            "model_id": args.model_id,
            "dtype": args.dtype,
            "max_pairs_per_dim": int(args.max_pairs_per_dim),
            "top_k": int(args.top_k),
            "top_n_ablate": int(args.top_n_ablate),
            "seeds": seeds,
        },
        "n_runs": len(runs),
        "runs": runs,
        "aggregate": agg,
        "hypotheses": hypotheses,
    }

    json_path = out_dir / "multidim_multiseed_stability.json"
    md_path = out_dir / "MULTIDIM_MULTISEED_STABILITY_REPORT.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Multidim Multi-seed Stability Report",
        "",
        f"- n_runs: {len(runs)}",
        f"- max_pairs_per_dim: {args.max_pairs_per_dim} (total prompts >= {args.max_pairs_per_dim * 3})",
        "",
        "## Aggregate (95% CI)",
    ]
    for k, v in agg.items():
        lines.append(f"- {k}: mean={v['mean']:.6f}, 95%CI=[{v['ci_low']:.6f}, {v['ci_high']:.6f}]")
    lines.extend(["", "## Hypotheses"])
    for hk, hv in hypotheses.items():
        lines.append(f"- {hk}: {hv['decision']} ({hv['metric']} | {hv['rule']})")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": json_path.as_posix(), "markdown": md_path.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()

