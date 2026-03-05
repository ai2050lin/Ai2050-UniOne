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
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
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
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def pass_by_lower_ci(stat: Dict[str, float], threshold: float) -> bool:
    return float(stat.get("ci_low", 0.0)) > threshold


def main() -> None:
    ap = argparse.ArgumentParser(description="Build falsifiable stage report from multiseed v4 outputs")
    ap.add_argument("--pattern", default="tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed*/mass_noun_encoding_scan.json")
    ap.add_argument(
        "--plasticity-summary-json",
        default="tempdata/deepseek7b_plasticity_efficiency_benchmark_v2_multiseed_summary.json",
    )
    ap.add_argument(
        "--prompt-bootstrap-json",
        default="tempdata/deepseek7b_prompt_bootstrap_causal_stability_v1/prompt_bootstrap_causal_stability.json",
    )
    ap.add_argument("--output-json", default="tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_falsifiable_report.json")
    ap.add_argument("--output-md", default="tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_falsifiable_report.md")
    args = ap.parse_args()

    files = sorted(Path(".").glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {args.pattern}")

    runs = []
    for fp in files:
        data = read_json(fp)
        score = data.get("mechanism_scorecard", {})
        agg = (data.get("causal_ablation") or {}).get("aggregate", {})
        mc = ((data.get("causal_ablation") or {}).get("minimal_circuit") or {}).get("aggregate", {})
        cf = ((data.get("causal_ablation") or {}).get("counterfactual_validation") or {}).get("aggregate", {})
        runs.append(
            {
                "path": fp.as_posix(),
                "seed": int(fp.parent.name.split("seed")[-1]) if "seed" in fp.parent.name else None,
                "overall_score": float(score.get("overall_score", 0.0)),
                "grade": str(score.get("grade", "unknown")),
                "causal_seq_margin": float(agg.get("mean_causal_margin_seq_logprob", 0.0)),
                "causal_seq_z": float(agg.get("causal_margin_seq_logprob_z", 0.0)),
                "counterfactual_margin": float(cf.get("mean_specificity_margin_seq_logprob", 0.0)),
                "counterfactual_z": float(cf.get("specificity_margin_z", 0.0)),
                "mcs_recovery": float(mc.get("mean_recovery_ratio", 0.0)),
                "mcs_subset_size": float(mc.get("mean_subset_size", 0.0)),
            }
        )

    metrics = {
        "overall_score": stats([r["overall_score"] for r in runs]),
        "causal_seq_margin": stats([r["causal_seq_margin"] for r in runs]),
        "causal_seq_z": stats([r["causal_seq_z"] for r in runs]),
        "counterfactual_margin": stats([r["counterfactual_margin"] for r in runs]),
        "counterfactual_z": stats([r["counterfactual_z"] for r in runs]),
        "mcs_recovery": stats([r["mcs_recovery"] for r in runs]),
        "mcs_subset_size": stats([r["mcs_subset_size"] for r in runs]),
    }

    hypotheses = {
        "H1_seq_causal_margin_positive": {
            "threshold": 0.0,
            "metric": "causal_seq_margin",
            "decision": "pass" if pass_by_lower_ci(metrics["causal_seq_margin"], 0.0) else "fail",
            "rule": "95% CI lower bound > 0",
        },
        "H2_counterfactual_margin_positive": {
            "threshold": 0.0,
            "metric": "counterfactual_margin",
            "decision": "pass" if pass_by_lower_ci(metrics["counterfactual_margin"], 0.0) else "fail",
            "rule": "95% CI lower bound > 0",
        },
        "H3_mcs_recovery_ge_0_8": {
            "threshold": 0.8,
            "metric": "mcs_recovery",
            "decision": "pass" if pass_by_lower_ci(metrics["mcs_recovery"], 0.8) else "fail",
            "rule": "95% CI lower bound > 0.8",
        },
        "H4_overall_score_ge_0_42": {
            "threshold": 0.42,
            "metric": "overall_score",
            "decision": "pass" if pass_by_lower_ci(metrics["overall_score"], 0.42) else "fail",
            "rule": "95% CI lower bound > 0.42",
        },
        "H5_seq_causal_z_ge_1_96": {
            "threshold": 1.96,
            "metric": "causal_seq_z",
            "decision": "pass" if float(metrics["causal_seq_z"]["mean"]) >= 1.96 else "fail",
            "rule": "seed mean >= 1.96",
        },
    }

    plasticity = None
    p_path = Path(args.plasticity_summary_json)
    if p_path.exists():
        p_data = read_json(p_path)
        p_hebb = p_data.get("hebbian_one_shot_acc_stats", {})
        p_not_reached = float(p_data.get("not_reached_ratio", 0.0))
        p_sgd1000 = (p_data.get("sgd_step_curve_stats", {}) or {}).get("1000", {})
        plasticity = {
            "source": p_path.as_posix(),
            "hebbian_one_shot_acc_stats": p_hebb,
            "not_reached_ratio": p_not_reached,
            "sgd_step1000_stats": p_sgd1000,
        }
        hypotheses["H6_plasticity_not_reached_ratio_ge_0_8"] = {
            "threshold": 0.8,
            "metric": "plasticity.not_reached_ratio",
            "decision": "pass" if p_not_reached >= 0.8 else "fail",
            "rule": "ratio >= 0.8",
        }
        if p_hebb and p_sgd1000:
            hypotheses["H7_plasticity_hebb_gt_sgd1000"] = {
                "threshold": 0.0,
                "metric": "plasticity.hebbian_minus_sgd1000",
                "decision": "pass" if float(p_hebb.get("mean", 0.0)) > float(p_sgd1000.get("mean", 0.0)) else "fail",
                "rule": "mean(hebbian) > mean(sgd@1000)",
            }

    prompt_bootstrap = None
    pb_path = Path(args.prompt_bootstrap_json)
    if pb_path.exists():
        pb_data = read_json(pb_path)
        pb_agg = (pb_data.get("aggregate") or {}).get("stats", {})
        prompt_bootstrap = {
            "source": pb_path.as_posix(),
            "bootstrap_seq_margin_mean_stats": pb_agg.get("bootstrap_seq_margin_mean", {}),
            "bootstrap_positive_ratio_stats": pb_agg.get("bootstrap_positive_ratio", {}),
            "mean_prompt_std_seq_margin_stats": pb_agg.get("mean_prompt_std_seq_margin", {}),
            "necessity_ratio_stats": pb_agg.get("necessity_ratio", {}),
            "sufficiency_ratio_stats": pb_agg.get("sufficiency_ratio", {}),
            "overshoot_ratio_stats": pb_agg.get("overshoot_ratio", {}),
        }

        bs = prompt_bootstrap["bootstrap_seq_margin_mean_stats"] or {}
        hypotheses["H8_prompt_bootstrap_seq_margin_positive"] = {
            "threshold": 0.0,
            "metric": "prompt_bootstrap.bootstrap_seq_margin_mean",
            "decision": "pass" if float(bs.get("ci_low", 0.0)) > 0.0 else "fail",
            "rule": "95% CI lower bound > 0",
        }
        bpr = prompt_bootstrap["bootstrap_positive_ratio_stats"] or {}
        hypotheses["H9_prompt_bootstrap_positive_ratio_ge_0_95"] = {
            "threshold": 0.95,
            "metric": "prompt_bootstrap.bootstrap_positive_ratio",
            "decision": "pass" if float(bpr.get("mean", 0.0)) >= 0.95 else "fail",
            "rule": "seed mean >= 0.95",
        }

    out = {
        "pattern": args.pattern,
        "n_runs": len(runs),
        "metrics": metrics,
        "plasticity": plasticity,
        "prompt_bootstrap": prompt_bootstrap,
        "hypotheses": hypotheses,
        "runs": runs,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Falsifiable Stage Report",
        "",
        f"- Runs: {len(runs)}",
        f"- Pattern: `{args.pattern}`",
        "",
        "## Metric Summary (95% CI)",
    ]
    for k, v in metrics.items():
        lines.append(
            f"- {k}: mean={v['mean']:.6f}, std={v['std']:.6f}, "
            f"95%CI=[{v['ci_low']:.6f}, {v['ci_high']:.6f}]"
        )

    lines.extend(["", "## Hypothesis Check"])
    for h, v in hypotheses.items():
        lines.append(
            f"- {h}: {v['decision']} ({v['metric']} | {v['rule']} | threshold={v['threshold']})"
        )

    if plasticity:
        lines.extend(
            [
                "",
                "## Plasticity",
                f"- source: `{plasticity['source']}`",
                f"- hebbian_one_shot_acc_mean: {float((plasticity.get('hebbian_one_shot_acc_stats') or {}).get('mean', 0.0)):.6f}",
                f"- sgd_step1000_mean: {float((plasticity.get('sgd_step1000_stats') or {}).get('mean', 0.0)):.6f}",
                f"- not_reached_ratio: {float(plasticity.get('not_reached_ratio', 0.0)):.4f}",
            ]
        )

    if prompt_bootstrap:
        lines.extend(
            [
                "",
                "## Prompt Bootstrap",
                f"- source: `{prompt_bootstrap['source']}`",
                f"- bootstrap_seq_margin_mean: {float((prompt_bootstrap.get('bootstrap_seq_margin_mean_stats') or {}).get('mean', 0.0)):.6f}",
                f"- bootstrap_seq_margin_CI_low: {float((prompt_bootstrap.get('bootstrap_seq_margin_mean_stats') or {}).get('ci_low', 0.0)):.6f}",
                f"- bootstrap_positive_ratio_mean: {float((prompt_bootstrap.get('bootstrap_positive_ratio_stats') or {}).get('mean', 0.0)):.4f}",
                f"- necessity_ratio_mean: {float((prompt_bootstrap.get('necessity_ratio_stats') or {}).get('mean', 0.0)):.4f}",
                f"- sufficiency_ratio_mean: {float((prompt_bootstrap.get('sufficiency_ratio_stats') or {}).get('mean', 0.0)):.4f}",
                f"- overshoot_ratio_mean: {float((prompt_bootstrap.get('overshoot_ratio_stats') or {}).get('mean', 0.0)):.4f}",
            ]
        )

    lines.extend(["", "## Per Run"])
    for r in runs:
        lines.append(
            f"- seed={r['seed']}, overall={r['overall_score']:.4f}, "
            f"seq_margin={r['causal_seq_margin']:.6f}, cf_margin={r['counterfactual_margin']:.6f}, "
            f"mcs_recovery={r['mcs_recovery']:.4f}"
        )

    out_md = Path(args.output_md)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] JSON: {out_json}")
    print(f"[OK] MD: {out_md}")


if __name__ == "__main__":
    main()
