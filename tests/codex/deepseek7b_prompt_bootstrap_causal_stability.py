#!/usr/bin/env python
"""
Prompt-bootstrap causal stability analysis for DeepSeek7B mass noun scans.

This script does not run model inference. It reuses saved `mass_noun_encoding_scan.json`
files and evaluates:
1) prompt-template stability of causal margins,
2) bootstrap confidence for sequence causal margin,
3) minimal-circuit necessity/sufficiency proxy metrics,
4) cross-seed summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


def ci_mean(values: Sequence[float], z: float = 1.96) -> Dict[str, float]:
    xs = [float(x) for x in values]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "sem": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    mean = sum(xs) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        std = math.sqrt(max(var, 0.0))
        sem = std / math.sqrt(n)
    else:
        std = 0.0
        sem = 0.0
    return {
        "n": n,
        "mean": float(mean),
        "std": float(std),
        "sem": float(sem),
        "ci_low": float(mean - z * sem),
        "ci_high": float(mean + z * sem),
    }


def _safe_mean(xs: Sequence[float]) -> float:
    ys = [float(x) for x in xs]
    return float(sum(ys) / len(ys)) if ys else 0.0


@dataclass
class RunMetrics:
    path: str
    seed: int
    n_records: int
    n_prompts_per_record_mean: float
    mean_prompt_std_seq_margin: float
    mean_prompt_range_seq_margin: float
    bootstrap_seq_margin_mean: float
    bootstrap_seq_margin_ci_low: float
    bootstrap_seq_margin_ci_high: float
    bootstrap_positive_ratio: float
    bootstrap_prob_margin_mean: float
    necessity_ratio: float
    sufficiency_ratio: float
    overshoot_ratio: float
    counterfactual_positive_ratio: float
    counterfactual_mean_specificity: float
    causal_seq_z_reported: float
    overall_score_reported: float

    def to_dict(self) -> Dict[str, float | int | str]:
        return {
            "path": self.path,
            "seed": self.seed,
            "n_records": self.n_records,
            "n_prompts_per_record_mean": self.n_prompts_per_record_mean,
            "mean_prompt_std_seq_margin": self.mean_prompt_std_seq_margin,
            "mean_prompt_range_seq_margin": self.mean_prompt_range_seq_margin,
            "bootstrap_seq_margin_mean": self.bootstrap_seq_margin_mean,
            "bootstrap_seq_margin_ci_low": self.bootstrap_seq_margin_ci_low,
            "bootstrap_seq_margin_ci_high": self.bootstrap_seq_margin_ci_high,
            "bootstrap_positive_ratio": self.bootstrap_positive_ratio,
            "bootstrap_prob_margin_mean": self.bootstrap_prob_margin_mean,
            "necessity_ratio": self.necessity_ratio,
            "sufficiency_ratio": self.sufficiency_ratio,
            "overshoot_ratio": self.overshoot_ratio,
            "counterfactual_positive_ratio": self.counterfactual_positive_ratio,
            "counterfactual_mean_specificity": self.counterfactual_mean_specificity,
            "causal_seq_z_reported": self.causal_seq_z_reported,
            "overall_score_reported": self.overall_score_reported,
        }


def per_record_prompt_stats(record: Dict[str, object]) -> Dict[str, float]:
    metrics = record.get("prompt_metrics") or []
    seq = [float(x.get("causal_margin_seq_logprob", 0.0)) for x in metrics if isinstance(x, dict)]
    prob = [float(x.get("causal_margin_prob", 0.0)) for x in metrics if isinstance(x, dict)]
    if not seq:
        return {"n": 0.0, "seq_mean": 0.0, "seq_std": 0.0, "seq_range": 0.0, "prob_mean": 0.0}
    seq_mean = _safe_mean(seq)
    seq_std = math.sqrt(sum((x - seq_mean) ** 2 for x in seq) / len(seq))
    return {
        "n": float(len(seq)),
        "seq_mean": float(seq_mean),
        "seq_std": float(seq_std),
        "seq_range": float(max(seq) - min(seq)),
        "prob_mean": float(_safe_mean(prob)),
    }


def bootstrap_prompt_resample(
    records: List[Dict[str, object]],
    n_boot: int,
    seed: int,
) -> Dict[str, object]:
    rng = random.Random(seed + 7919)
    seq_boot = []
    prob_boot = []
    for _ in range(max(1, n_boot)):
        noun_seq = []
        noun_prob = []
        for rec in records:
            pms = rec.get("prompt_metrics") or []
            if not pms:
                continue
            draws = []
            for _k in range(len(pms)):
                draws.append(pms[rng.randrange(len(pms))])
            seq_vals = [float(x.get("causal_margin_seq_logprob", 0.0)) for x in draws if isinstance(x, dict)]
            prob_vals = [float(x.get("causal_margin_prob", 0.0)) for x in draws if isinstance(x, dict)]
            if seq_vals:
                noun_seq.append(_safe_mean(seq_vals))
            if prob_vals:
                noun_prob.append(_safe_mean(prob_vals))
        seq_boot.append(_safe_mean(noun_seq))
        prob_boot.append(_safe_mean(noun_prob))

    seq_ci = ci_mean(seq_boot)
    prob_ci = ci_mean(prob_boot)
    positive_ratio = float(sum(1 for x in seq_boot if x > 0.0) / len(seq_boot)) if seq_boot else 0.0
    return {
        "seq": seq_ci,
        "prob": prob_ci,
        "seq_positive_ratio": positive_ratio,
        "seq_boot_samples": seq_boot,
        "prob_boot_samples": prob_boot,
    }


def analyze_one(path: str, n_boot: int) -> RunMetrics:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seed = int(data.get("seed", 0))
    if seed == 0:
        m = re.search(r"seed(\d+)", path.replace("\\", "/"))
        if m:
            seed = int(m.group(1))
    score = data.get("mechanism_scorecard") or {}
    causal = data.get("causal_ablation") or {}
    records = causal.get("records") or []

    prs = [per_record_prompt_stats(r) for r in records]
    mean_n_prompts = _safe_mean([x["n"] for x in prs])
    mean_prompt_std_seq_margin = _safe_mean([x["seq_std"] for x in prs])
    mean_prompt_range_seq_margin = _safe_mean([x["seq_range"] for x in prs])

    boot = bootstrap_prompt_resample(records, n_boot=n_boot, seed=seed)
    boot_seq = boot["seq"]
    boot_prob = boot["prob"]

    min_c = (causal.get("minimal_circuit") or {})
    min_recs = min_c.get("records") or []
    target_ratio = float(min_c.get("target_ratio", 0.8))
    full_drop = [float(x.get("full_signature_drop_seq_logprob", 0.0)) for x in min_recs]
    subset_drop = [float(x.get("subset_drop_seq_logprob", 0.0)) for x in min_recs]
    necessity_ratio = float(sum(1 for x in full_drop if x > 0.0) / len(full_drop)) if full_drop else 0.0
    sufficiency_ratio = (
        float(sum(1 for i in range(len(min_recs)) if subset_drop[i] >= target_ratio * full_drop[i]) / len(min_recs))
        if min_recs
        else 0.0
    )
    overshoot_ratio = (
        float(sum(1 for i in range(len(min_recs)) if subset_drop[i] > full_drop[i]) / len(min_recs))
        if min_recs
        else 0.0
    )

    cf = (causal.get("counterfactual_validation") or {})
    cf_agg = (cf.get("aggregate") or {})
    cf_pos = float(cf_agg.get("positive_specificity_ratio", 0.0))
    cf_margin = float(cf_agg.get("mean_specificity_margin_seq_logprob", 0.0))

    agg = causal.get("aggregate") or {}
    return RunMetrics(
        path=path,
        seed=seed,
        n_records=len(records),
        n_prompts_per_record_mean=mean_n_prompts,
        mean_prompt_std_seq_margin=mean_prompt_std_seq_margin,
        mean_prompt_range_seq_margin=mean_prompt_range_seq_margin,
        bootstrap_seq_margin_mean=float(boot_seq["mean"]),
        bootstrap_seq_margin_ci_low=float(boot_seq["ci_low"]),
        bootstrap_seq_margin_ci_high=float(boot_seq["ci_high"]),
        bootstrap_positive_ratio=float(boot["seq_positive_ratio"]),
        bootstrap_prob_margin_mean=float(boot_prob["mean"]),
        necessity_ratio=necessity_ratio,
        sufficiency_ratio=sufficiency_ratio,
        overshoot_ratio=overshoot_ratio,
        counterfactual_positive_ratio=cf_pos,
        counterfactual_mean_specificity=cf_margin,
        causal_seq_z_reported=float(agg.get("causal_margin_seq_logprob_z", 0.0)),
        overall_score_reported=float(score.get("overall_score", 0.0)),
    )


def aggregate_runs(runs: List[RunMetrics]) -> Dict[str, object]:
    def field(name: str) -> List[float]:
        return [float(getattr(r, name)) for r in runs]

    keys = [
        "n_prompts_per_record_mean",
        "mean_prompt_std_seq_margin",
        "mean_prompt_range_seq_margin",
        "bootstrap_seq_margin_mean",
        "bootstrap_positive_ratio",
        "bootstrap_prob_margin_mean",
        "necessity_ratio",
        "sufficiency_ratio",
        "overshoot_ratio",
        "counterfactual_positive_ratio",
        "counterfactual_mean_specificity",
        "causal_seq_z_reported",
        "overall_score_reported",
    ]
    out = {"n_runs": len(runs), "stats": {}}
    for k in keys:
        out["stats"][k] = ci_mean(field(k))
    return out


def build_report(payload: Dict[str, object]) -> str:
    agg = payload["aggregate"]
    st = agg["stats"]
    lines = [
        "# Prompt Bootstrap Causal Stability Report",
        "",
        f"- n_runs: {agg['n_runs']}",
        f"- bootstrap_seq_margin_mean: {st['bootstrap_seq_margin_mean']['mean']:.6f} "
        f"(95%CI [{st['bootstrap_seq_margin_mean']['ci_low']:.6f}, {st['bootstrap_seq_margin_mean']['ci_high']:.6f}])",
        f"- bootstrap_positive_ratio: {st['bootstrap_positive_ratio']['mean']:.4f}",
        f"- prompt_std_seq_margin: {st['mean_prompt_std_seq_margin']['mean']:.6f}",
        f"- necessity_ratio: {st['necessity_ratio']['mean']:.4f}",
        f"- sufficiency_ratio: {st['sufficiency_ratio']['mean']:.4f}",
        f"- overshoot_ratio: {st['overshoot_ratio']['mean']:.4f}",
        f"- counterfactual_positive_ratio: {st['counterfactual_positive_ratio']['mean']:.4f}",
        f"- causal_seq_z_reported: {st['causal_seq_z_reported']['mean']:.4f}",
        f"- overall_score_reported: {st['overall_score_reported']['mean']:.4f}",
        "",
        "## Interpretation",
        "- If bootstrap_seq_margin CI stays >0, prompt-level causal direction is stable.",
        "- High necessity+sufficiency supports a compact causal subset hypothesis.",
        "- If global z/overall remain low, local mechanism exists but system-level evidence is not yet strong.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default="tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed*/mass_noun_encoding_scan.json",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", default="tempdata/deepseek7b_prompt_bootstrap_causal_stability_v1")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {args.pattern}")

    runs = [analyze_one(p, n_boot=args.n_bootstrap) for p in paths]
    payload = {
        "pattern": args.pattern,
        "n_bootstrap": int(args.n_bootstrap),
        "runs": [r.to_dict() for r in runs],
        "aggregate": aggregate_runs(runs),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "prompt_bootstrap_causal_stability.json"
    md_path = out_dir / "PROMPT_BOOTSTRAP_CAUSAL_STABILITY_REPORT.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_report(payload), encoding="utf-8")

    print(json.dumps({"json": json_path.as_posix(), "markdown": md_path.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
