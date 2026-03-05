#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stat(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main():
    ap = argparse.ArgumentParser(description="Summarize deepseek7b mass noun multiseed scans")
    ap.add_argument("--pattern", default="tempdata/deepseek7b_mass_noun_scan_n120_multiseed_seed*/mass_noun_encoding_scan.json")
    ap.add_argument("--output-json", default="tempdata/deepseek7b_mass_noun_scan_n120_multiseed_summary.json")
    ap.add_argument("--output-md", default="tempdata/deepseek7b_mass_noun_scan_n120_multiseed_summary.md")
    args = ap.parse_args()

    files = sorted(Path(".").glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.pattern}")

    rows = []
    for fp in files:
        data = read_json(fp)
        score = data.get("mechanism_scorecard", {})
        subs = score.get("subscores", {})
        agg = (data.get("causal_ablation") or {}).get("aggregate", {})
        min_circuit = (data.get("causal_ablation") or {}).get("minimal_circuit", {})
        min_agg = (min_circuit or {}).get("aggregate", {}) if isinstance(min_circuit, dict) else {}
        cf_val = (data.get("causal_ablation") or {}).get("counterfactual_validation", {})
        cf_agg = (cf_val or {}).get("aggregate", {}) if isinstance(cf_val, dict) else {}
        cfg = data.get("config", {})
        rows.append(
            {
                "path": fp.as_posix(),
                "seed": int(fp.parent.name.split("seed")[-1]) if "seed" in fp.parent.name else None,
                "n_nouns": int(cfg.get("n_nouns", 0)),
                "overall_score": float(score.get("overall_score", 0.0)),
                "grade": str(score.get("grade", "unknown")),
                "structure_separation": float(subs.get("structure_separation", 0.0)),
                "reuse_sparsity_structure": float(subs.get("reuse_sparsity_structure", 0.0)),
                "low_rank_compactness": float(subs.get("low_rank_compactness", 0.0)),
                "causal_evidence": float(subs.get("causal_evidence", 0.0)),
                "mean_causal_margin_prob": float(agg.get("mean_causal_margin_prob", 0.0)),
                "mean_causal_margin_logprob": float(agg.get("mean_causal_margin_logprob", 0.0)),
                "mean_causal_margin_rank_worse": float(agg.get("mean_causal_margin_rank_worse", 0.0)),
                "mean_causal_margin_seq_logprob": float(agg.get("mean_causal_margin_seq_logprob", 0.0)),
                "mean_causal_margin_seq_avg_logprob": float(agg.get("mean_causal_margin_seq_avg_logprob", 0.0)),
                "causal_margin_prob_z": float(agg.get("causal_margin_prob_z", 0.0)),
                "causal_margin_seq_logprob_z": float(agg.get("causal_margin_seq_logprob_z", 0.0)),
                "positive_causal_margin_ratio": float(agg.get("positive_causal_margin_ratio", 0.0)),
                "minimal_circuit_n_tested_nouns": float(min_circuit.get("n_tested_nouns", 0)),
                "minimal_circuit_mean_subset_size": float(min_agg.get("mean_subset_size", 0.0)),
                "minimal_circuit_mean_recovery_ratio": float(min_agg.get("mean_recovery_ratio", 0.0)),
                "counterfactual_n_pairs": float(cf_val.get("n_pairs", 0)),
                "counterfactual_mean_specificity_margin_seq_logprob": float(
                    cf_agg.get("mean_specificity_margin_seq_logprob", 0.0)
                ),
                "counterfactual_specificity_margin_z": float(cf_agg.get("specificity_margin_z", 0.0)),
            }
        )

    fields = [
        "overall_score",
        "structure_separation",
        "reuse_sparsity_structure",
        "low_rank_compactness",
        "causal_evidence",
        "mean_causal_margin_prob",
        "mean_causal_margin_logprob",
        "mean_causal_margin_rank_worse",
        "mean_causal_margin_seq_logprob",
        "mean_causal_margin_seq_avg_logprob",
        "causal_margin_prob_z",
        "causal_margin_seq_logprob_z",
        "positive_causal_margin_ratio",
        "minimal_circuit_n_tested_nouns",
        "minimal_circuit_mean_subset_size",
        "minimal_circuit_mean_recovery_ratio",
        "counterfactual_n_pairs",
        "counterfactual_mean_specificity_margin_seq_logprob",
        "counterfactual_specificity_margin_z",
    ]
    summary_stats = {k: stat([r[k] for r in rows]) for k in fields}
    grade_dist = dict(Counter(r["grade"] for r in rows))

    summary = {
        "n_runs": len(rows),
        "pattern": args.pattern,
        "grade_distribution": grade_dist,
        "stats": summary_stats,
        "runs": rows,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# DeepSeek7B Mass Noun Multiseed Summary",
        "",
        f"- Runs: {len(rows)}",
        f"- Pattern: `{args.pattern}`",
        f"- Grade distribution: {grade_dist}",
        "",
        "## Aggregated Stats",
    ]
    for k in fields:
        s = summary_stats[k]
        lines.append(f"- {k}: mean={s['mean']:.6f}, std={s['std']:.6f}, min={s['min']:.6f}, max={s['max']:.6f}")

    lines.extend(["", "## Per-Run", ""])
    for r in rows:
        lines.append(
            f"- seed={r['seed']}, grade={r['grade']}, overall={r['overall_score']:.4f}, "
            f"causal={r['causal_evidence']:.4f}, z={r['causal_margin_prob_z']:.4f}, "
            f"logprob_margin={r['mean_causal_margin_logprob']:.6f}, "
            f"seq_logprob_margin={r['mean_causal_margin_seq_logprob']:.6f}, "
            f"mcs_size={r['minimal_circuit_mean_subset_size']:.3f}, "
            f"cf_margin={r['counterfactual_mean_specificity_margin_seq_logprob']:.6f}"
        )

    out_md = Path(args.output_md)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] JSON: {out_json}")
    print(f"[OK] MD: {out_md}")


if __name__ == "__main__":
    main()
