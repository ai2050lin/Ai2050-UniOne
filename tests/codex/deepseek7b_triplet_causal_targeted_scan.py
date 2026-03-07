#!/usr/bin/env python
"""
Targeted causal scan for apple / king / queen.

This wrapper runs `deepseek7b_mass_noun_encoding_scan.py` on a fixed 3-noun set
with causal ablation enabled, then extracts triplet-focused causal metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from agi_research_result_schema import build_result_payload, write_result_bundle


TRIPLET_ROWS = [
    ("apple", "fruit"),
    ("king", "human"),
    ("queen", "human"),
]


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _find_by_noun(records: List[Dict], noun: str) -> List[Dict]:
    nk = str(noun).strip().lower()
    return [r for r in records if str(r.get("noun", "")).strip().lower() == nk]


def summarize_triplet(scan: Dict) -> Dict:
    causal = scan.get("causal_ablation") or {}
    min_records = ((causal.get("minimal_circuit") or {}).get("records") or [])
    cf_records = ((causal.get("counterfactual_validation") or {}).get("records") or [])

    apple_min = _find_by_noun(min_records, "apple")
    king_min = _find_by_noun(min_records, "king")
    queen_min = _find_by_noun(min_records, "queen")
    apple_cf = _find_by_noun(cf_records, "apple")
    king_cf = _find_by_noun(cf_records, "king")
    queen_cf = _find_by_noun(cf_records, "queen")

    def _mean(xs):
        if not xs:
            return 0.0
        return float(sum(float(x) for x in xs) / len(xs))

    return {
        "triplet_minimal_records": int(len(apple_min) + len(king_min) + len(queen_min)),
        "triplet_counterfactual_records": int(len(apple_cf) + len(king_cf) + len(queen_cf)),
        "apple_recovery_ratio_mean": _mean([x.get("recovery_ratio", 0.0) for x in apple_min]),
        "king_recovery_ratio_mean": _mean([x.get("recovery_ratio", 0.0) for x in king_min]),
        "queen_recovery_ratio_mean": _mean([x.get("recovery_ratio", 0.0) for x in queen_min]),
        "apple_specificity_margin_mean": _mean([x.get("specificity_margin_seq_logprob", 0.0) for x in apple_cf]),
        "king_specificity_margin_mean": _mean([x.get("specificity_margin_seq_logprob", 0.0) for x in king_cf]),
        "queen_specificity_margin_mean": _mean([x.get("specificity_margin_seq_logprob", 0.0) for x in queen_cf]),
        "global_mean_causal_margin_prob": float((causal.get("aggregate") or {}).get("mean_causal_margin_prob", 0.0)),
        "global_mean_causal_margin_seq_logprob": float((causal.get("aggregate") or {}).get("mean_causal_margin_seq_logprob", 0.0)),
        "global_positive_causal_margin_ratio": float((causal.get("aggregate") or {}).get("positive_causal_margin_ratio", 0.0)),
    }


def render_md(payload: Dict, scan_rel_path: str) -> str:
    m = payload["metrics"]
    lines = [
        "# Apple/King/Queen 定向因果扫描报告",
        "",
        f"- 扫描 JSON: `{scan_rel_path}`",
        "",
        "## 三联覆盖度",
        f"- 最小回路记录数: {m['triplet_minimal_records']}",
        f"- 反事实记录数: {m['triplet_counterfactual_records']}",
        "",
        "## 三联因果指标",
        f"- apple recovery: {m['apple_recovery_ratio_mean']:.4f}",
        f"- king recovery: {m['king_recovery_ratio_mean']:.4f}",
        f"- queen recovery: {m['queen_recovery_ratio_mean']:.4f}",
        f"- apple specificity: {m['apple_specificity_margin_mean']:.6f}",
        f"- king specificity: {m['king_specificity_margin_mean']:.6f}",
        f"- queen specificity: {m['queen_specificity_margin_mean']:.6f}",
        "",
        "## 全局因果统计",
        f"- mean causal margin (prob): {m['global_mean_causal_margin_prob']:.6f}",
        f"- mean causal margin (seq_logprob): {m['global_mean_causal_margin_seq_logprob']:.6f}",
        f"- positive causal margin ratio: {m['global_positive_causal_margin_ratio']:.4f}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Targeted causal scan for apple/king/queen")
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--ablation-random-trials", type=int, default=3)
    ap.add_argument("--ablation-top-k", type=int, default=24)
    ap.add_argument("--minimal-circuit-max-size", type=int, default=12)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_triplet_causal_targeted_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    nouns_file = out_dir / "triplet_nouns.csv"
    with nouns_file.open("w", encoding="utf-8") as f:
        for n, c in TRIPLET_ROWS:
            f.write(f"{n},{c}\n")

    scan_out_dir = out_dir / "raw_scan"
    scan_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "tests/codex/deepseek7b_mass_noun_encoding_scan.py",
        "--model-id",
        args.model_id,
        "--dtype",
        args.dtype,
        "--local-files-only",
        "--nouns-file",
        str(nouns_file),
        "--top-signature-k",
        "120",
        "--run-causal-ablation",
        "--ablation-top-k",
        str(args.ablation_top_k),
        "--ablation-random-trials",
        str(args.ablation_random_trials),
        "--ablation-max-nouns",
        "3",
        "--ablation-per-category-max",
        "3",
        "--ablation-sample-strategy",
        "head",
        "--minimal-circuit-max-nouns",
        "3",
        "--minimal-circuit-target-ratio",
        "0.8",
        "--minimal-circuit-max-size",
        str(args.minimal_circuit_max_size),
        "--counterfactual-max-pairs",
        "8",
        "--seed",
        str(args.seed),
        "--output-dir",
        str(scan_out_dir),
    ]

    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"targeted scan failed with exit code {proc.returncode}")

    scan_json = scan_out_dir / "mass_noun_encoding_scan.json"
    if not scan_json.exists():
        raise FileNotFoundError(f"scan output missing: {scan_json}")

    scan = load_json(scan_json)
    metrics = summarize_triplet(scan)
    scan_rel = scan_json.as_posix()

    payload = build_result_payload(
        experiment_id="triplet_targeted_causal_scan_v1",
        title="Apple/King/Queen 定向因果扫描",
        config={
            "model_id": args.model_id,
            "dtype": args.dtype,
            "seed": args.seed,
            "ablation_top_k": args.ablation_top_k,
            "ablation_random_trials": args.ablation_random_trials,
            "minimal_circuit_max_size": args.minimal_circuit_max_size,
            "scan_json": scan_rel,
        },
        metrics=metrics,
        hypotheses=[
            {
                "id": "H_triplet_covered",
                "rule": "triplet_minimal_records >= 3 and triplet_counterfactual_records >= 3",
                "pass": bool(metrics["triplet_minimal_records"] >= 3 and metrics["triplet_counterfactual_records"] >= 3),
            },
            {
                "id": "H_causal_margin_positive",
                "rule": "global_mean_causal_margin_prob > 0",
                "pass": bool(metrics["global_mean_causal_margin_prob"] > 0),
            },
        ],
        artifacts={
            "scan_json": scan_rel,
            "nouns_file": nouns_file.as_posix(),
        },
        notes=["定向三联扫描用于补齐 king/queen 因果抽样覆盖。"],
    )

    paths = write_result_bundle(
        out_dir=out_dir,
        base_name="triplet_targeted_causal_scan",
        payload=payload,
        report_md=render_md(payload, scan_rel),
    )
    print(f"[OK] {paths['json']}")
    print(f"[OK] {paths['md']}")


if __name__ == "__main__":
    main()

