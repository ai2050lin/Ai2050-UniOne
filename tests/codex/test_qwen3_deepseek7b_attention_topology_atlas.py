#!/usr/bin/env python
"""
Build a broader attention-topology atlas for Qwen3-4B and DeepSeek-7B.

Compared with the direct topology basis probe on apple/cat/truth, this atlas:
- expands to the full concept set from the three families
- measures family match and residual gap per concept
- summarizes how stable T-side family structure is across a larger concept pool
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import test_qwen3_deepseek7b_attention_topology_basis as topo_basis


def mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def family_of(word: str, selected_words: Dict[str, List[str]]) -> str:
    for family, words in selected_words.items():
        if word in words:
            return family
    raise KeyError(f"Cannot find family for word={word}")


def analyze_model(
    model_name: str,
    model_path: str,
    dtype_name: str,
    prefer_cuda: bool,
) -> Dict[str, Any]:
    t0 = time.time()
    model, tok = topo_basis.load_model(model_path, dtype_name, prefer_cuda)
    selected = topo_basis.compatible_words(tok)
    family_rows = topo_basis.family_candidates()

    concept_topology: Dict[str, np.ndarray] = {}
    concept_entropy: Dict[str, float] = {}
    for family, words in selected.items():
        for word in words:
            rows = []
            ents = []
            for text in topo_basis.template_texts(word):
                flat, ent = topo_basis.run_attn(model, tok, text)
                rows.append(flat)
                ents.append(ent)
            concept_topology[word] = topo_basis.mean_vector(rows)
            concept_entropy[word] = float(np.mean(ents))

    family_basis = {}
    family_summary = {}
    for family, words in selected.items():
        mu, basis = topo_basis.affine_basis(
            [concept_topology[w] for w in words],
            rank_k=min(3, max(1, len(words) - 1)),
        )
        family_basis[family] = {"mu": mu, "basis": basis}
        vals = [topo_basis.residual_ratio(concept_topology[w], mu, basis) for w in words]
        family_summary[family] = {
            "candidate_count": int(len(family_rows[family])),
            "selected_count": int(len(words)),
            "mean_topology_residual_ratio": float(np.mean(vals)),
            "max_topology_residual_ratio": float(np.max(vals)),
            "mean_last_token_entropy": float(np.mean([concept_entropy[w] for w in words])),
        }

    concepts = {}
    matched = 0
    margins = []
    true_residuals = []
    wrong_residuals = []
    support_flags = []
    per_family_match: Dict[str, List[bool]] = {family: [] for family in selected}
    for family, words in selected.items():
        for word in words:
            x = concept_topology[word]
            fit = {}
            for cand_family in selected.keys():
                mu = family_basis[cand_family]["mu"]
                basis = family_basis[cand_family]["basis"]
                proj = topo_basis.affine_project(x, mu, basis)
                delta = (x - proj).astype(np.float32)
                fit[cand_family] = {
                    "residual_ratio": topo_basis.residual_ratio(x, mu, basis),
                    "delta_top32_energy_ratio": topo_basis.topk_energy_ratio(delta, 32),
                    "delta_top128_energy_ratio": topo_basis.topk_energy_ratio(delta, 128),
                }
            ranked = sorted(
                [{"family": cand_family, **row} for cand_family, row in fit.items()],
                key=lambda row: row["residual_ratio"],
            )
            preferred = ranked[0]["family"]
            match = preferred == family
            best_wrong = min(row["residual_ratio"] for row in ranked if row["family"] != family)
            margin = float(best_wrong - fit[family]["residual_ratio"])
            concepts[word] = {
                "true_family": family,
                "preferred_family": preferred,
                "preferred_family_matches_truth": bool(match),
                "family_fit": fit,
                "summary": {
                    "margin_vs_best_wrong": margin,
                    "entropy": float(concept_entropy[word]),
                    "supports_family_topology_basis": bool(match),
                },
            }
            matched += int(match)
            margins.append(margin)
            true_residuals.append(float(fit[family]["residual_ratio"]))
            wrong_residuals.append(float(best_wrong))
            support_flags.append(1.0 if match else 0.0)
            per_family_match[family].append(match)

    del model, tok
    if topo_basis.torch.cuda.is_available():
        topo_basis.torch.cuda.empty_cache()

    return {
        "meta": {
            "model_name": model_name,
            "model_path": model_path,
            "runtime_sec": float(time.time() - t0),
        },
        "selected_words": selected,
        "family_summary": family_summary,
        "concepts": concepts,
        "global_summary": {
            "concept_count": int(len(concepts)),
            "preferred_family_match_rate": float(matched / max(1, len(concepts))),
            "support_rate": mean(support_flags),
            "mean_margin_vs_best_wrong": mean(margins),
            "mean_true_family_residual": mean(true_residuals),
            "mean_best_wrong_residual": mean(wrong_residuals),
            "family_match_rate": {
                family: float(sum(1 for item in flags if item) / max(1, len(flags)))
                for family, flags in per_family_match.items()
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Qwen3/DeepSeek7B attention-topology atlas")
    ap.add_argument("--dtype-qwen", type=str, default="bfloat16")
    ap.add_argument("--dtype-deepseek", type=str, default="bfloat16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_attention_topology_atlas_20260309.json",
    )
    args = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in topo_basis.default_model_specs():
        dtype_name = args.dtype_qwen if model_name == "qwen3_4b" else args.dtype_deepseek
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(
            model_name=model_name,
            model_path=model_path,
            dtype_name=dtype_name,
            prefer_cuda=not args.cpu_only,
        )
        results["models"][model_name] = row
        summary = row["global_summary"]
        print(
            f"[summary] {model_name} "
            f"match={summary['preferred_family_match_rate']:.4f} "
            f"margin={summary['mean_margin_vs_best_wrong']:.4f} "
            f"true={summary['mean_true_family_residual']:.4f} "
            f"wrong={summary['mean_best_wrong_residual']:.4f}"
        )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
