"""
Compress the current six-object framing

    shared_basis / offset / relation_protocol / gating / topology / integration

into a smaller four-factor framing

    base / adaptive_offset / routing / stabilization

and quantify how much explanatory power is retained against existing bridge
scores and task-block summaries.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: List[float]) -> float:
    return float(np.mean(np.array(xs, dtype=np.float64))) if xs else 0.0


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def correlation(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def stabilization_score(blocks: Dict[str, Any], d_atlas: Dict[str, Any]) -> Dict[str, float]:
    a_score = float(blocks["blocks"]["A"]["headline_score"])
    d_best = float(d_atlas["global_summary"]["best_overall_gain_across_methods"])
    d_shift = clip01(0.5 + 10.0 * d_best)
    score = mean([a_score, d_shift])
    return {
        "score": score,
        "long_horizon_component": a_score,
        "grounding_component": d_shift,
        "best_d_overall_gain": d_best,
    }


def build_dnn_view(model_name: str, row: Dict[str, Any], stab: Dict[str, float]) -> Dict[str, Any]:
    comps = row["components"]
    base = mean([float(comps["shared_basis"]["score"]), float(comps["abstraction_operator"]["score"])])
    adaptive_offset = float(comps["sparse_offset"]["score"])
    routing = mean(
        [
            float(comps["topology_basis"]["score"]),
            float(comps["analogy_path"]["score"]),
            float(comps["protocol_routing"]["score"]),
        ]
    )
    stabilization = float(comps["multi_timescale_control"]["score"])
    compressed = mean([base, adaptive_offset, routing, stabilization])
    original = float(row["overall_bridge_score"])
    return {
        "view_id": f"{model_name}_dnn_bridge",
        "model_name": model_name,
        "source": "dnn_brain_puzzle_bridge",
        "factors": {
            "base": base,
            "adaptive_offset": adaptive_offset,
            "routing": routing,
            "stabilization": stabilization,
        },
        "reference_score": original,
        "compressed_score": compressed,
        "compression_gap": float(compressed - original),
        "stabilization_detail": stab,
    }


def build_mech_view(model_name: str, row: Dict[str, Any], stab: Dict[str, float]) -> Dict[str, Any]:
    comps = row["components"]
    base = mean([float(comps["shared_basis"]), float(comps["H_representation"])])
    adaptive_offset = mean([float(comps["offset"]), float(comps["protocol_calling"])])
    routing = mean(
        [
            float(comps["G_gating"]),
            float(comps["R_relation"]),
            float(comps["T_topology"]),
        ]
    )
    stabilization = float(stab["score"])
    compressed = mean([base, adaptive_offset, routing, stabilization])
    original = float(row["mechanism_bridge_score"])
    return {
        "view_id": f"{model_name}_mechanism_bridge",
        "model_name": model_name,
        "source": "qwen3_deepseek7b_mechanism_bridge",
        "factors": {
            "base": base,
            "adaptive_offset": adaptive_offset,
            "routing": routing,
            "stabilization": stabilization,
        },
        "reference_score": original,
        "compressed_score": compressed,
        "compression_gap": float(compressed - original),
        "stabilization_detail": stab,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compress six-object AGI framing into four-factor unified structure")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/unified_structure_four_factor_compression_20260309.json")
    args = ap.parse_args()

    t0 = time.time()
    dnn_bridge = load_json(ROOT / "tests" / "codex_temp" / "dnn_brain_puzzle_bridge_20260308.json")
    mech_bridge = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_mechanism_bridge_20260309.json")
    task_blocks = load_json(ROOT / "tests" / "codex_temp" / "agi_task_block_summary_20260309.json")
    d_atlas = load_json(ROOT / "tests" / "codex_temp" / "d_problem_atlas_summary_20260309.json")

    stab = stabilization_score(task_blocks, d_atlas)

    views = []
    for model_name, row in dnn_bridge["models"].items():
        views.append(build_dnn_view(model_name, row, stab))
    for model_name, row in mech_bridge["models"].items():
        views.append(build_mech_view(model_name, row, stab))

    ref_scores = [row["reference_score"] for row in views]
    compressed_scores = [row["compressed_score"] for row in views]
    gaps = [abs(row["compression_gap"]) for row in views]

    factor_means = {
        name: mean([row["factors"][name] for row in views])
        for name in ["base", "adaptive_offset", "routing", "stabilization"]
    }

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "compression_target": "shared_basis / offset / relation_protocol / gating / topology / integration -> base / adaptive_offset / routing / stabilization",
        },
        "mapping": {
            "base": [
                "shared_basis",
                "H_representation",
                "abstraction_operator",
            ],
            "adaptive_offset": [
                "offset",
                "protocol_calling",
                "sparse_offset",
            ],
            "routing": [
                "G_gating",
                "R_relation",
                "T_topology",
                "protocol_routing",
                "topology_basis",
                "analogy_path",
            ],
            "stabilization": [
                "integration",
                "multi_timescale_control",
                "long_horizon_stability",
                "grounding_barrier",
            ],
        },
        "views": views,
        "factor_summary": {
            "means": factor_means,
            "strongest_factor": max(factor_means.items(), key=lambda kv: kv[1])[0],
            "weakest_factor": min(factor_means.items(), key=lambda kv: kv[1])[0],
        },
        "retention": {
            "reference_mean": mean(ref_scores),
            "compressed_mean": mean(compressed_scores),
            "mean_absolute_gap": mean(gaps),
            "score_correlation": correlation(ref_scores, compressed_scores),
            "compression_pass": bool(mean(gaps) < 0.12 and correlation(ref_scores, compressed_scores) > 0.6),
        },
        "project_readout": {
            "why_compress": [
                "把六个对象压成四个对象后，理论表达更接近“最小统一结构”。",
                "如果压缩后仍能保留桥接分数排序和大部分量级，说明当前六对象里存在可合并冗余。",
                "后续真正要寻找的，不是更多模块名词，而是更小的更新律。",
            ],
            "current_verdict": (
                "当前证据支持先把 shared_basis 与表征层压成 base，"
                "把 offset 与 concept calling 压成 adaptive_offset，"
                "把 relation / gating / topology 压成 routing，"
                "再把长程与接地瓶颈压成 stabilization。"
            ),
            "next_question": "能否进一步把 base / adaptive_offset / routing / stabilization 再写成一条统一更新律？",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["retention"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
