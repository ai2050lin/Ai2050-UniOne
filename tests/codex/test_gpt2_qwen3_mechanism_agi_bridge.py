#!/usr/bin/env python
"""
Bridge mechanistic structure metrics to AGI blocker metrics.

This is a lightweight post-analysis script over existing result files. It does
not rerun the heavy model probes. Instead, it condenses three axes:
1) gate-law dynamical predictability
2) relation protocol field boundaryability
3) toy grounding / delayed credit / continual retention gains

The output is a single JSON payload designed for frontend visualization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GATE_JSON = ROOT / "tests" / "codex_temp" / "gpt2_qwen3_gate_law_nonlinear_dynamics_20260308.json"
BOUNDARY_JSON = ROOT / "tests" / "codex_temp" / "gpt2_qwen3_relation_boundary_atlas_20260308.json"
TOY_JSON = ROOT / "tests" / "codex_temp" / "toy_grounding_credit_continual_benchmark_20260308.json"
REAL_JSON = ROOT / "tests" / "codex_temp" / "real_multistep_agi_closure_benchmark_20260308.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def norm_gain(value: float, target: float) -> float:
    if target <= 0:
        return 0.0
    return clamp01(float(value) / float(target))


def build_toy_summary(payload: dict) -> dict:
    improvements = payload.get("improvements", {})
    grounding_gain = float(improvements.get("overall_grounding_accuracy", 0.0))
    delayed_gain = float(improvements.get("overall_delayed_accuracy", 0.0))
    retention_gain = float(improvements.get("retention_after_phase2", 0.0))
    forgetting_reduction = float(improvements.get("retention_drop_reduction", 0.0))

    grounding_norm = norm_gain(grounding_gain, 0.20)
    delayed_norm = norm_gain(delayed_gain, 0.10)
    retention_norm = norm_gain(retention_gain, 0.30)
    forgetting_norm = norm_gain(forgetting_reduction, 0.30)

    toy_closure_score = (
        0.35 * grounding_norm
        + 0.20 * delayed_norm
        + 0.25 * retention_norm
        + 0.20 * forgetting_norm
    )

    return {
        "grounding_gain": grounding_gain,
        "delayed_credit_gain": delayed_gain,
        "retention_gain": retention_gain,
        "forgetting_reduction": forgetting_reduction,
        "normalized_components": {
            "grounding": grounding_norm,
            "delayed_credit": delayed_norm,
            "retention": retention_norm,
            "forgetting_reduction": forgetting_norm,
        },
        "toy_closure_score": toy_closure_score,
    }


def build_real_summary(payload: dict) -> dict:
    improvements = payload.get("improvements", {})
    scores = payload.get("real_closure_score", {})
    trace_score = float(scores.get("trace_gated_local", 0.0))
    plain_score = float(scores.get("plain_local", 0.0))
    score_gain = float(scores.get("score_gain", trace_score - plain_score))
    return {
        "overall_episode_gain": float(improvements.get("overall_episode_success", 0.0)),
        "route_gain": float(improvements.get("overall_route_accuracy", 0.0)),
        "final_gain": float(improvements.get("overall_final_accuracy", 0.0)),
        "retention_gain": float(improvements.get("retention_after_phase2", 0.0)),
        "retention_drop_reduction": float(improvements.get("retention_drop_reduction", 0.0)),
        "plain_score": plain_score,
        "trace_score": trace_score,
        "score_gain": score_gain,
        "real_closure_score": trace_score,
    }


def build_model_summary(model_name: str, gate_row: dict, boundary_row: dict, toy_summary: dict, real_summary: dict) -> dict:
    gate_summary = gate_row.get("global_summary", {})
    boundary_summary = boundary_row.get("global_summary", {})
    class_hist = boundary_summary.get("classification_histogram", {})
    total_relations = max(1, sum(int(v or 0) for v in class_hist.values()))

    compact_ratio = float(class_hist.get("compact_boundary", 0)) / total_relations
    layer_cluster_ratio = float(class_hist.get("layer_cluster_only", 0)) / total_relations
    distributed_ratio = float(class_hist.get("distributed_none", 0)) / total_relations
    boundaryability_score = compact_ratio + 0.5 * layer_cluster_ratio

    gate_predictability = float(gate_summary.get("mean_nonlinear_recurrence_r2", 0.0))
    linear_gain = float(gate_summary.get("mean_linear_gain", 0.0))
    nonlinear_gain = float(gate_summary.get("mean_nonlinear_gain", 0.0))

    linear_gain_norm = norm_gain(linear_gain, 0.50)
    nonlinear_gain_norm = norm_gain(nonlinear_gain, 0.12)

    mechanism_score = (
        0.55 * gate_predictability
        + 0.20 * linear_gain_norm
        + 0.10 * nonlinear_gain_norm
        + 0.15 * boundaryability_score
    )
    agi_bridge_toy_score = 0.60 * mechanism_score + 0.40 * float(toy_summary["toy_closure_score"])
    agi_bridge_real_score = 0.45 * mechanism_score + 0.55 * float(real_summary["real_closure_score"])

    if distributed_ratio >= 0.50:
        field_shape = "distributed_mesofield"
    elif layer_cluster_ratio >= 0.15:
        field_shape = "layer_cluster_mesofield"
    else:
        field_shape = "compact_mesofield"

    next_steps = []
    if gate_predictability < 0.90:
        next_steps.append("Improve G recurrence fitting, especially cross-layer nonlinear terms.")
    if distributed_ratio >= 0.40:
        next_steps.append("Prioritize cross-layer head-group and layer-cluster ablations over single-head search.")
    if boundaryability_score < 0.60:
        next_steps.append("Expand more relation families and concepts to refine protocol-field boundaries.")
    if float(real_summary["real_closure_score"]) < 0.80:
        next_steps.append("Scale the real multi-step benchmark to harder task graphs and longer horizons.")
    if float(real_summary["retention_gain"]) < 0.65:
        next_steps.append("Strengthen continual retention under phase switches in the real multi-step setting.")
    next_steps.append("Test whether mechanism score remains predictive under real agent tasks.")

    return {
        "model_name": model_name,
        "gate": {
            "gate_predictability": gate_predictability,
            "linear_gain": linear_gain,
            "nonlinear_gain": nonlinear_gain,
            "linear_gain_norm": linear_gain_norm,
            "nonlinear_gain_norm": nonlinear_gain_norm,
        },
        "field": {
            "compact_ratio": compact_ratio,
            "layer_cluster_ratio": layer_cluster_ratio,
            "distributed_ratio": distributed_ratio,
            "boundaryability_score": boundaryability_score,
            "field_shape": field_shape,
            "classification_histogram": class_hist,
        },
        "bridge": {
            "mechanism_score": mechanism_score,
            "toy_closure_score": float(toy_summary["toy_closure_score"]),
            "real_closure_score": float(real_summary["real_closure_score"]),
            "agi_bridge_toy_score": agi_bridge_toy_score,
            "agi_bridge_real_score": agi_bridge_real_score,
            "agi_bridge_score": agi_bridge_real_score,
        },
        "recommended_next_steps": next_steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build mechanism-to-AGI bridge summary")
    ap.add_argument(
        "--json-out",
        type=str,
        default=str(ROOT / "tests" / "codex_temp" / "gpt2_qwen3_mechanism_agi_bridge_20260308.json"),
    )
    args = ap.parse_args()

    gate_payload = load_json(GATE_JSON)
    boundary_payload = load_json(BOUNDARY_JSON)
    toy_payload = load_json(TOY_JSON)
    real_payload = load_json(REAL_JSON)
    toy_summary = build_toy_summary(toy_payload)
    real_summary = build_real_summary(real_payload)

    models = {}
    for model_name, gate_row in gate_payload.get("models", {}).items():
        boundary_row = boundary_payload.get("models", {}).get(model_name, {})
        models[model_name] = build_model_summary(model_name, gate_row, boundary_row, toy_summary, real_summary)

    ranking = [
        {
            "model_name": model_name,
            "agi_bridge_score": row["bridge"]["agi_bridge_score"],
            "mechanism_score": row["bridge"]["mechanism_score"],
            "field_shape": row["field"]["field_shape"],
        }
        for model_name, row in sorted(models.items(), key=lambda item: item[1]["bridge"]["agi_bridge_score"], reverse=True)
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_gate_json": str(GATE_JSON.relative_to(ROOT)).replace("\\", "/"),
            "source_boundary_json": str(BOUNDARY_JSON.relative_to(ROOT)).replace("\\", "/"),
            "source_toy_json": str(TOY_JSON.relative_to(ROOT)).replace("\\", "/"),
            "source_real_json": str(REAL_JSON.relative_to(ROOT)).replace("\\", "/"),
        },
        "toy_bridge": toy_summary,
        "real_bridge": real_summary,
        "models": models,
        "ranking": ranking,
        "global_conclusion": {
            "statement": "Mechanistic interpretability is now quantifiable, and the bridge score should be anchored on real multi-step closure rather than toy-only proxies.",
            "why": [
                "Gate recurrence is already highly predictable, but larger models rely more on distributed meso-fields.",
                "Protocol-field boundaries differ by model, so interpretability cannot collapse to single-module search.",
                "The real multi-step closure benchmark now shows a large trace-driven gain, so bridge scoring can move from proxy-only to task-grounded evaluation.",
            ],
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in ranking:
      print(
          f"[bridge] {row['model_name']} agi_bridge={row['agi_bridge_score']:.4f} "
          f"mechanism={row['mechanism_score']:.4f} field={row['field_shape']}"
      )
    print(f"[toy] closure_score={toy_summary['toy_closure_score']:.4f}")
    print(f"[real] closure_score={real_summary['real_closure_score']:.4f} gain={real_summary['score_gain']:.4f}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
