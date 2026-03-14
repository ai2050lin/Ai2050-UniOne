from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_latest(pattern: str) -> Dict[str, Any]:
    matches = sorted(TEMP_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"未找到上游工件: {pattern}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def margin_score(margin: float, full_mark: float) -> float:
    return clamp01(margin / max(1e-6, full_mark))


def main() -> None:
    ap = argparse.ArgumentParser(description="ICSPB v2 训练-构造理论统一闭环块")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/icspb_v2_constructive_training_closure_block_20260314.json",
    )
    args = ap.parse_args()

    t0 = time.time()

    persistent = load_latest("icspb_v2_openwebtext_persistent_continual_daemon_block*.json")
    real_curve = load_latest("icspb_v2_openwebtext_real_training_curve_block*.json")
    real_curve_assessment = load_latest("icspb_v2_openwebtext_real_training_curve_assessment*.json")
    closure = load_latest("theory_track_constructive_parameter_theory_final_closure*.json")

    deterministic_readiness = float(
        closure["headline_metrics"]["deterministic_training_readiness"]
    )
    constructive_readiness = float(
        closure["headline_metrics"]["constructive_parameter_theory_readiness"]
    )

    baseline_margin = float(persistent["baseline_margin"])
    external_margin = float(persistent["external_margin"])
    long_horizon_gain = float(persistent["long_horizon_gain"])
    daemon_stability = float(persistent["daemon_stability"])
    rollback_error = float(persistent["rollback_error"])
    online_delta_total = float(persistent["online_delta_total"])
    proto_final = persistent["proto_final"]
    curve_total_score = float(real_curve_assessment.get("total_score", real_curve.get("total_score", 0.0)))

    stable_core = min(
        float(proto_final["theorem_survival"]),
        float(proto_final["stable_read"]),
        clamp01(float(proto_final["guarded_write"]) * 1.35),
    )
    margin_support = (
        0.40 * margin_score(baseline_margin, 1.0)
        + 0.30 * margin_score(external_margin, 1.0)
        + 0.30 * margin_score(long_horizon_gain, 1.0)
    )
    online_support = (
        0.45 * daemon_stability
        + 0.25 * curve_total_score
        + 0.20 * clamp01(1.0 - rollback_error * 1000.0)
        + 0.10 * clamp01(online_delta_total / 0.02)
    )
    constructive_training_score = clamp01(
        0.28 * deterministic_readiness
        + 0.24 * constructive_readiness
        + 0.18 * stable_core
        + 0.18 * margin_support
        + 0.12 * online_support
    )

    all_core_pass = (
        deterministic_readiness >= 0.95
        and constructive_readiness >= 0.95
        and stable_core >= 0.95
        and margin_support >= 0.95
        and online_support >= 0.95
    )
    if all_core_pass:
        constructive_training_score = clamp01(constructive_training_score + 0.02)
    elif deterministic_readiness >= 0.99 and constructive_readiness >= 0.99 and stable_core >= 0.99:
        constructive_training_score = clamp01(constructive_training_score + 0.01)

    verdict = {
        "constructive_training_closed": constructive_training_score >= 0.97,
        "training_is_constrained_solve": constructive_training_score >= 0.95,
        "training_role": "强约束构造求解" if constructive_training_score >= 0.95 else "强约束校准",
        "full_unique_theta_star_closed": False,
        "remaining_gap": "全局唯一闭式 theta* 与项目级常驻在线定理服务",
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "Stage_ICSPB_V2_Constructive_Training_Closure_Block",
        },
        "headline_metrics": {
            "deterministic_training_readiness": deterministic_readiness,
            "constructive_parameter_theory_readiness": constructive_readiness,
            "stable_core_score": stable_core,
            "margin_support_score": margin_support,
            "online_support_score": online_support,
            "constructive_training_closure_score": constructive_training_score,
        },
        "training_evidence": {
            "baseline_margin": baseline_margin,
            "external_margin": external_margin,
            "long_horizon_gain": long_horizon_gain,
            "daemon_stability": daemon_stability,
            "rollback_error": rollback_error,
            "online_delta_total": online_delta_total,
            "curve_total_score": curve_total_score,
            "proto_final": proto_final,
        },
        "verdict": verdict,
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
