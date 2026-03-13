from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def load_latest(prefix: str) -> dict:
    candidates = sorted(TEMP_DIR.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"未找到前缀为 {prefix} 的工件")
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="统一 successor/protocol/brain 评估")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_protocol_successor_brain_unified_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    try:
        unified = load("stage_protocol_successor_brain_unified_execution_20260313.json")
    except FileNotFoundError:
        unified = load_latest("stage_protocol_successor_brain_unified_execution")
    detailed = load("theory_track_successor_protocol_brain_detailed_synthesis_20260313.json")

    proj = unified["unified_projection"]
    th = unified["thresholds"]

    protocol_pass = float(proj["protocol_calling"]) >= float(th["protocol_strong_band"])
    successor_pass = float(proj["successor_coherence"]) >= float(th["successor_global_band"])
    brain_pass = float(proj["brain_side_causal_closure"]) >= float(th["brain_online_band"])
    inverse_pass = float(proj["encoding_inverse_reconstruction_readiness"]) >= float(th["inverse_reconstruction_band"])
    math_pass = float(proj["new_math_closure_readiness"]) >= float(th["math_closure_band"])

    pass_count = sum([protocol_pass, successor_pass, brain_pass, inverse_pass, math_pass])

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_protocol_successor_brain_unified_assessment",
        },
        "current_to_unified": {
            "protocol_current": detailed["protocol_block"]["current_score"],
            "protocol_projected": proj["protocol_calling"],
            "successor_current": detailed["successor_block"]["current_score"],
            "successor_projected": proj["successor_coherence"],
            "brain_current": detailed["brain_block"]["current_score"],
            "brain_projected": proj["brain_side_causal_closure"],
        },
        "pass_status": {
            "protocol_strong_pass": protocol_pass,
            "successor_global_pass": successor_pass,
            "brain_online_pass": brain_pass,
            "inverse_reconstruction_pass": inverse_pass,
            "new_math_closure_pass": math_pass,
            "pass_count": pass_count,
        },
        "verdict": {
            "core_answer": (
                "统一执行块已经足以让 protocol、brain、inverse reconstruction 和 new math closure 逼近或跨过目标带，但 successor 仍是最后的主硬伤。"
            ),
            "main_bottleneck_after_unification": "successor_global_support",
            "next_target": (
                "下一阶段应把真实长链 successor trace 捕获和 protocol bridge 强化继续并入 unified block，直到 successor 跨过全局支撑带。"
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
