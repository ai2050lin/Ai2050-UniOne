from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Route from DNN extraction to brain encoding inverse reconstruction and new math")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inverse_brain_math_route_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    suff = load("theory_track_dnn_extraction_sufficiency_assessment_20260313.json")
    overall = load("theory_track_encoding_math_progress_overall_20260313.json")
    systemic = load("stage_systemic_closure_master_block_20260312.json")

    extraction = float(suff["assessment"]["dnn_extraction_sufficiency_score"])
    encoding = float(overall["progress"]["encoding_mechanism_readiness"])
    math_ready = float(overall["progress"]["new_math_system_readiness"])
    protocol = float(systemic["headline_metrics"]["protocol_calling"])
    successor = float(systemic["headline_metrics"]["successor_coherence"])
    brain = float(systemic["headline_metrics"]["brain_side_causal_closure"])

    inverse_route = clamp01(0.30 * extraction + 0.30 * encoding + 0.12 * protocol + 0.14 * successor + 0.14 * brain)
    math_route = clamp01(0.48 * math_ready + 0.20 * extraction + 0.12 * protocol + 0.10 * successor + 0.10 * brain)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inverse_brain_math_route",
        },
        "route_progress": {
            "dnn_extraction_to_inverse_brain_encoding": inverse_route,
            "dnn_extraction_to_new_math_closure": math_route,
        },
        "systemic_method": [
            "从深度神经网络中提取 family patch / relation fiber / protocol bridge / successor trace",
            "把这些结构压成统计不变量而不是单个案例",
            "用不变量收缩 ICSPB theorem、A(I)、M_feas(I) 和 intervention 空间",
            "再用 online brain-side causal execution 做 survival / rollback / recovery",
        ],
        "main_open_questions": [
            "如何把 successor 从 local support 推到 global support",
            "如何让 protocol bridge 贯穿 object/readout/successor",
            "如何把 brain-side causal execution 真正接进循环执行层",
            "如何让 stress_guarded_update 与 anchored_bridge_lift 进入 strict core",
        ],
        "verdict": {
            "core_answer": (
                "The viable route is now clear: large-scale DNN structural extraction is already enough to drive inverse reconstruction and mathematical pruning, but final closure still depends on protocol-successor-brain execution rather than more local statistics alone."
            ),
            "next_target": (
                "Keep extraction and intervention coupled; extraction without execution will saturate, and execution without extraction will lose theoretical constraint."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
