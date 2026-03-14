from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    start = time.time()

    feasibility = load_json(TEMP / "theory_track_dnn_language_plus_instant_learning_feasibility.json")
    scaleup = load_json(TEMP / "theory_track_agi_chat_language_scaleup_assessment.json")
    runtime_summary = load_json(TEMP / "system_status_runtime_summary_block.json")

    language_readiness = float(feasibility["headline_metrics"]["dnn_language_target_readiness"])
    instant_readiness = float(feasibility["headline_metrics"]["human_instant_learning_readiness"])
    joint_feasibility = float(feasibility["headline_metrics"]["route_joint_feasibility"])
    scaleup_score = float(scaleup["headline_metrics"]["assessment_score"])
    runtime_semantic = float(runtime_summary["headline_metrics"]["semantic_benchmark_score"])

    language_emergence_support = clamp01(
        0.45 * language_readiness
        + 0.35 * scaleup_score
        + 0.20 * runtime_semantic
    )
    instant_learning_gap = clamp01(1.0 - instant_readiness)
    dual_timescale_balance = clamp01(
        0.50 * language_emergence_support
        + 0.50 * instant_readiness
    )
    route_validity = clamp01(
        0.40 * joint_feasibility
        + 0.35 * language_emergence_support
        + 0.25 * dual_timescale_balance
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Language_Emergence_Instant_Learning_Route_Reassessment",
        },
        "headline_metrics": {
            "language_emergence_support": language_emergence_support,
            "instant_learning_gap": instant_learning_gap,
            "dual_timescale_balance": dual_timescale_balance,
            "route_validity": route_validity,
        },
        "verdict": {
            "route_is_fundamentally_wrong": route_validity < 0.45,
            "route_is_valid_but_incomplete": route_validity >= 0.45 and dual_timescale_balance < 0.82,
            "route_requires_dual_track_execution": True,
            "core_answer": (
                "当前路线不是错在“既想要大模型式语言能力，又想要人类式即时学习”，"
                "而是错在如果把两者当成同一训练阶段自然同时完成，就会严重高估即时学习进度。"
                "更准确的做法是：一条大规模语言预训练主干负责语言涌现，"
                "一条 fast-write / slow-consolidation / replay / retention 主干负责人类式即时学习，"
                "两条主干在统一 ICSPB 结构下汇合。"
            ),
        },
        "recommended_shift": {
            "track_1": "继续扩大 token-level 语言主干，把语言能力推进到接近现有强 DNN 的水平",
            "track_2": "把即时学习从附属能力改成独立主目标，专门训练 retention / low-interference / carryover",
            "track_3": "在统一模型中通过 dual-timescale write/read regime 汇合两条主线",
        },
    }

    out_file = TEMP / "theory_track_language_emergence_instant_learning_route_reassessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
