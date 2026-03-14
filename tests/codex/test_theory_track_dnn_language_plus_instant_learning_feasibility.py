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

    language = load_json(TEMP / "theory_track_agi_chat_language_training_closure_assessment.json")
    human_progress = load_json(TEMP / "theory_track_human_level_training_progress_reassessment.json")
    f7 = load_json(TEMP / "f7_human_language_instant_learning_architecture_20260311.json")
    g3 = load_json(TEMP / "g3_instant_learning_boundary_stress_20260311.json")
    g7 = load_json(TEMP / "g7_strong_retention_instant_learning_closure_20260311.json")

    language_score = float(language["headline_metrics"]["assessment_score"])
    semantic_dialog_raw = float(human_progress["headline_metrics"]["semantic_dialog_raw"])
    open_domain_dialog_raw = float(human_progress["headline_metrics"]["open_domain_dialog_raw"])
    human_level_training = float(human_progress["headline_metrics"]["human_level_training_progress_score"])

    instant_architecture = float(f7["headline_metrics"]["instant_learning_readiness_score"])
    instant_boundary = float(g3["headline_metrics"]["overall_g3_score"])
    strong_retention = float(g7["headline_metrics"]["overall_g7_score"])

    dnn_language_target_readiness = clamp01(
        0.35 * language_score
        + 0.35 * semantic_dialog_raw
        + 0.30 * open_domain_dialog_raw
    )
    human_instant_learning_readiness = clamp01(
        0.35 * instant_architecture
        + 0.30 * instant_boundary
        + 0.35 * strong_retention
    )
    route_joint_feasibility = clamp01(
        0.45 * dnn_language_target_readiness
        + 0.35 * human_instant_learning_readiness
        + 0.20 * human_level_training
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_DNN_Language_Plus_Instant_Learning_Feasibility",
        },
        "headline_metrics": {
            "dnn_language_target_readiness": dnn_language_target_readiness,
            "human_instant_learning_readiness": human_instant_learning_readiness,
            "route_joint_feasibility": route_joint_feasibility,
            "current_human_level_training_progress": human_level_training,
        },
        "supporting_readout": {
            "language_training_closure_score": language_score,
            "semantic_dialog_raw": semantic_dialog_raw,
            "open_domain_dialog_raw": open_domain_dialog_raw,
            "instant_learning_architecture": instant_architecture,
            "instant_learning_boundary": instant_boundary,
            "strong_retention_score": strong_retention,
        },
        "verdict": {
            "route_can_reach_target_in_principle": route_joint_feasibility >= 0.62,
            "current_implementation_is_sufficient": False,
            "language_side_is_much_stronger_than_instant_learning": dnn_language_target_readiness > human_instant_learning_readiness,
            "main_open_gap": "strong_retention_and_low_interference_instant_learning",
            "core_answer": (
                "当前路线在原理上可以同时指向强语言能力和即时学习，但当前实现只是在语言原型上接近成熟，"
                "即时学习仍然明显卡在强保留、低干扰和跨环境稳定迁移。"
            ),
        },
        "recommended_route": {
            "stage_1": "把语言主回路从原型问答系统升级成更大规模的 token-level 语言训练主干",
            "stage_2": "把 fast-write / slow-consolidation / replay 变成训练主目标，而不是只做分析模块",
            "stage_3": "用高干扰 one-shot / few-shot 任务训练 strong retention 和 cross-environment carryover",
            "stage_4": "把多模态 grounding 与 dialogue memory 直接接入答案生成和在线更新",
            "stage_5": "再做真实外部世界下的 always-on semantic validation",
        },
    }

    out_file = TEMP / "theory_track_dnn_language_plus_instant_learning_feasibility.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
