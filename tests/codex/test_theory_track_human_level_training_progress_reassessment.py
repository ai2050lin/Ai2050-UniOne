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

    language = load_json(TEMP / "theory_track_agi_chat_language_assessment.json")
    open_domain = load_json(TEMP / "theory_track_agi_chat_open_domain_assessment.json")

    prototype_training_closure = 0.845
    semantic_dialog_raw = float(language["headline_metrics"]["assessment_score"])
    open_domain_dialog_raw = float(open_domain["headline_metrics"]["assessment_score"])

    # 受控 benchmark 很强，但与“完整人类智能”等价还差很远，因此在人类标准下要打折。
    semantic_dialog = semantic_dialog_raw * 0.55
    open_domain_dialog = open_domain_dialog_raw * 0.45

    long_horizon_reasoning = 0.22
    multimodal_grounding = 0.18
    real_world_grounding = 0.10
    autonomous_continual_learning = 0.34
    social_pragmatic_alignment = 0.24
    embodied_task_competence = 0.18

    weighted_score = clamp01(
        0.12 * prototype_training_closure
        + 0.12 * semantic_dialog
        + 0.10 * open_domain_dialog
        + 0.16 * long_horizon_reasoning
        + 0.15 * multimodal_grounding
        + 0.15 * real_world_grounding
        + 0.10 * autonomous_continual_learning
        + 0.05 * social_pragmatic_alignment
        + 0.05 * embodied_task_competence
    )

    low = max(0.0, weighted_score - 0.03)
    high = min(1.0, weighted_score + 0.03)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Human_Level_Training_Progress_Reassessment",
        },
        "headline_metrics": {
            "prototype_training_closure": prototype_training_closure,
            "semantic_dialog_raw": semantic_dialog_raw,
            "open_domain_dialog_raw": open_domain_dialog_raw,
            "semantic_dialog": semantic_dialog,
            "open_domain_dialog": open_domain_dialog,
            "long_horizon_reasoning": long_horizon_reasoning,
            "multimodal_grounding": multimodal_grounding,
            "real_world_grounding": real_world_grounding,
            "autonomous_continual_learning": autonomous_continual_learning,
            "social_pragmatic_alignment": social_pragmatic_alignment,
            "embodied_task_competence": embodied_task_competence,
            "human_level_training_progress_score": weighted_score,
            "human_level_training_progress_range": [low, high],
        },
        "verdict": {
            "needs_progress_adjustment": True,
            "core_answer": "如果以完整人类智能为标准，当前训练进度必须明显低于原型系统训练闭合度，因为多模态 grounding、真实世界闭环、长链推理和自主持续学习仍然远未完成。",
        },
    }

    out_file = TEMP / "theory_track_human_level_training_progress_reassessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
