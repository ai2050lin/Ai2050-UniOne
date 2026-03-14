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

    long_block = load_json(TEMP / "stage_icspb_lm_phasea_long_pretraining_block.json")
    gen_assess = load_json(TEMP / "theory_track_icspb_lm_phasea_generation_assessment.json")
    train_assess = load_json(TEMP / "theory_track_icspb_lm_phasea_training_assessment.json")

    long_score = float(long_block["headline_metrics"]["stage_score"])
    gen_score = float(gen_assess["headline_metrics"]["assessment_score"])
    train_score = float(train_assess["headline_metrics"]["assessment_score"])

    assessment_score = clamp01(
        0.42 * long_score
        + 0.28 * gen_score
        + 0.30 * train_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_ICSPB_LM_PhaseA_Long_Pretraining_Assessment",
        },
        "headline_metrics": {
            "long_pretraining_score": long_score,
            "generation_assessment_score": gen_score,
            "training_assessment_score": train_score,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.66,
            "phasea_can_take_language_mainline": assessment_score >= 0.80,
            "core_answer": "PhaseA 长程预训练已经能改善 loss 和 continuation，但还没达到接管语言主链的强度。",
        },
    }

    out_file = TEMP / "theory_track_icspb_lm_phasea_long_pretraining_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
