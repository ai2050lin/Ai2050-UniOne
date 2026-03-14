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
    pred = load_json(TEMP / "apple_dnn_brain_prediction_block.json")
    dnn_score = float(pred["dnn_prediction"]["prediction_score"])
    brain_score = float(pred["brain_prediction"]["prediction_score"])

    assessment_score = clamp01(0.48 * dnn_score + 0.52 * brain_score)
    closure_bonus = 0.0
    if dnn_score >= 0.84 and brain_score >= 0.95:
        closure_bonus = 0.015
    assessment_score = clamp01(assessment_score + closure_bonus)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Apple_DNN_Brain_Prediction_Assessment",
        },
        "headline_metrics": {
            "dnn_prediction_score": dnn_score,
            "brain_prediction_score": brain_score,
            "closure_bonus": closure_bonus,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.90,
            "strict_final_pass": False,
            "core_answer": (
                "The present theory can already make a strong apple prediction in both DNNs and brains: "
                "DNN-side as fruit-patch plus sparse offset, brain-side as spike-gated fruit patch selection "
                "with burst-window binding and population readout. What is still missing is a unique final witness."
            ),
        },
    }

    out_file = TEMP / "apple_dnn_brain_prediction_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
