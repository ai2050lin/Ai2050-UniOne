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

    arch = load_json(TEMP / "stage_icspb_lm_phasea_architecture_block.json")
    train = load_json(TEMP / "stage_icspb_lm_phasea_openwebtext_training_block.json")
    readiness = load_json(TEMP / "theory_track_icspb_lm_phasea_readiness_assessment.json")

    arch_score = float(arch["headline_metrics"]["phasea_score"])
    train_score = float(train["headline_metrics"]["stage_score"])
    readiness_score = float(readiness["headline_metrics"]["readiness_score"])

    assessment_score = clamp01(
        0.34 * arch_score
        + 0.40 * train_score
        + 0.26 * readiness_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_ICSPB_LM_PhaseA_Training_Assessment",
        },
        "headline_metrics": {
            "architecture_score": arch_score,
            "training_score": train_score,
            "readiness_score": readiness_score,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.74,
            "phasea_training_program_ready": assessment_score >= 0.85,
            "core_answer": "PhaseA 现在已经从纯架构设计推进到小规模真实文本训练可行，但还没有进入正式长程训练闭合。",
        },
    }

    out_file = TEMP / "theory_track_icspb_lm_phasea_training_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
