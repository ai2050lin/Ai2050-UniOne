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

    train = load_json(TEMP / "stage_icspb_lm_phasea_openwebtext_training_block.json")
    gen = load_json(TEMP / "stage_icspb_lm_phasea_openwebtext_generation_benchmark.json")
    total = load_json(TEMP / "theory_track_icspb_lm_phasea_training_assessment.json")

    train_score = float(train["headline_metrics"]["stage_score"])
    gen_score = float(gen["headline_metrics"]["benchmark_score"])
    total_score = float(total["headline_metrics"]["assessment_score"])

    assessment_score = clamp01(
        0.35 * train_score
        + 0.35 * gen_score
        + 0.30 * total_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_ICSPB_LM_PhaseA_Generation_Assessment",
        },
        "headline_metrics": {
            "training_score": train_score,
            "generation_score": gen_score,
            "phasea_total_score": total_score,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.55,
            "phasea_language_chain_ready": assessment_score >= 0.70,
            "core_answer": "PhaseA 已经从参数扩容进入训练和真实文本 continuation 阶段，但还没达到能接管完整语言主链的强度。",
        },
    }

    out_file = TEMP / "theory_track_icspb_lm_phasea_generation_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
