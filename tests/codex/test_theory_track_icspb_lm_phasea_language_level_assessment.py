from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"
TEMP.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    stage = load_json(TEMP / "stage_icspb_lm_phasea_language_level_block.json")
    long_pre = load_json(TEMP / "stage_icspb_lm_phasea_long_pretraining_block.json")
    gen = load_json(TEMP / "stage_icspb_lm_phasea_openwebtext_generation_benchmark.json")

    stage_score = float(stage["headline_metrics"]["stage_score"])
    final_match = float(stage["headline_metrics"]["final_match"])
    final_distinct = float(stage["headline_metrics"]["final_distinct"])
    final_collapse = float(stage["headline_metrics"]["final_collapse"])
    long_score = float(long_pre["headline_metrics"]["stage_score"])
    gen_score = float(gen["headline_metrics"]["benchmark_score"])

    assessment_score = min(
        1.0,
        0.42 * stage_score
        + 0.24 * long_score
        + 0.20 * gen_score
        + 0.08 * min(1.0, final_match / 0.22)
        + 0.06 * min(1.0, final_distinct / 0.50),
    )
    if final_collapse <= 0.35:
        assessment_score = min(1.0, assessment_score + 0.03)

    if assessment_score >= 0.88 and final_match >= 0.18:
        level = "可用原型级语言主干"
    elif assessment_score >= 0.72 and final_match >= 0.10:
        level = "早期语言先验形成"
    elif assessment_score >= 0.58:
        level = "弱 continuation 原型"
    else:
        level = "未形成可用语言主干"

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "TheoryTrack_ICSPB_LM_PhaseA_Language_Level_Assessment",
        },
        "headline_metrics": {
            "stage_score": stage_score,
            "long_pretraining_score": long_score,
            "generation_score": gen_score,
            "final_match": final_match,
            "final_distinct": final_distinct,
            "final_collapse": final_collapse,
            "assessment_score": assessment_score,
            "language_level": level,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.70,
            "phasea_language_level_ready": assessment_score >= 0.84,
            "core_answer": "PhaseA 训练后语言能力目前属于哪一档，取决于 continuation、distinctness、collapse 和长程预训练综合结果。",
        },
    }

    out_file = TEMP / "theory_track_icspb_lm_phasea_language_level_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
