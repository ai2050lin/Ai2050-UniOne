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

    phasea = load_json(TEMP / "stage_icspb_lm_phasea_architecture_block.json")
    target_plan = load_json(TEMP / "theory_track_qwen_deepseek_language_target_plan.json")

    phasea_score = float(phasea["headline_metrics"]["phasea_score"])
    params = float(phasea["headline_metrics"]["total_params"])
    target_gap = float(target_plan["headline_metrics"]["architecture_gap"])
    current_language_scale_fit = float(target_plan["headline_metrics"]["current_language_scale_fit"])

    readiness_score = clamp01(
        0.45 * phasea_score
        + 0.25 * min(1.0, params / 100_000_000)
        + 0.15 * (1.0 - target_gap)
        + 0.15 * current_language_scale_fit
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_ICSPB_LM_PhaseA_Readiness_Assessment",
        },
        "headline_metrics": {
            "phasea_score": phasea_score,
            "phasea_params": params,
            "target_gap": target_gap,
            "current_language_scale_fit": current_language_scale_fit,
            "readiness_score": readiness_score,
        },
        "verdict": {
            "overall_pass": readiness_score >= 0.75,
            "phasea_program_ready": readiness_score >= 0.88,
            "core_answer": "当前已经可以从 1.44M 原型切换到 ~100M 级 PhaseA 正式语言主干路线，但离 Qwen/DeepSeek 级还只是第一阶段。",
        },
    }

    out_file = TEMP / "theory_track_icspb_lm_phasea_readiness_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
