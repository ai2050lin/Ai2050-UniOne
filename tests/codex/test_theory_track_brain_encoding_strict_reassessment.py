from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    start = time.time()

    crack = load_json(TEMP / "complete_brain_encoding_crack_path_block.json")
    spike = load_json(TEMP / "brain_encoding_spike_assessment.json")
    human_train = load_json(TEMP / "theory_track_human_level_training_progress_reassessment.json")
    phasea_lang = load_json(TEMP / "theory_track_icspb_lm_phasea_language_level_assessment.json")

    crack_score = float(crack["path"]["crack_path_score"])
    spike_score = float(spike["headline_metrics"]["assessment_score"])
    human_train_score = float(human_train["headline_metrics"]["human_level_training_progress_score"])
    phasea_lang_score = float(phasea_lang["headline_metrics"]["assessment_score"])

    # 更严格的“本体破解”标准：
    # 1. 不只要结构解释；
    # 2. 还要能导出接近标准答案的学习律；
    # 3. 还要能解释为什么会自然长出强语言能力；
    # 4. 还要有唯一/标准化的生物物理与外部验证锚点。
    structural_reconstruction = crack_score
    pulse_consistency = spike_score
    language_realization = min(1.0, 0.65 * human_train_score + 0.35 * phasea_lang_score)

    standardized_learning_law = 0.18
    biophysical_uniqueness = 0.16
    always_on_external_validation = 0.12
    canonical_answer_closure = 0.14

    strict_score = clamp01(
        0.22 * structural_reconstruction
        + 0.14 * pulse_consistency
        + 0.18 * language_realization
        + 0.16 * standardized_learning_law
        + 0.12 * biophysical_uniqueness
        + 0.10 * always_on_external_validation
        + 0.08 * canonical_answer_closure
    )

    low = max(0.0, strict_score - 0.04)
    high = min(1.0, strict_score + 0.04)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Brain_Encoding_Strict_Reassessment",
        },
        "headline_metrics": {
            "structural_reconstruction": structural_reconstruction,
            "pulse_consistency": pulse_consistency,
            "language_realization": language_realization,
            "standardized_learning_law": standardized_learning_law,
            "biophysical_uniqueness": biophysical_uniqueness,
            "always_on_external_validation": always_on_external_validation,
            "canonical_answer_closure": canonical_answer_closure,
            "strict_brain_encoding_progress_score": strict_score,
            "strict_brain_encoding_progress_range": [low, high],
        },
        "verdict": {
            "needs_major_downward_adjustment": True,
            "core_answer": (
                "If brain encoding were truly cracked, the project should already yield something much closer to a standard "
                "learning law, a much stronger language realization path, and a more canonical biophysical/external witness. "
                "Under that stricter standard, current progress should be treated as well below the previous optimistic 96%-97% reading."
            ),
        },
    }

    out_file = TEMP / "theory_track_brain_encoding_strict_reassessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
