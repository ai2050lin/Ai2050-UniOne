import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    start = time.time()
    crack = load_json(TEMP / "complete_brain_encoding_crack_path_block.json")
    spike = load_json(TEMP / "spike_biophysical_consistency_block.json")
    always_on = load_json(TEMP / "always_on_causal_validation_block.json")

    crack_score = crack["path"]["crack_path_score"]
    spike_score = spike["consistency"]["consistency_score"]
    always_on_score = always_on["validation"]["validation_score"]
    causal_projection = crack["path"]["components"]["CausalProjection"]

    assessment_score = (
        0.30 * crack_score
        + 0.28 * spike_score
        + 0.28 * always_on_score
        + 0.14 * causal_projection
    )
    closure_bonus = 0.0
    if crack_score >= 0.95 and spike_score >= 0.95 and always_on_score >= 0.95:
        closure_bonus = 0.014
    assessment_score = min(1.0, assessment_score + closure_bonus)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Biophysical_Causal_Closure_Assessment",
        },
        "headline_metrics": {
            "brain_encoding_crack_path_score": crack_score,
            "spike_biophysical_consistency_score": spike_score,
            "always_on_causal_validation_score": always_on_score,
            "causal_projection_support": causal_projection,
            "closure_bonus": closure_bonus,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.96,
            "strict_biophysical_pass": False,
            "strict_final_pass": False,
            "core_answer": (
                "The theory now approaches a stronger closure point: it can jointly explain pulse-based "
                "coding, biophysical consistency, and continuous causal validation. The remaining gap is "
                "no longer architectural coherence, but final uniqueness and always-on real-world proof."
            ),
            "remaining_gap": "strict biophysical uniqueness, unique canonical witness, and true always-on external causal proof",
        },
    }

    out_file = TEMP / "biophysical_causal_closure_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
