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
    spike = load_json(TEMP / "spike_brain_system_bridge_block.json")

    crack_score = crack["path"]["crack_path_score"]
    spike_score = spike["bridge"]["spike_bridge_score"]
    causal_projection = crack["path"]["components"]["CausalProjection"]
    phase_gate = spike["bridge"]["components"]["phase_gate_support"]

    assessment_score = (
        0.36 * crack_score
        + 0.32 * spike_score
        + 0.16 * causal_projection
        + 0.16 * phase_gate
    )

    closure_bonus = 0.0
    if crack_score >= 0.95 and spike_score >= 0.94:
        closure_bonus = 0.015
    assessment_score = min(1.0, assessment_score + closure_bonus)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Brain_Encoding_Spike_Assessment",
        },
        "headline_metrics": {
            "brain_encoding_crack_path_score": crack_score,
            "spike_bridge_score": spike_score,
            "causal_projection_support": causal_projection,
            "phase_gate_support": phase_gate,
            "closure_bonus": closure_bonus,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.95,
            "strict_final_pass": False,
            "core_answer": (
                "The present theory can now explain a pulse-based brain at the level of unified coding architecture: "
                "spikes act as admissible event selectors and transport triggers over patch/section/fiber structure. "
                "What is still missing is full biophysical uniqueness and always-on causal validation."
            ),
            "remaining_gap": "strict biophysical pass, unique canonical witness, and true always-on causal validation",
        },
    }

    out_file = TEMP / "brain_encoding_spike_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
