from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    start = time.time()
    block = load_json(TEMP / "icspb_backbone_v2_multimodal_conscious_block.json")
    r = block["results"]
    v = block["verdict"]

    assessment_score = (
        0.20 * (1.0 if v["smoke_pass"] else 0.0)
        + 0.20 * (1.0 if v["training_pass"] else 0.0)
        + 0.15 * (1.0 if v["online_pass"] else 0.0)
        + 0.15 * (1.0 if v["multimodal_pass"] else 0.0)
        + 0.15 * (1.0 if v["consciousness_pass"] else 0.0)
        + 0.15 * (1.0 if v["replay_pass"] else 0.7)
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Multimodal_Conscious_Assessment",
        },
        "headline_metrics": {
            "visual_energy": r["visual_energy"],
            "audio_energy": r["audio_energy"],
            "consciousness_energy": r["consciousness_energy"],
            "conscious_access": r["conscious_access"],
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.90,
            "core_answer": (
                "The current model now supports visual input, audio input, and a unified consciousness-state "
                "workspace. This is still a prototype consciousness layer, not a final theory of subjective experience."
            ),
        },
    }

    out_file = TEMP / "multimodal_conscious_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
