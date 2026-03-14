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
    replay = load_json(TEMP / "icspb_backbone_v2_strict_replay_closure_block.json")
    r = replay["results"]

    recovery_ratio = float(r["replay_recovery_ratio"])
    stable_read = float(r["stable_read"])
    guarded_write = float(r["guarded_write"])
    theorem_survival = float(r["theorem_survival"])
    assessment_score = (
        0.40 * recovery_ratio
        + 0.18 * stable_read
        + 0.18 * guarded_write
        + 0.14 * theorem_survival
        + 0.10 * (1.0 if r["replay_total_loss"] < 2.0 else 0.75)
    )
    closure_bonus = 0.0
    if recovery_ratio > 0.75 and stable_read >= 1.0 and guarded_write >= 1.0 and theorem_survival >= 1.0:
        closure_bonus = 0.10
    assessment_score = min(1.0, assessment_score + closure_bonus)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Strict_Replay_Closure_Assessment",
        },
        "headline_metrics": {
            "replay_recovery_ratio": recovery_ratio,
            "stable_read": stable_read,
            "guarded_write": guarded_write,
            "theorem_survival": theorem_survival,
            "closure_bonus": closure_bonus,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.80,
            "strict_replay_pass": (
                recovery_ratio > 0.75 and stable_read >= 1.0 and guarded_write >= 1.0 and theorem_survival >= 1.0
            ),
            "core_answer": (
                "Strict replay closure requires recovery of structure and recovery of the read/write theorem-safe operating regime."
            ),
        },
    }

    out_file = TEMP / "strict_replay_closure_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
