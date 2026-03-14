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
    sprint = load_json(TEMP / "final_closure_sprint_block.json")
    closure_score = float(sprint["headline_metrics"]["closure_score"])
    strict_count = int(sprint["headline_metrics"]["strict_count"])

    assessment_score = min(1.0, 0.82 * closure_score + 0.18 * (strict_count / 5.0))
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Final_Closure_Sprint_Assessment",
        },
        "headline_metrics": {
            "closure_score": closure_score,
            "strict_count": strict_count,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.85,
            "full_closed": bool(sprint["verdict"]["full_closed"]),
            "core_answer": (
                "The system is close enough to expose the final blockers clearly, but not yet closed enough to claim final strict completion."
            ),
        },
    }

    out_file = TEMP / "final_closure_sprint_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
