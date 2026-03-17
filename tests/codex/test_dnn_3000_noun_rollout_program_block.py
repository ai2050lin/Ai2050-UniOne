from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gpt5.code.dnn_3000_noun_rollout_program import Dnn3000NounRolloutProgram  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    program = Dnn3000NounRolloutProgram.from_repo(ROOT)
    summary = program.summary()
    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "dnn_3000_noun_rollout_program_block",
        },
        "strict_goal": {
            "statement": "Complete the three-stage noun-scaling route: hundreds-scale baseline, 1000+ source expansion, and a concrete 3000-noun rollout program.",
            "boundary": "This block completes the rollout structure. It does not run the full 3000-noun dense harvest itself.",
        },
        "headline_metrics": {
            "metric_lines_cn": summary["metric_lines_cn"],
        },
        "phases": summary["phases"],
        "gap_from_1000plus_to_3000": summary["gap_from_1000plus_to_3000"],
        "strict_conclusion": summary["strict_conclusion"],
    }


def test_dnn_3000_noun_rollout_program_block() -> None:
    payload = build_payload()
    assert len(payload["headline_metrics"]["metric_lines_cn"]) >= 6
    assert payload["headline_metrics"]["metric_lines_cn"][0].startswith("（")
    assert len(payload["phases"]) == 3
    assert payload["phases"][0]["target_nouns"] == 280
    assert payload["phases"][1]["target_nouns"] >= 1000
    assert payload["phases"][2]["target_nouns"] == 3000
    assert payload["gap_from_1000plus_to_3000"] > 1000


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN 3000 noun rollout program block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_3000_noun_rollout_program_block_20260316.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
