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

from research.gpt5.code.dnn_thousand_noun_family_patch_offset_program import DnnThousandNounFamilyPatchOffsetProgram  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    program = DnnThousandNounFamilyPatchOffsetProgram.from_repo(ROOT)
    summary = program.summary()
    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "dnn_thousand_noun_family_patch_offset_program_block",
        },
        "strict_goal": {
            "statement": "Evaluate whether thousands-scale noun analysis is feasible for family patch plus concept offset, and convert that into a concrete execution program.",
            "boundary": "This block evaluates feasibility and program structure. It does not claim that scaling nouns alone is enough to crack full language coding or final theorem closure.",
        },
        "headline_metrics": {
            "metric_lines_cn": summary["metric_lines_cn"],
        },
        "program": summary["program"],
        "strict_conclusion": summary["strict_conclusion"],
    }


def test_dnn_thousand_noun_family_patch_offset_program_block() -> None:
    payload = build_payload()
    assert len(payload["headline_metrics"]["metric_lines_cn"]) >= 6
    assert payload["headline_metrics"]["metric_lines_cn"][0].startswith("（")
    program = payload["program"]
    assert program["current_base"]["current_mass_scan_count"] >= 5
    assert program["current_base"]["inventory_concepts"] >= 300
    assert program["thousand_noun_target"]["projected_batches"] >= 20
    assert len(program["execution_blocks"]) >= 3


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN thousand-noun family patch/offset program block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_thousand_noun_family_patch_offset_program_block_20260316.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
