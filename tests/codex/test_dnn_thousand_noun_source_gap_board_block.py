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

from research.gpt5.code.dnn_thousand_noun_source_gap_board import DnnThousandNounSourceGapBoard  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    board = DnnThousandNounSourceGapBoard.from_repo(ROOT)
    summary = board.summary()
    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "dnn_thousand_noun_source_gap_board_block",
        },
        "strict_goal": {
            "statement": "Audit the real unique noun source behind the thousands-scale plan, and separate true noun scaling from prompt multiplication.",
            "boundary": "This block does not increase noun count. It clarifies the honest source gap behind the thousands-scale route.",
        },
        "headline_metrics": {
            "metric_lines_cn": summary["metric_lines_cn"],
        },
        "source_state": summary["source_state"],
        "gap_analysis": summary["gap_analysis"],
        "execution_blocks": summary["execution_blocks"],
        "strict_conclusion": summary["strict_conclusion"],
    }


def test_dnn_thousand_noun_source_gap_board_block() -> None:
    payload = build_payload()
    assert len(payload["headline_metrics"]["metric_lines_cn"]) >= 6
    assert payload["headline_metrics"]["metric_lines_cn"][0].startswith("（")
    assert payload["source_state"]["base_unique_nouns"] >= 200
    assert payload["gap_analysis"]["unique_gap"] >= 2000
    assert payload["gap_analysis"]["prompt_multiplier_not_enough"] is True
    assert len(payload["execution_blocks"]) >= 3


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN thousand-noun source gap board block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_thousand_noun_source_gap_board_block_20260316.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
