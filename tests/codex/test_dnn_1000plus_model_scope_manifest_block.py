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

from research.gpt5.code.dnn_1000plus_model_scope_manifest import Dnn1000PlusModelScopeManifest  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    manifest = Dnn1000PlusModelScopeManifest.from_repo(ROOT)
    summary = manifest.summary()
    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "dnn_1000plus_model_scope_manifest_block",
        },
        "strict_goal": {
            "statement": "Clarify which model is actually used by the current 1000+ noun execution bundle, and distinguish it from the broader multi-model analysis stack.",
            "boundary": "This block clarifies execution scope. It does not run the heavy batch harvest itself.",
        },
        "headline_metrics": {
            "metric_lines_cn": summary["metric_lines_cn"],
        },
        "current_execution": summary["current_execution"],
        "analysis_scope": summary["analysis_scope"],
        "strict_conclusion": summary["strict_conclusion"],
    }


def test_dnn_1000plus_model_scope_manifest_block() -> None:
    payload = build_payload()
    assert payload["headline_metrics"]["metric_lines_cn"][0].startswith("（")
    assert payload["current_execution"]["model_id"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    assert payload["current_execution"]["launchable_batch_count"] >= 9
    assert "qwen3_4b" in payload["analysis_scope"]["historical_structure_analysis_models"]


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN 1000+ model scope manifest block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_1000plus_model_scope_manifest_block_20260316.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
