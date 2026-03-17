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

from research.gpt5.code.dnn_family_specific_dense_target_manifest import DnnFamilySpecificDenseTargetManifest  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    manifest = DnnFamilySpecificDenseTargetManifest.from_artifacts(ROOT)
    summary = manifest.summary()
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "dnn_family_specific_dense_target_manifest_block",
        },
        "strict_goal": {
            "statement": "Turn family-to-specific exact closure into a concrete dense harvesting target manifest, with explicit concept groups, prompt groups, tensor axes, and hard-gap reasoning.",
            "boundary": "This block prepares execution. It does not claim dense family-to-specific exact closure has already been achieved.",
        },
        "headline_metrics": {
            "metric_lines_cn": summary["metric_lines_cn"],
            "closure_gap": summary["closure_gap"],
            "projected_dense_uplift": summary["projected_dense_uplift"],
        },
        "target_groups": summary["target_groups"],
        "specific_schema": summary["specific_schema"],
        "strict_conclusion": summary["strict_conclusion"],
    }
    return payload


def test_dnn_family_specific_dense_target_manifest_block() -> None:
    payload = build_payload()
    assert len(payload["headline_metrics"]["metric_lines_cn"]) >= 6
    assert payload["headline_metrics"]["metric_lines_cn"][0].startswith("（")
    assert len(payload["target_groups"]) >= 3
    assert payload["specific_schema"]["launchable"] is True
    assert payload["headline_metrics"]["closure_gap"] > 0.5
    assert payload["headline_metrics"]["projected_dense_uplift"] > 0.6


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN family-specific dense target manifest block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_family_specific_dense_target_manifest_block_20260316.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
