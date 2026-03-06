#!/usr/bin/env python
"""统一实验结果 schema（v1）。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List


SCHEMA_VERSION = "agi_research_result.v1"


def build_result_payload(
    *,
    experiment_id: str,
    title: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    hypotheses: List[Dict[str, Any]],
    artifacts: Dict[str, Any] | None = None,
    notes: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "title": title,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
        "metrics": metrics,
        "hypotheses": hypotheses,
        "artifacts": artifacts or {},
        "notes": notes or [],
    }


def write_result_bundle(
    *,
    out_dir: Path,
    base_name: str,
    payload: Dict[str, Any],
    report_md: str,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{base_name}.json"
    md_path = out_dir / f"{base_name.upper()}_REPORT.md"
    with json_path.open("w", encoding="utf-8-sig") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with md_path.open("w", encoding="utf-8-sig") as f:
        f.write(report_md)
    return {"json": json_path, "md": md_path}


def is_v1_payload(obj: Dict[str, Any] | None) -> bool:
    if not isinstance(obj, dict):
        return False
    return (
        obj.get("schema_version") == SCHEMA_VERSION
        and isinstance(obj.get("experiment_id"), str)
        and isinstance(obj.get("metrics"), dict)
    )
