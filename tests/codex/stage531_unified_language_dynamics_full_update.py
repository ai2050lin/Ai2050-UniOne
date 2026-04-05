#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage531_unified_language_dynamics_full_update_20260404"
)
SOURCES = {
    "stage526": PROJECT_ROOT / "tests" / "codex_temp" / "stage526_language_band_circuit_dynamics_20260404" / "summary.json",
    "stage528": PROJECT_ROOT / "tests" / "codex_temp" / "stage528_wordclass_encoding_structure_synthesis_20260404" / "summary.json",
    "stage530": PROJECT_ROOT / "tests" / "codex_temp" / "stage530_four_model_wordclass_bridge_typology_20260404" / "summary.json",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    source_payloads = {name: load_json(path) for name, path in SOURCES.items()}
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage531_unified_language_dynamics_full_update",
        "title": "统一语言动力学全量更新",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": {name: str(path) for name, path in SOURCES.items()},
        "symbolic_update": (
            "h_t^{l+1} = h_t^l + E_l(route_t, function_t) + M_l(bind_t, role_t) + "
            "L_l(content_t, readout_t)，其中词类决定主要进入哪类功能带，桥接项决定这些带如何被拼成实例。"
        ),
        "core_answer": (
            "当前统一动力学已经可以同时容纳四块拼图：词类层带、概念骨干、桥接因果、晚层收束。"
            "更接近事实的结构不是均匀网络，而是“宽带功能区 + 尖锐控制杆 + 小型桥接回路”。"
        ),
        "payloads": source_payloads,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(
        "# stage531 统一语言动力学全量更新\n\n"
        + summary["core_answer"]
        + "\n\n"
        + summary["symbolic_update"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
