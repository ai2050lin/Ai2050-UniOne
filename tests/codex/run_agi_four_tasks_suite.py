#!/usr/bin/env python
"""
一键运行四项任务：
1) 变量绑定硬验证
2) 最小因果回路搜索
3) 统一坐标系测试
4) 规模化概念族实验
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def run_cmd(cmd: List[str]) -> int:
    print("[RUN]", " ".join(cmd))
    p = subprocess.run(cmd, check=False)
    return int(p.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run AGI four tasks suite")
    ap.add_argument(
        "--mass-json",
        default="tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed101/mass_noun_encoding_scan.json",
    )
    ap.add_argument("--root", default="tempdata")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/agi_four_tasks_suite_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (
            "variable_binding",
            [
                sys.executable,
                "tests/codex/deepseek7b_variable_binding_hard_verification.py",
                "--mass-json",
                args.mass_json,
                "--seed",
                str(args.seed),
                "--output-dir",
                str(out_dir / "task1_variable_binding"),
            ],
        ),
        (
            "minimal_causal_circuit",
            [
                sys.executable,
                "tests/codex/deepseek7b_minimal_causal_circuit_search.py",
                "--mass-json",
                args.mass_json,
                "--output-dir",
                str(out_dir / "task2_minimal_causal"),
            ],
        ),
        (
            "unified_coordinate",
            [
                sys.executable,
                "tests/codex/deepseek7b_unified_coordinate_system_test.py",
                "--root",
                args.root,
                "--mass-json",
                args.mass_json,
                "--output-dir",
                str(out_dir / "task3_unified_coordinate"),
            ],
        ),
        (
            "concept_family_parallel",
            [
                sys.executable,
                "tests/codex/deepseek7b_concept_family_parallel_scale.py",
                "--root",
                args.root,
                "--output-dir",
                str(out_dir / "task4_concept_family"),
            ],
        ),
    ]

    rc_map: Dict[str, int] = {}
    for name, cmd in jobs:
        rc_map[name] = run_cmd(cmd)

    manifest = {
        "suite_id": "agi_four_tasks_suite_v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "mass_json": args.mass_json,
            "root": args.root,
            "seed": args.seed,
        },
        "return_codes": rc_map,
        "all_success": all(v == 0 for v in rc_map.values()),
        "artifacts": {
            "task1": str(out_dir / "task1_variable_binding"),
            "task2": str(out_dir / "task2_minimal_causal"),
            "task3": str(out_dir / "task3_unified_coordinate"),
            "task4": str(out_dir / "task4_concept_family"),
        },
    }
    mf = out_dir / "agi_four_tasks_suite_manifest.json"
    with mf.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] manifest: {mf}")
    if not manifest["all_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

