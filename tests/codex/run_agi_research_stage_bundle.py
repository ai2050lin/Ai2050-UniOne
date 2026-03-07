#!/usr/bin/env python
"""
四计划批量执行入口：
1) 三个硬伤实验（动态绑定 / 长程因果 / 局部信用分配）
2) 三元组定向因果（apple/king/queen）
3) 统一解码（可选）
4) 输出 bundle manifest，供 Main 导入
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict


def run_cmd(cmd: list[str], cwd: Path) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(cwd), check=False)
    return int(p.returncode)


def latest_file(base_dir: Path, pattern: str) -> Path | None:
    files = sorted(base_dir.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_json(path: Path | None) -> Dict:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def rel_or_empty(path: Path | None, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix() if path else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Run AGI research stage bundle")
    ap.add_argument(
        "--mass-json",
        default="tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed101/mass_noun_encoding_scan.json",
    )
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--run-unified-decoder", action="store_true")
    ap.add_argument("--skip-triplet-targeted", action="store_true", help="Skip apple/king/queen targeted causal scan")
    ap.add_argument("--run-triplet-multiseed", action="store_true", help="Run triplet targeted multi-seed stability")
    ap.add_argument("--triplet-multiseed-list", default="101,202,303,404,505")
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else root / f"tempdata/agi_research_stage_bundle_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    rc_map = {}

    dynamic_dir = out_dir / "hard_problem_dynamic_binding"
    rc_map["dynamic_binding"] = run_cmd(
        [
            py,
            "tests/codex/deepseek7b_dynamic_binding_stress_test.py",
            "--mass-json",
            args.mass_json,
            "--seed",
            str(args.seed),
            "--output-dir",
            str(dynamic_dir),
        ],
        cwd=root,
    )

    long_dir = out_dir / "hard_problem_long_horizon"
    rc_map["long_horizon"] = run_cmd(
        [
            py,
            "tests/codex/deepseek7b_long_horizon_causal_trace_test.py",
            "--mass-json",
            args.mass_json,
            "--seed",
            str(args.seed),
            "--output-dir",
            str(long_dir),
        ],
        cwd=root,
    )

    local_dir = out_dir / "hard_problem_local_credit"
    rc_map["local_credit"] = run_cmd(
        [
            py,
            "tests/codex/deepseek7b_local_credit_assignment_proxy_test.py",
            "--mass-json",
            args.mass_json,
            "--seed",
            str(args.seed),
            "--output-dir",
            str(local_dir),
        ],
        cwd=root,
    )

    triplet_dir = out_dir / "hard_problem_triplet_targeted"
    if args.skip_triplet_targeted:
        rc_map["triplet_targeted"] = -1
    else:
        rc_map["triplet_targeted"] = run_cmd(
            [
                py,
                "tests/codex/deepseek7b_triplet_causal_targeted_scan.py",
                "--seed",
                str(args.seed),
                "--output-dir",
                str(triplet_dir),
            ],
            cwd=root,
        )

    unified_dir = out_dir / "unified_decode"
    if args.run_unified_decoder:
        rc_map["unified_decode"] = run_cmd(
            [
                py,
                "tests/codex/deepseek7b_unified_math_structure_decoder.py",
                "--root",
                "tempdata",
                "--output-dir",
                str(unified_dir),
            ],
            cwd=root,
        )
    else:
        rc_map["unified_decode"] = -1

    triplet_stability_dir = out_dir / "hard_problem_triplet_multiseed"
    if args.run_triplet_multiseed:
        rc_map["triplet_multiseed"] = run_cmd(
            [
                py,
                "tests/codex/deepseek7b_triplet_causal_multiseed_stability.py",
                "--seeds",
                str(args.triplet_multiseed_list),
                "--output-dir",
                str(triplet_stability_dir),
            ],
            cwd=root,
        )
    else:
        rc_map["triplet_multiseed"] = -1

    dynamic_json = latest_file(dynamic_dir, "dynamic_binding_stress_test.json")
    long_json = latest_file(long_dir, "long_horizon_causal_trace_test.json")
    local_json = latest_file(local_dir, "local_credit_assignment_proxy_test.json")
    triplet_json = latest_file(triplet_dir, "triplet_targeted_causal_scan.json")
    triplet_multiseed_json = latest_file(triplet_stability_dir, "triplet_targeted_multiseed_stability.json")
    unified_json = latest_file(unified_dir, "unified_math_structure_decode.json")

    manifest = {
        "bundle_id": "agi_research_stage_bundle_v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "mass_json": args.mass_json,
            "seed": args.seed,
            "run_unified_decoder": bool(args.run_unified_decoder),
            "run_triplet_targeted": bool(not args.skip_triplet_targeted),
            "run_triplet_multiseed": bool(args.run_triplet_multiseed),
            "triplet_multiseed_list": str(args.triplet_multiseed_list),
        },
        "return_codes": rc_map,
        "artifacts": {
            "dynamic_binding_json": rel_or_empty(dynamic_json, root),
            "long_horizon_json": rel_or_empty(long_json, root),
            "local_credit_json": rel_or_empty(local_json, root),
            "triplet_targeted_json": rel_or_empty(triplet_json, root),
            "triplet_multiseed_json": rel_or_empty(triplet_multiseed_json, root),
            "unified_decode_json": rel_or_empty(unified_json, root),
        },
        "metrics_snapshot": {
            "dynamic_binding": (load_json(dynamic_json).get("metrics") or {}),
            "long_horizon": (load_json(long_json).get("metrics") or {}),
            "local_credit": (load_json(local_json).get("metrics") or {}),
            "triplet_targeted": (load_json(triplet_json).get("metrics") or {}),
            "triplet_multiseed": (load_json(triplet_multiseed_json).get("metrics") or {}),
            "unified_decode": {
                "pass_ratio": ((load_json(unified_json).get("hypothesis_test") or {}).get("pass_ratio"))
                if unified_json
                else None
            },
        },
    }

    manifest_path = out_dir / "agi_research_stage_bundle_manifest.json"
    with manifest_path.open("w", encoding="utf-8-sig") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {manifest_path}")
    for k, v in manifest["artifacts"].items():
        if v:
            print(f"[OK] {k}: {v}")


if __name__ == "__main__":
    main()
