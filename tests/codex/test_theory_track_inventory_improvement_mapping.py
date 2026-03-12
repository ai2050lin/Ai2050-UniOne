from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track inventory improvement mapping")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_improvement_mapping_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    info = load("theory_track_inventory_information_gain_summary_20260312.json")
    p2 = load("stage_p2_stress_coupled_update_pruned_search_20260312.json")
    p3 = load("stage_p3_path_conditioned_transport_filtered_search_20260312.json")
    bline = load("stage_b_path_conditioned_bridge_filtered_search_20260312.json")
    p4 = load("stage_p4_brain_probe_execution_bundle_20260312.json")

    mapping = {
        "family_patch_structure": [
            "排除了 global_isotropic_transport",
            "把 P3 收缩到 family-conditioned filtered search space",
        ],
        "low_rank_family_axes": [
            "把 operator family 写成 Omega^(f)_upd / Omega^(f)_read / Omega^(f)_bridge",
            "把 bridge-role 搜索改成 family-anchored role kernel",
        ],
        "recurrent_dimensions": [
            "为后续 operator-form change 提供跨 family 的共享维度骨架",
        ],
        "restricted_overlap": [
            "排除了 direct_object_to_disc_collapse",
            "把 readout 改写成 restricted-overlap + switching-aware path",
        ],
        "operator_families": [
            "P2 用 guarded-write / stable-read family 过滤 update-law",
            "B-line 用 path-conditioned bridge family 过滤桥律空间",
        ],
        "stress_profiles": [
            "把 Q2/Q3 重写成 stress-coupled write/read law",
            "把 P4 扩成 stress probe bundle",
        ],
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_improvement_mapping",
        },
        "information_to_improvement": mapping,
        "affected_execution_blocks": {
            "P2": p2["bridge_to_engineering"]["core_statement"],
            "P3": p3["bridge_to_engineering"]["core_statement"],
            "B_line": bline["bridge_to_engineering"]["core_statement"],
            "P4": p4["verdict"]["core_answer"],
        },
        "verdict": {
            "core_answer": "Most of the recent engineering improvements can now be traced back to specific information extracted from the encoding inventory.",
            "next_theory_target": "use the same inventory signals to drive operator-form changes instead of only search-space pruning",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
