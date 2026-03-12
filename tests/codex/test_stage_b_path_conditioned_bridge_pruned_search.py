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
    ap = argparse.ArgumentParser(description="Stage B path-conditioned bridge pruned search")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_b_path_conditioned_bridge_pruned_search_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    bridge_law = load("theory_track_path_conditioned_bridge_lift_law_20260312.json")
    bridge_entries = bridge_law["concept_bridge_entries"]

    family_rows: dict[str, dict] = {}
    for row in bridge_entries.values():
        family = row["family"]
        family_rows.setdefault(
            family,
            {"count": 0, "mean_relation_overlap": 0.0, "conditional_count": 0},
        )
        family_rows[family]["count"] += 1
        family_rows[family]["mean_relation_overlap"] += float(row["relation_overlap"])
        if row["path_condition"] == "conditional":
            family_rows[family]["conditional_count"] += 1

    retained: dict[str, dict] = {}
    pruned: dict[str, dict] = {}
    for family, row in family_rows.items():
        mean_overlap = row["mean_relation_overlap"] / max(1, row["count"])
        conditional_ratio = row["conditional_count"] / max(1, row["count"])
        report = {
            "mean_relation_overlap": float(mean_overlap),
            "conditional_ratio": float(conditional_ratio),
            "bridge_rule": "Pi_bridge(c) = 1[relation_overlap(f_c) > tau_rel] * 1[sigma_rel(c) > tau_sigma] * 1[Delta in A(I)]",
        }
        if mean_overlap > 0.52 and conditional_ratio >= 1.0:
            retained[family] = report
        else:
            pruned[family] = report

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageB_path_conditioned_bridge_pruned_search",
        },
        "candidate_classes": {
            "path_conditioned_bridge_lift": "kept",
            "family_anchored_role_kernel": "kept",
            "free_symbolic_role_layer": "excluded",
            "family_agnostic_bridge_head": "excluded",
        },
        "retained_families": retained,
        "pruned_families": pruned,
        "bridge_to_engineering": {
            "core_statement": "The next B-line search should keep only path-conditioned, family-anchored bridge families.",
            "kept_family_count": int(len(retained)),
            "pruned_family_count": int(len(pruned)),
        },
        "verdict": {
            "core_answer": "B-line can now be explicitly pruned by path-conditioned bridge openness instead of broader role heuristics.",
            "next_engineering_target": "run the next bridge-role engineering search only inside the retained path-conditioned bridge family space",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
