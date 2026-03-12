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
    ap = argparse.ArgumentParser(description="Theory-track large inventory relation-context synthesis")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_large_inventory_relation_context_synthesis_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    large_concept = load("theory_track_large_scale_concept_inventory_analysis_20260312.json")
    large_relctx = load("theory_track_large_scale_concept_relation_context_inventory_20260312.json")
    closure = load("theory_track_icspb_stronger_closure_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_large_inventory_relation_context_synthesis",
        },
        "inventory_scale": {
            "concept_only_count": large_concept["headline_metrics"]["num_concepts"],
            "concept_relation_context_count": large_relctx["headline_metrics"]["num_concepts"],
            "num_contexts": large_relctx["headline_metrics"]["num_contexts"],
            "num_relations": large_relctx["headline_metrics"]["num_relations"],
        },
        "global_constraints": {
            "family_cross_to_within_ratio": large_relctx["headline_metrics"]["family_cross_to_within_ratio"],
            "context_cross_to_within_ratio": large_relctx["headline_metrics"]["context_cross_to_within_ratio"],
            "relation_cross_to_within_ratio": large_relctx["headline_metrics"]["relation_cross_to_within_ratio"],
            "global_recurrent_dims": large_concept["global_recurrent_dims"],
        },
        "reconstruction_update": {
            "what_becomes_clearer": [
                "大脑编码不只是概念 patch，还带 context fibers 与 relation fibers",
                "推理过程中的因果与联系可被看作 inventory 上的条件化结构，而不是额外外挂模块",
                "编码机制更像 patch-statistics + path-conditioned causal transitions 的联合系统",
            ],
            "why_it_matters": "这让逆向还原从『概念编码』扩展到『概念+关系+上下文+推理过程编码』。",
        },
        "new_math_update": {
            "icspb_upgrade_direction": [
                "from stratified path-bundle theory",
                "to patch-statistics + attached fibers + causal transition theory",
            ],
            "closure_dependency": closure["needed_for_strict_new_math"],
            "theory_message": "Large inventory with relation/context structure can now serve as a population-level source of invariants for theorem pruning and transport-law tightening.",
        },
        "verdict": {
            "core_answer": "Yes. Expanding to concept + relation + context inventory is a stronger route than concept-only scaling, because it begins to capture the coding structure of reasoning itself.",
            "next_theory_target": "grow this inventory further, add temporal/causal chains, and use the resulting invariants as hard constraints on ICSPB theorem survival and P3/P4 interventions.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
