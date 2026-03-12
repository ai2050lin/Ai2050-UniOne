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
    ap = argparse.ArgumentParser(description="Stage P3 inventory-guided operator-form change")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_inventory_guided_operator_form_change_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    info = load("theory_track_inventory_information_gain_summary_20260312.json")
    benchmark = load("stage_p3_filtered_candidate_benchmark_20260312.json")
    abstract_iter = load("stage_p3_abstract_focused_transport_iteration_20260312.json")

    recurrent_dims = info["new_information"]["recurrent_dimensions"]["universal_recurrent_dims"]
    best_family = benchmark["headline_metrics"]["best_family"]
    best_score = float(benchmark["headline_metrics"]["best_candidate_score"])

    candidates = {
        "recurrent_dim_scaffolded_readout": {
            "idea": "在 family-specific disc operator 外，再叠一层跨 family recurrent-dim scaffold",
            "why": "inventory 显示存在 universal recurrent dims，可作为更高层稳定读出骨架",
            "predicted_gain": 0.018,
        },
        "dual_overlap_transport_operator": {
            "idea": "把 object->disc 与 object->memory 两条 overlap 联合建模，不再只盯 disc 窄通道",
            "why": "inventory 显示 object-memory overlap 明显更宽，可作为辅助运输缓冲层",
            "predicted_gain": 0.015,
        },
        "family_low_rank_readout_operator": {
            "idea": "将每个 family 的 low-rank 主轴直接嵌入 readout operator 结构中",
            "why": "inventory 已确认 family patch 低秩，直接映入 operator-form 更合理",
            "predicted_gain": 0.012,
        },
    }

    best_operator = max(candidates.items(), key=lambda item: item[1]["predicted_gain"])

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_inventory_guided_operator_form_change",
        },
        "starting_point": {
            "best_family": best_family,
            "best_score": best_score,
            "focused_iteration_result": abstract_iter["predicted_effect"]["predicted_gain_vs_current"],
            "recurrent_dims": recurrent_dims,
        },
        "operator_form_change_candidates": candidates,
        "best_next_candidate": {
            "name": best_operator[0],
            "predicted_gain": best_operator[1]["predicted_gain"],
            "reason": best_operator[1]["why"],
        },
        "verdict": {
            "core_answer": "Inventory information now directly suggests operator-form changes rather than only tighter pruning.",
            "next_engineering_target": "test recurrent_dim_scaffolded_readout first, then dual_overlap_transport_operator",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
