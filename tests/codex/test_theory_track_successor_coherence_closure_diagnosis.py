from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> Dict[str, Any]:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory track successor coherence closure diagnosis")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_successor_coherence_closure_diagnosis_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    mech = load("theory_track_successor_coherence_mechanism_analysis_20260312.json")
    v3 = load("theory_track_10round_excavation_loop_v3_20260312.json")
    qd = load("qwen_deepseek_naturalized_trace_bundle_20260312.json")

    successor_score = float(v3["ending_point"]["final_scores"]["successor_coherence"])
    protocol_score = float(v3["ending_point"]["final_scores"]["protocol_calling"])
    brain_score = float(v3["ending_point"]["final_scores"]["brain_side_causal_closure"])
    online_trace_gap = 1.0 - float(qd["naturalized_trace_axes"]["successor_coherence"])

    blockers: List[Dict[str, Any]] = [
        {
            "name": "successor_is_still_local_not_global",
            "severity": "highest",
            "evidence": f"observed_global_successor_score={successor_score:.4f} is still far below stable-closure band",
        },
        {
            "name": "protocol_layer_still_underpowered",
            "severity": "high",
            "evidence": f"protocol_calling={protocol_score:.4f}, meaning successor transitions still lack enough protocol/task bridge support",
        },
        {
            "name": "brain_side_causal_projection_not_yet_closed",
            "severity": "high",
            "evidence": f"brain_side_causal_closure={brain_score:.4f}, successor paths still do not survive strong causal projection",
        },
        {
            "name": "trace_source_is_not_true_online_natural_trace",
            "severity": "high",
            "evidence": f"successor online trace gap remains {online_trace_gap:.4f}, indicating current loop still uses artifact-led rather than live internal traces",
        },
        {
            "name": "successor_theorem_pass_does_not_equal_system_closure",
            "severity": "medium",
            "evidence": "theorem strict-pass under strengthened inventory only proves local support, not full-system closure across readout, stress, and brain-side intervention",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_successor_coherence_closure_diagnosis",
        },
        "closure_status": {
            "successor_theorem": mech["strict_frontier_context"]["successor_theorem_status"],
            "effective_successor_bundle": mech["components"]["effective_successor_bundle"],
            "observed_global_successor_score": successor_score,
            "closure_band_threshold_hint": 0.45,
            "is_closed": False,
        },
        "blockers": blockers,
        "verdict": {
            "core_answer": (
                "successor_coherence 现在无法闭环，不是因为完全没信号，而是因为它只在 strengthened inventory 下形成了局部 theorem-support，"
                "还没有变成跨 protocol、readout、brain-side causal projection 的全系统主导结构。"
            ),
            "next_theory_target": (
                "把 successor 从 local theorem-support 推到 global system-support，关键是在线自然 trace、protocol bridge 和 brain-side causal closure 三条线一起加强。"
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
