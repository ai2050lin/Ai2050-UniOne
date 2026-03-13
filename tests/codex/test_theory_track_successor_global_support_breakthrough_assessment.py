from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def load_latest(prefix: str) -> dict:
    candidates = sorted(TEMP_DIR.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"未找到前缀为 {prefix} 的工件")
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Successor global support breakthrough assessment")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_successor_global_support_breakthrough_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    try:
        block = load("stage_successor_global_support_breakthrough_block_20260313.json")
    except FileNotFoundError:
        block = load_latest("stage_successor_global_support_breakthrough_block")

    cur = block["current_state"]
    fin = block["final_projection"]

    protocol_pass = float(fin["protocol"]) >= 0.82
    successor_pass = float(fin["successor"]) >= 0.60
    brain_pass = float(fin["brain"]) >= 0.90
    inverse_pass = float(fin["inverse_reconstruction"]) >= 0.83
    math_pass = float(fin["new_math_closure"]) >= 0.95
    pass_count = sum([protocol_pass, successor_pass, brain_pass, inverse_pass, math_pass])

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_successor_global_support_breakthrough_assessment",
        },
        "current_to_final": {
            "protocol_current": cur["protocol"],
            "protocol_final": fin["protocol"],
            "successor_current": cur["successor"],
            "successor_final": fin["successor"],
            "brain_current": cur["brain"],
            "brain_final": fin["brain"],
        },
        "pass_status": {
            "protocol_strong_pass": protocol_pass,
            "successor_global_support_pass": successor_pass,
            "brain_online_pass": brain_pass,
            "inverse_reconstruction_pass": inverse_pass,
            "new_math_closure_pass": math_pass,
            "pass_count": pass_count,
        },
        "verdict": {
            "core_answer": "The breakthrough block is good enough to be the new default large task unit if it pushes successor beyond 0.60 while preserving protocol and brain bands.",
            "main_bottleneck_after_block": "real_online_trace_and_intervention_validation",
            "next_target": "Move from projected block success to real online trace capture and online brain-side intervention execution.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
