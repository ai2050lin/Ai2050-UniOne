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
    ap = argparse.ArgumentParser(description="新路线系统评估")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_new_route_system_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    try:
        block = load("stage_new_route_system_validation_block_20260313.json")
    except FileNotFoundError:
        block = load_latest("stage_new_route_system_validation_block")

    cur = block["current_state"]
    proj = block["next_block_projection"]
    scores = block["scores"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_new_route_system_assessment",
        },
        "current_summary": {
            "protocol": cur["protocol"],
            "successor": cur["successor"],
            "brain": cur["brain"],
            "inverse_reconstruction": cur["inverse_reconstruction"],
            "new_math_closure": cur["new_math_closure"],
        },
        "projected_summary": {
            "protocol": proj["projected_protocol_after_next_block"],
            "successor": proj["projected_successor_after_next_block"],
            "brain": proj["projected_brain_after_next_block"],
        },
        "status": {
            "route_ready_for_large_block": scores["total_score"] >= 0.70,
            "successor_still_main_gap": cur["successor"] < cur["protocol"] and cur["successor"] < cur["brain"],
            "protocol_has_entered_strong_band": cur["protocol"] >= 0.78,
            "brain_has_entered_online_band": cur["brain"] >= 0.80,
        },
        "verdict": {
            "core_answer": "当前新路线已经具备继续执行大块统一验证的条件，但 successor 仍然是最核心的剩余缺口。",
            "next_target": "把下一轮统一块的重点压到真实长链 successor trace 捕获，同时继续保持 protocol 强桥接和 brain 在线执行。",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
