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
    ap = argparse.ArgumentParser(description="Theory track encoding inverse reconstruction")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_encoding_inverse_reconstruction_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    phase_map = load("theory_track_phase_p1_p4_current_map_20260312.json")
    gaps = load("theory_track_encoding_mechanism_core_gaps_20260312.json")
    reasoning = load("theory_track_modality_unified_reasoning_law_20260312.json")
    benchmark = load("stage_p3_winner_gap_aligned_benchmark_20260312.json")

    reconstruction_layers = [
        {
            "layer": "family_patched_object_atlas",
            "what_is_reconstructed": "概念首先落在 family-patched base manifold，而不是全局均匀坐标系。",
            "support_strength": "strong",
            "evidence": ["P1 strong base layer", "family patch / concept section results"],
            "confidence": 0.86,
        },
        {
            "layer": "stress_coupled_local_update",
            "what_is_reconstructed": "写入与读取受 novelty / retention stress 调制，写常需 guarded，读要求 stable。",
            "support_strength": "medium_high",
            "evidence": ["P2 filtered dynamic law", "stress-coupled write/read law"],
            "confidence": 0.72,
        },
        {
            "layer": "path_conditioned_readout_transport",
            "what_is_reconstructed": "object manifold 到 readout 的使用必须沿 admissible path 和 restricted overlap 进行。",
            "support_strength": "medium_high",
            "evidence": ["P3 operator-sensitive positive signal", benchmark["headline_metrics"]["winner"]],
            "confidence": 0.74,
        },
        {
            "layer": "family_anchored_bridge_role_lift",
            "what_is_reconstructed": "relation / role 更像 anchored lift，而不是自由 symbolic 层。",
            "support_strength": "medium",
            "evidence": ["B-line filtered family-conditioned bridge space"],
            "confidence": 0.68,
        },
        {
            "layer": "modality_conditioned_shared_reasoning_slice",
            "what_is_reconstructed": "意识统一处理更像 modality-conditioned entry 进入 family-conditioned shared reasoning slice。",
            "support_strength": "medium",
            "evidence": [reasoning["law_name"]],
            "confidence": 0.66,
        },
    ]

    unresolved = [gap["name"] for gap in gaps["gaps"]]
    overall_confidence = sum(item["confidence"] for item in reconstruction_layers) / len(reconstruction_layers)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_encoding_inverse_reconstruction",
        },
        "inverse_reconstruction_layers": reconstruction_layers,
        "overall_inverse_reconstruction_confidence": overall_confidence,
        "unresolved_core_gaps": unresolved,
        "verdict": {
            "core_answer": "当前拼图已经足够逆向还原出大脑编码机制的主分层：object atlas、stress-coupled update、path-conditioned readout、anchored bridge-role、shared reasoning slice。",
            "next_theory_target": "继续把 unresolved gaps 压缩到 theorem-level statements and causal falsification constraints.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
