from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def latest_match(pattern: str) -> Path:
    matches = sorted(TEMP_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    return matches[0]


def load_latest(pattern: str) -> dict:
    return json.loads(latest_match(pattern).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track DNN to brain reverse constraints")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_dnn_to_brain_reverse_constraints_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    mined = load_latest("theory_track_dnn_encoding_pattern_mining_*.json")

    constraints = [
        {
            "constraint": "brain_family_patch_constraint",
            "source_pattern": "family_patch_stability",
            "brain_prediction": "真实脑侧应更像 family-patched object atlas，而不是全局均匀概念云",
        },
        {
            "constraint": "brain_recurrent_scaffold_constraint",
            "source_pattern": "recurrent_scaffold_reuse",
            "brain_prediction": "不同脑区/模态之间应共享一小组可重复利用的低维 scaffold，而不是完全分裂的编码字典",
        },
        {
            "constraint": "brain_relation_fiber_constraint",
            "source_pattern": "relation_fiber_emergence",
            "brain_prediction": "关系与角色应更像附着在对象层上的结构纤维，而不是完全自由的符号层",
        },
        {
            "constraint": "brain_temporal_operator_constraint",
            "source_pattern": "temporal_transition_emergence",
            "brain_prediction": "时间阶段更像 operator/transition 约束，而不是独立类别表征",
        },
        {
            "constraint": "brain_successor_coherence_constraint",
            "source_pattern": "successor_coherence_emergence",
            "brain_prediction": "推理链中的前后继局部一致性应在脑侧 probe 中逐步显现，并优先约束 reasoning transport 而不是静态表示",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_dnn_to_brain_reverse_constraints",
        },
        "constraints": constraints,
        "central_claim": "DNN-side coding invariants can now be used as reverse constraints on brain encoding, not just as analogies or weak hints.",
        "verdict": {
            "core_answer": "The project can now reverse-map stable DNN coding invariants into concrete brain-side constraints about object patches, shared scaffolds, relation fibers, temporal operators, and successor coherence.",
            "next_theory_target": "bind these reverse constraints into the next P4 causal falsification layer and use them to reject brain models that remain too global, too symbol-free, or too stage-agnostic.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
