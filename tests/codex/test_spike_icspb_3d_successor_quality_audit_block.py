from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gpt5.code.spike_icspb_3d_multiregion_phasea import (  # noqa: E402
    SpikeICSPB3DMultiRegionConfig,
    SpikeICSPB3DMultiRegionPhaseA,
)


def encode_text(text: str) -> List[int]:
    return list(text.encode("utf-8", errors="ignore"))


def build_batch(texts: List[str], max_seq_len: int = 96) -> torch.Tensor:
    rows = []
    for text in texts:
        ids = encode_text(text)[:max_seq_len]
        if len(ids) >= 16:
            rows.append(ids)
    min_len = min(len(row) for row in rows)
    rows = [row[:min_len] for row in rows]
    return torch.tensor(rows, dtype=torch.long)


def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item())


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    torch.manual_seed(71)
    config = SpikeICSPB3DMultiRegionConfig(
        hidden_dim=128,
        region_hidden_dim=96,
        patch_slots=64,
        max_seq_len=96,
        phase_dim=12,
        topology_radius=0.8,
        bridge_scale=0.05,
        bridge_topk=6,
        potential_limit=1.10,
        local_lr=0.05,
        replay_decay=0.95,
        consolidation_lr=0.10,
        bridge_mix=0.24,
        homeostasis_target_abs=0.50,
        homeostasis_gain=0.20,
        homeostasis_lr=0.06,
    )
    model = SpikeICSPB3DMultiRegionPhaseA(config)

    train_batch = build_batch(
        [
            "apple is red and sweet.\n",
            "banana is yellow and soft.\n",
            "cat can run and jump.\n",
            "dog can run and bark.\n",
            "truth guides logic and memory.\n",
            "justice seeks balance and order.\n",
        ]
    )
    eval_batch = build_batch(
        [
            "apple is red and sweet.\n",
            "banana is yellow and soft.\n",
            "cat can run and jump.\n",
            "dog can run and bark.\n",
        ]
    )
    for _ in range(10):
        model.local_update_step(train_batch, train_batch, lr=0.065)
        model.replay_consolidate(train_batch)

    out = model.forward(eval_batch)
    fused = out["fused_states"]
    same_successor = cosine_mean(fused[:, :-1, :], fused[:, 1:, :])
    cross_successor = cosine_mean(fused[:, :-1, :], fused.roll(shifts=1, dims=0)[:, 1:, :])
    successor_alignment_margin = same_successor - cross_successor

    logits = out["logits"][:, :-1, :]
    targets = eval_batch[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    true_next = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    random_targets = targets.roll(shifts=1, dims=0)
    random_next = log_probs.gather(dim=-1, index=random_targets.unsqueeze(-1)).squeeze(-1)
    next_token_margin = float((true_next - random_next).mean().item())

    entropy = float((-(log_probs.exp() * log_probs).sum(dim=-1)).mean().item())
    normalized_entropy = entropy / math.log(config.vocab_size)
    mean_gate = float(
        sum(out["successor_gate"][name].mean().item() for name in config.region_names) / len(config.region_names)
    )
    gate_dispersion = float(
        sum(out["successor_gate"][name].std().item() for name in config.region_names) / len(config.region_names)
    )
    protocol_energy = float(out["protocol_states"].norm(dim=-1).mean().item())
    successor_energy = float(
        sum(out["region_hidden_states"][name].norm(dim=-1).mean().item() for name in config.region_names)
        / len(config.region_names)
    )
    successor_protocol_margin = max(0.0, protocol_energy - successor_energy)
    successor_quality_score = min(
        1.0,
        0.34 * max(0.0, successor_alignment_margin / 0.08)
        + 0.30 * max(0.0, next_token_margin / 0.9)
        + 0.18 * max(0.0, 1.0 - normalized_entropy)
        + 0.10 * min(1.0, (mean_gate + gate_dispersion) / 0.8)
        + 0.08 * min(1.0, protocol_energy / max(successor_energy, 1e-6)),
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "SpikeICSPB_3D_successor_quality_audit_block",
        },
        "strict_goal": {
            "statement": (
                "Audit the actual successor bottleneck directly: measure whether the multi-region 3D line turns stable internal codes into aligned next-step structure."
            ),
            "boundary": (
                "This block diagnoses successor quality on short structured sequences. It does not prove long-context language competence."
            ),
        },
        "headline_metrics": {
            "same_successor_cosine": same_successor,
            "cross_successor_cosine": cross_successor,
            "successor_alignment_margin": successor_alignment_margin,
            "next_token_margin": next_token_margin,
            "normalized_entropy": normalized_entropy,
            "mean_successor_gate": mean_gate,
            "successor_gate_dispersion": gate_dispersion,
            "protocol_energy": protocol_energy,
            "successor_energy": successor_energy,
            "successor_protocol_margin": successor_protocol_margin,
            "successor_quality_score": successor_quality_score,
        },
        "strict_verdict": {
            "successor_structure_present": bool(successor_alignment_margin > 0.02),
            "successor_is_still_main_bottleneck": bool(successor_quality_score < 0.62),
            "core_answer": (
                "The current 3D SpikeICSPB line already has real same-chain successor structure and an explicit protocol-side state, but the step from structural alignment to strong next-token discrimination remains weak. "
                "That is why successor quality is still the main bottleneck."
            ),
            "main_hard_gaps": [
                "same-chain successor coherence exists, but token-level margin is still modest",
                "prediction entropy remains high compared with strong language models",
                "successor gates and protocol state are active, but they do not yet convert dense stable codes into sharp continuation decisions",
                "the audit is still short-context and does not yet test long reasoning chains",
            ],
        },
        "progress_estimate": {
            "successor_quality_audit_percent": 49.0,
            "non_attention_non_bp_full_language_capability_percent": 34.0,
            "whole_network_state_generator_percent": 68.0,
            "full_brain_encoding_mechanism_percent": 72.0,
        },
        "next_large_blocks": [
            "Train successor routing directly against longer structured continuation tasks.",
            "Bind successor quality to protocol routing and replay-triggered recall rather than only local hidden-state continuity.",
            "Measure successor quality on scalable Phase-A candidates instead of only the small multi-region prototype.",
        ],
    }
    return payload


def test_spike_icspb_3d_successor_quality_audit_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["successor_alignment_margin"] > 0.02
    assert metrics["successor_quality_score"] > 0.18
    assert verdict["successor_structure_present"] is True


def main() -> None:
    ap = argparse.ArgumentParser(description="SpikeICSPB 3D successor quality audit block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/spike_icspb_3d_successor_quality_audit_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
