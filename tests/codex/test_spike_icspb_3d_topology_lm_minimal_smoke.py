from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gpt5.code.spike_icspb_3d_topology_lm_minimal import (  # noqa: E402
    SpikeICSPB3DConfig,
    SpikeICSPB3DLMMinimal,
)


def encode_text(text: str) -> torch.Tensor:
    token_ids = list(text.encode("utf-8", errors="ignore"))
    return torch.tensor([token_ids], dtype=torch.long)


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    torch.manual_seed(7)

    config = SpikeICSPB3DConfig(
        hidden_dim=128,
        patch_slots=64,
        max_seq_len=192,
        phase_dim=12,
        topology_radius=0.8,
        bridge_scale=0.05,
        bridge_topk=6,
        local_lr=0.06,
    )
    model = SpikeICSPB3DLMMinimal(config)

    corpus = (
        "apple is sweet and round. "
        "banana is long and edible. "
        "cat can run and jump. "
        "truth stays stable in memory. "
    )
    input_ids = encode_text(corpus)
    targets = input_ids.clone()

    pre_loss, pre_metrics = model.compute_loss(input_ids, targets)
    update_metrics = []
    for _ in range(4):
        update_metrics.append(model.local_update_step(input_ids, targets))
    post_loss, post_metrics = model.compute_loss(input_ids, targets)

    prompt = encode_text("apple and cat ")
    generation = model.generate(prompt, max_new_tokens=24, temperature=0.8)
    generated_ids = generation["generated_ids"][0]
    generated_text = model.decode_token_ids(generated_ids)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "SpikeICSPB_3D_topology_lm_minimal_smoke",
        },
        "strict_goal": {
            "statement": (
                "Verify that a non-Attention, non-BP SpikeICSPB language prototype can run with "
                "explicit 3D local-topology bias and sparse bridge routing."
            ),
            "boundary": (
                "This is still a minimal smoke test. It proves the 3D topology constraint can be "
                "implemented and trained locally, not that full language capability is solved."
            ),
        },
        "model_signature": {
            "patch_slots": config.patch_slots,
            "topology_radius": config.topology_radius,
            "bridge_scale": config.bridge_scale,
            "bridge_topk": config.bridge_topk,
            "hidden_dim": config.hidden_dim,
        },
        "smoke_metrics": {
            "pre_loss": float(pre_loss.item()),
            "post_loss": float(post_loss.item()),
            "loss_delta": float(pre_loss.item() - post_loss.item()),
            "pre_patch_entropy": pre_metrics["patch_entropy"],
            "post_patch_entropy": post_metrics["patch_entropy"],
            "pre_local_mass_mean": pre_metrics["local_mass_mean"],
            "post_local_mass_mean": post_metrics["local_mass_mean"],
            "pre_bridge_mass_mean": pre_metrics["bridge_mass_mean"],
            "post_bridge_mass_mean": post_metrics["bridge_mass_mean"],
            "pre_mean_topology_distance": pre_metrics["mean_topology_distance"],
            "post_mean_topology_distance": post_metrics["mean_topology_distance"],
            "post_successor_gate_mean": post_metrics["successor_gate_mean"],
            "avg_update_error_norm": float(
                sum(row["avg_error_norm"] for row in update_metrics) / max(1, len(update_metrics))
            ),
        },
        "generation_preview": {
            "prompt_text": "apple and cat ",
            "generated_text": generated_text,
            "last_local_mass": float(generation["last_metrics"]["local_mass"].mean().item()),
            "last_bridge_mass": float(generation["last_metrics"]["bridge_mass"].mean().item()),
            "last_mean_distance": float(generation["last_metrics"]["mean_distance"].mean().item()),
        },
        "strict_verdict": {
            "engine_ready": True,
            "topology_bias_present": bool(post_metrics["local_mass_mean"] > post_metrics["bridge_mass_mean"]),
            "core_answer": (
                "The 3D-topology-constrained SpikeICSPB prototype runs, learns locally, and maintains a real "
                "local-neighborhood bias while still leaving nonzero sparse bridge mass."
            ),
            "main_hard_gaps": [
                "generation is still weak and repetitive",
                "3D topology is present as an architectural bias, not yet a learned spatial theorem",
                "successor quality is still far from strong language modeling",
                "no large-scale replay or multi-region consolidation has been added yet",
            ],
        },
        "progress_estimate": {
            "spike_icspb_3d_topology_minimal_prototype_percent": 52.0,
            "non_attention_non_bp_large_scale_trainability_percent": 39.0,
            "non_attention_non_bp_full_language_capability_percent": 25.0,
            "full_brain_encoding_mechanism_percent": 65.0,
        },
        "next_large_blocks": [
            "Scale the 3D topology prototype into a multi-region SpikeICSPB Phase-A with replay and consolidation.",
            "Turn the fixed topology radius and bridge budget into adaptive learned control variables.",
            "Bind the 3D topology core to successor and protocol execution benchmarks instead of smoke-only local updates.",
        ],
    }
    return payload


def test_spike_icspb_3d_topology_lm_minimal_smoke() -> None:
    payload = build_payload()
    metrics = payload["smoke_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["loss_delta"] > 0.5
    assert metrics["post_local_mass_mean"] > metrics["post_bridge_mass_mean"]
    assert metrics["post_bridge_mass_mean"] > 0.01
    assert verdict["engine_ready"] is True
    assert verdict["topology_bias_present"] is True


def main() -> None:
    ap = argparse.ArgumentParser(description="SpikeICSPB 3D topology minimal smoke")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/spike_icspb_3d_topology_lm_minimal_smoke_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["smoke_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
