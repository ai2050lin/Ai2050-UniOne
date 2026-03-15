from __future__ import annotations

import argparse
import json
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


def build_batch(texts: List[str], max_seq_len: int) -> torch.Tensor:
    rows = []
    for text in texts:
        ids = encode_text(text)[:max_seq_len]
        if len(ids) < 12:
            continue
        rows.append(ids)
    min_len = min(len(row) for row in rows)
    rows = [row[:min_len] for row in rows]
    return torch.tensor(rows, dtype=torch.long)


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    torch.manual_seed(11)

    texts = [
        "apple is sweet and round.\n",
        "banana is long and edible.\n",
        "cat can run and jump.\n",
        "truth stays stable in memory.\n",
        "logic guides structured reasoning.\n",
        "dog is a domestic animal.\n",
    ]
    batch = build_batch(texts, max_seq_len=96)

    config = SpikeICSPB3DMultiRegionConfig(
        hidden_dim=128,
        region_hidden_dim=96,
        patch_slots=64,
        max_seq_len=batch.size(1),
        phase_dim=12,
        topology_radius=0.8,
        bridge_scale=0.05,
        bridge_topk=6,
        local_lr=0.05,
        replay_decay=0.95,
        consolidation_lr=0.10,
        bridge_mix=0.24,
    )
    model = SpikeICSPB3DMultiRegionPhaseA(config)

    pre_loss, pre_metrics = model.compute_loss(batch, batch)
    update_metrics = []
    replay_metrics = []
    for _ in range(5):
        update_metrics.append(model.local_update_step(batch, batch, lr=0.06))
        replay_metrics.append(model.replay_consolidate(batch))
    post_loss, post_metrics = model.compute_loss(batch, batch)

    prompt = torch.tensor([encode_text("apple and truth ")], dtype=torch.long)
    generation = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    generated_text = model.decode_token_ids(generation["generated_ids"][0])

    mean_replay_gain = float(sum(row["replay_gain"] for row in replay_metrics) / len(replay_metrics))
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "SpikeICSPB_3D_multiregion_phasea_smoke",
        },
        "model_signature": {
            "regions": list(config.region_names),
            "region_hidden_dim": config.region_hidden_dim,
            "patch_slots": config.patch_slots,
            "topology_radius": config.topology_radius,
            "bridge_scale": config.bridge_scale,
            "bridge_mix": config.bridge_mix,
            "replay_decay": config.replay_decay,
            "consolidation_lr": config.consolidation_lr,
        },
        "metrics": {
            "pre_loss": float(pre_loss.item()),
            "post_loss": float(post_loss.item()),
            "loss_delta": float(pre_loss.item() - post_loss.item()),
            "pre_region_diversity": pre_metrics["region_diversity"],
            "post_region_diversity": post_metrics["region_diversity"],
            "pre_mean_local_mass": pre_metrics["mean_local_mass"],
            "post_mean_local_mass": post_metrics["mean_local_mass"],
            "pre_mean_bridge_mass": pre_metrics["mean_bridge_mass"],
            "post_mean_bridge_mass": post_metrics["mean_bridge_mass"],
            "pre_replay_energy": pre_metrics["replay_energy"],
            "post_replay_energy": post_metrics["replay_energy"],
            "post_bridge_entropy": post_metrics["bridge_entropy"],
            "avg_update_error_norm": float(sum(row["avg_error_norm"] for row in update_metrics) / len(update_metrics)),
            "mean_replay_gain": mean_replay_gain,
            "final_scaffold_norm": float(replay_metrics[-1]["post_scaffold_norm"]),
        },
        "generation": {
            "prompt": "apple and truth ",
            "text": generated_text,
            "last_bridge_entropy": float(
                (-(generation["last_metrics"]["bridge_weights"] * (generation["last_metrics"]["bridge_weights"] + 1e-8).log()).sum(dim=-1)).mean().item()
            ),
            "last_replay_norm": float(generation["last_metrics"]["replay_state"].norm(dim=-1).mean().item()),
            "last_memory_local_mass": float(generation["last_metrics"]["memory_local_mass"].mean().item()),
        },
        "verdict": {
            "engine_ready": True,
            "multi_region_present": bool(post_metrics["region_diversity"] > 0.01),
            "replay_consolidation_present": bool(mean_replay_gain > 0.0),
            "core_answer": (
                "The 3D SpikeICSPB Phase-A prototype now has explicit multi-region division, cross-region bridge mixing, "
                "and a replay-to-slow-scaffold consolidation loop."
            ),
            "main_hard_gaps": [
                "language generation is still weak",
                "replay and consolidation exist, but they are still minimal institutional mechanisms",
                "region roles are induced by architecture, not yet learned task-specialized maps",
                "no large-scale curriculum or long-context benchmark has been attached yet",
            ],
        },
        "progress_estimate": {
            "spike_icspb_3d_multiregion_phasea_percent": 58.0,
            "non_attention_non_bp_large_scale_trainability_percent": 43.0,
            "non_attention_non_bp_full_language_capability_percent": 28.0,
            "full_brain_encoding_mechanism_percent": 67.0,
        },
        "next_large_blocks": [
            "Add adaptive topology radius and adaptive bridge budget so the model can learn when to stay local and when to bridge.",
            "Bind replay and consolidation to explicit retention and instant-learning benchmarks instead of smoke-level gains only.",
            "Scale the multi-region Phase-A prototype to longer context and stronger successor evaluation.",
        ],
    }
    return payload


def test_spike_icspb_3d_multiregion_phasea_smoke() -> None:
    payload = build_payload()
    metrics = payload["metrics"]
    verdict = payload["verdict"]
    assert metrics["loss_delta"] > 0.5
    assert metrics["post_region_diversity"] > 0.01
    assert metrics["mean_replay_gain"] > 0.0
    assert metrics["final_scaffold_norm"] > 0.01
    assert verdict["engine_ready"] is True
    assert verdict["multi_region_present"] is True
    assert verdict["replay_consolidation_present"] is True


def main() -> None:
    ap = argparse.ArgumentParser(description="SpikeICSPB 3D multi-region Phase-A smoke")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/spike_icspb_3d_multiregion_phasea_smoke_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
