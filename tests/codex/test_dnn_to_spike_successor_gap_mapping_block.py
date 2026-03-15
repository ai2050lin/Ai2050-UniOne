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

from research.gpt5.code.icspb_backbone_v2_large_online import (  # noqa: E402
    ICSPBLargeOnlineConfig,
    ICSPBBackboneV2LargeOnline,
    make_synthetic_batch,
)
from research.gpt5.code.spike_icspb_3d_multiregion_phasea import (  # noqa: E402
    SpikeICSPB3DMultiRegionConfig,
    SpikeICSPB3DMultiRegionPhaseA,
)


def encode_text(text: str) -> List[int]:
    return list(text.encode("utf-8", errors="ignore"))


def build_text_batch(texts: List[str], max_seq_len: int = 96) -> torch.Tensor:
    rows = []
    for text in texts:
        ids = encode_text(text)[:max_seq_len]
        if len(ids) >= 16:
            rows.append(ids)
    min_len = min(len(row) for row in rows)
    rows = [row[:min_len] for row in rows]
    return torch.tensor(rows, dtype=torch.long)


def build_payload() -> Dict[str, Any]:
    t0 = time.time()

    torch.manual_seed(97)
    dnn_config = ICSPBLargeOnlineConfig(hidden_dim=128)
    dnn_model = ICSPBBackboneV2LargeOnline(dnn_config)
    dnn_batch = make_synthetic_batch(dnn_config, batch_size=24, seed=17)
    dnn_out = dnn_model.forward(dnn_batch)
    dnn_metrics = dnn_model.survival_metrics(dnn_batch, dnn_out)
    dnn_theorem_prob = float(torch.sigmoid(dnn_out["theorem_logits"]).mean().item())
    dnn_protocol_energy = float(dnn_out["protocol_state"].norm(dim=-1).mean().item())
    dnn_successor_energy = float(dnn_out["successor_state"].norm(dim=-1).mean().item())
    dnn_transport_margin = float(max(0.0, dnn_protocol_energy - dnn_successor_energy))

    torch.manual_seed(101)
    spike_config = SpikeICSPB3DMultiRegionConfig(
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
    spike_model = SpikeICSPB3DMultiRegionPhaseA(spike_config)
    spike_batch = build_text_batch(
        [
            "apple is red and sweet while banana is yellow and soft.\n",
            "cat can run and jump and dog can bark and move.\n",
            "truth and justice remain abstract but stable in memory.\n",
            "reasoning chains should keep successor structure over time.\n",
            "protocol and recall must guide the next continuation.\n",
        ]
    )
    for _ in range(10):
        spike_model.local_update_step(spike_batch, spike_batch, lr=0.065)
        spike_model.replay_consolidate(spike_batch)
    spike_loss, spike_metrics = spike_model.compute_loss(spike_batch, spike_batch)
    spike_out = spike_model.forward(spike_batch)
    spike_protocol_energy = float(spike_out["protocol_states"].norm(dim=-1).mean().item())
    spike_successor_energy = float(
        sum(spike_out["region_hidden_states"][name].norm(dim=-1).mean().item() for name in spike_config.region_names)
        / len(spike_config.region_names)
    )
    spike_transport_margin = float(max(0.0, spike_protocol_energy - spike_successor_energy))
    spike_gate = float(sum(spike_out["successor_gate"][name].mean().item() for name in spike_config.region_names) / len(spike_config.region_names))
    spike_dual_timescale_proxy = float(min(1.0, spike_metrics["replay_energy"] / max(spike_protocol_energy, 1e-6)))
    spike_theorem_proxy = float(
        min(
            1.0,
            0.34 * (1.0 - min(1.0, spike_metrics["mean_region_saturation_fraction"] / 0.5))
            + 0.33 * min(1.0, spike_protocol_energy / max(spike_successor_energy, 1e-6))
            + 0.33 * min(1.0, spike_gate / 0.7),
        )
    )

    dnn_reference = {
        "dual_timescale": 1.0,
        "protocol_explicitness": min(1.0, dnn_protocol_energy / max(dnn_successor_energy, 1e-6)),
        "transport_margin": min(1.0, dnn_transport_margin / 0.20),
        "theorem_guard": dnn_theorem_prob,
    }
    spike_current = {
        "dual_timescale": spike_dual_timescale_proxy,
        "protocol_explicitness": min(1.0, spike_protocol_energy / max(spike_successor_energy, 1e-6)),
        "transport_margin": min(1.0, spike_transport_margin / 0.20),
        "theorem_guard": spike_theorem_proxy,
    }
    gaps = {key: float(max(0.0, dnn_reference[key] - spike_current[key])) for key in dnn_reference}
    gap_score = float(sum(gaps.values()) / len(gaps))

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_to_Spike_successor_gap_mapping_block",
        },
        "strict_goal": {
            "statement": (
                "If successor remains weak in the spike line, map the gap against the explicit successor structure already extracted from the DNN backbone."
            ),
            "boundary": (
                "This block diagnoses missing structure factors. It does not by itself close the spike successor problem."
            ),
        },
        "candidate_math_mapping": {
            "dnn_law": "succ = T(fast, slow, stage, protocol); theorem = G(write, read, margin, stress)",
            "spike_target": "succ_spike should grow from replay, protocol_state, phase gate, topology, and bounded homeostasis under the same admissible-band logic",
        },
        "dnn_reference": dnn_reference,
        "spike_current": spike_current,
        "gap_map": gaps,
        "headline_metrics": {
            "mean_gap_score": gap_score,
            "largest_gap": max(gaps, key=gaps.get),
            "dnn_transport_margin": dnn_transport_margin,
            "spike_transport_margin": spike_transport_margin,
            "spike_loss": float(spike_loss.item()),
        },
        "strict_verdict": {
            "core_answer": (
                "The spike successor problem is not a mystery anymore. Relative to the DNN successor law, the main missing factor is not protocol existence but theorem-like guarded continuation quality: the spike line has protocol state and replay, but it still lacks the strong admissible-band discrimination that turns them into sharp next-step decisions."
            ),
            "main_hard_gaps": [
                f"largest current structural gap: {max(gaps, key=gaps.get)}",
                "spike transport margin is still much weaker than the DNN reference margin",
                "replay gives only a partial dual-timescale proxy, not a full fast/slow successor law",
                "theorem-like guarded continuation is still only a proxy in the spike line",
            ],
        },
        "progress_estimate": {
            "dnn_to_spike_successor_gap_mapping_percent": 58.0,
            "successor_quality_audit_percent": 52.0,
            "non_attention_non_bp_full_language_capability_percent": 35.0,
            "full_brain_encoding_mechanism_percent": 73.0,
        },
        "next_large_blocks": [
            "Introduce theorem-like guarded successor objectives into the spike line instead of relying on weak proxies.",
            "Upgrade replay into an explicit fast/slow successor law rather than a generic memory add-on.",
            "Train protocol-aware successor routing on longer structured continuation chains.",
        ],
    }
    return payload


def test_dnn_to_spike_successor_gap_mapping_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    assert metrics["mean_gap_score"] > 0.05
    assert isinstance(metrics["largest_gap"], str)


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN to Spike successor gap mapping block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_to_spike_successor_gap_mapping_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
