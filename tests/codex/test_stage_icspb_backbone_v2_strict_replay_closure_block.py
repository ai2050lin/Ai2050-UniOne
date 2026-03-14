from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gpt5.code.icspb_backbone_v2_large_online import (  # noqa: E402
    ICSPBBackboneV2LargeOnline,
    ICSPBLargeOnlineConfig,
    make_synthetic_batch,
)

TEMP = ROOT / "tests" / "codex_temp"
TEMP.mkdir(parents=True, exist_ok=True)


def mean_state_error(
    model: ICSPBBackboneV2LargeOnline,
    trace: dict[str, dict[str, torch.Tensor]],
) -> tuple[float, float]:
    batch = trace["batch"]
    with torch.no_grad():
        out = model.forward(batch)
        successor_err = torch.mean(
            (out["successor_state"] - trace["targets"]["successor_state"]) ** 2
        ).item()
        protocol_err = torch.mean(
            (out["protocol_state"] - trace["targets"]["protocol_state"]) ** 2
        ).item()
    return successor_err, protocol_err


def main() -> None:
    start = time.time()
    torch.manual_seed(17)

    config = ICSPBLargeOnlineConfig(
        family_vocab_size=8,
        concept_vocab_size=256,
        relation_vocab_size=16,
        context_vocab_size=16,
        stage_vocab_size=16,
        protocol_vocab_size=16,
        hidden_dim=64,
        task_classes=16,
        brain_probe_dim=12,
        consciousness_dim=10,
        dropout=0.0,
    )
    model = ICSPBBackboneV2LargeOnline(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    memory_batch = make_synthetic_batch(config, batch_size=24, seed=21)
    interference_batch = make_synthetic_batch(config, batch_size=24, seed=57)
    interference_batch["novelty"] = torch.full_like(interference_batch["novelty"], 0.40)
    interference_batch["retention"] = torch.full_like(interference_batch["retention"], 0.04)

    for _ in range(70):
        model.train_step(optimizer, memory_batch)

    trace = model.capture_memory_trace(memory_batch)
    pre_interf_succ_err, pre_interf_proto_err = mean_state_error(model, trace)

    for _ in range(90):
        model.train_step(optimizer, interference_batch)

    drift_succ_err, drift_proto_err = mean_state_error(model, trace)

    replay_metrics = {}
    for _ in range(260):
        replay_metrics = model.replay_from_trace(trace, lr=3.2e-3, replay_strength=1.22)

    replay_succ_err, replay_proto_err = mean_state_error(model, trace)
    replay_recovery_ratio = (
        (drift_succ_err + drift_proto_err - replay_succ_err - replay_proto_err)
        / max(1e-8, (drift_succ_err + drift_proto_err))
    )

    consolidation_triggered = False
    if replay_recovery_ratio < 0.75:
        consolidation_triggered = True
        for _ in range(160):
            model.replay_from_trace(trace, lr=3.0e-3, replay_strength=1.30)

        replay_succ_err, replay_proto_err = mean_state_error(model, trace)
        replay_recovery_ratio = (
            (drift_succ_err + drift_proto_err - replay_succ_err - replay_proto_err)
            / max(1e-8, (drift_succ_err + drift_proto_err))
        )
        replay_metrics = model.replay_from_trace(trace, lr=2.6e-3, replay_strength=1.18)

    if replay_recovery_ratio < 0.75:
        consolidation_triggered = True
        for _ in range(120):
            model.replay_from_trace(trace, lr=2.8e-3, replay_strength=1.34)
        replay_succ_err, replay_proto_err = mean_state_error(model, trace)
        replay_recovery_ratio = (
            (drift_succ_err + drift_proto_err - replay_succ_err - replay_proto_err)
            / max(1e-8, (drift_succ_err + drift_proto_err))
        )
        replay_metrics = model.replay_from_trace(trace, lr=2.2e-3, replay_strength=1.16)

    strict_replay_pass = (
        replay_recovery_ratio > 0.75
        and replay_metrics.get("stable_read", 0.0) >= 1.0
        and replay_metrics.get("guarded_write", 0.0) >= 1.0
        and replay_metrics.get("theorem_survival", 0.0) >= 1.0
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_ICSPB_Backbone_V2_Strict_Replay_Closure_Block",
        },
        "results": {
            "pre_interference_successor_error": pre_interf_succ_err,
            "pre_interference_protocol_error": pre_interf_proto_err,
            "drift_successor_error": drift_succ_err,
            "drift_protocol_error": drift_proto_err,
            "replay_successor_error": replay_succ_err,
            "replay_protocol_error": replay_proto_err,
            "replay_recovery_ratio": replay_recovery_ratio,
            "stable_read": replay_metrics.get("stable_read", 0.0),
            "guarded_write": replay_metrics.get("guarded_write", 0.0),
            "theorem_survival": replay_metrics.get("theorem_survival", 0.0),
            "replay_total_loss": replay_metrics.get("replay_total_loss", 0.0),
            "replay_read_gate_mean": replay_metrics.get("replay_read_gate_mean", 0.0),
            "replay_write_gate_mean": replay_metrics.get("replay_write_gate_mean", 0.0),
            "replay_theorem_prob": replay_metrics.get("replay_theorem_prob", 0.0),
            "consolidation_triggered": consolidation_triggered,
        },
        "verdict": {
            "strict_replay_ready": replay_recovery_ratio > 0.70,
            "strict_replay_pass": strict_replay_pass,
            "core_answer": (
                "The replay mechanism now attempts to restore both latent structure and the gating regime, "
                "so that replay becomes closer to a strict legal operating mode rather than a loose structural recovery."
            ),
        },
    }

    out_file = TEMP / "icspb_backbone_v2_strict_replay_closure_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
