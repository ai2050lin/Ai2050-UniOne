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


def main() -> None:
    start = time.time()
    torch.manual_seed(13)

    config = ICSPBLargeOnlineConfig(
        family_vocab_size=8,
        concept_vocab_size=256,
        relation_vocab_size=16,
        context_vocab_size=16,
        stage_vocab_size=16,
        protocol_vocab_size=16,
        hidden_dim=64,
        visual_input_dim=24,
        audio_input_dim=20,
        consciousness_dim=12,
        task_classes=16,
        brain_probe_dim=10,
        dropout=0.0,
    )
    model = ICSPBBackboneV2LargeOnline(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    batch = make_synthetic_batch(config, batch_size=20, seed=23)
    replay_batch = make_synthetic_batch(config, batch_size=20, seed=29)
    replay_batch["novelty"] = torch.full_like(replay_batch["novelty"], 0.28)
    replay_batch["retention"] = torch.full_like(replay_batch["retention"], 0.06)

    with torch.no_grad():
        out0 = model.forward(batch)
    smoke_pass = all(
        key in out0
        for key in (
            "visual_state",
            "audio_state",
            "consciousness_state",
            "consciousness_logits",
            "task_logits",
        )
    )

    initial_loss, _ = model.compute_loss(batch)
    for _ in range(80):
        train_metrics = model.train_step(optimizer, batch)
    final_loss = train_metrics["total_loss"]
    training_pass = final_loss < float(initial_loss.detach())

    online_metrics = model.online_update_step(replay_batch, lr=1e-3)
    online_pass = online_metrics["conscious_access"] > 0.45

    trace = model.capture_memory_trace(batch)
    replay_metrics = {}
    for _ in range(60):
        replay_metrics = model.replay_from_trace(trace, lr=2e-3, replay_strength=1.0)

    with torch.no_grad():
        out1 = model.forward(batch)
        visual_energy = float(out1["visual_state"].norm(dim=-1).mean().detach())
        audio_energy = float(out1["audio_state"].norm(dim=-1).mean().detach())
        consciousness_energy = float(out1["consciousness_state"].norm(dim=-1).mean().detach())

    multimodal_pass = visual_energy > 0.1 and audio_energy > 0.1
    consciousness_pass = consciousness_energy > 0.1 and replay_metrics["conscious_access"] > 0.45
    replay_pass = replay_metrics["theorem_survival"] >= 1.0

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_ICSPB_Backbone_V2_Multimodal_Conscious_Block",
        },
        "results": {
            "initial_total_loss": float(initial_loss.detach()),
            "trained_total_loss": final_loss,
            "loss_drop": float(initial_loss.detach()) - final_loss,
            "visual_energy": visual_energy,
            "audio_energy": audio_energy,
            "consciousness_energy": consciousness_energy,
            "conscious_access": replay_metrics.get("conscious_access", 0.0),
            "theorem_survival": replay_metrics.get("theorem_survival", 0.0),
            "transport_margin": replay_metrics.get("transport_margin", 0.0),
        },
        "verdict": {
            "smoke_pass": smoke_pass,
            "training_pass": training_pass,
            "online_pass": online_pass,
            "multimodal_pass": multimodal_pass,
            "consciousness_pass": consciousness_pass,
            "replay_pass": replay_pass,
            "implementation_ready": (
                smoke_pass
                and training_pass
                and online_pass
                and multimodal_pass
                and consciousness_pass
            ),
        },
    }

    out_file = TEMP / "icspb_backbone_v2_multimodal_conscious_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
