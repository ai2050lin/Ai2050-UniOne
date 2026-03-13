from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"

import sys

sys.path.append(str(ROOT))

from research.gpt5.code.icspb_backbone_v2_large_online import (  # noqa: E402
    ICSPBBackboneV2LargeOnline,
    ICSPBLargeOnlineConfig,
    make_synthetic_batch,
)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description="ICSPB Backbone v2 code-level prototype validation")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_icspb_backbone_v2_large_online_prototype_impl_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(7)

    cfg = ICSPBLargeOnlineConfig(
        hidden_dim=64,
        task_classes=16,
        brain_probe_dim=12,
        concept_vocab_size=512,
        family_vocab_size=8,
        relation_vocab_size=24,
        context_vocab_size=24,
        stage_vocab_size=24,
        protocol_vocab_size=24,
        dropout=0.0,
    )
    model = ICSPBBackboneV2LargeOnline(cfg)
    batch = make_synthetic_batch(cfg, batch_size=12, seed=11)

    out = model(batch)
    smoke_ok = (
        out["task_logits"].shape == (12, cfg.task_classes)
        and out["brain_probe"].shape == (12, cfg.brain_probe_dim)
        and out["successor_state"].shape == (12, cfg.hidden_dim)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    initial_loss, initial_metrics = model.compute_loss(batch)
    last_metrics = initial_metrics
    for _ in range(8):
        last_metrics = model.train_step(optimizer, batch)
    trained_loss, trained_metrics = model.compute_loss(batch)
    model.eval()
    trained_out = model(batch)

    model.snapshot()
    before_online_state = trained_out["protocol_state"].detach().clone()
    online_metrics = model.online_update_step(batch, lr=5e-3)
    model.eval()
    after_online_state = model(batch)["protocol_state"].detach().clone()
    online_delta = float((after_online_state - before_online_state).norm(dim=-1).mean().detach())
    rollback_ok = model.rollback()
    model.eval()
    after_rollback_state = model(batch)["protocol_state"].detach().clone()
    rollback_error = float((after_rollback_state - before_online_state).norm(dim=-1).mean().detach())

    loss_drop = float(initial_loss.detach() - trained_loss.detach())
    guarded_write = float(trained_out["write_gate"].mean().detach())
    stable_read = float(trained_out["read_gate"].mean().detach())
    theorem_survival = float(trained_metrics["theorem_survival"])
    transport_margin = float(trained_metrics["transport_margin"])
    online_write_scale = float(online_metrics["online_write_scale"])
    smoke_pass = smoke_ok
    training_pass = trained_metrics["total_loss"] <= initial_metrics["total_loss"]
    online_update_pass = online_delta > 0.0 and online_write_scale >= cfg.guarded_write_floor
    rollback_pass = rollback_ok and rollback_error < 1e-6
    implementation_bonus = 0.15 if (smoke_pass and training_pass and online_update_pass and rollback_pass) else 0.0

    implementation_score = clamp01(
        0.14 * float(smoke_pass)
        + 0.16 * float(training_pass)
        + 0.14 * float(online_update_pass)
        + 0.14 * float(rollback_pass)
        + 0.12 * clamp01(loss_drop / 0.25)
        + 0.08 * theorem_survival
        + 0.08 * clamp01(transport_margin / 0.5)
        + 0.05 * clamp01(guarded_write / max(cfg.guarded_write_floor, 1e-6))
        + 0.04 * clamp01(stable_read / max(cfg.stable_read_floor, 1e-6))
        + implementation_bonus
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "Stage_ICSPB_Backbone_v2_Large_Online_Prototype_Impl",
        },
        "shapes": {
            "task_logits": list(out["task_logits"].shape),
            "brain_probe": list(out["brain_probe"].shape),
            "successor_state": list(out["successor_state"].shape),
        },
        "training": {
            "initial_total_loss": float(initial_metrics["total_loss"]),
            "trained_total_loss": float(trained_metrics["total_loss"]),
            "loss_drop": loss_drop,
            "guarded_write": guarded_write,
            "stable_read": stable_read,
            "theorem_survival": theorem_survival,
            "transport_margin": transport_margin,
        },
        "online_update": {
            "online_write_scale": online_write_scale,
            "online_delta": online_delta,
            "rollback_ok": rollback_ok,
            "rollback_error": rollback_error,
        },
        "pass_status": {
            "smoke_pass": smoke_pass,
            "training_pass": training_pass,
            "online_update_pass": online_update_pass,
            "rollback_pass": rollback_pass,
            "implementation_ready": implementation_score >= 0.90,
        },
        "headline_metrics": {
            "implementation_score": implementation_score,
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
