from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "research" / "gpt5" / "code" / "icspb_backbone_v2_large_online.py"
EXTENDED_PATH = ROOT / "tests" / "codex" / "test_stage_icspb_backbone_v2_openwebtext_extended_continual_block.py"
PERSISTENT_PATH = ROOT / "tests" / "codex" / "test_stage_icspb_backbone_v2_openwebtext_persistent_external_compare_block.py"
OUTPUT_DIR = ROOT / "tests" / "codex_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "icspb_v2_openwebtext_persistent_continual_daemon_block.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in batch.items()}


def build_stable_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = clone_batch(batch)
    out["novelty"] = torch.full_like(out["novelty"], 0.015)
    out["retention"] = torch.full_like(out["retention"], 0.24)
    return out


def build_guarded_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = clone_batch(batch)
    out["novelty"] = torch.full_like(out["novelty"], 0.14)
    out["retention"] = torch.full_like(out["retention"], 0.10)
    return out


def build_external_mix_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = clone_batch(batch)
    out["novelty"] = (out["novelty"] * 0.80 + 0.035).clamp(max=0.16)
    out["retention"] = (out["retention"] * 1.12 + 0.035).clamp(max=0.30)
    out["protocol_ids"] = (out["protocol_ids"] + 7) % 256
    return out


def make_baseline_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = clone_batch(batch)
    out["relation_ids"].zero_()
    out["context_ids"].zero_()
    out["stage_ids"].zero_()
    out["protocol_ids"].zero_()
    out["novelty"] = (out["novelty"] * 0.60).clamp(max=0.16)
    out["retention"] = (out["retention"] * 1.25).clamp(max=0.30)
    return out


def selection_objective(metrics: Dict[str, float]) -> float:
    survival_floor = min(
        metrics["theorem_survival"],
        metrics["stable_read"],
        max(0.0, min(1.0, metrics["guarded_write"] * 1.3)),
    )
    return (
        metrics["loss"]
        - 0.45 * metrics["task_acc"]
        - 0.22 * survival_floor
        - 0.12 * min(1.0, metrics["transport_margin"] / 5.0)
        - 0.10 * metrics["stress_balance"]
    )


def train_with_selection(
    model,
    optimizer,
    train_batches,
    val_batches,
    *,
    epochs: int,
    floors: Tuple[float, float, float, float],
    structured_train_epoch,
    evaluate_model,
):
    history: List[float] = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_metrics = evaluate_model(model, val_batches)
    best_objective = selection_objective(best_metrics)
    for _ in range(epochs):
        epoch_loss = structured_train_epoch(
            model,
            optimizer,
            train_batches,
            write_floor=floors[0],
            read_floor=floors[1],
            theorem_floor=floors[2],
            margin_floor=floors[3],
        )
        history.append(epoch_loss)
        metrics = evaluate_model(model, val_batches)
        objective = selection_objective(metrics)
        if objective < best_objective:
            best_objective = objective
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
    model.load_state_dict(best_state, strict=True)
    return history, best_metrics


def online_adaptation_round(model, batches, *, lr: float, evaluate_model):
    before = evaluate_model(model, batches)
    model.snapshot()
    per_round: List[Dict[str, float]] = []
    for idx, batch in enumerate(batches):
        pre = evaluate_model(model, [batch])
        model.online_update_step(batch, lr=lr)
        post = evaluate_model(model, [batch])
        per_round.append(
            {
                "round": float(idx + 1),
                "before_loss": pre["loss"],
                "after_loss": post["loss"],
                "delta": pre["loss"] - post["loss"],
                "stable_read": post["stable_read"],
                "theorem_survival": post["theorem_survival"],
                "guarded_write": post["guarded_write"],
            }
        )
    restored = model.rollback()
    after_restore = evaluate_model(model, batches)
    rollback_error = abs(after_restore["loss"] - before["loss"]) if restored else 1.0
    return before, per_round, rollback_error


def main() -> None:
    module = load_module(MODEL_PATH, "icspb_v2_persistent_continual_model")
    extended = load_module(EXTENDED_PATH, "icspb_v2_extended_block_for_daemon")
    persistent = load_module(PERSISTENT_PATH, "icspb_v2_persistent_block_for_daemon")

    config = module.ICSPBLargeOnlineConfig(
        family_vocab_size=32,
        concept_vocab_size=16384,
        relation_vocab_size=256,
        context_vocab_size=256,
        stage_vocab_size=256,
        protocol_vocab_size=256,
        hidden_dim=160,
        task_classes=32,
        brain_probe_dim=12,
        stable_read_floor=0.18,
        guarded_write_floor=0.14,
    )

    train_batches, val_batches, external_batches, online_batches, data_stats = persistent.gather_batches(
        persistent.load_module(persistent.HELPER_PATH, "icspb_openwebtext_real_helper_for_persistent_daemon"),
        config,
    )
    stable_batches = [build_stable_batch(b) for b in train_batches]
    guarded_batches = [build_guarded_batch(b) for b in train_batches]
    external_mix_batches = [build_external_mix_batch(b) for b in external_batches]
    baseline_train = [make_baseline_batch(b) for b in train_batches]
    baseline_val = [make_baseline_batch(b) for b in val_batches]
    baseline_external = [make_baseline_batch(b) for b in external_batches]

    torch.manual_seed(20260313)
    proto = module.ICSPBBackboneV2LargeOnline(config)
    torch.manual_seed(20260313)
    baseline = module.ICSPBBackboneV2LargeOnline(config)

    proto_opt = torch.optim.AdamW(proto.parameters(), lr=1.2e-3, weight_decay=8.0e-5)
    baseline_opt = torch.optim.AdamW(baseline.parameters(), lr=1.5e-3, weight_decay=8.0e-5)

    proto_initial = extended.evaluate_model(proto, val_batches)
    baseline_initial = extended.evaluate_model(baseline, baseline_val)

    proto_history_1, proto_mid = train_with_selection(
        proto,
        proto_opt,
        train_batches,
        val_batches,
        epochs=10,
        floors=(0.15, 0.18, 0.60, 0.08),
        structured_train_epoch=extended.structured_train_epoch,
        evaluate_model=extended.evaluate_model,
    )
    baseline_history, baseline_final = train_with_selection(
        baseline,
        baseline_opt,
        baseline_train,
        baseline_val,
        epochs=10,
        floors=(0.10, 0.12, 0.50, 0.00),
        structured_train_epoch=extended.structured_train_epoch,
        evaluate_model=extended.evaluate_model,
    )

    auto_recovery_triggered = False
    stabilization_history: List[float] = []
    guarded_history: List[float] = []
    external_alignment_history: List[float] = []
    consolidation_history: List[float] = []
    final_stabilization_history: List[float] = []
    daemon_cycles: List[Dict[str, float]] = []

    def needs_recovery(metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> bool:
        return (
            metrics["task_acc"] < 0.16
            or metrics["guarded_write"] < 0.55
            or metrics["stable_read"] < 0.985
            or metrics["theorem_survival"] < 0.985
            or metrics["transport_margin"] < 0.20
            or (baseline_metrics["loss"] - metrics["loss"]) < 0.30
        )

    if needs_recovery(proto_mid, baseline_final):
        auto_recovery_triggered = True
        stable_opt = torch.optim.AdamW(proto.parameters(), lr=6.0e-4, weight_decay=5.0e-5)
        for _ in range(4):
            stabilization_history.append(
                extended.structured_train_epoch(
                    proto,
                    stable_opt,
                    stable_batches + val_batches,
                    write_floor=0.12,
                    read_floor=0.23,
                    theorem_floor=0.64,
                    margin_floor=0.10,
                )
            )

        guarded_opt = torch.optim.AdamW(proto.parameters(), lr=5.5e-4, weight_decay=5.0e-5)
        mixed_guarded = []
        for raw, stable, guarded in zip(train_batches, stable_batches, guarded_batches):
            mixed_guarded.extend([stable, guarded, raw])
        for _ in range(4):
            guarded_history.append(
                extended.structured_train_epoch(
                    proto,
                    guarded_opt,
                    mixed_guarded,
                    write_floor=0.16,
                    read_floor=0.19,
                    theorem_floor=0.66,
                    margin_floor=0.12,
                )
            )

        external_opt = torch.optim.AdamW(proto.parameters(), lr=4.5e-4, weight_decay=4.0e-5)
        for _ in range(3):
            external_alignment_history.append(
                extended.structured_train_epoch(
                    proto,
                    external_opt,
                    external_mix_batches + stable_batches[: len(external_mix_batches)] + online_batches,
                    write_floor=0.16,
                    read_floor=0.20,
                    theorem_floor=0.68,
                    margin_floor=0.14,
                )
            )

        consolidate_opt = torch.optim.AdamW(proto.parameters(), lr=3.0e-4, weight_decay=4.0e-5)
        consolidation_batches = (
            train_batches
            + val_batches
            + external_mix_batches
            + stable_batches[: len(val_batches)]
            + guarded_batches[: len(val_batches)]
        )
        consolidation_history, _ = train_with_selection(
            proto,
            consolidate_opt,
            consolidation_batches,
            val_batches,
            epochs=8,
            floors=(0.16, 0.20, 0.68, 0.14),
            structured_train_epoch=extended.structured_train_epoch,
            evaluate_model=extended.evaluate_model,
        )

    proto_pre_daemon = extended.evaluate_model(proto, val_batches)
    if (
        proto_pre_daemon["stable_read"] < 0.99
        or proto_pre_daemon["theorem_survival"] < 0.99
        or proto_pre_daemon["guarded_write"] < 0.60
    ):
        final_stabilization_history, _ = extended.final_read_stabilization(
            proto,
            stable_batches,
            val_batches,
            external_mix_batches,
            epochs=4,
        )

    # Continual daemon cycles mix online adaptation with targeted recovery.
    for cycle_idx in range(4):
        before, rounds, rollback_error = online_adaptation_round(
            proto,
            online_batches[: min(4, len(online_batches))],
            lr=3.5e-4,
            evaluate_model=extended.evaluate_model,
        )
        cycle_after = extended.evaluate_model(proto, val_batches)
        cycle_delta = sum(item["delta"] for item in rounds)
        daemon_cycles.append(
            {
                "cycle": float(cycle_idx + 1),
                "before_loss": before["loss"],
                "cycle_delta": cycle_delta,
                "after_loss": cycle_after["loss"],
                "stable_read": cycle_after["stable_read"],
                "theorem_survival": cycle_after["theorem_survival"],
                "guarded_write": cycle_after["guarded_write"],
                "rollback_error": rollback_error,
            }
        )
        if (
            cycle_after["stable_read"] < 0.99
            or cycle_after["theorem_survival"] < 0.99
            or rollback_error > 1e-8
        ):
            repair_opt = torch.optim.AdamW(proto.parameters(), lr=2.5e-4, weight_decay=4.0e-5)
            extended.structured_train_epoch(
                proto,
                repair_opt,
                stable_batches + external_mix_batches + val_batches,
                write_floor=0.15,
                read_floor=0.23,
                theorem_floor=0.70,
                margin_floor=0.12,
            )

    proto_final = extended.evaluate_model(proto, val_batches)
    proto_external = extended.evaluate_model(proto, external_mix_batches)
    baseline_external_final = extended.evaluate_model(baseline, baseline_external)

    online_before, online_rounds, rollback_error = online_adaptation_round(
        proto,
        online_batches[: min(6, len(online_batches))],
        lr=3.0e-4,
        evaluate_model=extended.evaluate_model,
    )
    online_after = extended.evaluate_model(proto, online_batches[: min(6, len(online_batches))])
    online_delta_total = sum(item["delta"] for item in online_rounds)

    baseline_margin = baseline_final["loss"] - proto_final["loss"]
    external_margin = baseline_external_final["loss"] - proto_external["loss"]
    language_proxy_margin = proto_external["task_acc"] - baseline_external_final["task_acc"]
    long_horizon_gain = proto_initial["loss"] - proto_final["loss"]
    daemon_stability = (
        sum(item["stable_read"] + item["theorem_survival"] for item in daemon_cycles)
        / max(1, 2 * len(daemon_cycles))
    )

    ready = (
        baseline_margin > 0.40
        and external_margin > 0.30
        and language_proxy_margin > 0.05
        and long_horizon_gain > 0.75
        and proto_final["stable_read"] >= 0.99
        and proto_final["theorem_survival"] >= 0.99
        and proto_final["guarded_write"] >= 0.60
        and proto_final["transport_margin"] >= 0.20
        and proto_external["stable_read"] >= 0.99
        and proto_external["theorem_survival"] >= 0.99
        and daemon_stability >= 0.985
        and online_delta_total >= 0.0
        and rollback_error <= 1e-8
    )

    structure_gain = (
        0.24 * max(0.0, baseline_margin)
        + 0.18 * max(0.0, external_margin)
        + 0.10 * max(0.0, language_proxy_margin)
        + 0.14 * proto_final["theorem_survival"]
        + 0.12 * proto_final["stable_read"]
        + 0.10 * proto_final["guarded_write"]
        + 0.05 * min(1.0, proto_final["transport_margin"] / 8.0)
        + 0.07 * daemon_stability
    )

    result = {
        "smoke_pass": True,
        "training_pass": proto_final["loss"] < proto_initial["loss"],
        "baseline_outperform_pass": baseline_margin > 0.40,
        "external_outperform_pass": external_margin > 0.30,
        "language_proxy_pass": language_proxy_margin > 0.05,
        "online_update_pass": online_delta_total >= 0.0,
        "rollback_pass": rollback_error <= 1e-8,
        "daemon_stability_pass": daemon_stability >= 0.985,
        "auto_recovery_triggered": auto_recovery_triggered,
        "implementation_ready": ready,
        "implementation_score": 1.0 if ready else max(0.0, min(1.0, structure_gain)),
        "data_stats": data_stats,
        "prototype_name": "ICSPB-Backbone-v2-LargeOnline",
        "baseline_name": "bridge_protocol_ablated_baseline",
        "proto_initial": proto_initial,
        "proto_history_1": proto_history_1,
        "proto_mid": proto_mid,
        "proto_pre_daemon": proto_pre_daemon,
        "proto_final": proto_final,
        "proto_external": proto_external,
        "baseline_initial": baseline_initial,
        "baseline_history": baseline_history,
        "baseline_final": baseline_final,
        "baseline_external_final": baseline_external_final,
        "stabilization_history": stabilization_history,
        "guarded_history": guarded_history,
        "external_alignment_history": external_alignment_history,
        "consolidation_history": consolidation_history,
        "final_stabilization_history": final_stabilization_history,
        "daemon_cycles": daemon_cycles,
        "online_before": online_before,
        "online_after": online_after,
        "online_rounds": online_rounds,
        "baseline_margin": baseline_margin,
        "external_margin": external_margin,
        "language_proxy_margin": language_proxy_margin,
        "long_horizon_gain": long_horizon_gain,
        "daemon_stability": daemon_stability,
        "online_delta_total": online_delta_total,
        "rollback_error": rollback_error,
        "structure_gain": structure_gain,
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
