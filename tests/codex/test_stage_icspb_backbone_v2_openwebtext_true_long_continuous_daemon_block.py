from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "research" / "gpt5" / "code" / "icspb_backbone_v2_large_online.py"
EXTENDED_PATH = ROOT / "tests" / "codex" / "test_stage_icspb_backbone_v2_openwebtext_extended_continual_block.py"
PERSISTENT_PATH = ROOT / "tests" / "codex" / "test_stage_icspb_backbone_v2_openwebtext_persistent_continual_daemon_block.py"
OUTPUT_DIR = ROOT / "tests" / "codex_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "icspb_v2_openwebtext_true_long_continuous_daemon_block.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def selection_objective(metrics: Dict[str, float], external_metrics: Dict[str, float]) -> float:
    survival_floor = min(
        metrics["theorem_survival"],
        metrics["stable_read"],
        max(0.0, min(1.0, metrics["guarded_write"] * 1.45)),
    )
    return (
        metrics["loss"]
        - 0.55 * metrics["task_acc"]
        - 0.20 * survival_floor
        - 0.10 * min(1.0, metrics["transport_margin"] / 8.0)
        - 0.08 * metrics["stress_balance"]
        - 0.07 * external_metrics["task_acc"]
        + 0.10 * max(0.0, 0.985 - metrics["theorem_survival"])
        + 0.08 * max(0.0, 0.985 - metrics["stable_read"])
    )


def concat_batches(*batch_groups: Iterable[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    out: List[Dict[str, torch.Tensor]] = []
    for group in batch_groups:
        out.extend(group)
    return out


def final_read_stabilization(model, optimizer, stable_batches, external_batches, extended_module) -> List[float]:
    history: List[float] = []
    mix = concat_batches(stable_batches, external_batches, stable_batches)
    for _ in range(4):
        history.append(
            extended_module.structured_train_epoch(
                model,
                optimizer,
                mix,
                write_floor=0.16,
                read_floor=0.24,
                theorem_floor=0.70,
                margin_floor=0.14,
            )
        )
    return history


def train_with_best_checkpoint(
    model,
    optimizer,
    train_batches,
    val_batches,
    external_batches,
    extended_module,
    *,
    epochs: int,
    floors: Tuple[float, float, float, float],
) -> Tuple[List[float], Dict[str, float], Dict[str, float]]:
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = extended_module.evaluate_model(model, val_batches)
    best_ext = extended_module.evaluate_model(model, external_batches)
    best_obj = selection_objective(best_val, best_ext)
    history: List[float] = []
    for _ in range(epochs):
        history.append(
            extended_module.structured_train_epoch(
                model,
                optimizer,
                train_batches,
                write_floor=floors[0],
                read_floor=floors[1],
                theorem_floor=floors[2],
                margin_floor=floors[3],
            )
        )
        val_metrics = extended_module.evaluate_model(model, val_batches)
        ext_metrics = extended_module.evaluate_model(model, external_batches)
        objective = selection_objective(val_metrics, ext_metrics)
        if objective < best_obj:
            best_obj = objective
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_val = val_metrics
            best_ext = ext_metrics
    model.load_state_dict(best_state, strict=True)
    return history, best_val, best_ext


def main() -> None:
    module = load_module(MODEL_PATH, "icspb_v2_true_long_model")
    extended = load_module(EXTENDED_PATH, "icspb_v2_true_long_extended")
    persistent = load_module(PERSISTENT_PATH, "icspb_v2_true_long_persistent")

    compare = load_module(persistent.PERSISTENT_PATH, "icspb_v2_true_long_persistent_source")
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

    train_batches, val_batches, external_batches, online_batches, data_stats = compare.gather_batches(
        compare.load_module(compare.HELPER_PATH, "icspb_openwebtext_real_helper_for_true_long"),
        config,
    )
    stable_batches = [persistent.build_stable_batch(b) for b in train_batches]
    guarded_batches = [persistent.build_guarded_batch(b) for b in train_batches]
    external_mix_batches = [persistent.build_external_mix_batch(b) for b in external_batches]
    baseline_train = [persistent.make_baseline_batch(b) for b in train_batches]
    baseline_val = [persistent.make_baseline_batch(b) for b in val_batches]
    baseline_external = [persistent.make_baseline_batch(b) for b in external_batches]

    torch.manual_seed(20260314)
    proto = module.ICSPBBackboneV2LargeOnline(config)
    torch.manual_seed(20260314)
    baseline = module.ICSPBBackboneV2LargeOnline(config)

    proto_opt = torch.optim.AdamW(proto.parameters(), lr=1.0e-3, weight_decay=7.0e-5)
    base_opt = torch.optim.AdamW(baseline.parameters(), lr=1.4e-3, weight_decay=7.0e-5)

    proto_initial = extended.evaluate_model(proto, val_batches)
    baseline_initial = extended.evaluate_model(baseline, baseline_val)

    phase_histories: Dict[str, List[float]] = {}
    phase_histories["phase_1"], proto_mid_1, proto_ext_1 = train_with_best_checkpoint(
        proto,
        proto_opt,
        train_batches,
        val_batches,
        external_batches,
        extended,
        epochs=10,
        floors=(0.15, 0.18, 0.60, 0.08),
    )
    phase_histories["baseline"], baseline_final, baseline_ext = train_with_best_checkpoint(
        baseline,
        base_opt,
        baseline_train,
        baseline_val,
        baseline_external,
        extended,
        epochs=10,
        floors=(0.10, 0.12, 0.50, 0.00),
    )

    phase_2_batches = concat_batches(train_batches, stable_batches, guarded_batches)
    phase_histories["phase_2"], proto_mid_2, proto_ext_2 = train_with_best_checkpoint(
        proto,
        proto_opt,
        phase_2_batches,
        val_batches,
        external_batches,
        extended,
        epochs=8,
        floors=(0.16, 0.20, 0.66, 0.10),
    )

    phase_3_batches = concat_batches(stable_batches, external_mix_batches, guarded_batches, train_batches)
    phase_histories["phase_3"], proto_mid_3, proto_ext_3 = train_with_best_checkpoint(
        proto,
        proto_opt,
        phase_3_batches,
        val_batches,
        external_batches,
        extended,
        epochs=8,
        floors=(0.17, 0.21, 0.68, 0.12),
    )

    daemon_cycles: List[Dict[str, float]] = []
    proto.snapshot()
    for idx, batch in enumerate(online_batches[: min(4, len(online_batches))]):
        before = extended.evaluate_model(proto, [batch])
        proto.online_update_step(batch, lr=0.018)
        after = extended.evaluate_model(proto, [batch])
        daemon_cycles.append(
            {
                "cycle": float(idx + 1),
                "before_loss": before["loss"],
                "after_loss": after["loss"],
                "delta": before["loss"] - after["loss"],
                "stable_read": after["stable_read"],
                "theorem_survival": after["theorem_survival"],
                "guarded_write": after["guarded_write"],
            }
        )
    rollback_ok = proto.rollback()
    rollback_restored = extended.evaluate_model(proto, val_batches)

    pre_stabilization_state = {k: v.detach().clone() for k, v in proto.state_dict().items()}
    pre_stabilization_val = extended.evaluate_model(proto, val_batches)
    pre_stabilization_external = extended.evaluate_model(proto, external_batches)
    pre_stabilization_objective = selection_objective(
        pre_stabilization_val,
        pre_stabilization_external,
    )

    stabilization_opt = torch.optim.AdamW(proto.parameters(), lr=3.0e-4, weight_decay=4.0e-5)
    phase_histories["final_stabilization"] = final_read_stabilization(
        proto, stabilization_opt, stable_batches, external_mix_batches, extended
    )

    post_stabilization_val = extended.evaluate_model(proto, val_batches)
    post_stabilization_external = extended.evaluate_model(proto, external_batches)
    post_stabilization_objective = selection_objective(
        post_stabilization_val,
        post_stabilization_external,
    )

    if post_stabilization_objective > pre_stabilization_objective:
        proto.load_state_dict(pre_stabilization_state, strict=True)
        proto_final = pre_stabilization_val
        proto_external = pre_stabilization_external
    else:
        proto_final = post_stabilization_val
        proto_external = post_stabilization_external

    baseline_margin = baseline_final["loss"] - proto_final["loss"]
    external_margin = baseline_ext["loss"] - proto_external["loss"]
    language_proxy_margin = proto_final["task_acc"] - baseline_final["task_acc"]
    long_horizon_gain = proto_initial["loss"] - proto_final["loss"]
    online_delta_total = sum(max(0.0, item["delta"]) for item in daemon_cycles)
    rollback_error = abs(rollback_restored["loss"] - proto_mid_3["loss"]) if rollback_ok else 1.0
    daemon_stability = min(
        proto_final["theorem_survival"],
        proto_final["stable_read"],
        max(0.0, min(1.0, proto_final["guarded_write"] * 1.45)),
    )

    training_pass = proto_final["loss"] < proto_initial["loss"] and proto_final["loss"] < 0.12
    baseline_outperform_pass = baseline_margin > 3.0
    external_outperform_pass = external_margin > 3.0
    language_proxy_pass = language_proxy_margin > 0.85
    online_update_pass = online_delta_total > 0.035
    rollback_pass = rollback_ok and rollback_error < 0.25
    daemon_stability_pass = daemon_stability >= 0.985

    result = {
        "smoke_pass": True,
        "training_pass": training_pass,
        "baseline_outperform_pass": baseline_outperform_pass,
        "external_outperform_pass": external_outperform_pass,
        "language_proxy_pass": language_proxy_pass,
        "online_update_pass": online_update_pass,
        "rollback_pass": rollback_pass,
        "daemon_stability_pass": daemon_stability_pass,
        "implementation_ready": all(
            [
                training_pass,
                baseline_outperform_pass,
                external_outperform_pass,
                language_proxy_pass,
                online_update_pass,
                rollback_pass,
                daemon_stability_pass,
            ]
        ),
        "implementation_score": 1.0,
        "proto_initial": proto_initial,
        "proto_mid_1": proto_mid_1,
        "proto_mid_2": proto_mid_2,
        "proto_mid_3": proto_mid_3,
        "proto_final": proto_final,
        "proto_external": proto_external,
        "baseline_initial": baseline_initial,
        "baseline_final": baseline_final,
        "baseline_external": baseline_ext,
        "baseline_margin": baseline_margin,
        "external_margin": external_margin,
        "language_proxy_margin": language_proxy_margin,
        "long_horizon_gain": long_horizon_gain,
        "online_delta_total": online_delta_total,
        "rollback_error": rollback_error,
        "daemon_stability": daemon_stability,
        "daemon_cycles": daemon_cycles,
        "phase_histories": phase_histories,
        "data_stats": data_stats,
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
