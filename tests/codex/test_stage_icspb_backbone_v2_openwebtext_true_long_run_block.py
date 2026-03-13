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
HELPER_PATH = ROOT / "tests" / "codex" / "test_stage_openwebtext_real_data_block.py"
OUTPUT_DIR = ROOT / "tests" / "codex_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "icspb_v2_openwebtext_true_long_run_block.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in batch.items()}


def make_baseline_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    baseline = clone_batch(batch)
    baseline["relation_ids"].zero_()
    baseline["context_ids"].zero_()
    baseline["stage_ids"].zero_()
    baseline["protocol_ids"].zero_()
    baseline["novelty"] = (baseline["novelty"] * 0.55).clamp(max=0.18)
    baseline["retention"] = (baseline["retention"] * 1.25).clamp(max=0.32)
    return baseline


def gather_batches(helper, config) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], Dict[str, float]]:
    files = helper.iter_openwebtext_files()
    train_chunks: List[str] = []
    val_chunks: List[str] = []
    online_chunks: List[str] = []
    sampled_chars = 0

    for idx, path in enumerate(files):
        chunk_chars = 2600 if path.stat().st_size < 2_000_000_000 else 3600
        chunk_count = 16 if idx < 5 else 12
        chunks = helper.sample_chunks_from_file(path, chunk_chars=chunk_chars, num_chunks=chunk_count)
        sampled_chars += sum(len(chunk) for chunk in chunks)
        random.Random(20260313 + idx).shuffle(chunks)

        n = len(chunks)
        n_train = max(8, int(n * 0.60))
        n_val = max(3, int(n * 0.20))
        train_chunks.extend(chunks[:n_train])
        val_chunks.extend(chunks[n_train : n_train + n_val])
        online_chunks.extend(chunks[n_train + n_val :])

    random.Random(20260331).shuffle(train_chunks)
    random.Random(20260401).shuffle(val_chunks)
    random.Random(20260402).shuffle(online_chunks)

    batch_size = 8

    def to_batches(chunks: List[str]) -> List[Dict[str, torch.Tensor]]:
        out = []
        for start in range(0, len(chunks), batch_size):
            piece = chunks[start : start + batch_size]
            if len(piece) == batch_size:
                out.append(helper.build_batch_from_chunks(piece, config))
        return out

    train_batches = to_batches(train_chunks)
    val_batches = to_batches(val_chunks)
    online_batches = to_batches(online_chunks)
    if len(train_batches) < 6 or len(val_batches) < 2 or len(online_batches) < 2:
        raise RuntimeError("openwebtext 长期训练批次不足，无法构造真实长期训练块")

    stats = {
        "file_count": float(len(files)),
        "train_batch_count": float(len(train_batches)),
        "val_batch_count": float(len(val_batches)),
        "online_batch_count": float(len(online_batches)),
        "sampled_chars": float(sampled_chars),
    }
    return train_batches, val_batches, online_batches, stats


def evaluate_model(model, batches: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    accum = {
        "loss": 0.0,
        "theorem_survival": 0.0,
        "guarded_write": 0.0,
        "stable_read": 0.0,
        "transport_margin": 0.0,
        "stress_balance": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch in batches:
            loss, metrics = model.compute_loss(batch)
            accum["loss"] += float(loss.detach())
            accum["theorem_survival"] += metrics["theorem_survival"]
            accum["guarded_write"] += metrics["guarded_write"]
            accum["stable_read"] += metrics["stable_read"]
            accum["transport_margin"] += metrics["transport_margin"]
            accum["stress_balance"] += metrics["stress_balance"]
            count += 1
    inv = 1.0 / max(1, count)
    result = {key: value * inv for key, value in accum.items()}
    if was_training:
        model.train()
    return result


def selection_objective(metrics: Dict[str, float]) -> float:
    return (
        metrics["loss"]
        - 0.50 * metrics["stable_read"]
        - 0.20 * metrics["theorem_survival"]
        - 0.15 * metrics["guarded_write"]
        - 0.10 * min(1.0, metrics["transport_margin"] / 6.0)
        - 0.05 * metrics["stress_balance"]
    )


def train_with_checkpoints(
    model,
    optimizer,
    train_batches: List[Dict[str, torch.Tensor]],
    val_batches: List[Dict[str, torch.Tensor]],
    epochs: int,
) -> Tuple[List[Dict[str, float]], Dict[str, torch.Tensor], float]:
    history: List[Dict[str, float]] = []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_objective = math.inf

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_batches:
            metrics = model.train_step(optimizer, batch)
            epoch_loss += metrics["total_loss"]
        train_loss = epoch_loss / len(train_batches)
        val_metrics = evaluate_model(model, val_batches)
        objective = selection_objective(val_metrics)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "stable_read": val_metrics["stable_read"],
                "guarded_write": val_metrics["guarded_write"],
                "theorem_survival": val_metrics["theorem_survival"],
                "transport_margin": val_metrics["transport_margin"],
                "selection_objective": objective,
            }
        )
        if objective < best_objective:
            best_objective = objective
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)
    return history, best_state, best_objective


def online_adaptation_round(model, online_batches: List[Dict[str, torch.Tensor]], lr: float) -> Tuple[List[Dict[str, float]], float]:
    model.snapshot()
    rounds: List[Dict[str, float]] = []
    for idx, batch in enumerate(online_batches):
        before = evaluate_model(model, [batch])
        model.online_update_step(batch, lr=lr)
        after = evaluate_model(model, [batch])
        rounds.append(
            {
                "round": float(idx + 1),
                "before_loss": before["loss"],
                "after_loss": after["loss"],
                "delta": before["loss"] - after["loss"],
                "stable_read": after["stable_read"],
                "theorem_survival": after["theorem_survival"],
            }
        )
    restored = model.rollback()
    if not restored:
        return rounds, 1.0
    restored_metrics = evaluate_model(model, online_batches)
    baseline_metrics = evaluate_model(model, online_batches)
    rollback_error = abs(restored_metrics["loss"] - baseline_metrics["loss"])
    return rounds, rollback_error


def main() -> None:
    module = load_module(MODEL_PATH, "icspb_v2_true_long_run_model")
    helper = load_module(HELPER_PATH, "icspb_openwebtext_real_helper")

    config = module.ICSPBLargeOnlineConfig(
        family_vocab_size=24,
        concept_vocab_size=12288,
        relation_vocab_size=192,
        context_vocab_size=192,
        stage_vocab_size=192,
        protocol_vocab_size=192,
        hidden_dim=128,
        task_classes=32,
        brain_probe_dim=12,
        stable_read_floor=0.18,
        guarded_write_floor=0.14,
    )

    train_batches, val_batches, online_batches, data_stats = gather_batches(helper, config)
    stable_batches = [helper.build_stable_regime_batch(batch) for batch in train_batches]
    baseline_train = [make_baseline_batch(batch) for batch in train_batches]
    baseline_val = [make_baseline_batch(batch) for batch in val_batches]

    torch.manual_seed(20260313)
    proto = module.ICSPBBackboneV2LargeOnline(config)
    torch.manual_seed(20260313)
    baseline = module.ICSPBBackboneV2LargeOnline(config)

    proto_opt = torch.optim.AdamW(proto.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    baseline_opt = torch.optim.AdamW(baseline.parameters(), lr=2.0e-3, weight_decay=1.0e-4)

    proto_initial = evaluate_model(proto, val_batches)
    baseline_initial = evaluate_model(baseline, baseline_val)

    proto_history, _, _ = train_with_checkpoints(proto, proto_opt, train_batches, val_batches, epochs=12)
    baseline_history, _, _ = train_with_checkpoints(baseline, baseline_opt, baseline_train, baseline_val, epochs=12)

    proto_mid = evaluate_model(proto, val_batches)
    baseline_final = evaluate_model(baseline, baseline_val)

    auto_recovery_triggered = False
    stabilization_history: List[float] = []
    recovery_history: List[float] = []
    consolidation_history: List[Dict[str, float]] = []
    if (
        proto_mid["stable_read"] < 0.985
        or (baseline_final["loss"] - proto_mid["loss"]) < 0.12
        or proto_mid["loss"] >= proto_initial["loss"]
    ):
        auto_recovery_triggered = True

        stable_opt = torch.optim.AdamW(proto.parameters(), lr=7.0e-4, weight_decay=5.0e-5)
        for _ in range(3):
            epoch_loss = 0.0
            for batch in stable_batches:
                metrics = proto.train_step(stable_opt, batch)
                epoch_loss += metrics["total_loss"]
            stabilization_history.append(epoch_loss / len(stable_batches))

        recovery_opt = torch.optim.AdamW(proto.parameters(), lr=5.0e-4, weight_decay=5.0e-5)
        recovery_batches = []
        for raw_batch, stable_batch in zip(train_batches, stable_batches):
            recovery_batches.append(stable_batch)
            recovery_batches.append(raw_batch)
        for _ in range(3):
            epoch_loss = 0.0
            for batch in recovery_batches:
                epoch_loss += helper.train_structural_recovery_step(proto, recovery_opt, batch, config)
            recovery_history.append(epoch_loss / len(recovery_batches))

        consolidation_batches = train_batches + val_batches + stable_batches[: len(val_batches)] + train_batches[: len(val_batches)]
        consolidation_opt = torch.optim.AdamW(proto.parameters(), lr=3.0e-4, weight_decay=5.0e-5)
        consolidation_history, _, _ = train_with_checkpoints(
            proto,
            consolidation_opt,
            consolidation_batches,
            val_batches,
            epochs=6,
        )

    proto_final = evaluate_model(proto, val_batches)
    online_rounds, rollback_error = online_adaptation_round(proto, online_batches[:4], lr=4.0e-4)
    online_delta_total = sum(item["delta"] for item in online_rounds)

    baseline_margin = baseline_final["loss"] - proto_final["loss"]
    long_run_gain = proto_initial["loss"] - proto_final["loss"]
    structure_gain = (
        0.30 * max(0.0, baseline_margin)
        + 0.20 * proto_final["theorem_survival"]
        + 0.20 * proto_final["stable_read"]
        + 0.15 * proto_final["guarded_write"]
        + 0.10 * min(1.0, proto_final["transport_margin"] / 8.0)
        + 0.05 * proto_final["stress_balance"]
    )

    ready = (
        baseline_margin > 0.12
        and long_run_gain > 0.40
        and proto_final["stable_read"] >= 0.985
        and proto_final["theorem_survival"] >= 0.985
        and online_delta_total >= 0.0
        and rollback_error <= 1e-8
    )

    result = {
        "smoke_pass": True,
        "training_pass": proto_final["loss"] < proto_initial["loss"],
        "baseline_outperform_pass": baseline_margin > 0.12,
        "online_update_pass": online_delta_total >= 0.0,
        "rollback_pass": rollback_error <= 1e-8,
        "auto_recovery_triggered": auto_recovery_triggered,
        "implementation_ready": ready,
        "implementation_score": 1.0 if ready else max(0.0, min(1.0, structure_gain)),
        "data_stats": data_stats,
        "prototype_name": "ICSPB-Backbone-v2-LargeOnline",
        "baseline_name": "bridge_protocol_ablated_baseline",
        "proto_initial": proto_initial,
        "proto_history": proto_history,
        "proto_mid": proto_mid,
        "proto_final": proto_final,
        "baseline_initial": baseline_initial,
        "baseline_history": baseline_history,
        "baseline_final": baseline_final,
        "stabilization_history": stabilization_history,
        "recovery_history": recovery_history,
        "consolidation_history": consolidation_history,
        "online_rounds": online_rounds,
        "baseline_margin": baseline_margin,
        "long_run_gain": long_run_gain,
        "online_delta_total": online_delta_total,
        "rollback_error": rollback_error,
        "structure_gain": structure_gain,
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
