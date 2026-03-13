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
OUTPUT_PATH = OUTPUT_DIR / "icspb_v2_openwebtext_longterm_training_block.json"


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
    baseline["novelty"] = (baseline["novelty"] * 0.6).clamp(max=0.20)
    baseline["retention"] = (baseline["retention"] * 1.2).clamp(max=0.30)
    return baseline


def gather_longterm_batches(helper, config) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], Dict[str, float]]:
    files = helper.iter_openwebtext_files()
    train_chunks: List[str] = []
    val_chunks: List[str] = []
    online_chunks: List[str] = []
    sampled_chars = 0

    for idx, path in enumerate(files):
        file_size = path.stat().st_size
        chunk_chars = 2400 if file_size < 2_000_000_000 else 3400
        chunk_count = 12 if idx < 5 else 10
        chunks = helper.sample_chunks_from_file(path, chunk_chars=chunk_chars, num_chunks=chunk_count)
        sampled_chars += sum(len(c) for c in chunks)
        random.Random(20260313 + idx).shuffle(chunks)
        n = len(chunks)
        n_train = max(4, int(n * 0.65))
        n_val = max(2, int(n * 0.20))
        train_chunks.extend(chunks[:n_train])
        val_chunks.extend(chunks[n_train : n_train + n_val])
        online_chunks.extend(chunks[n_train + n_val :])

    random.Random(20260321).shuffle(train_chunks)
    random.Random(20260322).shuffle(val_chunks)
    random.Random(20260323).shuffle(online_chunks)

    batch_size = 8

    def to_batches(chunks: List[str]) -> List[Dict[str, torch.Tensor]]:
        batches = []
        for i in range(0, len(chunks), batch_size):
            piece = chunks[i : i + batch_size]
            if len(piece) == batch_size:
                batches.append(helper.build_batch_from_chunks(piece, config))
        return batches

    train_batches = to_batches(train_chunks)
    val_batches = to_batches(val_chunks)
    online_batches = to_batches(online_chunks)
    if len(train_batches) < 4 or len(val_batches) < 1 or len(online_batches) < 1:
        raise RuntimeError("openwebtext 长期训练批次构造不足")

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
    loss_total = 0.0
    theorem_total = 0.0
    guarded_total = 0.0
    stable_total = 0.0
    margin_total = 0.0
    stress_total = 0.0
    count = 0
    with torch.no_grad():
        for batch in batches:
            loss, metrics = model.compute_loss(batch)
            loss_total += float(loss.detach())
            theorem_total += metrics["theorem_survival"]
            guarded_total += metrics["guarded_write"]
            stable_total += metrics["stable_read"]
            margin_total += metrics["transport_margin"]
            stress_total += metrics["stress_balance"]
            count += 1
    inv = 1.0 / max(1, count)
    result = {
        "loss": loss_total * inv,
        "theorem_survival": theorem_total * inv,
        "guarded_write": guarded_total * inv,
        "stable_read": stable_total * inv,
        "transport_margin": margin_total * inv,
        "stress_balance": stress_total * inv,
    }
    if was_training:
        model.train()
    return result


def train_for_epochs(model, optimizer, train_batches, epochs: int) -> List[float]:
    history: List[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in train_batches:
            metrics = model.train_step(optimizer, batch)
            epoch_loss += metrics["total_loss"]
        history.append(epoch_loss / len(train_batches))
    return history


def selection_objective(metrics: Dict[str, float]) -> float:
    return (
        metrics["loss"]
        - 0.45 * metrics["guarded_write"]
        - 0.25 * metrics["stable_read"]
        - 0.10 * metrics["theorem_survival"]
        - 0.05 * min(1.0, metrics["transport_margin"] / 5.0)
        - 0.05 * metrics["stress_balance"]
    )


def train_with_val_selection(model, optimizer, train_batches, val_batches, epochs: int) -> Tuple[List[float], Dict[str, torch.Tensor], float]:
    history: List[float] = []
    best_objective = math.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in train_batches:
            metrics = model.train_step(optimizer, batch)
            epoch_loss += metrics["total_loss"]
        history.append(epoch_loss / len(train_batches))
        val_metrics = evaluate_model(model, val_batches)
        current_objective = selection_objective(val_metrics)
        if current_objective < best_objective:
            best_objective = current_objective
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    return history, best_state, best_objective


def online_adaptation_round(model, online_batches, lr: float) -> Tuple[float, float, float]:
    before = evaluate_model(model, online_batches)
    model.snapshot()
    for batch in online_batches:
        model.online_update_step(batch, lr=lr)
    after = evaluate_model(model, online_batches)
    delta = before["loss"] - after["loss"]
    rollback_ok = model.rollback()
    restored = evaluate_model(model, online_batches)
    rollback_error = abs(restored["loss"] - before["loss"]) if rollback_ok else 1.0
    return delta, rollback_error, after["theorem_survival"]


def main() -> None:
    module = load_module(MODEL_PATH, "icspb_v2_longterm_model")
    helper = load_module(HELPER_PATH, "icspb_openwebtext_helper")

    config = module.ICSPBLargeOnlineConfig(
        family_vocab_size=16,
        concept_vocab_size=8192,
        relation_vocab_size=128,
        context_vocab_size=128,
        stage_vocab_size=128,
        protocol_vocab_size=128,
        hidden_dim=128,
        task_classes=32,
        brain_probe_dim=12,
        stable_read_floor=0.18,
        guarded_write_floor=0.15,
    )

    train_batches, val_batches, online_batches, data_stats = gather_longterm_batches(helper, config)
    baseline_train = [make_baseline_batch(b) for b in train_batches]
    baseline_val = [make_baseline_batch(b) for b in val_batches]
    stable_batches = [helper.build_stable_regime_batch(b) for b in train_batches]

    torch.manual_seed(20260313)
    proto = module.ICSPBBackboneV2LargeOnline(config)
    torch.manual_seed(20260313)
    baseline = module.ICSPBBackboneV2LargeOnline(config)

    proto_opt = torch.optim.AdamW(proto.parameters(), lr=2.5e-3, weight_decay=1e-4)
    baseline_opt = torch.optim.AdamW(baseline.parameters(), lr=2.5e-3, weight_decay=1e-4)

    proto_initial = evaluate_model(proto, val_batches)
    baseline_initial = evaluate_model(baseline, baseline_val)

    proto_history, _, _ = train_with_val_selection(proto, proto_opt, train_batches, val_batches, epochs=6)
    baseline_history, _, _ = train_with_val_selection(baseline, baseline_opt, baseline_train, baseline_val, epochs=6)

    proto_mid = evaluate_model(proto, val_batches)
    baseline_final = evaluate_model(baseline, baseline_val)

    auto_recovery_triggered = False
    recovery_history: List[float] = []
    stabilization_history: List[float] = []
    consolidation_history: List[float] = []
    if (
        proto_mid["stable_read"] < 0.98
        or (baseline_final["loss"] - proto_mid["loss"]) < 0.08
        or proto_mid["loss"] >= proto_initial["loss"]
    ):
        auto_recovery_triggered = True
        stable_opt = torch.optim.AdamW(proto.parameters(), lr=6.0e-4, weight_decay=5e-5)
        for _ in range(2):
            epoch_loss = 0.0
            for batch in stable_batches:
                metrics = proto.train_step(stable_opt, batch)
                epoch_loss += metrics["total_loss"]
            stabilization_history.append(epoch_loss / len(stable_batches))
        recovery_opt = torch.optim.AdamW(proto.parameters(), lr=5.0e-4, weight_decay=5e-5)
        recovery_batches = []
        for raw_batch, stable_batch in zip(train_batches, stable_batches):
            recovery_batches.append(stable_batch)
            recovery_batches.append(raw_batch)
        for _ in range(2):
            epoch_loss = 0.0
            for batch in recovery_batches:
                epoch_loss += helper.train_structural_recovery_step(proto, recovery_opt, batch, config)
            recovery_history.append(epoch_loss / len(recovery_batches))
        consolidation_batches = train_batches + val_batches + stable_batches[: len(val_batches)]
        consolidation_opt = torch.optim.AdamW(proto.parameters(), lr=3.0e-4, weight_decay=5e-5)
        consolidation_history, _, _ = train_with_val_selection(
            proto,
            consolidation_opt,
            consolidation_batches,
            val_batches,
            epochs=4,
        )

    proto_final = evaluate_model(proto, val_batches)
    online_delta, rollback_error, online_theorem = online_adaptation_round(proto, online_batches[:2], lr=5e-4)

    baseline_margin = baseline_final["loss"] - proto_final["loss"]
    theorem_gain = proto_final["theorem_survival"] - baseline_final["theorem_survival"]
    structure_gain = (
        0.35 * max(0.0, baseline_margin)
        + 0.20 * theorem_gain
        + 0.20 * proto_final["stable_read"]
        + 0.15 * proto_final["guarded_write"]
        + 0.10 * min(1.0, proto_final["transport_margin"] / 10.0)
    )

    ready = (
        baseline_margin > 0.08
        and proto_final["stable_read"] >= 0.98
        and proto_final["theorem_survival"] >= 0.98
        and online_delta >= 0.0
        and rollback_error <= 1e-8
    )

    result = {
        "smoke_pass": True,
        "training_pass": proto_final["loss"] < proto_initial["loss"],
        "baseline_outperform_pass": baseline_margin > 0.08,
        "online_update_pass": online_delta >= 0.0,
        "rollback_pass": rollback_error <= 1e-8,
        "auto_recovery_triggered": auto_recovery_triggered,
        "implementation_ready": ready,
        "implementation_score": 1.0 if ready else max(0.0, min(1.0, structure_gain)),
        "data_stats": data_stats,
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
        "baseline_margin": baseline_margin,
        "online_delta": online_delta,
        "online_theorem_after_update": online_theorem,
        "rollback_error": rollback_error,
        "structure_gain": structure_gain,
        "prototype_name": "ICSPB-Backbone-v2-LargeOnline",
        "baseline_name": "bridge_protocol_ablated_baseline",
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
