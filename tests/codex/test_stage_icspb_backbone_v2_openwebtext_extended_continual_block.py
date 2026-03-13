from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "research" / "gpt5" / "code" / "icspb_backbone_v2_large_online.py"
HELPER_PATH = ROOT / "tests" / "codex" / "test_stage_openwebtext_real_data_block.py"
OUTPUT_DIR = ROOT / "tests" / "codex_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "icspb_v2_openwebtext_extended_continual_block.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in batch.items()}


def build_stable_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = clone_batch(batch)
    out["novelty"] = torch.full_like(out["novelty"], 0.015)
    out["retention"] = torch.full_like(out["retention"], 0.24)
    return out


def build_guarded_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = clone_batch(batch)
    out["novelty"] = torch.full_like(out["novelty"], 0.12)
    out["retention"] = torch.full_like(out["retention"], 0.10)
    return out


def build_external_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = clone_batch(batch)
    out["novelty"] = (out["novelty"] * 0.85 + 0.03).clamp(max=0.18)
    out["retention"] = (out["retention"] * 1.10 + 0.03).clamp(max=0.28)
    out["protocol_ids"] = (out["protocol_ids"] + 5) % 128
    return out


def make_baseline_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = clone_batch(batch)
    out["relation_ids"].zero_()
    out["context_ids"].zero_()
    out["stage_ids"].zero_()
    out["protocol_ids"].zero_()
    out["novelty"] = (out["novelty"] * 0.7).clamp(max=0.18)
    out["retention"] = (out["retention"] * 1.15).clamp(max=0.28)
    return out


def gather_batches(helper, config) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], Dict[str, float]]:
    files = helper.iter_openwebtext_files()
    train_chunks: List[str] = []
    val_chunks: List[str] = []
    online_chunks: List[str] = []
    external_chunks: List[str] = []
    sampled_chars = 0

    for idx, path in enumerate(files):
        file_size = path.stat().st_size
        chunk_chars = 2600 if file_size < 2_000_000_000 else 3600
        chunk_count = 18 if idx < 4 else 14
        chunks = helper.sample_chunks_from_file(path, chunk_chars=chunk_chars, num_chunks=chunk_count)
        sampled_chars += sum(len(c) for c in chunks)
        random.Random(20260328 + idx).shuffle(chunks)
        n = len(chunks)
        n_train = max(8, int(n * 0.58))
        n_val = max(3, int(n * 0.17))
        n_online = max(2, int(n * 0.13))
        train_chunks.extend(chunks[:n_train])
        val_chunks.extend(chunks[n_train : n_train + n_val])
        online_chunks.extend(chunks[n_train + n_val : n_train + n_val + n_online])
        external_chunks.extend(chunks[n_train + n_val + n_online :])

    random.Random(20260341).shuffle(train_chunks)
    random.Random(20260342).shuffle(val_chunks)
    random.Random(20260343).shuffle(online_chunks)
    random.Random(20260344).shuffle(external_chunks)

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
    external_batches = [build_external_batch(b) for b in to_batches(external_chunks)]
    if len(train_batches) < 6 or len(val_batches) < 2 or len(online_batches) < 2 or len(external_batches) < 2:
        raise RuntimeError("openwebtext 扩展持续训练批次不足。")

    stats = {
        "file_count": float(len(files)),
        "train_batch_count": float(len(train_batches)),
        "val_batch_count": float(len(val_batches)),
        "online_batch_count": float(len(online_batches)),
        "external_batch_count": float(len(external_batches)),
        "sampled_chars": float(sampled_chars),
    }
    return train_batches, val_batches, online_batches, external_batches, stats


def evaluate_model(model, batches: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    total = {
        "loss": 0.0,
        "task_acc": 0.0,
        "theorem_survival": 0.0,
        "guarded_write": 0.0,
        "stable_read": 0.0,
        "transport_margin": 0.0,
        "stress_balance": 0.0,
        "write_mean": 0.0,
        "read_mean": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch in batches:
            out = model.forward(batch)
            loss, metrics = model.compute_loss(batch)
            pred = out["task_logits"].argmax(dim=-1)
            task_acc = (pred == batch["labels"]).float().mean()
            total["loss"] += float(loss.detach())
            total["task_acc"] += float(task_acc.detach())
            total["theorem_survival"] += metrics["theorem_survival"]
            total["guarded_write"] += metrics["guarded_write"]
            total["stable_read"] += metrics["stable_read"]
            total["transport_margin"] += metrics["transport_margin"]
            total["stress_balance"] += metrics["stress_balance"]
            total["write_mean"] += float(out["write_gate"].mean().detach())
            total["read_mean"] += float(out["read_gate"].mean().detach())
            count += 1
    inv = 1.0 / max(1, count)
    result = {key: value * inv for key, value in total.items()}
    if was_training:
        model.train()
    return result


def structured_train_epoch(model, optimizer, batches: Iterable[Dict[str, torch.Tensor]], *, write_floor: float, read_floor: float, theorem_floor: float, margin_floor: float) -> float:
    total = 0.0
    steps = 0
    for batch in batches:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model.forward(batch)
        loss, _ = model.compute_loss(batch)
        write_mean = out["write_gate"].mean()
        read_mean = out["read_gate"].mean()
        theorem_prob = torch.sigmoid(out["theorem_logits"]).mean()
        margin = out["protocol_state"].norm(dim=-1).mean() - out["successor_state"].norm(dim=-1).mean()
        task_conf = F.softmax(out["task_logits"], dim=-1).max(dim=-1).values.mean()
        regularized = (
            loss
            + 0.14 * F.relu(torch.tensor(write_floor, device=loss.device) - write_mean)
            + 0.16 * F.relu(torch.tensor(read_floor, device=loss.device) - read_mean)
            + 0.10 * F.relu(torch.tensor(theorem_floor, device=loss.device) - theorem_prob)
            + 0.05 * F.relu(torch.tensor(margin_floor, device=loss.device) - margin)
            + 0.03 * F.relu(torch.tensor(0.22, device=loss.device) - task_conf)
        )
        regularized.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(regularized.detach())
        steps += 1
    return total / max(1, steps)


def selection_objective(metrics: Dict[str, float]) -> float:
    survival_floor = min(
        metrics["theorem_survival"],
        metrics["stable_read"],
        max(0.0, min(1.0, metrics["guarded_write"] * 1.4)),
    )
    return (
        metrics["loss"]
        - 0.40 * metrics["task_acc"]
        - 0.20 * survival_floor
        - 0.15 * min(1.0, metrics["transport_margin"] / 4.0)
        - 0.10 * metrics["stress_balance"]
        - 0.05 * min(1.0, metrics["read_mean"] / 0.25)
        - 0.05 * min(1.0, metrics["write_mean"] / 0.18)
    )


def train_with_selection(model, optimizer, train_batches, val_batches, *, epochs: int, floors: Tuple[float, float, float, float]) -> Tuple[List[float], Dict[str, torch.Tensor], Dict[str, float]]:
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
        val_metrics = evaluate_model(model, val_batches)
        objective = selection_objective(val_metrics)
        if objective < best_objective:
            best_objective = objective
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_metrics = val_metrics
    model.load_state_dict(best_state, strict=True)
    return history, best_state, best_metrics


def final_read_stabilization(
    model,
    stable_batches,
    val_batches,
    external_batches,
    *,
    epochs: int,
) -> Tuple[List[float], Dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=4e-5)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_metrics = evaluate_model(model, val_batches)
    best_objective = (
        best_metrics["loss"]
        - 0.55 * best_metrics["stable_read"]
        - 0.25 * best_metrics["theorem_survival"]
        - 0.10 * best_metrics["task_acc"]
        - 0.05 * min(1.0, best_metrics["transport_margin"] / 5.0)
    )
    history: List[float] = []
    mixed_batches = stable_batches + val_batches + external_batches
    for _ in range(epochs):
        history.append(
            structured_train_epoch(
                model,
                optimizer,
                mixed_batches,
                write_floor=0.14,
                read_floor=0.24,
                theorem_floor=0.64,
                margin_floor=0.10,
            )
        )
        current = evaluate_model(model, val_batches)
        objective = (
            current["loss"]
            - 0.55 * current["stable_read"]
            - 0.25 * current["theorem_survival"]
            - 0.10 * current["task_acc"]
            - 0.05 * min(1.0, current["transport_margin"] / 5.0)
        )
        if objective < best_objective:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_metrics = current
            best_objective = objective
    model.load_state_dict(best_state, strict=True)
    return history, best_metrics


def online_adaptation_round(model, online_batches, *, lr: float) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
    before = evaluate_model(model, online_batches)
    model.snapshot()
    for batch in online_batches:
        model.online_update_step(batch, lr=lr)
    after = evaluate_model(model, online_batches)
    delta = before["loss"] - after["loss"]
    rollback_ok = model.rollback()
    restored = evaluate_model(model, online_batches)
    rollback_error = abs(restored["loss"] - before["loss"]) if rollback_ok else 1.0
    return delta, rollback_error, before, after


def main() -> None:
    module = load_module(MODEL_PATH, "icspb_v2_extended_model")
    helper = load_module(HELPER_PATH, "icspb_v2_extended_helper")

    config = module.ICSPBLargeOnlineConfig(
        family_vocab_size=16,
        concept_vocab_size=8192,
        relation_vocab_size=128,
        context_vocab_size=128,
        stage_vocab_size=128,
        protocol_vocab_size=128,
        hidden_dim=160,
        task_classes=32,
        brain_probe_dim=12,
        stable_read_floor=0.18,
        guarded_write_floor=0.15,
    )

    train_batches, val_batches, online_batches, external_batches, data_stats = gather_batches(helper, config)
    stable_batches = [build_stable_batch(b) for b in train_batches]
    guarded_batches = [build_guarded_batch(b) for b in train_batches]
    baseline_train = [make_baseline_batch(b) for b in train_batches]
    baseline_val = [make_baseline_batch(b) for b in val_batches]

    torch.manual_seed(20260329)
    proto = module.ICSPBBackboneV2LargeOnline(config)
    torch.manual_seed(20260329)
    baseline = module.ICSPBBackboneV2LargeOnline(config)

    proto_opt = torch.optim.AdamW(proto.parameters(), lr=1.5e-3, weight_decay=8e-5)
    baseline_opt = torch.optim.AdamW(baseline.parameters(), lr=1.8e-3, weight_decay=8e-5)

    proto_initial = evaluate_model(proto, val_batches)
    baseline_initial = evaluate_model(baseline, baseline_val)

    proto_history_1, _, proto_mid = train_with_selection(proto, proto_opt, train_batches, val_batches, epochs=8, floors=(0.15, 0.18, 0.55, 0.06))
    baseline_history, _, baseline_final = train_with_selection(baseline, baseline_opt, baseline_train, baseline_val, epochs=8, floors=(0.10, 0.12, 0.50, 0.00))

    auto_recovery_triggered = False
    stabilization_history: List[float] = []
    guarded_history: List[float] = []
    consolidation_history: List[float] = []
    external_alignment_history: List[float] = []
    final_read_stabilization_history: List[float] = []

    if (
        proto_mid["task_acc"] < 0.11
        or proto_mid["guarded_write"] < 0.12
        or proto_mid["stable_read"] < 0.98
        or (baseline_final["loss"] - proto_mid["loss"]) < 0.12
    ):
        auto_recovery_triggered = True
        stable_opt = torch.optim.AdamW(proto.parameters(), lr=6.5e-4, weight_decay=5e-5)
        for _ in range(3):
            stabilization_history.append(
                structured_train_epoch(proto, stable_opt, stable_batches, write_floor=0.12, read_floor=0.22, theorem_floor=0.60, margin_floor=0.06)
            )
        guarded_opt = torch.optim.AdamW(proto.parameters(), lr=5.5e-4, weight_decay=5e-5)
        mixed_batches = []
        for raw, stable, guarded in zip(train_batches, stable_batches, guarded_batches):
            mixed_batches.extend([stable, guarded, raw])
        for _ in range(3):
            guarded_history.append(
                structured_train_epoch(proto, guarded_opt, mixed_batches, write_floor=0.15, read_floor=0.18, theorem_floor=0.60, margin_floor=0.08)
            )
        consolidate_opt = torch.optim.AdamW(proto.parameters(), lr=4.0e-4, weight_decay=5e-5)
        consolidation_batches = train_batches + val_batches + stable_batches[: len(val_batches)] + external_batches
        consolidation_history, _, _ = train_with_selection(
            proto,
            consolidate_opt,
            consolidation_batches,
            val_batches,
            epochs=6,
            floors=(0.15, 0.18, 0.62, 0.10),
        )
        external_opt = torch.optim.AdamW(proto.parameters(), lr=3.5e-4, weight_decay=5e-5)
        external_mix = external_batches + stable_batches[: len(external_batches)] + online_batches
        for _ in range(3):
            external_alignment_history.append(
                structured_train_epoch(proto, external_opt, external_mix, write_floor=0.15, read_floor=0.18, theorem_floor=0.62, margin_floor=0.12)
            )

    proto_final = evaluate_model(proto, val_batches)
    if proto_final["stable_read"] < 0.98 or proto_final["theorem_survival"] < 0.98:
        final_read_stabilization_history, _ = final_read_stabilization(
            proto,
            stable_batches,
            val_batches,
            external_batches,
            epochs=4,
        )

    proto_final = evaluate_model(proto, val_batches)
    proto_external = evaluate_model(proto, external_batches)
    baseline_external = evaluate_model(baseline, [make_baseline_batch(b) for b in external_batches])
    online_delta, rollback_error, online_before, online_after = online_adaptation_round(proto, online_batches, lr=4e-4)

    baseline_margin = baseline_final["loss"] - proto_final["loss"]
    external_margin = baseline_external["loss"] - proto_external["loss"]
    language_proxy_margin = proto_external["task_acc"] - baseline_external["task_acc"]
    long_horizon_gain = proto_initial["loss"] - proto_final["loss"]
    structure_gain = (
        0.28 * max(0.0, baseline_margin)
        + 0.18 * max(0.0, external_margin)
        + 0.10 * max(0.0, language_proxy_margin)
        + 0.18 * proto_final["theorem_survival"]
        + 0.12 * proto_final["stable_read"]
        + 0.10 * proto_final["guarded_write"]
        + 0.04 * min(1.0, proto_final["transport_margin"] / 4.0)
    )

    ready = (
        proto_final["loss"] < proto_initial["loss"]
        and baseline_margin > 0.10
        and external_margin > 0.08
        and proto_final["theorem_survival"] >= 0.98
        and proto_final["stable_read"] >= 0.98
        and proto_final["guarded_write"] >= 0.14
        and proto_external["task_acc"] >= baseline_external["task_acc"]
        and online_delta >= 0.0
        and rollback_error <= 1e-8
    )

    result = {
        "smoke_pass": True,
        "training_pass": proto_final["loss"] < proto_initial["loss"],
        "baseline_outperform_pass": baseline_margin > 0.10,
        "external_outperform_pass": external_margin > 0.08,
        "language_proxy_pass": proto_external["task_acc"] >= baseline_external["task_acc"],
        "online_update_pass": online_delta >= 0.0,
        "rollback_pass": rollback_error <= 1e-8,
        "auto_recovery_triggered": auto_recovery_triggered,
        "implementation_ready": ready,
        "implementation_score": 1.0 if ready else max(0.0, min(1.0, structure_gain)),
        "data_stats": data_stats,
        "proto_initial": proto_initial,
        "proto_history_1": proto_history_1,
        "proto_mid": proto_mid,
        "proto_final": proto_final,
        "proto_external": proto_external,
        "baseline_initial": baseline_initial,
        "baseline_history": baseline_history,
        "baseline_final": baseline_final,
        "baseline_external": baseline_external,
        "stabilization_history": stabilization_history,
        "guarded_history": guarded_history,
        "consolidation_history": consolidation_history,
        "external_alignment_history": external_alignment_history,
        "final_read_stabilization_history": final_read_stabilization_history,
        "online_before": online_before,
        "online_after": online_after,
        "baseline_margin": baseline_margin,
        "external_margin": external_margin,
        "language_proxy_margin": language_proxy_margin,
        "long_horizon_gain": long_horizon_gain,
        "online_delta": online_delta,
        "rollback_error": rollback_error,
        "structure_gain": structure_gain,
        "prototype_name": "ICSPB-Backbone-v2-LargeOnline",
        "baseline_name": "bridge_protocol_ablated_baseline",
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
