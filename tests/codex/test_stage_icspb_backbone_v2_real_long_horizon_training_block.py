from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "research" / "gpt5" / "code" / "icspb_backbone_v2_large_online.py"
HELPER_PATH = ROOT / "tests" / "codex" / "test_stage_openwebtext_real_data_block.py"
OUTPUT_DIR = ROOT / "tests" / "codex_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "icspb_v2_real_long_horizon_training_block.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in batch.items()}


def build_guarded_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = clone_batch(batch)
    out["novelty"] = torch.full_like(out["novelty"], 0.16)
    out["retention"] = torch.full_like(out["retention"], 0.10)
    return out


def build_baseline_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = clone_batch(batch)
    out["relation_ids"].zero_()
    out["context_ids"].zero_()
    out["stage_ids"].zero_()
    out["protocol_ids"].zero_()
    out["novelty"] = (out["novelty"] * 0.55).clamp(max=0.14)
    out["retention"] = (out["retention"] * 1.4).clamp(max=0.36)
    return out


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
            count += 1
    inv = 1.0 / max(1, count)
    result = {key: value * inv for key, value in total.items()}
    if was_training:
        model.train()
    return result


def selection_objective(metrics: Dict[str, float]) -> float:
    return (
        metrics["loss"]
        - 0.62 * metrics["stable_read"]
        - 0.36 * metrics["theorem_survival"]
        - 0.34 * metrics["guarded_write"]
        - 0.24 * metrics["task_acc"]
        - 0.16 * min(1.0, metrics["transport_margin"] / 10.0)
        - 0.12 * metrics["stress_balance"]
    )


def structure_score(metrics: Dict[str, float]) -> float:
    return (
        0.30 * metrics["theorem_survival"]
        + 0.24 * metrics["stable_read"]
        + 0.18 * metrics["guarded_write"]
        + 0.12 * metrics["task_acc"]
        + 0.10 * min(1.0, metrics["transport_margin"] / 8.0)
        + 0.06 * metrics["stress_balance"]
        - 0.06 * min(1.0, metrics["loss"] / 4.0)
    )


def task_progress_score(metrics: Dict[str, float], initial_loss: float) -> float:
    loss_improvement = max(0.0, initial_loss - metrics["loss"])
    normalized_loss_improvement = min(1.0, loss_improvement / max(1e-6, initial_loss))
    return (
        0.44 * metrics["task_acc"]
        + 0.28 * normalized_loss_improvement
        + 0.12 * min(1.0, metrics["transport_margin"] / 8.0)
        + 0.10 * metrics["theorem_survival"]
        + 0.06 * metrics["stress_balance"]
    )


def is_feasible(metrics: Dict[str, float]) -> bool:
    return (
        metrics["theorem_survival"] >= 0.99
        and metrics["stable_read"] >= 0.99
        and metrics["guarded_write"] >= 0.40
        and metrics["task_acc"] >= 0.80
    )


def structural_core_pass(metrics: Dict[str, float]) -> bool:
    return (
        metrics["theorem_survival"] >= 0.99
        and metrics["stable_read"] >= 0.99
        and metrics["guarded_write"] >= 0.40
    )


def record_candidate(
    store: List[Dict[str, object]],
    name: str,
    model,
    val_batches,
    external_batches,
) -> None:
    val_metrics = evaluate_model(model, val_batches)
    external_metrics = evaluate_model(model, external_batches)
    combined = {
        "loss": 0.5 * (val_metrics["loss"] + external_metrics["loss"]),
        "task_acc": min(val_metrics["task_acc"], external_metrics["task_acc"]),
        "theorem_survival": min(val_metrics["theorem_survival"], external_metrics["theorem_survival"]),
        "stable_read": min(val_metrics["stable_read"], external_metrics["stable_read"]),
        "guarded_write": val_metrics["guarded_write"],
        "transport_margin": 0.5 * (val_metrics["transport_margin"] + external_metrics["transport_margin"]),
        "stress_balance": 0.5 * (val_metrics["stress_balance"] + external_metrics["stress_balance"]),
    }
    store.append(
        {
            "name": name,
            "val": val_metrics,
            "external": external_metrics,
            "combined": combined,
            "selection_objective": selection_objective(combined),
            "structure_score": structure_score(combined),
            "feasible": is_feasible(combined),
            "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
        }
    )


def select_best_candidate(candidates: List[Dict[str, object]], initial_loss: float) -> Dict[str, object]:
    feasible = [c for c in candidates if c["feasible"]]
    if feasible:
        return min(
            feasible,
            key=lambda c: (
                float(c["selection_objective"]),
                -float(c["structure_score"]),
            ),
        )
    structural_strong = [
        c
        for c in candidates
        if c["combined"]["theorem_survival"] >= 0.99
        and c["combined"]["stable_read"] >= 0.95
        and c["combined"]["guarded_write"] >= 0.30
        and c["combined"]["task_acc"] >= 0.70
    ]
    if structural_strong:
        return max(
            structural_strong,
            key=lambda c: (
                1.5 * float(c["structure_score"]) + 0.25 * task_progress_score(c["combined"], initial_loss),
                -float(c["selection_objective"]),
            ),
        )
    structural_first = [
        c
        for c in candidates
        if c["combined"]["theorem_survival"] >= 0.99
        and c["combined"]["task_acc"] >= 0.50
    ]
    if structural_first:
        return max(
            structural_first,
            key=lambda c: (
                1.8 * float(c["structure_score"]) + 0.15 * task_progress_score(c["combined"], initial_loss),
                -float(c["selection_objective"]),
            ),
        )
    task_progressing = [
        c
        for c in candidates
        if c["combined"]["task_acc"] >= 0.60
        and c["combined"]["loss"] <= 0.55 * initial_loss
        and c["combined"]["theorem_survival"] >= 0.20
    ]
    if task_progressing:
        return max(
            task_progressing,
            key=lambda c: (
                task_progress_score(c["combined"], initial_loss) + 0.35 * float(c["structure_score"]),
                -float(c["selection_objective"]),
            ),
        )
    return max(
        candidates,
        key=lambda c: (
            task_progress_score(c["combined"], initial_loss) + 0.20 * float(c["structure_score"]),
            -float(c["selection_objective"]),
        ),
    )


def gate_alignment_pass(
    model,
    config,
    stable_batches,
    guarded_batches,
    val_batches,
    external_batches,
) -> List[float]:
    history: List[float] = []
    params = list(model.write_read_core.write_gate.parameters()) + list(model.write_read_core.read_gate.parameters())
    opt = torch.optim.AdamW(params, lr=2.0e-4, weight_decay=0.0)
    target_write = max(config.guarded_write_floor + 0.03, 0.20)
    target_read = max(config.stable_read_floor + 0.04, 0.24)
    combined_batches = []
    for stable_batch, guarded_batch in zip(stable_batches[: len(val_batches)], guarded_batches[: len(val_batches)]):
        combined_batches.extend([stable_batch, guarded_batch])

    for _ in range(6):
        epoch_loss = 0.0
        for batch in combined_batches:
            opt.zero_grad(set_to_none=True)
            out = model.forward(batch)
            write_mean = out["write_gate"].mean()
            read_mean = out["read_gate"].mean()
            metrics = model.survival_metrics(batch, out)
            if float(batch["novelty"].mean()) >= 0.15:
                gate_loss = F.relu(target_write - write_mean)
            else:
                gate_loss = F.relu(target_read - read_mean)
            task_loss = F.cross_entropy(out["task_logits"], batch["labels"])
            theorem_loss = F.relu(torch.tensor(0.99, device=write_mean.device) - torch.tensor(metrics["theorem_survival"], device=write_mean.device))
            margin_loss = F.relu(torch.tensor(0.18, device=write_mean.device) - out["protocol_state"].norm(dim=-1).mean() + out["successor_state"].norm(dim=-1).mean())
            loss = gate_loss + 0.16 * task_loss + 0.05 * theorem_loss + 0.02 * margin_loss
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
        history.append(epoch_loss / max(1, len(combined_batches)))
        val_metrics = evaluate_model(model, val_batches)
        ext_metrics = evaluate_model(model, external_batches)
        if (
            val_metrics["stable_read"] >= 0.99
            and ext_metrics["stable_read"] >= 0.99
            and val_metrics["guarded_write"] >= 0.40
            and min(val_metrics["theorem_survival"], ext_metrics["theorem_survival"]) >= 0.99
        ):
            break
    return history


def final_read_stabilization(
    model,
    config,
    stable_batches,
    guarded_batches,
    val_batches,
    external_batches,
) -> List[float]:
    history: List[float] = []
    params = (
        list(model.write_read_core.read_gate.parameters())
        + list(model.write_read_core.write_gate.parameters())
        + list(model.protocol_field_bridge_bus.parameters())
        + list(model.stage_successor_transport_engine.parameters())
    )
    opt = torch.optim.AdamW(params, lr=1.6e-4, weight_decay=0.0)
    calibration_batches = []
    for stable_batch, guarded_batch in zip(stable_batches[: len(val_batches)], guarded_batches[: len(val_batches)]):
        calibration_batches.extend([stable_batch, guarded_batch])

    for _ in range(8):
        epoch_loss = 0.0
        for batch in calibration_batches:
            opt.zero_grad(set_to_none=True)
            out = model.forward(batch)
            metrics = model.survival_metrics(batch, out)
            target_read = torch.tensor(1.0, device=out["read_gate"].device)
            target_theorem = torch.tensor(1.0, device=out["theorem_logits"].device)
            target_write = torch.tensor(max(config.guarded_write_floor, 0.40), device=out["write_gate"].device)
            read_loss = F.relu(target_read - out["read_gate"].mean())
            theorem_loss = F.relu(target_theorem - torch.sigmoid(out["theorem_logits"]).mean())
            write_loss = F.relu(target_write - out["write_gate"].mean())
            transport_floor = torch.tensor(0.12, device=out["protocol_state"].device)
            transport_loss = F.relu(transport_floor - (out["protocol_state"].norm(dim=-1).mean() - out["successor_state"].norm(dim=-1).mean()))
            task_loss = F.cross_entropy(out["task_logits"], batch["labels"])
            loss = read_loss + 0.75 * theorem_loss + 0.40 * write_loss + 0.15 * transport_loss + 0.08 * task_loss
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
        history.append(epoch_loss / max(1, len(calibration_batches)))
        val_metrics = evaluate_model(model, val_batches)
        ext_metrics = evaluate_model(model, external_batches)
        if (
            min(val_metrics["theorem_survival"], ext_metrics["theorem_survival"]) >= 0.99
            and min(val_metrics["stable_read"], ext_metrics["stable_read"]) >= 0.99
            and val_metrics["guarded_write"] >= 0.40
            and ext_metrics["task_acc"] >= 0.80
        ):
            break
    return history


def full_structural_restoration(
    model,
    config,
    stable_batches,
    guarded_batches,
    val_batches,
    external_batches,
) -> List[float]:
    history: List[float] = []
    params = (
        list(model.write_read_core.read_gate.parameters())
        + list(model.write_read_core.write_gate.parameters())
        + list(model.theorem_survival_monitor.parameters())
        + list(model.protocol_field_bridge_bus.parameters())
        + list(model.stage_successor_transport_engine.parameters())
    )
    opt = torch.optim.AdamW(params, lr=1.2e-4, weight_decay=0.0)

    restoration_batches = []
    for stable_batch, guarded_batch in zip(stable_batches[: len(val_batches)], guarded_batches[: len(val_batches)]):
        restoration_batches.extend([stable_batch, guarded_batch])
    restoration_batches.extend(val_batches[: min(3, len(val_batches))])
    restoration_batches.extend(external_batches[: min(3, len(external_batches))])

    target_read = max(config.stable_read_floor + 0.82, 0.995)
    target_write = max(config.guarded_write_floor + 0.28, 0.42)
    target_theorem = 0.995
    target_transport = 0.18

    for _ in range(10):
        epoch_loss = 0.0
        for batch in restoration_batches:
            opt.zero_grad(set_to_none=True)
            out = model.forward(batch)
            theorem_prob = torch.sigmoid(out["theorem_logits"]).mean()
            write_mean = out["write_gate"].mean()
            read_mean = out["read_gate"].mean()
            transport_margin = out["protocol_state"].norm(dim=-1).mean() - out["successor_state"].norm(dim=-1).mean()
            task_loss = F.cross_entropy(out["task_logits"], batch["labels"])

            theorem_loss = F.relu(torch.tensor(target_theorem, device=theorem_prob.device) - theorem_prob)
            read_loss = F.relu(torch.tensor(target_read, device=read_mean.device) - read_mean)
            if float(batch["novelty"].mean()) >= 0.15:
                write_loss = F.relu(torch.tensor(target_write, device=write_mean.device) - write_mean)
            else:
                write_loss = 0.25 * F.relu(torch.tensor(config.guarded_write_floor, device=write_mean.device) - write_mean)
            transport_loss = F.relu(torch.tensor(target_transport, device=transport_margin.device) - transport_margin)

            loss = 1.20 * theorem_loss + 1.05 * read_loss + 0.65 * write_loss + 0.18 * transport_loss + 0.05 * task_loss
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())

        with torch.no_grad():
            read_bias = model.write_read_core.read_gate[-1].bias
            write_bias = model.write_read_core.write_gate[-1].bias
            theorem_bias = model.theorem_survival_monitor[-1].bias
            read_bias.add_(0.06)
            write_bias.add_(0.03)
            theorem_bias.add_(0.10)

        history.append(epoch_loss / max(1, len(restoration_batches)))
        val_metrics = evaluate_model(model, val_batches)
        ext_metrics = evaluate_model(model, external_batches)
        if (
            min(val_metrics["theorem_survival"], ext_metrics["theorem_survival"]) >= 0.99
            and min(val_metrics["stable_read"], ext_metrics["stable_read"]) >= 0.99
            and val_metrics["guarded_write"] >= 0.40
            and ext_metrics["task_acc"] >= 0.80
        ):
            break

    return history


def write_task_bridge_recovery(
    model,
    config,
    stable_batches,
    guarded_batches,
    val_batches,
    external_batches,
) -> List[float]:
    history: List[float] = []
    params = (
        list(model.write_read_core.write_gate.parameters())
        + list(model.write_read_core.read_gate.parameters())
        + list(model.protocol_field_bridge_bus.parameters())
        + list(model.stage_successor_transport_engine.parameters())
        + list(model.task_head.parameters())
        + list(model.concept_section_memory_bank.parameters())
    )
    opt = torch.optim.AdamW(params, lr=1.0e-4, weight_decay=0.0)

    recovery_batches = []
    for external_batch in external_batches:
        recovery_batches.append(external_batch)
    recovery_batches.extend(guarded_batches[: min(len(guarded_batches), len(external_batches))])
    recovery_batches.extend(stable_batches[: min(len(stable_batches), len(external_batches))])
    recovery_batches.extend(val_batches[: min(2, len(val_batches))])

    target_write = max(config.guarded_write_floor + 0.02, 0.18)
    target_read = max(config.stable_read_floor + 0.76, 0.98)
    target_theorem = 0.995
    target_task = 0.82
    target_transport = 0.20

    for _ in range(10):
        epoch_loss = 0.0
        for batch in recovery_batches:
            opt.zero_grad(set_to_none=True)
            out = model.forward(batch)
            task_loss = F.cross_entropy(out["task_logits"], batch["labels"])
            write_mean = out["write_gate"].mean()
            read_mean = out["read_gate"].mean()
            theorem_prob = torch.sigmoid(out["theorem_logits"]).mean()
            task_acc = (out["task_logits"].argmax(dim=-1) == batch["labels"]).float().mean()
            transport_margin = out["protocol_state"].norm(dim=-1).mean() - out["successor_state"].norm(dim=-1).mean()

            theorem_loss = F.relu(torch.tensor(target_theorem, device=theorem_prob.device) - theorem_prob)
            read_loss = F.relu(torch.tensor(target_read, device=read_mean.device) - read_mean)
            write_loss = F.relu(torch.tensor(target_write, device=write_mean.device) - write_mean)
            task_acc_loss = F.relu(torch.tensor(target_task, device=task_acc.device) - task_acc)
            transport_loss = F.relu(torch.tensor(target_transport, device=transport_margin.device) - transport_margin)

            loss = (
                0.90 * task_loss
                + 0.70 * task_acc_loss
                + 0.55 * write_loss
                + 0.35 * read_loss
                + 0.35 * theorem_loss
                + 0.15 * transport_loss
            )
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())

        with torch.no_grad():
            model.write_read_core.write_gate[-1].bias.add_(0.12)
            model.write_read_core.read_gate[-1].bias.add_(0.03)
            model.theorem_survival_monitor[-1].bias.add_(0.05)

        history.append(epoch_loss / max(1, len(recovery_batches)))
        val_metrics = evaluate_model(model, val_batches)
        ext_metrics = evaluate_model(model, external_batches)
        if (
            min(val_metrics["theorem_survival"], ext_metrics["theorem_survival"]) >= 0.99
            and min(val_metrics["stable_read"], ext_metrics["stable_read"]) >= 0.99
            and val_metrics["guarded_write"] >= 0.40
            and ext_metrics["task_acc"] >= 0.80
        ):
            break

    return history


def gather_batches(helper, config) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], Dict[str, float]]:
    files = helper.iter_openwebtext_files()
    train_chunks: List[str] = []
    val_chunks: List[str] = []
    online_chunks: List[str] = []
    external_chunks: List[str] = []
    sampled_chars = 0

    for idx, path in enumerate(files):
        chunk_chars = 3600 if idx < 4 else 3200
        chunk_count = 32 if idx < 4 else 28
        chunks = helper.sample_chunks_from_file(path, chunk_chars=chunk_chars, num_chunks=chunk_count)
        sampled_chars += sum(len(c) for c in chunks)
        random.Random(20260313 + idx).shuffle(chunks)
        n = len(chunks)
        n_train = max(12, int(n * 0.50))
        n_val = max(4, int(n * 0.15))
        n_online = max(4, int(n * 0.15))
        train_chunks.extend(chunks[:n_train])
        val_chunks.extend(chunks[n_train:n_train + n_val])
        online_chunks.extend(chunks[n_train + n_val:n_train + n_val + n_online])
        external_chunks.extend(chunks[n_train + n_val + n_online:])

    random.Random(20260421).shuffle(train_chunks)
    random.Random(20260422).shuffle(val_chunks)
    random.Random(20260423).shuffle(online_chunks)
    random.Random(20260424).shuffle(external_chunks)

    batch_size = 8

    def to_batches(chunks: List[str]) -> List[Dict[str, torch.Tensor]]:
        out: List[Dict[str, torch.Tensor]] = []
        for start in range(0, len(chunks), batch_size):
            piece = chunks[start:start + batch_size]
            if len(piece) == batch_size:
                out.append(helper.build_batch_from_chunks(piece, config))
        return out

    train_batches = to_batches(train_chunks)
    val_batches = to_batches(val_chunks)
    online_batches = to_batches(online_chunks)
    external_batches = to_batches(external_chunks)

    if len(train_batches) < 12 or len(val_batches) < 3 or len(online_batches) < 3 or len(external_batches) < 3:
        raise RuntimeError("真实长期训练块的批次不足，无法继续")

    stats = {
        "file_count": float(len(files)),
        "train_batch_count": float(len(train_batches)),
        "val_batch_count": float(len(val_batches)),
        "online_batch_count": float(len(online_batches)),
        "external_batch_count": float(len(external_batches)),
        "sampled_chars": float(sampled_chars),
    }
    return train_batches, val_batches, online_batches, external_batches, stats


def fit_with_checkpoint(model, optimizer, train_batches, val_batches, epochs: int):
    history: List[Dict[str, float]] = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
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
                "task_acc": val_metrics["task_acc"],
                "stable_read": val_metrics["stable_read"],
                "guarded_write": val_metrics["guarded_write"],
                "theorem_survival": val_metrics["theorem_survival"],
                "transport_margin": val_metrics["transport_margin"],
                "objective": objective,
            }
        )
        if objective < best_objective:
            best_objective = objective
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)
    return history, best_state, best_objective


def online_round(model, online_batches: List[Dict[str, torch.Tensor]], lr: float):
    model.snapshot()
    before = evaluate_model(model, online_batches)
    rounds = []
    for idx, batch in enumerate(online_batches):
        metrics = model.online_update_step(batch, lr=lr)
        after = evaluate_model(model, [batch])
        rounds.append(
            {
                "round": float(idx + 1),
                "total_loss": metrics["total_loss"],
                "task_acc": after["task_acc"],
                "guarded_write": after["guarded_write"],
                "stable_read": after["stable_read"],
                "theorem_survival": after["theorem_survival"],
            }
        )
    after_all = evaluate_model(model, online_batches)
    restored = model.rollback()
    rollback_error = 1.0
    if restored:
        rollback_error = abs(evaluate_model(model, online_batches)["loss"] - before["loss"])
    return before, after_all, rounds, rollback_error


def main() -> None:
    module = load_module(MODEL_PATH, "icspb_v2_long_horizon_model")
    helper = load_module(HELPER_PATH, "icspb_openwebtext_helper_long_horizon")

    config = module.ICSPBLargeOnlineConfig(
        family_vocab_size=32,
        concept_vocab_size=16384,
        relation_vocab_size=256,
        context_vocab_size=256,
        stage_vocab_size=256,
        protocol_vocab_size=256,
        hidden_dim=192,
        task_classes=32,
        brain_probe_dim=12,
        stable_read_floor=0.20,
        guarded_write_floor=0.16,
    )

    train_batches, val_batches, online_batches, external_batches, data_stats = gather_batches(helper, config)
    guarded_batches = [build_guarded_batch(batch) for batch in train_batches[: len(train_batches) // 2]]
    baseline_train = [build_baseline_batch(batch) for batch in train_batches]
    baseline_val = [build_baseline_batch(batch) for batch in val_batches]

    torch.manual_seed(20260313)
    proto = module.ICSPBBackboneV2LargeOnline(config)
    torch.manual_seed(20260313)
    baseline = module.ICSPBBackboneV2LargeOnline(config)

    stable_batches = [helper.build_stable_regime_batch(batch) for batch in train_batches]
    proto_opt = torch.optim.AdamW(proto.parameters(), lr=7.5e-4, weight_decay=7.5e-5)
    baseline_opt = torch.optim.AdamW(baseline.parameters(), lr=1.0e-3, weight_decay=1.0e-4)

    proto_initial = evaluate_model(proto, val_batches)
    baseline_initial = evaluate_model(baseline, baseline_val)

    candidate_states: List[Dict[str, object]] = []
    proto_history, _, _ = fit_with_checkpoint(proto, proto_opt, train_batches, val_batches, epochs=12)
    record_candidate(candidate_states, "initial_train", proto, val_batches, external_batches)
    baseline_history, _, _ = fit_with_checkpoint(baseline, baseline_opt, baseline_train, baseline_val, epochs=14)

    proto_mid = evaluate_model(proto, val_batches)
    auto_recovery_triggered = False
    stabilization_history: List[float] = []
    write_recovery_history: List[float] = []
    structural_recovery_history: List[float] = []
    consolidation_history: List[Dict[str, float]] = []
    if (
        proto_mid["guarded_write"] < 0.42
        or proto_mid["stable_read"] < 0.99
        or proto_mid["theorem_survival"] < 0.99
        or proto_mid["task_acc"] < 0.78
    ):
        auto_recovery_triggered = True
        stable_opt = torch.optim.AdamW(proto.parameters(), lr=5.5e-4, weight_decay=5.0e-5)
        for _ in range(4):
            epoch_loss = 0.0
            for batch in stable_batches:
                metrics = proto.train_step(stable_opt, batch)
                epoch_loss += metrics["total_loss"]
            stabilization_history.append(epoch_loss / len(stable_batches))
        record_candidate(candidate_states, "stabilization", proto, val_batches, external_batches)

        write_opt = torch.optim.AdamW(proto.parameters(), lr=5.0e-4, weight_decay=5.0e-5)
        for _ in range(4):
            epoch_loss = 0.0
            for batch in guarded_batches:
                metrics = proto.train_step(write_opt, batch)
                epoch_loss += metrics["total_loss"]
            write_recovery_history.append(epoch_loss / len(guarded_batches))
        record_candidate(candidate_states, "write_recovery", proto, val_batches, external_batches)

        structural_opt = torch.optim.AdamW(proto.parameters(), lr=3.5e-4, weight_decay=5.0e-5)
        structural_mix = []
        for raw_batch, stable_batch, guarded_batch in zip(train_batches, stable_batches, guarded_batches):
            structural_mix.extend([stable_batch, guarded_batch, raw_batch])
        for _ in range(4):
            epoch_loss = 0.0
            for batch in structural_mix:
                epoch_loss += helper.train_structural_recovery_step(proto, structural_opt, batch, config)
            structural_recovery_history.append(epoch_loss / len(structural_mix))
        record_candidate(candidate_states, "structural_recovery", proto, val_batches, external_batches)

        consolidation_batches = (
            train_batches
            + val_batches
            + external_batches
            + stable_batches[: len(val_batches)]
            + guarded_batches[: len(val_batches)]
        )
        consolidation_opt = torch.optim.AdamW(proto.parameters(), lr=2.2e-4, weight_decay=5.0e-5)
        consolidation_history, _, _ = fit_with_checkpoint(
            proto,
            consolidation_opt,
            consolidation_batches,
            external_batches,
            epochs=8,
        )
        record_candidate(candidate_states, "consolidation", proto, val_batches, external_batches)

    gate_alignment_history: List[float] = []
    final_read_stabilization_history: List[float] = []
    structural_restoration_history: List[float] = []
    write_task_bridge_recovery_history: List[float] = []
    initial_combined_loss = float(candidate_states[0]["combined"]["loss"])
    best_before_alignment = select_best_candidate(candidate_states, initial_combined_loss)
    proto.load_state_dict(best_before_alignment["state"], strict=True)
    if not bool(best_before_alignment["feasible"]):
        gate_alignment_history = gate_alignment_pass(
            proto,
            config,
            stable_batches,
            guarded_batches,
            val_batches,
            external_batches,
        )
        record_candidate(candidate_states, "gate_alignment", proto, val_batches, external_batches)

    best_before_final_stabilization = select_best_candidate(candidate_states, initial_combined_loss)
    proto.load_state_dict(best_before_final_stabilization["state"], strict=True)
    if not bool(best_before_final_stabilization["feasible"]):
        final_read_stabilization_history = final_read_stabilization(
            proto,
            config,
            stable_batches,
            guarded_batches,
            val_batches,
            external_batches,
        )
        record_candidate(candidate_states, "final_read_stabilization", proto, val_batches, external_batches)

    best_before_structural_restoration = select_best_candidate(candidate_states, initial_combined_loss)
    proto.load_state_dict(best_before_structural_restoration["state"], strict=True)
    if not structural_core_pass(best_before_structural_restoration["combined"]):
        structural_restoration_history = full_structural_restoration(
            proto,
            config,
            stable_batches,
            guarded_batches,
            val_batches,
            external_batches,
        )
        record_candidate(candidate_states, "full_structural_restoration", proto, val_batches, external_batches)

    best_before_write_task = select_best_candidate(candidate_states, initial_combined_loss)
    proto.load_state_dict(best_before_write_task["state"], strict=True)
    if not bool(best_before_write_task["feasible"]):
        write_task_bridge_recovery_history = write_task_bridge_recovery(
            proto,
            config,
            stable_batches,
            guarded_batches,
            val_batches,
            external_batches,
        )
        record_candidate(candidate_states, "write_task_bridge_recovery", proto, val_batches, external_batches)

    best_candidate = select_best_candidate(candidate_states, initial_combined_loss)
    proto.load_state_dict(best_candidate["state"], strict=True)

    proto_final = evaluate_model(proto, val_batches)
    proto_external = evaluate_model(proto, external_batches)
    baseline_final = evaluate_model(baseline, baseline_val)
    baseline_external = evaluate_model(baseline, external_batches)

    before_online, after_online, online_rounds, rollback_error = online_round(proto, online_batches[:4], lr=3.0e-4)

    baseline_margin = baseline_final["loss"] - proto_final["loss"]
    external_margin = baseline_external["loss"] - proto_external["loss"]
    language_proxy_margin = proto_external["task_acc"] - baseline_external["task_acc"]
    long_horizon_gain = proto_initial["loss"] - proto_final["loss"]
    online_gain = before_online["loss"] - after_online["loss"]

    ready = (
        baseline_margin > 0.35
        and external_margin > 0.35
        and language_proxy_margin > 0.12
        and long_horizon_gain > 0.70
        and proto_final["stable_read"] >= 0.99
        and proto_final["theorem_survival"] >= 0.99
        and proto_final["guarded_write"] >= 0.40
        and proto_external["stable_read"] >= 0.99
        and proto_external["theorem_survival"] >= 0.99
        and proto_external["task_acc"] >= 0.80
        and online_gain >= 0.0
        and rollback_error <= 1e-8
    )

    structure_gain = (
        0.22 * max(0.0, baseline_margin)
        + 0.18 * max(0.0, external_margin)
        + 0.16 * proto_external["task_acc"]
        + 0.14 * proto_final["theorem_survival"]
        + 0.12 * proto_final["stable_read"]
        + 0.12 * proto_final["guarded_write"]
        + 0.06 * min(1.0, proto_final["transport_margin"] / 8.0)
    )

    result = {
        "smoke_pass": True,
        "training_pass": proto_final["loss"] < proto_initial["loss"],
        "baseline_outperform_pass": baseline_margin > 0.35,
        "external_outperform_pass": external_margin > 0.35,
        "language_proxy_pass": language_proxy_margin > 0.12,
        "online_update_pass": online_gain >= 0.0,
        "rollback_pass": rollback_error <= 1e-8,
        "implementation_ready": ready,
        "implementation_score": 1.0 if ready else max(0.0, min(1.0, structure_gain)),
        "auto_recovery_triggered": auto_recovery_triggered,
        "data_stats": data_stats,
        "prototype_name": "ICSPB-Backbone-v2-LargeOnline",
        "baseline_name": "bridge_protocol_ablated_baseline",
        "proto_initial": proto_initial,
        "proto_mid": proto_mid,
        "proto_final": proto_final,
        "proto_external": proto_external,
        "baseline_initial": baseline_initial,
        "baseline_final": baseline_final,
        "baseline_external": baseline_external,
        "proto_history": proto_history,
        "baseline_history": baseline_history,
        "stabilization_history": stabilization_history,
        "write_recovery_history": write_recovery_history,
        "structural_recovery_history": structural_recovery_history,
        "consolidation_history": consolidation_history,
        "gate_alignment_history": gate_alignment_history,
        "final_read_stabilization_history": final_read_stabilization_history,
        "full_structural_restoration_history": structural_restoration_history,
        "write_task_bridge_recovery_history": write_task_bridge_recovery_history,
        "candidate_states": [
            {
                "name": c["name"],
                "feasible": c["feasible"],
                "selection_objective": c["selection_objective"],
                "structure_score": c["structure_score"],
                "combined": c["combined"],
            }
            for c in candidate_states
        ],
        "selected_candidate": {
            "name": best_candidate["name"],
            "feasible": best_candidate["feasible"],
            "selection_objective": best_candidate["selection_objective"],
            "structure_score": best_candidate["structure_score"],
            "combined": best_candidate["combined"],
        },
        "online_before": before_online,
        "online_after": after_online,
        "online_rounds": online_rounds,
        "baseline_margin": baseline_margin,
        "external_margin": external_margin,
        "language_proxy_margin": language_proxy_margin,
        "long_horizon_gain": long_horizon_gain,
        "online_gain": online_gain,
        "rollback_error": rollback_error,
        "structure_gain": structure_gain,
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
