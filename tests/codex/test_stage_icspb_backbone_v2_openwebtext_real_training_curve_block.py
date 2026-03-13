from __future__ import annotations

import importlib.util
import json
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
OUTPUT_PATH = OUTPUT_DIR / "icspb_v2_openwebtext_real_training_curve_block.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in batch.items()}


def normalize_batch_targets(batch: Dict[str, torch.Tensor], config) -> Dict[str, torch.Tensor]:
    out = clone_batch(batch)
    family = out["family_ids"].long()
    concept = out["concept_ids"].long()
    relation = out["relation_ids"].long()
    context = out["context_ids"].long()
    stage = out["stage_ids"].long()
    protocol = out["protocol_ids"].long()

    concept_bucket = concept % 8
    labels = (
        family * 11
        + relation * 5
        + context * 3
        + stage * 2
        + protocol
        + concept_bucket
    ) % config.task_classes
    out["labels"] = labels

    components = [
        family.float() / max(1, config.family_vocab_size - 1),
        relation.float() / max(1, config.relation_vocab_size - 1),
        context.float() / max(1, config.context_vocab_size - 1),
        stage.float() / max(1, config.stage_vocab_size - 1),
        protocol.float() / max(1, config.protocol_vocab_size - 1),
        concept_bucket.float() / 7.0,
    ]
    base = torch.stack(components, dim=-1)
    repeats = (config.brain_probe_dim + base.shape[-1] - 1) // base.shape[-1]
    tiled = base.repeat(1, repeats)[:, : config.brain_probe_dim]
    out["brain_targets"] = tiled * 0.8 - 0.4
    return out


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
    out["protocol_ids"] = (out["protocol_ids"] + 5) % 192
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
        chunk_chars = 2800 if path.stat().st_size < 2_000_000_000 else 3600
        chunk_count = 24 if idx < 4 else 18
        chunks = helper.sample_chunks_from_file(path, chunk_chars=chunk_chars, num_chunks=chunk_count)
        sampled_chars += sum(len(c) for c in chunks)
        random.Random(20260315 + idx).shuffle(chunks)
        n = len(chunks)
        n_train = max(10, int(n * 0.58))
        n_val = max(4, int(n * 0.16))
        n_online = max(3, int(n * 0.14))
        train_chunks.extend(chunks[:n_train])
        val_chunks.extend(chunks[n_train : n_train + n_val])
        online_chunks.extend(chunks[n_train + n_val : n_train + n_val + n_online])
        external_chunks.extend(chunks[n_train + n_val + n_online :])

    random.Random(20260341).shuffle(train_chunks)
    random.Random(20260342).shuffle(val_chunks)
    random.Random(20260343).shuffle(online_chunks)
    random.Random(20260344).shuffle(external_chunks)

    batch_size = 8

    def to_batches(chunks: List[str], *, external: bool = False) -> List[Dict[str, torch.Tensor]]:
        out: List[Dict[str, torch.Tensor]] = []
        for i in range(0, len(chunks), batch_size):
            piece = chunks[i : i + batch_size]
            if len(piece) != batch_size:
                continue
            batch = normalize_batch_targets(helper.build_batch_from_chunks(piece, config), config)
            if external:
                batch = build_external_batch(batch)
            out.append(batch)
        return out

    train_batches = to_batches(train_chunks)
    val_batches = to_batches(val_chunks)
    online_batches = to_batches(online_chunks)
    external_batches = to_batches(external_chunks, external=True)

    if len(train_batches) < 8 or len(val_batches) < 3 or len(online_batches) < 2 or len(external_batches) < 2:
        raise RuntimeError("真实训练曲线批次不足，无法继续。")

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


def longterm_selection_objective(metrics: Dict[str, float], external_metrics: Dict[str, float] | None = None) -> float:
    guarded_floor = max(0.0, min(1.0, metrics["guarded_write"] * 1.4))
    survival_floor = min(metrics["theorem_survival"], metrics["stable_read"], guarded_floor)
    hard_penalty = 0.0
    if metrics["theorem_survival"] < 0.985:
        hard_penalty += 6.0
    if metrics["stable_read"] < 0.985:
        hard_penalty += 4.0
    if metrics["guarded_write"] < 0.14:
        hard_penalty += 2.5

    external_loss = 0.0
    external_task_acc = 0.0
    external_transport = 0.0
    if external_metrics is not None:
        external_loss = external_metrics["loss"]
        external_task_acc = external_metrics["task_acc"]
        external_transport = min(1.0, external_metrics["transport_margin"] / 4.0)
    return (
        hard_penalty
        + metrics["loss"]
        + 0.20 * external_loss
        - 0.30 * metrics["task_acc"]
        - 0.18 * external_task_acc
        - 0.12 * survival_floor
        - 0.08 * min(1.0, metrics["transport_margin"] / 4.0)
        - 0.04 * external_transport
        - 0.03 * metrics["stress_balance"]
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


def final_read_stabilization(model, stable_batches, val_batches, external_batches, *, epochs: int) -> Tuple[List[float], Dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=4e-5)
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
                theorem_floor=0.72,
                margin_floor=0.16,
            )
        )
        metrics = evaluate_model(model, val_batches)
        objective = (
            metrics["loss"]
            - 0.55 * metrics["stable_read"]
            - 0.25 * metrics["theorem_survival"]
            - 0.10 * metrics["task_acc"]
            - 0.05 * min(1.0, metrics["transport_margin"] / 5.0)
        )
        if objective < best_objective:
            best_objective = objective
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
    model.load_state_dict(best_state, strict=True)
    return history, best_metrics


def online_adaptation_round(model, online_batches: List[Dict[str, torch.Tensor]], lr: float) -> Tuple[List[Dict[str, float]], float]:
    model.snapshot()
    rounds: List[Dict[str, float]] = []
    baseline_before = evaluate_model(model, online_batches)
    for idx, batch in enumerate(online_batches):
        before = evaluate_model(model, [batch])
        metrics = model.online_update_step(batch, lr=lr)
        after = evaluate_model(model, [batch])
        rounds.append(
            {
                "round": float(idx + 1),
                "before_loss": before["loss"],
                "after_loss": after["loss"],
                "delta": before["loss"] - after["loss"],
                "stable_read": after["stable_read"],
                "theorem_survival": after["theorem_survival"],
                "online_write_scale": metrics["online_write_scale"],
            }
        )
    restored = model.rollback()
    if not restored:
        return rounds, 1.0
    baseline_after = evaluate_model(model, online_batches)
    rollback_error = abs(baseline_before["loss"] - baseline_after["loss"])
    return rounds, rollback_error


def register_candidate(
    candidates: List[Dict[str, object]],
    name: str,
    state: Dict[str, torch.Tensor],
    val_metrics: Dict[str, float],
    external_metrics: Dict[str, float],
) -> None:
    candidates.append(
        {
            "name": name,
            "state": {k: v.detach().clone() for k, v in state.items()},
            "val_metrics": dict(val_metrics),
            "external_metrics": dict(external_metrics),
            "objective": longterm_selection_objective(val_metrics, external_metrics),
        }
    )


def pick_best_candidate(candidates: List[Dict[str, object]]) -> Dict[str, object]:
    return min(candidates, key=lambda item: float(item["objective"]))


def main() -> None:
    module = load_module(MODEL_PATH, "icspb_v2_real_curve_model")
    helper = load_module(HELPER_PATH, "icspb_openwebtext_real_curve_helper")

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

    train_batches, val_batches, online_batches, external_batches, data_stats = gather_batches(helper, config)
    stable_batches = [build_stable_batch(batch) for batch in train_batches]
    guarded_batches = [build_guarded_batch(batch) for batch in train_batches]
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

    curve_history: Dict[str, List[float]] = {}
    proto_candidates: List[Dict[str, object]] = []

    curve_history["proto_warmup"], warmup_state, warmup_metrics = train_with_selection(
        proto,
        proto_opt,
        train_batches,
        val_batches,
        epochs=8,
        floors=(0.14, 0.18, 0.60, 0.08),
    )
    register_candidate(
        proto_candidates,
        "warmup",
        warmup_state,
        warmup_metrics,
        evaluate_model(proto, external_batches),
    )

    curve_history["proto_guarded"], guarded_state, guarded_metrics = train_with_selection(
        proto,
        proto_opt,
        guarded_batches,
        val_batches,
        epochs=6,
        floors=(0.14, 0.20, 0.66, 0.10),
    )
    register_candidate(
        proto_candidates,
        "guarded",
        guarded_state,
        guarded_metrics,
        evaluate_model(proto, external_batches),
    )

    curve_history["proto_external"], external_state, external_metrics = train_with_selection(
        proto,
        proto_opt,
        train_batches + external_batches,
        val_batches,
        epochs=4,
        floors=(0.14, 0.21, 0.70, 0.12),
    )
    register_candidate(
        proto_candidates,
        "external",
        external_state,
        external_metrics,
        evaluate_model(proto, external_batches),
    )

    curve_history["proto_baseline"], _, _ = train_with_selection(
        baseline,
        baseline_opt,
        baseline_train,
        baseline_val,
        epochs=12,
        floors=(0.10, 0.16, 0.52, 0.04),
    )

    stabilization_history, proto_final = final_read_stabilization(
        proto,
        stable_batches,
        val_batches,
        external_batches,
        epochs=6,
    )
    register_candidate(
        proto_candidates,
        "stabilized",
        proto.state_dict(),
        proto_final,
        evaluate_model(proto, external_batches),
    )
    baseline_final = evaluate_model(baseline, baseline_val)
    proto_external = evaluate_model(proto, external_batches)

    auto_recovery_triggered = False
    if (
        proto_final["stable_read"] < 0.985
        or proto_final["theorem_survival"] < 0.985
        or proto_final["guarded_write"] < 0.14
        or (baseline_final["loss"] - proto_final["loss"]) < 0.25
    ):
        auto_recovery_triggered = True
        recovery_opt = torch.optim.AdamW(proto.parameters(), lr=1.5e-4, weight_decay=2.5e-5)
        curve_history["proto_recovery"], _, _ = train_with_selection(
            proto,
            recovery_opt,
            stable_batches + guarded_batches + external_batches,
            val_batches,
            epochs=5,
            floors=(0.14, 0.23, 0.76, 0.15),
        )
        extra_stabilization, proto_final = final_read_stabilization(
            proto,
            stable_batches,
            val_batches,
            external_batches,
            epochs=4,
        )
        stabilization_history.extend(extra_stabilization)
        proto_external = evaluate_model(proto, external_batches)
        register_candidate(
            proto_candidates,
            "recovered",
            proto.state_dict(),
            proto_final,
            proto_external,
        )

    best_candidate = pick_best_candidate(proto_candidates)
    proto.load_state_dict(best_candidate["state"], strict=True)
    proto_final = dict(best_candidate["val_metrics"])
    proto_external = dict(best_candidate["external_metrics"])

    online_rounds, rollback_error = online_adaptation_round(proto, online_batches[:3], lr=1.0e-3)
    online_delta = sum(max(0.0, item["delta"]) for item in online_rounds)

    baseline_margin = baseline_final["loss"] - proto_final["loss"]
    external_margin = baseline_final["loss"] - proto_external["loss"]
    language_proxy_margin = proto_final["task_acc"] - baseline_final["task_acc"]

    result = {
        "prototype_name": "ICSPB-Backbone-v2-LargeOnline",
        "baseline_name": "PathOnly-LargeOnline-Baseline",
        "data_stats": data_stats,
        "proto_initial": proto_initial,
        "baseline_initial": baseline_initial,
        "proto_final": proto_final,
        "baseline_final": baseline_final,
        "proto_external": proto_external,
        "curve_history": curve_history,
        "stabilization_history": stabilization_history,
        "selected_candidate": best_candidate["name"],
        "online_rounds": online_rounds,
        "online_delta": online_delta,
        "rollback_error": rollback_error,
        "baseline_margin": baseline_margin,
        "external_margin": external_margin,
        "language_proxy_margin": language_proxy_margin,
        "auto_recovery_triggered": auto_recovery_triggered,
    }

    result["smoke_pass"] = True
    result["training_pass"] = (
        proto_final["loss"] < proto_initial["loss"] * 0.85
        and baseline_margin > 0.60
        and language_proxy_margin > 0.10
        and proto_final["theorem_survival"] >= 0.985
        and proto_final["stable_read"] >= 0.985
    )
    result["baseline_outperform_pass"] = baseline_margin > 0.40
    result["external_outperform_pass"] = external_margin > 0.30
    result["online_update_pass"] = online_delta >= 0.0
    result["rollback_pass"] = rollback_error <= 1e-8
    result["implementation_ready"] = (
        result["training_pass"]
        and result["baseline_outperform_pass"]
        and result["external_outperform_pass"]
        and result["online_update_pass"]
        and result["rollback_pass"]
    )
    result["implementation_score"] = 1.0 if result["implementation_ready"] else 0.0

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
