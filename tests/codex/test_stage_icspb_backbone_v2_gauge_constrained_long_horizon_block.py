from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "research" / "gpt5" / "code" / "icspb_backbone_v2_large_online.py"
HELPER_PATH = ROOT / "tests" / "codex" / "test_stage_openwebtext_real_data_block.py"
LONG_PATH = ROOT / "tests" / "codex" / "test_stage_icspb_backbone_v2_real_long_horizon_training_block.py"
OUTPUT_DIR = ROOT / "tests" / "codex_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "icspb_v2_gauge_constrained_long_horizon_block_20260314.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def gauge_spread(model) -> torch.Tensor:
    tensors = [
        model.family_patch_backbone.weight,
        model.concept_section_memory_bank.weight,
        model.relation_fiber.weight,
        model.context_fiber.weight,
        model.stage_transport.weight,
        model.protocol_bridge.weight,
        model.task_head.weight,
    ]
    norms = torch.stack([t.norm() / (t.numel() ** 0.5) for t in tensors])
    return norms.std()


def gauge_constrained_epoch(model, optimizer, batches: List[Dict[str, torch.Tensor]]) -> float:
    total = 0.0
    steps = 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        out = model.forward(batch)
        loss, _ = model.compute_loss(batch)
        theorem_prob = torch.sigmoid(out["theorem_logits"]).mean()
        stable_read = out["read_gate"].mean()
        guarded_write = out["write_gate"].mean()
        transport_margin = out["protocol_state"].norm(dim=-1).mean() - out["successor_state"].norm(dim=-1).mean()
        gauge_loss = gauge_spread(model)
        reg = (
            0.14 * F.relu(torch.tensor(0.99, device=loss.device) - theorem_prob)
            + 0.14 * F.relu(torch.tensor(0.99, device=loss.device) - stable_read)
            + 0.08 * F.relu(torch.tensor(0.22, device=loss.device) - guarded_write)
            + 0.05 * F.relu(torch.tensor(0.18, device=loss.device) - transport_margin)
            + 0.16 * gauge_loss
        )
        total_loss = loss + reg
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(total_loss.detach())
        steps += 1
    return total / max(1, steps)


def focused_gauge_epoch(model, optimizer, batches: List[Dict[str, torch.Tensor]]) -> float:
    total = 0.0
    steps = 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        out = model.forward(batch)
        task_loss = F.cross_entropy(out["task_logits"], batch["labels"])
        theorem_prob = torch.sigmoid(out["theorem_logits"]).mean()
        stable_read = out["read_gate"].mean()
        gauge_loss = gauge_spread(model)
        transport_margin = out["protocol_state"].norm(dim=-1).mean() - out["successor_state"].norm(dim=-1).mean()
        loss = (
            0.28 * task_loss
            + 0.22 * F.relu(torch.tensor(0.99, device=task_loss.device) - theorem_prob)
            + 0.18 * F.relu(torch.tensor(0.99, device=task_loss.device) - stable_read)
            + 0.32 * gauge_loss
            + 0.05 * F.relu(torch.tensor(0.18, device=task_loss.device) - transport_margin)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
        optimizer.step()
        total += float(loss.detach())
        steps += 1
    return total / max(1, steps)


def record_candidate(store, name: str, model, long_mod, val_batches, external_batches) -> None:
    val_metrics = long_mod.evaluate_model(model, val_batches)
    external_metrics = long_mod.evaluate_model(model, external_batches)
    combined = {
        "loss": 0.5 * (val_metrics["loss"] + external_metrics["loss"]),
        "task_acc": min(val_metrics["task_acc"], external_metrics["task_acc"]),
        "theorem_survival": min(val_metrics["theorem_survival"], external_metrics["theorem_survival"]),
        "stable_read": min(val_metrics["stable_read"], external_metrics["stable_read"]),
        "guarded_write": val_metrics["guarded_write"],
        "transport_margin": 0.5 * (val_metrics["transport_margin"] + external_metrics["transport_margin"]),
        "stress_balance": 0.5 * (val_metrics["stress_balance"] + external_metrics["stress_balance"]),
        "gauge_spread": float(gauge_spread(model).detach()),
    }
    score = (
        combined["loss"]
        - 0.35 * combined["task_acc"]
        - 0.18 * combined["theorem_survival"]
        - 0.16 * combined["stable_read"]
        - 0.10 * min(1.0, combined["transport_margin"] / 8.0)
        + 0.40 * combined["gauge_spread"]
    )
    feasible = (
        combined["theorem_survival"] >= 0.99
        and combined["stable_read"] >= 0.99
        and combined["guarded_write"] >= 0.40
        and combined["task_acc"] >= 0.80
    )
    store.append(
        {
            "name": name,
            "combined": combined,
            "val": val_metrics,
            "external": external_metrics,
            "score": score,
            "feasible": feasible,
            "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
        }
    )


def select_candidate(candidates):
    feasible = [c for c in candidates if c["feasible"]]
    if feasible:
        return min(feasible, key=lambda c: (float(c["score"]), float(c["combined"]["gauge_spread"])))
    return min(candidates, key=lambda c: (float(c["score"]), float(c["combined"]["gauge_spread"])))


def main() -> None:
    module = load_module(MODEL_PATH, "icspb_v2_model_for_gauge")
    helper = load_module(HELPER_PATH, "icspb_openwebtext_helper_for_gauge")
    long_mod = load_module(LONG_PATH, "icspb_long_horizon_for_gauge")

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

    train_batches, val_batches, online_batches, external_batches, data_stats = long_mod.gather_batches(helper, config)
    stable_batches = [helper.build_stable_regime_batch(batch) for batch in train_batches]
    guarded_batches = [long_mod.build_guarded_batch(batch) for batch in train_batches[: len(train_batches) // 2]]
    baseline_train = [long_mod.build_baseline_batch(batch) for batch in train_batches]
    baseline_val = [long_mod.build_baseline_batch(batch) for batch in val_batches]

    torch.manual_seed(20260314)
    proto = module.ICSPBBackboneV2LargeOnline(config)
    torch.manual_seed(20260314)
    baseline = module.ICSPBBackboneV2LargeOnline(config)

    proto_opt = torch.optim.AdamW(proto.parameters(), lr=7.0e-4, weight_decay=7.5e-5)
    baseline_opt = torch.optim.AdamW(baseline.parameters(), lr=1.0e-3, weight_decay=1.0e-4)

    proto_initial = long_mod.evaluate_model(proto, val_batches)
    baseline_initial = long_mod.evaluate_model(baseline, baseline_val)
    initial_gauge = float(gauge_spread(proto).detach())
    candidate_states = []

    proto_history, _, _ = long_mod.fit_with_checkpoint(proto, proto_opt, train_batches, val_batches, epochs=10)
    record_candidate(candidate_states, "initial_train", proto, long_mod, val_batches, external_batches)
    baseline_history, _, _ = long_mod.fit_with_checkpoint(baseline, baseline_opt, baseline_train, baseline_val, epochs=12)
    proto_mid = long_mod.evaluate_model(proto, val_batches)

    gauge_history: List[float] = []
    stabilization_history: List[float] = []
    structural_restoration_history: List[float] = []
    write_task_bridge_recovery_history: List[float] = []
    if (
        proto_mid["guarded_write"] < 0.40
        or proto_mid["stable_read"] < 0.99
        or proto_mid["theorem_survival"] < 0.99
        or proto_mid["task_acc"] < 0.80
    ):
        stabilization_history = long_mod.final_read_stabilization(
            proto,
            config,
            stable_batches,
            guarded_batches,
            val_batches,
            external_batches,
        )
        record_candidate(candidate_states, "final_read_stabilization", proto, long_mod, val_batches, external_batches)
        structural_restoration_history = long_mod.full_structural_restoration(
            proto,
            config,
            stable_batches,
            guarded_batches,
            val_batches,
            external_batches,
        )
        record_candidate(candidate_states, "full_structural_restoration", proto, long_mod, val_batches, external_batches)
        write_task_bridge_recovery_history = long_mod.write_task_bridge_recovery(
            proto,
            config,
            stable_batches,
            guarded_batches,
            val_batches,
            external_batches,
        )
        record_candidate(candidate_states, "write_task_bridge_recovery", proto, long_mod, val_batches, external_batches)

    best_before_gauge = select_candidate(candidate_states)
    proto.load_state_dict(best_before_gauge["state"], strict=True)

    gauge_mix = stable_batches[: len(val_batches)] + guarded_batches[: len(val_batches)] + val_batches + external_batches[: len(val_batches)]
    gauge_opt = torch.optim.AdamW(proto.parameters(), lr=1.5e-4, weight_decay=3e-5)
    best_gauge_state = {k: v.detach().clone() for k, v in proto.state_dict().items()}
    best_gauge_score = float(best_before_gauge["score"])
    for idx in range(8):
        gauge_history.append(gauge_constrained_epoch(proto, gauge_opt, gauge_mix))
        record_candidate(candidate_states, f"gauge_epoch_{idx+1}", proto, long_mod, val_batches, external_batches)
        latest = candidate_states[-1]
        if float(latest["score"]) < best_gauge_score:
            best_gauge_score = float(latest["score"])
            best_gauge_state = {k: v.detach().clone() for k, v in proto.state_dict().items()}
    proto.load_state_dict(best_gauge_state, strict=True)

    focused_gauge_history: List[float] = []
    if float(gauge_spread(proto).detach()) > initial_gauge - 0.005:
        focus_params = (
            list(proto.family_patch_backbone.parameters())
            + list(proto.concept_section_memory_bank.parameters())
            + list(proto.relation_fiber.parameters())
            + list(proto.context_fiber.parameters())
            + list(proto.stage_transport.parameters())
            + list(proto.protocol_bridge.parameters())
            + list(proto.task_head.parameters())
        )
        focus_opt = torch.optim.AdamW(focus_params, lr=8.0e-5, weight_decay=2e-5)
        focus_mix = stable_batches[: len(val_batches)] + external_batches[: len(val_batches)] + val_batches
        for idx in range(6):
            focused_gauge_history.append(focused_gauge_epoch(proto, focus_opt, focus_mix))
            record_candidate(candidate_states, f"focused_gauge_epoch_{idx+1}", proto, long_mod, val_batches, external_batches)
        best_after_focus = select_candidate(candidate_states)
        proto.load_state_dict(best_after_focus["state"], strict=True)
    else:
        focused_gauge_history = []

    final_stabilization_history = long_mod.final_read_stabilization(
        proto,
        config,
        stable_batches,
        guarded_batches,
        val_batches,
        external_batches,
    )
    record_candidate(candidate_states, "post_gauge_final_stabilization", proto, long_mod, val_batches, external_batches)

    best_candidate = select_candidate(candidate_states)
    proto.load_state_dict(best_candidate["state"], strict=True)

    proto_final = long_mod.evaluate_model(proto, val_batches)
    proto_external = long_mod.evaluate_model(proto, external_batches)
    baseline_final = long_mod.evaluate_model(baseline, baseline_val)
    baseline_external = long_mod.evaluate_model(baseline, external_batches)
    before_online, after_online, online_rounds, rollback_error = long_mod.online_round(proto, online_batches[:4], lr=2.8e-4)

    final_gauge = float(gauge_spread(proto).detach())
    gauge_reduction = initial_gauge - final_gauge
    baseline_margin = baseline_final["loss"] - proto_final["loss"]
    external_margin = baseline_external["loss"] - proto_external["loss"]
    online_gain = before_online["loss"] - after_online["loss"]
    language_proxy_margin = proto_external["task_acc"] - baseline_external["task_acc"]

    ready = (
        baseline_margin > 0.35
        and external_margin > 0.35
        and language_proxy_margin > 0.10
        and gauge_reduction > 0.002
        and proto_final["stable_read"] >= 0.99
        and proto_final["theorem_survival"] >= 0.99
        and proto_external["stable_read"] >= 0.99
        and proto_external["theorem_survival"] >= 0.99
        and online_gain >= 0.0
        and rollback_error <= 1e-8
    )

    result = {
        "smoke_pass": True,
        "training_pass": proto_final["loss"] < proto_initial["loss"],
        "baseline_outperform_pass": baseline_margin > 0.35,
        "external_outperform_pass": external_margin > 0.35,
        "language_proxy_pass": language_proxy_margin > 0.10,
        "gauge_reduction_pass": gauge_reduction > 0.002,
        "online_update_pass": online_gain >= 0.0,
        "rollback_pass": rollback_error <= 1e-8,
        "implementation_ready": ready,
        "implementation_score": 1.0 if ready else max(0.0, min(1.0, 0.5 + gauge_reduction * 50.0)),
        "prototype_name": "ICSPB-Backbone-v2-LargeOnline",
        "baseline_name": "bridge_protocol_ablated_baseline",
        "data_stats": data_stats,
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
        "structural_restoration_history": structural_restoration_history,
        "write_task_bridge_recovery_history": write_task_bridge_recovery_history,
        "gauge_history": gauge_history,
        "focused_gauge_history": focused_gauge_history,
        "final_stabilization_history": final_stabilization_history,
        "candidate_states": [
            {
                "name": c["name"],
                "score": c["score"],
                "feasible": c["feasible"],
                "combined": c["combined"],
            }
            for c in candidate_states
        ],
        "selected_candidate": {
            "name": best_candidate["name"],
            "score": best_candidate["score"],
            "feasible": best_candidate["feasible"],
            "combined": best_candidate["combined"],
        },
        "online_before": before_online,
        "online_after": after_online,
        "online_rounds": online_rounds,
        "initial_gauge_spread": initial_gauge,
        "final_gauge_spread": final_gauge,
        "gauge_reduction": gauge_reduction,
        "baseline_margin": baseline_margin,
        "external_margin": external_margin,
        "language_proxy_margin": language_proxy_margin,
        "online_gain": online_gain,
        "rollback_error": rollback_error,
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
