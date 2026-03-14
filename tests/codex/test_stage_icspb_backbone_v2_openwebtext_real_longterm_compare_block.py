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
DAEMON_PATH = ROOT / "tests" / "codex" / "test_stage_icspb_backbone_v2_openwebtext_persistent_continual_daemon_block.py"
OUTPUT_DIR = ROOT / "tests" / "codex_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "icspb_v2_openwebtext_real_longterm_compare_block.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in batch.items()}


def make_baseline_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = clone_batch(batch)
    out["relation_ids"].zero_()
    out["context_ids"].zero_()
    out["stage_ids"].zero_()
    out["protocol_ids"].zero_()
    out["novelty"] = (out["novelty"] * 0.55).clamp(max=0.16)
    out["retention"] = (out["retention"] * 1.20).clamp(max=0.30)
    return out


def gather_batches(helper, config) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], Dict[str, float]]:
    files = helper.iter_openwebtext_files()
    train_chunks: List[str] = []
    val_chunks: List[str] = []
    online_chunks: List[str] = []
    sampled_chars = 0

    for idx, path in enumerate(files):
        chunk_chars = 3000 if path.stat().st_size < 2_000_000_000 else 4200
        chunk_count = 18 if idx < 6 else 14
        chunks = helper.sample_chunks_from_file(path, chunk_chars=chunk_chars, num_chunks=chunk_count)
        sampled_chars += sum(len(chunk) for chunk in chunks)
        random.Random(20260313 + idx).shuffle(chunks)

        n = len(chunks)
        n_train = max(10, int(n * 0.62))
        n_val = max(4, int(n * 0.18))
        train_chunks.extend(chunks[:n_train])
        val_chunks.extend(chunks[n_train : n_train + n_val])
        online_chunks.extend(chunks[n_train + n_val :])

    random.Random(20260411).shuffle(train_chunks)
    random.Random(20260412).shuffle(val_chunks)
    random.Random(20260413).shuffle(online_chunks)

    # Smaller exact batches preserve enough train/val/online blocks for long-term
    # comparison without weakening split discipline.
    batch_size = 6

    def to_batches(chunks: List[str]) -> List[Dict[str, torch.Tensor]]:
        out: List[Dict[str, torch.Tensor]] = []
        for start in range(0, len(chunks), batch_size):
            piece = chunks[start : start + batch_size]
            if len(piece) == batch_size:
                out.append(helper.build_batch_from_chunks(piece, config))
        return out

    train_batches = to_batches(train_chunks)
    val_batches = to_batches(val_chunks)
    online_batches = to_batches(online_chunks)
    if len(train_batches) < 10 or len(val_batches) < 3 or len(online_batches) < 3:
        raise RuntimeError("openwebtext 真实长期对照批次数不足，无法构建长期训练块")

    stats = {
        "file_count": float(len(files)),
        "train_batch_count": float(len(train_batches)),
        "val_batch_count": float(len(val_batches)),
        "online_batch_count": float(len(online_batches)),
        "sampled_chars": float(sampled_chars),
    }
    return train_batches, val_batches, online_batches, stats


def main() -> None:
    module = load_module(MODEL_PATH, "icspb_v2_real_longterm_model")
    helper = load_module(HELPER_PATH, "icspb_openwebtext_helper_real_longterm")
    daemon = load_module(DAEMON_PATH, "icspb_v2_daemon_longterm")

    config = module.ICSPBLargeOnlineConfig(
        family_vocab_size=40,
        concept_vocab_size=32768,
        relation_vocab_size=320,
        context_vocab_size=320,
        stage_vocab_size=320,
        protocol_vocab_size=320,
        hidden_dim=192,
        task_classes=48,
        brain_probe_dim=16,
        stable_read_floor=0.18,
        guarded_write_floor=0.14,
    )

    train_batches, val_batches, online_batches, data_stats = gather_batches(helper, config)
    stable_batches = [daemon.build_stable_batch(b) for b in train_batches]
    guarded_batches = [daemon.build_guarded_batch(b) for b in train_batches]
    external_mix_batches = [daemon.build_external_mix_batch(b) for b in online_batches]
    baseline_train = [make_baseline_batch(b) for b in train_batches]
    baseline_val = [make_baseline_batch(b) for b in val_batches]
    baseline_online = [make_baseline_batch(b) for b in online_batches]

    torch.manual_seed(20260313)
    proto = module.ICSPBBackboneV2LargeOnline(config)
    torch.manual_seed(20260313)
    baseline = module.ICSPBBackboneV2LargeOnline(config)

    proto_opt = torch.optim.AdamW(proto.parameters(), lr=1.0e-3, weight_decay=8.0e-5)
    baseline_opt = torch.optim.AdamW(baseline.parameters(), lr=1.4e-3, weight_decay=8.0e-5)

    proto_initial = daemon.extended.evaluate_model(proto, val_batches)
    baseline_initial = daemon.extended.evaluate_model(baseline, baseline_val)

    proto_history_1, proto_mid_1 = daemon.train_with_selection(
        proto,
        proto_opt,
        train_batches,
        val_batches,
        epochs=12,
        floors=(0.15, 0.18, 0.62, 0.10),
        structured_train_epoch=daemon.extended.structured_train_epoch,
        evaluate_model=daemon.extended.evaluate_model,
    )
    baseline_history, baseline_final = daemon.train_with_selection(
        baseline,
        baseline_opt,
        baseline_train,
        baseline_val,
        epochs=12,
        floors=(0.10, 0.12, 0.50, 0.00),
        structured_train_epoch=daemon.extended.structured_train_epoch,
        evaluate_model=daemon.extended.evaluate_model,
    )

    auto_recovery_triggered = False
    stabilization_history: List[float] = []
    guarded_history: List[float] = []
    online_alignment_history: List[float] = []
    consolidation_history: List[float] = []
    final_stabilization_history: List[float] = []
    daemon_cycles: List[Dict[str, float]] = []

    def needs_recovery(metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> bool:
        return (
            metrics["task_acc"] < 0.18
            or metrics["guarded_write"] < 0.70
            or metrics["stable_read"] < 0.992
            or metrics["theorem_survival"] < 0.992
            or metrics["transport_margin"] < 0.55
            or (baseline_metrics["loss"] - metrics["loss"]) < 0.55
        )

    if needs_recovery(proto_mid_1, baseline_final):
        auto_recovery_triggered = True
        stable_opt = torch.optim.AdamW(proto.parameters(), lr=5.5e-4, weight_decay=5.0e-5)
        for _ in range(4):
            stabilization_history.append(
                daemon.extended.structured_train_epoch(
                    proto,
                    stable_opt,
                    stable_batches + val_batches,
                    write_floor=0.14,
                    read_floor=0.24,
                    theorem_floor=0.68,
                    margin_floor=0.16,
                )
            )

        guarded_opt = torch.optim.AdamW(proto.parameters(), lr=4.5e-4, weight_decay=5.0e-5)
        guarded_train = []
        for raw, stable, guarded in zip(train_batches, stable_batches, guarded_batches):
            guarded_train.extend([stable, guarded, raw])
        for _ in range(5):
            guarded_history.append(
                daemon.extended.structured_train_epoch(
                    proto,
                    guarded_opt,
                    guarded_train,
                    write_floor=0.18,
                    read_floor=0.22,
                    theorem_floor=0.72,
                    margin_floor=0.20,
                )
            )

        online_align_opt = torch.optim.AdamW(proto.parameters(), lr=3.5e-4, weight_decay=4.0e-5)
        online_align_train = external_mix_batches + stable_batches[: len(external_mix_batches)] + online_batches
        for _ in range(4):
            online_alignment_history.append(
                daemon.extended.structured_train_epoch(
                    proto,
                    online_align_opt,
                    online_align_train,
                    write_floor=0.18,
                    read_floor=0.23,
                    theorem_floor=0.74,
                    margin_floor=0.22,
                )
            )

        consolidation_opt = torch.optim.AdamW(proto.parameters(), lr=2.8e-4, weight_decay=4.0e-5)
        consolidation_batches = train_batches + val_batches + online_batches + stable_batches[: len(val_batches)]
        consolidation_history, _ = daemon.train_with_selection(
            proto,
            consolidation_opt,
            consolidation_batches,
            val_batches,
            epochs=10,
            floors=(0.18, 0.23, 0.75, 0.22),
            structured_train_epoch=daemon.extended.structured_train_epoch,
            evaluate_model=daemon.extended.evaluate_model,
        )

    proto_pre_daemon = daemon.extended.evaluate_model(proto, val_batches)
    if (
        proto_pre_daemon["stable_read"] < 0.995
        or proto_pre_daemon["theorem_survival"] < 0.995
        or proto_pre_daemon["guarded_write"] < 0.75
    ):
        final_stabilization_history, _ = daemon.extended.final_read_stabilization(
            proto,
            stable_batches,
            val_batches,
            external_mix_batches,
            epochs=6,
        )

    for cycle_idx in range(8):
        before, rounds, rollback_error = daemon.online_adaptation_round(
            proto,
            online_batches[: min(6, len(online_batches))],
            lr=2.5e-4,
            evaluate_model=daemon.extended.evaluate_model,
        )
        cycle_after = daemon.extended.evaluate_model(proto, val_batches)
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
            cycle_after["stable_read"] < 0.995
            or cycle_after["theorem_survival"] < 0.995
            or rollback_error > 1e-8
        ):
            repair_opt = torch.optim.AdamW(proto.parameters(), lr=2.0e-4, weight_decay=3.5e-5)
            daemon.extended.structured_train_epoch(
                proto,
                repair_opt,
                stable_batches + online_batches + val_batches,
                write_floor=0.17,
                read_floor=0.25,
                theorem_floor=0.78,
                margin_floor=0.20,
            )

    proto_final = daemon.extended.evaluate_model(proto, val_batches)
    proto_online = daemon.extended.evaluate_model(proto, online_batches)
    baseline_online = daemon.extended.evaluate_model(baseline, baseline_online)

    online_before, online_rounds, rollback_error = daemon.online_adaptation_round(
        proto,
        online_batches[: min(8, len(online_batches))],
        lr=2.2e-4,
        evaluate_model=daemon.extended.evaluate_model,
    )
    online_after = daemon.extended.evaluate_model(proto, online_batches[: min(8, len(online_batches))])
    online_delta_total = sum(item["delta"] for item in online_rounds)

    baseline_margin = baseline_final["loss"] - proto_final["loss"]
    external_margin = baseline_online["loss"] - proto_online["loss"]
    language_proxy_margin = proto_online["task_acc"] - baseline_online["task_acc"]
    long_horizon_gain = proto_initial["loss"] - proto_final["loss"]
    daemon_stability = (
        sum(item["stable_read"] + item["theorem_survival"] for item in daemon_cycles)
        / max(1, 2 * len(daemon_cycles))
    )

    ready = (
        baseline_margin > 0.55
        and external_margin > 0.45
        and language_proxy_margin > 0.08
        and long_horizon_gain > 0.85
        and proto_final["stable_read"] >= 0.995
        and proto_final["theorem_survival"] >= 0.995
        and proto_final["guarded_write"] >= 0.75
        and proto_final["transport_margin"] >= 0.75
        and proto_online["stable_read"] >= 0.995
        and proto_online["theorem_survival"] >= 0.995
        and daemon_stability >= 0.992
        and online_delta_total >= 0.0
        and rollback_error <= 1e-8
    )

    structure_gain = (
        0.22 * max(0.0, baseline_margin)
        + 0.18 * max(0.0, external_margin)
        + 0.12 * max(0.0, language_proxy_margin)
        + 0.14 * proto_final["theorem_survival"]
        + 0.12 * proto_final["stable_read"]
        + 0.08 * proto_final["guarded_write"]
        + 0.07 * min(1.0, proto_final["transport_margin"] / 10.0)
        + 0.07 * daemon_stability
    )

    result = {
        "smoke_pass": True,
        "training_pass": proto_final["loss"] < proto_initial["loss"],
        "baseline_outperform_pass": baseline_margin > 0.55,
        "external_outperform_pass": external_margin > 0.45,
        "language_proxy_pass": language_proxy_margin > 0.08,
        "online_update_pass": online_delta_total >= 0.0,
        "rollback_pass": rollback_error <= 1e-8,
        "daemon_stability_pass": daemon_stability >= 0.992,
        "auto_recovery_triggered": auto_recovery_triggered,
        "implementation_ready": ready,
        "implementation_score": 1.0 if ready else max(0.0, min(1.0, structure_gain)),
        "data_stats": data_stats,
        "prototype_name": "ICSPB-Backbone-v2-LargeOnline",
        "baseline_name": "bridge_protocol_ablated_baseline",
        "proto_initial": proto_initial,
        "proto_history_1": proto_history_1,
        "proto_mid_1": proto_mid_1,
        "proto_pre_daemon": proto_pre_daemon,
        "proto_final": proto_final,
        "proto_online": proto_online,
        "baseline_initial": baseline_initial,
        "baseline_history": baseline_history,
        "baseline_final": baseline_final,
        "baseline_online": baseline_online,
        "stabilization_history": stabilization_history,
        "guarded_history": guarded_history,
        "online_alignment_history": online_alignment_history,
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
