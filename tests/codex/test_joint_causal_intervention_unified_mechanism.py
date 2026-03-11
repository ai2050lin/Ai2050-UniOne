#!/usr/bin/env python
"""
Run a first joint causal intervention benchmark in one unified environment.

The benchmark keeps shared support, stage gating, recovery memory, and online
success in the same toy chain so we can ask a stricter question:
when we perturb candidate "same-family" components together, do
representation, routing, recovery, and chain success fall together?
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from test_shared_atom_causal_unification_benchmark import (
    SharedAtomModel,
    TrainConfig,
    apply_unified_ablation,
    atom_indices_by_mode,
    build_loaders,
    collect_usage,
    train_one_model,
)


ROOT = Path(__file__).resolve().parents[2]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def top_margin(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def sparse_overlap(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    numer = torch.minimum(a.abs(), b.abs()).sum(dim=-1)
    denom = torch.maximum(a.abs(), b.abs()).sum(dim=-1) + eps
    return numer / denom


def build_model(cfg: TrainConfig, seed: int, device: torch.device) -> tuple[SharedAtomModel, Any]:
    train_loader, val_loader = build_loaders(cfg, seed)
    model = SharedAtomModel(len(__import__("test_shared_atom_causal_unification_benchmark").VOCAB), cfg.d_model, cfg.dict_size, cfg.top_k).to(device)
    train_one_model(model, train_loader, device, cfg)
    return model, val_loader


def intervention_cases(shared_top: List[int], random_control: List[int]) -> Dict[str, Dict[str, Any]]:
    return {
        "baseline": {
            "ablate_indices": [],
            "relation_gate_scale": 1.0,
            "recovery_gate_scale": 1.0,
            "recovery_memory_scale": 1.0,
        },
        "random_atom_control": {
            "ablate_indices": random_control,
            "relation_gate_scale": 1.0,
            "recovery_gate_scale": 1.0,
            "recovery_memory_scale": 1.0,
        },
        "shared_support_only": {
            "ablate_indices": shared_top,
            "relation_gate_scale": 1.0,
            "recovery_gate_scale": 1.0,
            "recovery_memory_scale": 1.0,
        },
        "stage_gate_only": {
            "ablate_indices": [],
            "relation_gate_scale": 0.0,
            "recovery_gate_scale": 0.0,
            "recovery_memory_scale": 1.0,
        },
        "recovery_node_only": {
            "ablate_indices": [],
            "relation_gate_scale": 1.0,
            "recovery_gate_scale": 1.0,
            "recovery_memory_scale": 0.0,
        },
        "joint_shared_gate_recovery": {
            "ablate_indices": shared_top,
            "relation_gate_scale": 0.0,
            "recovery_gate_scale": 0.0,
            "recovery_memory_scale": 0.0,
        },
    }


def simulate_chain(
    model: SharedAtomModel,
    val_loader: Any,
    device: torch.device,
    case: Dict[str, Any],
) -> Dict[str, float]:
    work_model = model
    if case["ablate_indices"]:
        work_model = apply_unified_ablation(model, list(case["ablate_indices"])).to(device)
    work_model.eval()

    representation_rows: List[float] = []
    topology_rows: List[float] = []
    concept_rows: List[float] = []
    relation_rows: List[float] = []
    recovery_rows: List[float] = []
    online_rows: List[float] = []
    tool_rows: List[float] = []
    verify_rows: List[float] = []
    relation_gate_rows: List[float] = []
    recovery_gate_rows: List[float] = []

    with torch.no_grad():
        for clean_tokens, noisy_tokens, concept_target, relation_target in val_loader:
            clean_tokens = clean_tokens.to(device)
            noisy_tokens = noisy_tokens.to(device)
            concept_target = concept_target.to(device)
            relation_target = relation_target.to(device)

            out = work_model(clean_tokens, noisy_tokens)
            clean_concept_probs = torch.softmax(out["concept_logits"], dim=-1)
            clean_relation_probs = torch.softmax(out["relation_logits"], dim=-1)
            noisy_concept_probs = torch.softmax(out["noisy_concept_logits"], dim=-1)
            noisy_relation_probs = torch.softmax(out["noisy_relation_logits"], dim=-1)

            concept_margin = top_margin(out["concept_logits"])
            relation_margin = top_margin(out["relation_logits"])
            noisy_concept_margin = top_margin(out["noisy_concept_logits"])
            noisy_relation_margin = top_margin(out["noisy_relation_logits"])

            shared_signal_clean = sparse_overlap(out["concept_sparse"], out["relation_sparse"])
            shared_signal_noisy = sparse_overlap(out["noisy_concept_sparse"], out["noisy_relation_sparse"])
            shared_signal = 0.5 * (shared_signal_clean + shared_signal_noisy)

            disagreement = 0.5 * (
                torch.abs(clean_concept_probs - noisy_concept_probs).sum(dim=-1)
                + torch.abs(clean_relation_probs - noisy_relation_probs).sum(dim=-1)
            )

            # Gates are local functions of current confidence and disagreement.
            relation_gate = torch.sigmoid(8.0 * (shared_signal + concept_margin - 0.55))
            recovery_gate = torch.sigmoid(7.0 * (shared_signal + disagreement - 0.90))
            relation_gate = case["relation_gate_scale"] * relation_gate
            recovery_gate = case["recovery_gate_scale"] * recovery_gate

            relation_stage_probs = (
                relation_gate.unsqueeze(-1) * clean_relation_probs
                + (1.0 - relation_gate.unsqueeze(-1)) * noisy_relation_probs
            )
            recovered_concept_probs = (
                noisy_concept_probs
                + case["recovery_memory_scale"] * recovery_gate.unsqueeze(-1) * (clean_concept_probs - noisy_concept_probs)
            )
            recovered_relation_probs = (
                relation_stage_probs
                + case["recovery_memory_scale"] * recovery_gate.unsqueeze(-1) * (clean_relation_probs - relation_stage_probs)
            )

            concept_pred = clean_concept_probs.argmax(dim=-1)
            relation_pred = relation_stage_probs.argmax(dim=-1)
            recovered_concept_pred = recovered_concept_probs.argmax(dim=-1)
            recovered_relation_pred = recovered_relation_probs.argmax(dim=-1)

            concept_ok = (concept_pred == concept_target).float()
            relation_ok = (relation_pred == relation_target).float()
            recovery_ok = ((recovered_concept_pred == concept_target) & (recovered_relation_pred == relation_target)).float()

            route_alignment = 0.5 * (relation_gate + recovery_gate)
            confidence_floor = torch.minimum(
                recovered_concept_probs.max(dim=-1).values,
                recovered_relation_probs.max(dim=-1).values,
            )
            tool_ok = (
                recovery_ok
                * (route_alignment > 0.38).float()
                * (confidence_floor > 0.52).float()
            )
            verify_ok = (
                recovery_ok
                * ((0.55 * confidence_floor + 0.25 * concept_margin + 0.20 * noisy_relation_margin) > 0.46).float()
            )
            online_ok = tool_ok * verify_ok

            representation_signal = (
                0.40 * shared_signal
                + 0.25 * concept_margin
                + 0.20 * relation_margin
                + 0.15 * noisy_concept_margin
            )

            representation_rows.extend(representation_signal.detach().cpu().tolist())
            topology_rows.extend(route_alignment.detach().cpu().tolist())
            concept_rows.extend(concept_ok.detach().cpu().tolist())
            relation_rows.extend(relation_ok.detach().cpu().tolist())
            recovery_rows.extend(recovery_ok.detach().cpu().tolist())
            online_rows.extend(online_ok.detach().cpu().tolist())
            tool_rows.extend(tool_ok.detach().cpu().tolist())
            verify_rows.extend(verify_ok.detach().cpu().tolist())
            relation_gate_rows.extend(relation_gate.detach().cpu().tolist())
            recovery_gate_rows.extend(recovery_gate.detach().cpu().tolist())

    return {
        "representation_score": mean(representation_rows),
        "topology_score": mean(topology_rows),
        "concept_accuracy": mean(concept_rows),
        "relation_accuracy": mean(relation_rows),
        "recovery_accuracy": mean(recovery_rows),
        "tool_success_rate": mean(tool_rows),
        "verify_success_rate": mean(verify_rows),
        "online_success_rate": mean(online_rows),
        "mean_relation_gate": mean(relation_gate_rows),
        "mean_recovery_gate": mean(recovery_gate_rows),
    }


def summarize_drop(base_metrics: Dict[str, float], metrics: Dict[str, float]) -> Dict[str, float]:
    representation_drop = float(base_metrics["representation_score"] - metrics["representation_score"])
    topology_drop = float(base_metrics["topology_score"] - metrics["topology_score"])
    recovery_drop = float(base_metrics["recovery_accuracy"] - metrics["recovery_accuracy"])
    online_drop = float(base_metrics["online_success_rate"] - metrics["online_success_rate"])
    relation_drop = float(base_metrics["relation_accuracy"] - metrics["relation_accuracy"])
    return {
        "representation_drop": representation_drop,
        "topology_drop": topology_drop,
        "relation_drop": relation_drop,
        "recovery_drop": recovery_drop,
        "online_drop": online_drop,
        "joint_drop": float(mean([representation_drop, topology_drop, recovery_drop, online_drop])),
        "coupled_drop": float(min(representation_drop, topology_drop) + 0.5 * (recovery_drop + online_drop)),
    }


def run_benchmark(cfg: TrainConfig, seed: int, device: torch.device) -> Dict[str, Any]:
    model, val_loader = build_model(cfg, seed, device)
    usage = collect_usage(model, val_loader, device)
    ablate_count = min(cfg.dict_size // 3, max(cfg.top_k * 2, cfg.top_k + 2))
    groups = atom_indices_by_mode(usage["concept_usage"], usage["relation_usage"], ablate_count)
    cases = intervention_cases(groups["shared_top"], groups["random_control"])

    results = {}
    for name, case in cases.items():
        metrics = simulate_chain(model, val_loader, device, case)
        results[name] = {
            "config": {
                "ablate_indices": list(case["ablate_indices"]),
                "relation_gate_scale": float(case["relation_gate_scale"]),
                "recovery_gate_scale": float(case["recovery_gate_scale"]),
                "recovery_memory_scale": float(case["recovery_memory_scale"]),
            },
            "metrics": metrics,
        }

    base_metrics = results["baseline"]["metrics"]
    for name, row in results.items():
        row["drops"] = summarize_drop(base_metrics, row["metrics"])

    hypotheses = {
        "H1_shared_support_hits_representation_and_recovery": bool(
            results["shared_support_only"]["drops"]["representation_drop"] > 0.05
            and results["shared_support_only"]["drops"]["recovery_drop"] > 0.08
        ),
        "H2_stage_gate_hits_topology_and_online_chain": bool(
            results["stage_gate_only"]["drops"]["topology_drop"] > 0.20
            and results["stage_gate_only"]["drops"]["online_drop"] > 0.10
        ),
        "H3_recovery_node_hits_recovery_more_than_relation": bool(
            results["recovery_node_only"]["drops"]["recovery_drop"]
            > results["recovery_node_only"]["drops"]["relation_drop"] + 0.05
        ),
        "H4_joint_intervention_beats_single_interventions": bool(
            results["joint_shared_gate_recovery"]["drops"]["joint_drop"]
            > max(
                results["shared_support_only"]["drops"]["joint_drop"],
                results["stage_gate_only"]["drops"]["joint_drop"],
                results["recovery_node_only"]["drops"]["joint_drop"],
            )
            + 0.05
        ),
        "H5_joint_intervention_beats_random_atom_control": bool(
            results["joint_shared_gate_recovery"]["drops"]["joint_drop"]
            > results["random_atom_control"]["drops"]["joint_drop"] + 0.10
        ),
        "H6_joint_intervention_causes_synchronized_collapse": bool(
            min(
                results["joint_shared_gate_recovery"]["drops"]["representation_drop"],
                results["joint_shared_gate_recovery"]["drops"]["topology_drop"],
                results["joint_shared_gate_recovery"]["drops"]["recovery_drop"],
                results["joint_shared_gate_recovery"]["drops"]["online_drop"],
            )
            > 0.08
        ),
    }

    return {
        "config": {
            "train": cfg.__dict__,
            "ablate_count": int(ablate_count),
            "shared_top_indices": groups["shared_top"],
            "random_control_indices": groups["random_control"],
        },
        "cases": results,
        "headline_metrics": {
            "baseline_online_success": float(base_metrics["online_success_rate"]),
            "shared_support_online_drop": float(results["shared_support_only"]["drops"]["online_drop"]),
            "stage_gate_online_drop": float(results["stage_gate_only"]["drops"]["online_drop"]),
            "recovery_node_online_drop": float(results["recovery_node_only"]["drops"]["online_drop"]),
            "joint_online_drop": float(results["joint_shared_gate_recovery"]["drops"]["online_drop"]),
            "joint_joint_drop": float(results["joint_shared_gate_recovery"]["drops"]["joint_drop"]),
            "joint_coupled_drop": float(results["joint_shared_gate_recovery"]["drops"]["coupled_drop"]),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "This first same-environment benchmark no longer asks whether shared atoms, gates, and recovery exist "
                "separately. It asks whether perturbing them in one chain makes representation, routing, recovery, "
                "and final success fail together."
            ),
            "next_question": (
                "If the joint intervention synchronizes collapse in one environment, the next step is to map the same "
                "intervention template onto a real-model proxy so that layer support, protocol routing, and recovery "
                "fragility are perturbed together."
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Joint causal intervention benchmark for the unified mechanism")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--dict-size", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--train-size", type=int, default=6000)
    ap.add_argument("--val-size", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/joint_causal_intervention_unified_mechanism_20260311.json",
    )
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    cfg = TrainConfig(
        d_model=int(args.d_model),
        dict_size=int(args.dict_size),
        top_k=int(args.top_k),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        train_size=int(args.train_size),
        val_size=int(args.val_size),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    payload = run_benchmark(cfg, int(args.seed), device)
    payload["meta"] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "runtime_sec": float(time.time() - t0),
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
