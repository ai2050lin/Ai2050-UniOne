#!/usr/bin/env python
"""
Shared-atom causal benchmark for the unified encoding hypothesis.

This script tests whether the same low-level atoms jointly support:
1. concept decoding
2. relation decoding
3. noisy-input recovery

The goal is to move from correlation evidence to causal coupling evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]

FAMILIES = {
    "fruit": ["apple", "banana", "orange", "pear"],
    "animal": ["cat", "dog", "horse", "rabbit"],
    "tool": ["hammer", "wrench", "saw", "drill"],
    "abstract": ["truth", "logic", "memory", "justice"],
}
RELATIONS = ["uses", "contains", "guides", "opposes", "protects", "extends"]
FAMILY_RELATION_PRIORS = {
    "fruit": ["contains", "protects", "extends"],
    "animal": ["guides", "protects", "opposes"],
    "tool": ["uses", "extends", "protects"],
    "abstract": ["guides", "opposes", "extends"],
}
CONTEXT_BY_RELATION = {
    "uses": ["builds", "fixes", "holds"],
    "contains": ["inside", "basket", "cluster"],
    "guides": ["toward", "path", "signal"],
    "opposes": ["against", "contrast", "conflict"],
    "protects": ["shield", "guard", "safe"],
    "extends": ["longer", "chain", "bridge"],
}
MODIFIERS = ["bright", "silent", "small", "rapid", "stable", "novel", "shared", "distant"]
PHASES = ["phase_1", "phase_2", "phase_3"]
NOISE_TOKEN = "<noise>"
PAD_TOKEN = "<pad>"


def build_vocab() -> Dict[str, int]:
    words = [PAD_TOKEN, NOISE_TOKEN]
    words.extend(FAMILIES.keys())
    for concepts in FAMILIES.values():
        words.extend(concepts)
    words.extend(RELATIONS)
    for rows in CONTEXT_BY_RELATION.values():
        words.extend(rows)
    words.extend(MODIFIERS)
    words.extend(PHASES)
    words.extend(["links", "near", "with", "and", "through", "field", "kernel"])
    vocab = {}
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


VOCAB = build_vocab()
CONCEPTS = [concept for family in FAMILIES.values() for concept in family]
CONCEPT_TO_IDX = {concept: idx for idx, concept in enumerate(CONCEPTS)}
RELATION_TO_IDX = {relation: idx for idx, relation in enumerate(RELATIONS)}


def sample_sequence(rng: random.Random) -> Tuple[List[int], List[int], int, int]:
    family = rng.choice(list(FAMILIES.keys()))
    concept = rng.choice(FAMILIES[family])
    if rng.random() < 0.82:
        relation = rng.choice(FAMILY_RELATION_PRIORS[family])
    else:
        relation = rng.choice(RELATIONS)
    partner_family = rng.choice(list(FAMILIES.keys()))
    partner = rng.choice(FAMILIES[partner_family])
    context = rng.choice(CONTEXT_BY_RELATION[relation])
    modifier = rng.choice(MODIFIERS)
    phase = rng.choice(PHASES)
    clean_words = [
        family,
        concept,
        relation,
        partner,
        context,
        modifier,
        phase,
        "links",
        "through",
        "kernel",
    ]
    noisy_words = clean_words[:]
    for idx in range(len(noisy_words)):
        if rng.random() < 0.18:
            if rng.random() < 0.55:
                noisy_words[idx] = NOISE_TOKEN
            else:
                noisy_words[idx] = rng.choice(list(VOCAB.keys())[2:])
    clean_ids = [VOCAB[word] for word in clean_words]
    noisy_ids = [VOCAB[word] for word in noisy_words]
    return clean_ids, noisy_ids, CONCEPT_TO_IDX[concept], RELATION_TO_IDX[relation]


class SyntheticConceptRelationDataset(Dataset):
    def __init__(self, size: int, seed: int):
        rng = random.Random(seed)
        rows = [sample_sequence(rng) for _ in range(size)]
        self.clean = torch.tensor([row[0] for row in rows], dtype=torch.long)
        self.noisy = torch.tensor([row[1] for row in rows], dtype=torch.long)
        self.concept = torch.tensor([row[2] for row in rows], dtype=torch.long)
        self.relation = torch.tensor([row[3] for row in rows], dtype=torch.long)

    def __len__(self) -> int:
        return self.clean.shape[0]

    def __getitem__(self, idx: int):
        return self.clean[idx], self.noisy[idx], self.concept[idx], self.relation[idx]


def sparse_topk(coeffs: torch.Tensor, top_k: int) -> torch.Tensor:
    values, indices = coeffs.abs().topk(top_k, dim=-1)
    signed_values = coeffs.gather(-1, indices)
    sparse = torch.zeros_like(coeffs)
    sparse.scatter_(-1, indices, signed_values)
    return sparse


class SharedAtomModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dict_size: int, top_k: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.shared_atoms = nn.Parameter(torch.randn(dict_size, d_model) * 0.04)
        self.shared_router = nn.Linear(d_model, dict_size, bias=False)
        self.concept_delta = nn.Linear(d_model, dict_size, bias=False)
        self.relation_delta = nn.Linear(d_model, dict_size, bias=False)
        self.top_k = top_k
        self.concept_head = nn.Linear(d_model, len(CONCEPTS))
        self.relation_head = nn.Linear(d_model, len(RELATIONS))

    def pooled(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens).mean(dim=1)
        return self.encoder(x)

    def encode_with_atoms(self, pooled: torch.Tensor, head: str) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared_router(pooled)
        if head == "concept":
            coeffs = shared + 0.30 * self.concept_delta(pooled)
        else:
            coeffs = shared + 0.30 * self.relation_delta(pooled)
        sparse = sparse_topk(coeffs, self.top_k)
        rep = sparse @ self.shared_atoms
        return rep, sparse

    def forward(self, clean_tokens: torch.Tensor, noisy_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        clean_pooled = self.pooled(clean_tokens)
        noisy_pooled = self.pooled(noisy_tokens)

        concept_rep, concept_sparse = self.encode_with_atoms(clean_pooled, "concept")
        relation_rep, relation_sparse = self.encode_with_atoms(clean_pooled, "relation")
        noisy_concept_rep, noisy_concept_sparse = self.encode_with_atoms(noisy_pooled, "concept")
        noisy_relation_rep, noisy_relation_sparse = self.encode_with_atoms(noisy_pooled, "relation")

        fused_clean = 0.20 * clean_pooled + 0.90 * concept_rep + 0.90 * relation_rep
        fused_noisy = 0.20 * noisy_pooled + 0.90 * noisy_concept_rep + 0.90 * noisy_relation_rep
        return {
            "concept_logits": self.concept_head(fused_clean),
            "relation_logits": self.relation_head(fused_clean),
            "noisy_concept_logits": self.concept_head(fused_noisy),
            "noisy_relation_logits": self.relation_head(fused_noisy),
            "concept_sparse": concept_sparse,
            "relation_sparse": relation_sparse,
            "noisy_concept_sparse": noisy_concept_sparse,
            "noisy_relation_sparse": noisy_relation_sparse,
        }


class IndependentAtomModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dict_size: int, top_k: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.concept_atoms = nn.Parameter(torch.randn(dict_size, d_model) * 0.04)
        self.relation_atoms = nn.Parameter(torch.randn(dict_size, d_model) * 0.04)
        self.concept_encoder = nn.Linear(d_model, dict_size, bias=False)
        self.relation_encoder = nn.Linear(d_model, dict_size, bias=False)
        self.top_k = top_k
        self.concept_head = nn.Linear(d_model, len(CONCEPTS))
        self.relation_head = nn.Linear(d_model, len(RELATIONS))

    def pooled(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens).mean(dim=1)
        return self.encoder(x)

    def encode_with_atoms(self, pooled: torch.Tensor, head: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if head == "concept":
            coeffs = self.concept_encoder(pooled)
            sparse = sparse_topk(coeffs, self.top_k)
            rep = sparse @ self.concept_atoms
        else:
            coeffs = self.relation_encoder(pooled)
            sparse = sparse_topk(coeffs, self.top_k)
            rep = sparse @ self.relation_atoms
        return rep, sparse

    def forward(self, clean_tokens: torch.Tensor, noisy_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        clean_pooled = self.pooled(clean_tokens)
        noisy_pooled = self.pooled(noisy_tokens)

        concept_rep, concept_sparse = self.encode_with_atoms(clean_pooled, "concept")
        relation_rep, relation_sparse = self.encode_with_atoms(clean_pooled, "relation")
        noisy_concept_rep, noisy_concept_sparse = self.encode_with_atoms(noisy_pooled, "concept")
        noisy_relation_rep, noisy_relation_sparse = self.encode_with_atoms(noisy_pooled, "relation")

        fused_clean = 0.20 * clean_pooled + 0.90 * concept_rep + 0.90 * relation_rep
        fused_noisy = 0.20 * noisy_pooled + 0.90 * noisy_concept_rep + 0.90 * noisy_relation_rep
        return {
            "concept_logits": self.concept_head(fused_clean),
            "relation_logits": self.relation_head(fused_clean),
            "noisy_concept_logits": self.concept_head(fused_noisy),
            "noisy_relation_logits": self.relation_head(fused_noisy),
            "concept_sparse": concept_sparse,
            "relation_sparse": relation_sparse,
            "noisy_concept_sparse": noisy_concept_sparse,
            "noisy_relation_sparse": noisy_relation_sparse,
        }


class DenseBaselineModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
        self.concept_head = nn.Linear(d_model, len(CONCEPTS))
        self.relation_head = nn.Linear(d_model, len(RELATIONS))

    def pooled(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens).mean(dim=1)
        return self.encoder(x)

    def forward(self, clean_tokens: torch.Tensor, noisy_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        clean_pooled = self.pooled(clean_tokens)
        noisy_pooled = self.pooled(noisy_tokens)
        return {
            "concept_logits": self.concept_head(clean_pooled),
            "relation_logits": self.relation_head(clean_pooled),
            "noisy_concept_logits": self.concept_head(noisy_pooled),
            "noisy_relation_logits": self.relation_head(noisy_pooled),
        }


@dataclass
class TrainConfig:
    d_model: int = 64
    dict_size: int = 32
    top_k: int = 6
    batch_size: int = 64
    epochs: int = 18
    lr: float = 3e-3
    train_size: int = 6000
    val_size: int = 1200


def build_loaders(cfg: TrainConfig, seed: int) -> Tuple[DataLoader, DataLoader]:
    train_ds = SyntheticConceptRelationDataset(cfg.train_size, seed)
    val_ds = SyntheticConceptRelationDataset(cfg.val_size, seed + 1)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_model(model: nn.Module, train_loader: DataLoader, device: torch.device, cfg: TrainConfig) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    for _epoch in range(cfg.epochs):
        model.train()
        for clean_tokens, noisy_tokens, concept_target, relation_target in train_loader:
            clean_tokens = clean_tokens.to(device)
            noisy_tokens = noisy_tokens.to(device)
            concept_target = concept_target.to(device)
            relation_target = relation_target.to(device)

            out = model(clean_tokens, noisy_tokens)
            loss = (
                F.cross_entropy(out["concept_logits"], concept_target)
                + F.cross_entropy(out["relation_logits"], relation_target)
                + 0.8 * F.cross_entropy(out["noisy_concept_logits"], concept_target)
                + 0.8 * F.cross_entropy(out["noisy_relation_logits"], relation_target)
            )
            if "concept_sparse" in out:
                clean_cos = F.cosine_similarity(out["concept_sparse"].abs(), out["relation_sparse"].abs(), dim=-1).mean()
                noisy_cos = F.cosine_similarity(out["noisy_concept_sparse"].abs(), out["noisy_relation_sparse"].abs(), dim=-1).mean()
                loss = loss + 0.0008 * (
                    out["concept_sparse"].abs().mean()
                    + out["relation_sparse"].abs().mean()
                    + out["noisy_concept_sparse"].abs().mean()
                    + out["noisy_relation_sparse"].abs().mean()
                )
                loss = loss + 0.035 * ((1.0 - clean_cos) + 0.7 * (1.0 - noisy_cos))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    concept_ok = 0
    relation_ok = 0
    noisy_concept_ok = 0
    noisy_relation_ok = 0
    noisy_joint_ok = 0
    total = 0
    with torch.no_grad():
        for clean_tokens, noisy_tokens, concept_target, relation_target in val_loader:
            clean_tokens = clean_tokens.to(device)
            noisy_tokens = noisy_tokens.to(device)
            concept_target = concept_target.to(device)
            relation_target = relation_target.to(device)
            out = model(clean_tokens, noisy_tokens)
            concept_pred = out["concept_logits"].argmax(dim=-1)
            relation_pred = out["relation_logits"].argmax(dim=-1)
            noisy_concept_pred = out["noisy_concept_logits"].argmax(dim=-1)
            noisy_relation_pred = out["noisy_relation_logits"].argmax(dim=-1)
            concept_ok += int((concept_pred == concept_target).sum().item())
            relation_ok += int((relation_pred == relation_target).sum().item())
            noisy_concept_ok += int((noisy_concept_pred == concept_target).sum().item())
            noisy_relation_ok += int((noisy_relation_pred == relation_target).sum().item())
            noisy_joint_ok += int(((noisy_concept_pred == concept_target) & (noisy_relation_pred == relation_target)).sum().item())
            total += clean_tokens.shape[0]
    return {
        "concept_accuracy": float(concept_ok / max(1, total)),
        "relation_accuracy": float(relation_ok / max(1, total)),
        "noisy_concept_accuracy": float(noisy_concept_ok / max(1, total)),
        "noisy_relation_accuracy": float(noisy_relation_ok / max(1, total)),
        "recovery_accuracy": float(noisy_joint_ok / max(1, total)),
    }


def collect_usage(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    concept_usage = []
    relation_usage = []
    with torch.no_grad():
        for clean_tokens, noisy_tokens, _concept_target, _relation_target in val_loader:
            clean_tokens = clean_tokens.to(device)
            noisy_tokens = noisy_tokens.to(device)
            out = model(clean_tokens, noisy_tokens)
            if "concept_sparse" not in out:
                continue
            concept_usage.append(out["concept_sparse"].abs().mean(dim=0).cpu().numpy())
            relation_usage.append(out["relation_sparse"].abs().mean(dim=0).cpu().numpy())
    if not concept_usage:
        return {"concept_usage": np.zeros(0), "relation_usage": np.zeros(0)}
    return {
        "concept_usage": np.mean(np.stack(concept_usage, axis=0), axis=0),
        "relation_usage": np.mean(np.stack(relation_usage, axis=0), axis=0),
    }


def atom_indices_by_mode(concept_usage: np.ndarray, relation_usage: np.ndarray, count: int) -> Dict[str, List[int]]:
    shared_score = np.sqrt(np.maximum(concept_usage, 0.0) * np.maximum(relation_usage, 0.0))
    shared_top = np.argsort(shared_score)[::-1][:count].tolist()
    concept_only = [idx for idx in np.argsort(concept_usage - relation_usage)[::-1].tolist() if idx not in shared_top][:count]
    relation_only = [idx for idx in np.argsort(relation_usage - concept_usage)[::-1].tolist() if idx not in shared_top][:count]
    pool = [idx for idx in range(concept_usage.shape[0]) if idx not in shared_top[: count // 2]]
    rng = random.Random(7)
    rng.shuffle(pool)
    random_control = pool[:count]
    return {
        "shared_top": shared_top,
        "concept_only": concept_only,
        "relation_only": relation_only,
        "random_control": random_control,
    }


def apply_unified_ablation(model: SharedAtomModel, indices: List[int]) -> SharedAtomModel:
    ablated = copy.deepcopy(model)
    with torch.no_grad():
        ablated.shared_atoms[indices] = 0.0
    return ablated


def apply_independent_ablation(model: IndependentAtomModel, concept_indices: List[int], relation_indices: List[int]) -> IndependentAtomModel:
    ablated = copy.deepcopy(model)
    with torch.no_grad():
        ablated.concept_atoms[concept_indices] = 0.0
        ablated.relation_atoms[relation_indices] = 0.0
    return ablated


def summarize_drop(base_metrics: Dict[str, float], ablated_metrics: Dict[str, float]) -> Dict[str, float]:
    concept_drop = float(base_metrics["concept_accuracy"] - ablated_metrics["concept_accuracy"])
    relation_drop = float(base_metrics["relation_accuracy"] - ablated_metrics["relation_accuracy"])
    recovery_drop = float(base_metrics["recovery_accuracy"] - ablated_metrics["recovery_accuracy"])
    joint_drop = float((concept_drop + relation_drop + recovery_drop) / 3.0)
    coupled_drop = float(min(concept_drop, relation_drop) + 0.5 * recovery_drop)
    return {
        "concept_drop": concept_drop,
        "relation_drop": relation_drop,
        "recovery_drop": recovery_drop,
        "joint_drop": joint_drop,
        "coupled_drop": coupled_drop,
    }


def run_benchmark(cfg: TrainConfig, seed: int, device: torch.device) -> Dict[str, object]:
    train_loader, val_loader = build_loaders(cfg, seed)

    unified = SharedAtomModel(len(VOCAB), cfg.d_model, cfg.dict_size, cfg.top_k).to(device)
    independent = IndependentAtomModel(len(VOCAB), cfg.d_model, cfg.dict_size, cfg.top_k).to(device)
    baseline = DenseBaselineModel(len(VOCAB), cfg.d_model).to(device)

    train_one_model(unified, train_loader, device, cfg)
    train_one_model(independent, train_loader, device, cfg)
    train_one_model(baseline, train_loader, device, cfg)

    metrics_unified = evaluate_model(unified, val_loader, device)
    metrics_independent = evaluate_model(independent, val_loader, device)
    metrics_baseline = evaluate_model(baseline, val_loader, device)

    usage_unified = collect_usage(unified, val_loader, device)
    usage_independent = collect_usage(independent, val_loader, device)
    ablate_count = min(cfg.dict_size // 3, max(cfg.top_k * 2, cfg.top_k + 2))
    unified_groups = atom_indices_by_mode(usage_unified["concept_usage"], usage_unified["relation_usage"], ablate_count)
    independent_groups = atom_indices_by_mode(usage_independent["concept_usage"], usage_independent["relation_usage"], ablate_count)

    unified_ablations = {}
    for name, indices in unified_groups.items():
        metrics = evaluate_model(apply_unified_ablation(unified, indices).to(device), val_loader, device)
        unified_ablations[name] = {
            "indices": indices,
            "metrics": metrics,
            "drops": summarize_drop(metrics_unified, metrics),
        }

    half = max(1, ablate_count // 2)
    independent_ablations = {}
    paired_metrics = evaluate_model(
        apply_independent_ablation(
            independent,
            independent_groups["concept_only"][:half],
            independent_groups["relation_only"][:half],
        ).to(device),
        val_loader,
        device,
    )
    independent_ablations["paired_dual_topk"] = {
        "concept_indices": independent_groups["concept_only"][:half],
        "relation_indices": independent_groups["relation_only"][:half],
        "metrics": paired_metrics,
        "drops": summarize_drop(metrics_independent, paired_metrics),
    }
    random_metrics = evaluate_model(
        apply_independent_ablation(
            independent,
            independent_groups["random_control"][:half],
            independent_groups["random_control"][half : half * 2],
        ).to(device),
        val_loader,
        device,
    )
    independent_ablations["random_control"] = {
        "concept_indices": independent_groups["random_control"][:half],
        "relation_indices": independent_groups["random_control"][half : half * 2],
        "metrics": random_metrics,
        "drops": summarize_drop(metrics_independent, random_metrics),
    }

    concept_usage = usage_unified["concept_usage"]
    relation_usage = usage_unified["relation_usage"]
    cross_dim_corr = float(np.corrcoef(concept_usage, relation_usage)[0, 1]) if concept_usage.size > 1 else 0.0
    top_overlap = len(set(unified_groups["shared_top"]) & set(np.argsort(concept_usage)[::-1][: cfg.top_k].tolist())) / max(1, cfg.top_k)

    hypotheses = {
        "H1_shared_ablation_beats_random_joint_drop": bool(
            unified_ablations["shared_top"]["drops"]["joint_drop"] > unified_ablations["random_control"]["drops"]["joint_drop"] + 0.04
        ),
        "H2_shared_ablation_couples_concept_and_relation": bool(
            min(
                unified_ablations["shared_top"]["drops"]["concept_drop"],
                unified_ablations["shared_top"]["drops"]["relation_drop"],
            )
            > 0.10
        ),
        "H3_shared_ablation_hits_recovery": bool(unified_ablations["shared_top"]["drops"]["recovery_drop"] > 0.08),
        "H4_unified_joint_drop_beats_independent_paired": bool(
            unified_ablations["shared_top"]["drops"]["joint_drop"] > independent_ablations["paired_dual_topk"]["drops"]["joint_drop"] + 0.03
        ),
    }

    return {
        "config": cfg.__dict__,
        "systems": {
            "unified_shared_atoms": {
                "metrics": metrics_unified,
                "usage": {
                    "concept_usage": [float(v) for v in concept_usage.tolist()],
                    "relation_usage": [float(v) for v in relation_usage.tolist()],
                    "cross_dim_corr": cross_dim_corr,
                    "top_overlap_ratio": float(top_overlap),
                },
                "ablations": unified_ablations,
            },
            "independent_atoms": {
                "metrics": metrics_independent,
                "usage": {
                    "concept_usage": [float(v) for v in usage_independent["concept_usage"].tolist()],
                    "relation_usage": [float(v) for v in usage_independent["relation_usage"].tolist()],
                },
                "ablations": independent_ablations,
            },
            "dense_baseline": {
                "metrics": metrics_baseline,
            },
        },
        "headline_metrics": {
            "unified_clean_concept_accuracy": metrics_unified["concept_accuracy"],
            "unified_clean_relation_accuracy": metrics_unified["relation_accuracy"],
            "unified_recovery_accuracy": metrics_unified["recovery_accuracy"],
            "shared_joint_drop": unified_ablations["shared_top"]["drops"]["joint_drop"],
            "shared_recovery_drop": unified_ablations["shared_top"]["drops"]["recovery_drop"],
            "cross_dim_corr": cross_dim_corr,
        },
        "gains": {
            "shared_vs_random_joint_drop": float(
                unified_ablations["shared_top"]["drops"]["joint_drop"] - unified_ablations["random_control"]["drops"]["joint_drop"]
            ),
            "shared_vs_concept_only_joint_drop": float(
                unified_ablations["shared_top"]["drops"]["joint_drop"] - unified_ablations["concept_only"]["drops"]["joint_drop"]
            ),
            "shared_vs_relation_only_joint_drop": float(
                unified_ablations["shared_top"]["drops"]["joint_drop"] - unified_ablations["relation_only"]["drops"]["joint_drop"]
            ),
            "shared_vs_independent_paired_joint_drop": float(
                unified_ablations["shared_top"]["drops"]["joint_drop"] - independent_ablations["paired_dual_topk"]["drops"]["joint_drop"]
            ),
            "unified_vs_independent_recovery": float(
                metrics_unified["recovery_accuracy"] - metrics_independent["recovery_accuracy"]
            ),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": "这一版不再停在共享字典相关性，而是直接做共享原子因果消融，检测同一批低层原子是否会同时打掉概念解码、关系解码和噪声恢复。",
            "next_question": "如果共享原子消融确实造成多维联动下跌，下一步就该把这种同源性因果结构接回真实模型和真实工具闭环。",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Shared atom causal benchmark for unified encoding")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--dict-size", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--train-size", type=int, default=6000)
    ap.add_argument("--val-size", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/shared_atom_causal_unification_benchmark_20260310.json")
    args = ap.parse_args()

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

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
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
    print(json.dumps(payload["gains"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
